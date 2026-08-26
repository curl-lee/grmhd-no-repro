#!/usr/bin/env python
"""Decompose frozen H0 by channel, direction, radial shell, and angular region."""

from __future__ import annotations

from collections import defaultdict
import math
import statistics
from typing import Any, Mapping

import torch

from grmhd import CHANNELS
from grmhd.paper_config import (
    build_paper_training_loss,
    load_paper_experiment_config,
)
from grmhd.paper_h1_diagnostics import additive_region_rows, h1_diagnostic
from grmhd.paper_stage_g import tensor_state_sha256
from grmhd.paper_stage_h import (
    FULL_CONFIG_PATH,
    MODEL_STATES,
    OUTPUT_ROOT,
    collate_fixed_pair,
    fixed_samples,
    forward_fixed_pair,
    freeze_or_validate_inputs,
    load_coordinates,
    load_model_state,
    make_fixed_datasets,
    make_processor,
    project_root_from_file,
    require_cuda_device,
    write_csv,
    write_json,
)


def region_masks(
    shells: torch.Tensor,
    theta: torch.Tensor,
) -> dict[str, torch.Tensor]:
    shell_masks = shells[0].bool()
    masks = {f"shell_{index}": shell_masks[index] for index in range(8)}
    masks.update(
        {
            "inner_two": shell_masks[0] | shell_masks[1],
            "middle_four": torch.any(shell_masks[2:6], dim=0),
            "outer_two": shell_masks[6] | shell_masks[7],
        }
    )
    theta = theta.to(device=shells.device, dtype=shells.dtype)
    equatorial_1d = torch.abs(theta - math.pi / 2) <= math.pi / 6
    polar_1d = (theta <= math.pi / 6) | (theta >= 5 * math.pi / 6)
    spatial_shape = tuple(shell_masks.shape[1:])
    equatorial = equatorial_1d[None, :, None].expand(spatial_shape)
    polar = polar_1d[None, :, None].expand(spatial_shape)
    masks.update(
        {
            "equatorial_band": equatorial,
            "polar_caps": polar,
            "remaining_angular": ~(equatorial | polar),
            "all": torch.ones(spatial_shape, dtype=torch.bool, device=shells.device),
        }
    )
    return masks


def envelope_violation_mask(
    state: torch.Tensor,
    baseline: torch.Tensor,
    *,
    delta_rho: float,
    delta_press: float,
) -> torch.Tensor:
    residual = state - baseline
    return (residual[:, 3].abs() > delta_rho) | (
        residual[:, 4].abs() > delta_press
    )


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def main() -> None:
    root = project_root_from_file()
    output_root = root / OUTPUT_ROOT
    device = require_cuda_device()
    frozen = freeze_or_validate_inputs(root, output_root)
    full_config = load_paper_experiment_config(
        root / FULL_CONFIG_PATH, project_root=root
    )
    full_loss = build_paper_training_loss(full_config).to(device).eval()
    coordinates = {
        name: value.to(device=device)
        for name, value in load_coordinates(full_config).items()
    }
    datasets = make_fixed_datasets(root)

    rows: list[dict[str, Any]] = []
    case_summaries: list[dict[str, Any]] = []
    state_hashes: dict[str, Any] = {}
    for spec in MODEL_STATES:
        model, epoch, metadata, before_hash = load_model_state(
            root, spec, device=device
        )
        processor = make_processor(root, device=device, epoch=epoch)
        masks = region_masks(processor.shells, coordinates["theta"])
        with torch.no_grad():
            for split, source, target in fixed_samples():
                batch = collate_fixed_pair(
                    datasets, split=split, source=source, target=target
                )
                prediction, fields = forward_fixed_pair(
                    model=model, processor=processor, batch=batch
                )
                states = {
                    "persistence_error": fields["normalized_input"],
                    "model_error": prediction,
                }
                for error_case, state in states.items():
                    with torch.enable_grad():
                        error = (
                            state - fields["normalized_target"]
                        ).detach().requires_grad_(True)
                        diagnostic = h1_diagnostic(
                            error, variant="H0_current_upstream"
                        )
                        gradient = torch.autograd.grad(
                            diagnostic.total, error, create_graph=False
                        )[0]
                    target_clamp = (
                        fields["normalized_target"].abs()
                        > float(full_loss.gamma * full_loss.inverse_clamp_fraction)
                    )
                    envelope = envelope_violation_mask(
                        state,
                        fields["radial_baseline_normalized"],
                        delta_rho=full_loss.envelope.delta_rho,
                        delta_press=full_loss.envelope.delta_press,
                    )
                    overlap_masks = {
                        "target_clamp": target_clamp[0],
                        "roi": fields["canonical_roi_mask"][0],
                        "envelope_violation": envelope[0],
                    }
                    decomposition = additive_region_rows(
                        diagnostic,
                        masks,
                        channel_names=CHANNELS,
                        gradient_density=gradient.square(),
                        overlap_masks=overlap_masks,
                    )
                    for row in decomposition:
                        row = {
                            key: (
                                float(value.detach())
                                if torch.is_tensor(value)
                                else value
                            )
                            for key, value in row.items()
                        }
                        row.update(
                            {
                                "model_state": spec.name,
                                "model_mode": spec.mode,
                                "checkpoint_epoch": epoch,
                                "split": split,
                                "source_snapshot": source,
                                "target_snapshot": target,
                                "error_case": error_case,
                            }
                        )
                        rows.append(row)
                    shell_total = sum(
                        item["contribution_fraction"]
                        for item in decomposition
                        if item["region"].startswith("shell_")
                        and item["channel"] == "all"
                        and item["direction"] == "all"
                    )
                    grouped_total = sum(
                        item["contribution_fraction"]
                        for item in decomposition
                        if item["region"] in {"inner_two", "middle_four", "outer_two"}
                        and item["channel"] == "all"
                        and item["direction"] == "all"
                    )
                    angular_total = sum(
                        item["contribution_fraction"]
                        for item in decomposition
                        if item["region"]
                        in {"equatorial_band", "polar_caps", "remaining_angular"}
                        and item["channel"] == "all"
                        and item["direction"] == "all"
                    )
                    one = diagnostic.total.new_tensor(1.0)
                    for name, value in (
                        ("shell", shell_total),
                        ("grouped radial", grouped_total),
                        ("angular", angular_total),
                    ):
                        if not torch.isclose(value, one, rtol=2.0e-6, atol=2.0e-6):
                            raise RuntimeError(
                                f"{name} H1 contributions do not add to one"
                            )
                    total_direction = sum(diagnostic.per_direction.values())
                    case_summaries.append(
                        {
                            "model_state": spec.name,
                            "model_mode": spec.mode,
                            "checkpoint_epoch": epoch,
                            "split": split,
                            "source_snapshot": source,
                            "target_snapshot": target,
                            "error_case": error_case,
                            "h0_total": float(diagnostic.total.detach()),
                            "per_channel": {
                                channel: float(value.detach())
                                for channel, value in zip(
                                    CHANNELS, diagnostic.per_channel
                                )
                            },
                            "per_direction_fraction": {
                                name: float(
                                    (
                                        value
                                        / total_direction.clamp_min(1.0e-30)
                                    ).detach()
                                )
                                for name, value in diagnostic.per_direction.items()
                            },
                            "shell_sum": float(shell_total.detach()),
                            "grouped_radial_sum": float(grouped_total.detach()),
                            "angular_sum": float(angular_total.detach()),
                        }
                    )
        after_hash = tensor_state_sha256(model.state_dict())
        if after_hash != before_hash:
            raise RuntimeError(f"Shell audit modified model state {spec.name}")
        state_hashes[spec.name] = {
            "before": before_hash,
            "after": after_hash,
            "unchanged": True,
            "epoch": epoch,
            "checkpoint_mode": metadata.get("mode"),
        }
        del processor, model
        torch.cuda.empty_cache()

    aggregate_rows = [
        row
        for row in rows
        if row["channel"] == "all" and row["direction"] == "all"
    ]
    model_error_rows = [
        row for row in aggregate_rows if row["error_case"] == "model_error"
    ]
    inner_rows = [row for row in model_error_rows if row["region"] == "inner_two"]
    full_inner = [
        float(row["contribution_fraction"])
        for row in inner_rows
        if row["model_mode"] == "full"
    ]
    plain_inner = [
        float(row["contribution_fraction"])
        for row in inner_rows
        if row["model_mode"] == "plain"
    ]
    full_inner_density = [
        float(row["per_voxel_mean"])
        for row in inner_rows
        if row["model_mode"] == "full"
    ]
    plain_inner_density = [
        float(row["per_voxel_mean"])
        for row in inner_rows
        if row["model_mode"] == "plain"
    ]
    full_inner_gradient = [
        float(row["gradient_norm_fraction"])
        for row in inner_rows
        if row["model_mode"] == "full"
    ]
    plain_inner_gradient = [
        float(row["gradient_norm_fraction"])
        for row in inner_rows
        if row["model_mode"] == "plain"
    ]
    all_rows = [row for row in model_error_rows if row["region"] == "all"]
    overlap_means = {
        key: mean([float(row[key]) for row in all_rows])
        for key in (
            "target_clamp_h1_overlap",
            "roi_h1_overlap",
            "envelope_violation_h1_overlap",
        )
    }
    direction_fractions = {
        direction: mean(
            [
                float(item["per_direction_fraction"][direction])
                for item in case_summaries
                if item["error_case"] == "model_error"
            ]
        )
        for direction in ("phi", "theta", "r")
    }
    channel_values = {
        channel: mean(
            [
                float(item["per_channel"][channel])
                for item in case_summaries
                if item["error_case"] == "model_error"
            ]
        )
        for channel in CHANNELS
    }
    dominant_direction = max(direction_fractions, key=direction_fractions.get)
    dominant_channels = sorted(
        channel_values, key=channel_values.get, reverse=True
    )
    answers = {
        "inner_two_contribution_fraction_mean": mean(
            [float(row["contribution_fraction"]) for row in inner_rows]
        ),
        "inner_two_voxel_fraction": mean(
            [float(row["voxel_fraction"]) for row in inner_rows]
        ),
        "inner_two_enrichment_mean": mean(
            [float(row["enrichment"]) for row in inner_rows]
        ),
        "full_inner_two_contribution_mean": mean(full_inner),
        "plain_inner_two_contribution_mean": mean(plain_inner),
        "full_inner_two_per_voxel_h1_mean": mean(full_inner_density),
        "plain_inner_two_per_voxel_h1_mean": mean(plain_inner_density),
        "full_inner_two_gradient_fraction_mean": mean(full_inner_gradient),
        "plain_inner_two_gradient_fraction_mean": mean(plain_inner_gradient),
        "dominant_direction": dominant_direction,
        "direction_fraction_means": direction_fractions,
        "dominant_channels_descending": dominant_channels,
        "channel_h0_value_means": channel_values,
        "global_h1_overlap_means": overlap_means,
        "derivative_shell_attribution": (
            "central-cell midpoint attribution: each periodic centered derivative "
            "density is assigned to the shell containing its evaluation cell"
        ),
    }
    output = {
        "schema_version": "paper-stage-h-h1-shell-decomposition-v1",
        "status": "completed",
        "frozen_inputs": frozen,
        "device": torch.cuda.get_device_name(device),
        "fixed_model_states": state_hashes,
        "case_summaries": case_summaries,
        "rows": rows,
        "answers": answers,
        "region_definitions": {
            "inner_two": "shells 0-1",
            "middle_four": "shells 2-5",
            "outer_two": "shells 6-7",
            "equatorial_band": "|theta-pi/2| <= pi/6",
            "polar_caps": "theta <= pi/6 or theta >= 5pi/6",
            "remaining_angular": "complement of equatorial band and polar caps",
            "attribution": answers["derivative_shell_attribution"],
        },
        "safety": {
            "eval_mode": True,
            "mixed_precision": False,
            "batch_size": 1,
            "prediction_space_autograd_only": True,
            "optimizer_created": False,
            "optimizer_step": False,
            "checkpoint_write": False,
        },
    }
    write_json(output_root / "h1_shell_decomposition.json", output)
    write_csv(output_root / "h1_shell_decomposition.csv", rows)
    markdown = f"""# Stage H H1 shell/radial decomposition

- Inner-two-shell H1 contribution: {answers['inner_two_contribution_fraction_mean']:.6g}
  for voxel fraction {answers['inner_two_voxel_fraction']:.6g};
  enrichment {answers['inner_two_enrichment_mean']:.6g}.
- Full inner contribution mean: {answers['full_inner_two_contribution_mean']:.6g}.
- Plain inner contribution mean: {answers['plain_inner_two_contribution_mean']:.6g}.
- Full/Plain inner per-voxel H1:
  {answers['full_inner_two_per_voxel_h1_mean']:.6g} /
  {answers['plain_inner_two_per_voxel_h1_mean']:.6g}.
- Full/Plain inner prediction-gradient fraction:
  {answers['full_inner_two_gradient_fraction_mean']:.6g} /
  {answers['plain_inner_two_gradient_fraction_mean']:.6g}.
- Dominant direction: `{dominant_direction}` with fractions
  `{direction_fractions}`.
- Dominant channels, descending: `{dominant_channels}`.
- Global H1 overlap means: `{overlap_means}`.

Derivative ownership uses central-cell midpoint attribution. Shells 0--7,
inner/middle/outer groups, and the three angular regions independently add to
the full H0 within tolerance. Gradient-norm fractions are prediction-space
diagnostics. No optimizer, Trainer, checkpoint write, or model update was
used.
"""
    (output_root / "h1_shell_decomposition.md").write_text(
        markdown, encoding="utf-8"
    )
    print(
        {
            "status": "completed",
            "rows": len(rows),
            "cases": len(case_summaries),
            "answers": answers,
        }
    )


if __name__ == "__main__":
    main()
