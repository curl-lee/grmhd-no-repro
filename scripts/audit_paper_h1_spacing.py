#!/usr/bin/env python
"""Run the frozen Stage H H0--H4 spacing audit without parameter updates."""

from __future__ import annotations

from collections import defaultdict
import statistics
from pathlib import Path
from typing import Any

import torch

from grmhd import CHANNELS
from grmhd.paper_config import load_paper_experiment_config
from grmhd.paper_h1_diagnostics import h1_diagnostic
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


VARIANTS = (
    ("H0_current_upstream", "uniform", "all_channels_naive_proxy"),
    ("H1_unit_index", "uniform", "all_channels_naive_proxy"),
    ("H2_normalized_axis", "uniform", "all_channels_naive_proxy"),
    ("H3_stored_r", "uniform", "all_channels_naive_proxy"),
    ("H3_stored_r", "volume_proxy", "all_channels_naive_proxy"),
    ("H3_stored_logr", "uniform", "all_channels_naive_proxy"),
    ("H3_stored_logr", "volume_proxy", "all_channels_naive_proxy"),
    ("H4_spherical_metric_proxy", "uniform", "all_channels_naive_proxy"),
    ("H4_spherical_metric_proxy", "volume_proxy", "all_channels_naive_proxy"),
    ("H4_spherical_metric_proxy", "uniform", "scalar_channel_proxy_only"),
    ("H4_spherical_metric_proxy", "volume_proxy", "scalar_channel_proxy_only"),
)


def mean(values: list[float]) -> float:
    return float(statistics.fmean(values))


def main() -> None:
    root = project_root_from_file()
    output_root = root / OUTPUT_ROOT
    device = require_cuda_device()
    frozen = freeze_or_validate_inputs(root, output_root)
    config = load_paper_experiment_config(
        root / FULL_CONFIG_PATH, project_root=root
    )
    coordinates = {
        name: value.to(device=device)
        for name, value in load_coordinates(config).items()
    }
    datasets = make_fixed_datasets(root)

    records: list[dict[str, Any]] = []
    state_hashes: dict[str, dict[str, Any]] = {}
    for spec in MODEL_STATES:
        model, epoch, metadata, before_hash = load_model_state(
            root, spec, device=device
        )
        processor = make_processor(root, device=device, epoch=epoch)
        with torch.no_grad():
            for split, source, target in fixed_samples():
                batch = collate_fixed_pair(
                    datasets, split=split, source=source, target=target
                )
                prediction, fields = forward_fixed_pair(
                    model=model, processor=processor, batch=batch
                )
                error_fields = {
                    "persistence_error": (
                        fields["normalized_input"] - fields["normalized_target"]
                    ),
                    "model_error": prediction - fields["normalized_target"],
                    "target_field": fields["normalized_target"],
                }
                for case, error in error_fields.items():
                    h0 = h1_diagnostic(
                        error,
                        variant="H0_current_upstream",
                        coordinates=coordinates,
                    )
                    h0_total = max(float(h0.total), 1.0e-30)
                    for variant, reduction, channel_mode in VARIANTS:
                        diagnostic = h1_diagnostic(
                            error,
                            variant=variant,
                            coordinates=coordinates,
                            reduction=reduction,
                            channel_mode=channel_mode,
                        )
                        variant_label = (
                            f"{variant}:{channel_mode}"
                            if variant == "H4_spherical_metric_proxy"
                            else variant
                        )
                        record = {
                            "model_state": spec.name,
                            "model_mode": spec.mode,
                            "checkpoint_epoch": epoch,
                            "split": split,
                            "source_snapshot": source,
                            "target_snapshot": target,
                            "error_case": case,
                            "variant": variant_label,
                            "reduction": reduction,
                            "global_value": float(diagnostic.total),
                            "relative_to_h0": float(diagnostic.total) / h0_total,
                            "per_channel": {
                                channel: float(value)
                                for channel, value in zip(
                                    CHANNELS
                                    if diagnostic.per_channel.numel() == 8
                                    else ("rho", "press"),
                                    diagnostic.per_channel,
                                )
                            },
                            "per_direction": {
                                name: float(value)
                                for name, value in diagnostic.per_direction.items()
                            },
                            "metadata": diagnostic.metadata,
                        }
                        records.append(record)
        after_hash = tensor_state_sha256(model.state_dict())
        if after_hash != before_hash:
            raise RuntimeError(f"Spacing audit modified model state {spec.name}")
        state_hashes[spec.name] = {
            "before": before_hash,
            "after": after_hash,
            "unchanged": True,
            "epoch": epoch,
            "checkpoint_mode": metadata.get("mode"),
        }
        del processor, model
        torch.cuda.empty_cache()

    grouped: dict[tuple[str, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            record["model_state"],
            record["split"],
            record["error_case"],
            record["variant"],
            record["reduction"],
        )
        grouped[key].append(record)
    aggregate = []
    for key, values in sorted(grouped.items()):
        aggregate.append(
            {
                "model_state": key[0],
                "split": key[1],
                "error_case": key[2],
                "variant": key[3],
                "reduction": key[4],
                "sample_count": len(values),
                "global_value_mean": mean([item["global_value"] for item in values]),
                "relative_to_h0_mean": mean(
                    [item["relative_to_h0"] for item in values]
                ),
                "per_channel_mean": {
                    channel: mean(
                        [
                            item["per_channel"][channel]
                            for item in values
                            if channel in item["per_channel"]
                        ]
                    )
                    for channel in values[0]["per_channel"]
                },
                "per_direction_mean": {
                    direction: mean(
                        [item["per_direction"][direction] for item in values]
                    )
                    for direction in ("phi", "theta", "r")
                },
            }
        )

    h0_h1_ratios = [
        1.0 / item["relative_to_h0_mean"]
        for item in aggregate
        if item["variant"] == "H1_unit_index"
    ]
    h2_ratios = [
        item["relative_to_h0_mean"]
        for item in aggregate
        if item["variant"] == "H2_normalized_axis"
    ]
    h0_model_rows = [
        item
        for item in aggregate
        if item["variant"] == "H0_current_upstream"
        and item["error_case"] == "model_error"
    ]
    direction_totals = {
        direction: mean(
            [item["per_direction_mean"][direction] for item in h0_model_rows]
        )
        for direction in ("phi", "theta", "r")
    }
    dominant_direction = max(direction_totals, key=direction_totals.get)
    answers = {
        "h0_over_unit_index_mean": mean(h0_h1_ratios),
        "h0_over_unit_index_min": min(h0_h1_ratios),
        "h0_over_unit_index_max": max(h0_h1_ratios),
        "h2_over_h0_mean": mean(h2_ratios),
        "h2_over_h0_max_absolute_deviation": max(
            abs(value - 1.0) for value in h2_ratios
        ),
        "dominant_h0_model_error_direction": dominant_direction,
        "h0_model_error_direction_means": direction_totals,
        "spacing_amplification_scale_interpretation": (
            "H0/H1 is expected to be N^2=4096 because all three axes have N=64 "
            "and only derivative spacing changes while uniform voxel reduction is held fixed."
        ),
    }
    result = {
        "schema_version": "paper-stage-h-h1-spacing-audit-v1",
        "status": "completed",
        "frozen_inputs": frozen,
        "device": torch.cuda.get_device_name(device),
        "fixed_model_states": state_hashes,
        "records": records,
        "aggregate": aggregate,
        "answers": answers,
        "safety": {
            "eval_mode": True,
            "mixed_precision": False,
            "batch_size": 1,
            "optimizer_created": False,
            "optimizer_step": False,
            "checkpoint_write": False,
        },
    }
    write_json(output_root / "h1_spacing_audit.json", result)
    write_csv(output_root / "h1_spacing_audit.csv", aggregate)
    markdown = f"""# Stage H H1 spacing audit

- Samples: 5 frozen train pairs and 5 frozen validation pairs.
- States: shared initial, Full best/last, Plain best/last.
- H0/H1 mean amplification: {answers['h0_over_unit_index_mean']:.6g}.
- H2/H0 mean: {answers['h2_over_h0_mean']:.9g}; maximum absolute parity deviation:
  {answers['h2_over_h0_max_absolute_deviation']:.3g}.
- Dominant H0 model-error direction: `{dominant_direction}`.
- Direction means: `{direction_totals}`.

H0 is exactly the frozen Stage E computational-grid seminorm. H1 holds the
uniform-voxel reduction and periodic centered stencil fixed while setting cell
spacing to one. H2 explicitly restates the upstream `1/N` spacing. H3 uses the
stored `r` or `log(r)` centers with open theta/r boundaries. H4 is only a
spherical scalar-metric proxy; the all-channel form is not covariant for stored
magnetic or velocity components. Volume weighting is
`r^2 sin(theta) dphi dtheta dr`, not a verified Kerr--Schild proper volume.

No Trainer, optimizer, scheduler, checkpoint write, or parameter update was
used. Detailed per-state, per-sample, per-channel, and per-direction values are
in `h1_spacing_audit.json`.
"""
    (output_root / "h1_spacing_audit.md").write_text(markdown, encoding="utf-8")
    print(
        {
            "status": "completed",
            "records": len(records),
            "aggregate_rows": len(aggregate),
            "answers": answers,
        }
    )


if __name__ == "__main__":
    main()
