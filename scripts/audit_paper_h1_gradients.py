#!/usr/bin/env python
"""Compare Full loss gradient directions without optimizer or parameter updates."""

from __future__ import annotations

from collections import defaultdict
import math
import statistics
from typing import Any, Mapping, Sequence

import torch

from grmhd import CHANNELS
from grmhd.paper_config import (
    build_paper_training_loss,
    load_paper_experiment_config,
)
from grmhd.paper_h1_diagnostics import (
    flatten_optional_gradients,
    gradient_pair_metrics,
    h1_diagnostic,
    simulate_global_norm_clip,
)
from grmhd.paper_losses import PlainL2Loss
from grmhd.paper_stage_g import tensor_state_sha256
from grmhd.paper_stage_h import (
    FULL_CONFIG_PATH,
    MODEL_STATES,
    OUTPUT_ROOT,
    collate_fixed_pair,
    fixed_samples,
    forward_fixed_pair,
    freeze_or_validate_inputs,
    load_model_state,
    make_fixed_datasets,
    make_processor,
    module_group,
    project_root_from_file,
    require_cuda_device,
    write_csv,
    write_json,
)


COMPONENT_ORDER = (
    "base",
    "h1_raw",
    "h1_weighted",
    "other",
    "total",
    "plain_l2",
)


def prediction_gradient_summary(
    gradient: torch.Tensor,
    *,
    shells: torch.Tensor,
) -> dict[str, Any]:
    absolute = gradient.abs()
    square = gradient.square()
    flat_square = square.reshape(-1)
    count = flat_square.numel()
    top_one = max(1, int(math.ceil(0.01 * count)))
    top_ten = max(1, int(math.ceil(0.10 * count)))
    total_energy = flat_square.sum().clamp_min(1.0e-30)
    sorted_energy = torch.sort(flat_square, descending=True).values
    h0 = h1_diagnostic(gradient, variant="H0_current_upstream")
    direction_total = sum(h0.per_direction.values()).clamp_min(1.0e-30)
    shell_masks = shells[0].bool()
    return {
        "norm": torch.linalg.vector_norm(gradient),
        "per_channel_norm": {
            channel: torch.linalg.vector_norm(gradient[:, index])
            for index, channel in enumerate(CHANNELS)
        },
        "per_shell_norm": {
            str(index): torch.linalg.vector_norm(
                gradient[..., shell_masks[index]]
            )
            for index in range(shell_masks.shape[0])
        },
        "maximum_absolute": absolute.max(),
        "q95_absolute": torch.quantile(absolute.reshape(-1), 0.95),
        "top_1_percent_energy_fraction": sorted_energy[:top_one].sum()
        / total_energy,
        "top_10_percent_energy_fraction": sorted_energy[:top_ten].sum()
        / total_energy,
        "direction_diagnostic_fraction": {
            direction: value / direction_total
            for direction, value in h0.per_direction.items()
        },
    }


def parameter_module_norms(
    gradients: Sequence[torch.Tensor | None],
    named_parameters: Sequence[tuple[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    squared: dict[str, torch.Tensor] = {}
    for gradient, (name, parameter) in zip(gradients, named_parameters):
        value = torch.zeros_like(parameter) if gradient is None else gradient
        norm_square = value.abs().square().sum()
        for group in module_group(name):
            squared[group] = squared.get(group, norm_square.new_zeros(())) + norm_square
    return {name: value.sqrt() for name, value in squared.items()}


def serial_pair_metrics(first: torch.Tensor, second: torch.Tensor) -> dict[str, Any]:
    return gradient_pair_metrics(first, second)


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
    full_loss = build_paper_training_loss(full_config).to(device)
    full_loss.eval()
    plain_loss = PlainL2Loss().to(device).eval()
    datasets = make_fixed_datasets(root)

    records: list[dict[str, Any]] = []
    state_hashes: dict[str, Any] = {}
    for spec in MODEL_STATES:
        model, epoch, metadata, before_hash = load_model_state(
            root, spec, device=device
        )
        processor = make_processor(root, device=device, epoch=epoch)
        named_parameters = tuple(model.named_parameters())
        parameters = tuple(parameter for _, parameter in named_parameters)
        for split, source, target in fixed_samples():
            batch = collate_fixed_pair(
                datasets, split=split, source=source, target=target
            )
            prediction, fields = forward_fixed_pair(
                model=model, processor=processor, batch=batch
            )
            result = full_loss.components(
                prediction, context=fields["context"]
            )
            other = (
                result.roi_weighted
                + result.bounds_weighted
                + result.envelope_weighted
                + result.dissipation_weighted
            )
            plain = plain_loss(prediction, fields["normalized_target"])
            components = {
                "base": result.base_fidelity_weighted,
                "h1_raw": result.h1_raw,
                "h1_weighted": result.h1_weighted,
                "other": other,
                "total": result.total,
                "plain_l2": plain,
            }
            prediction_gradients: dict[str, torch.Tensor] = {}
            parameter_vectors: dict[str, torch.Tensor] = {}
            module_norms: dict[str, Mapping[str, torch.Tensor]] = {}
            for index, name in enumerate(COMPONENT_ORDER):
                gradients = torch.autograd.grad(
                    components[name],
                    (prediction, *parameters),
                    retain_graph=index < len(COMPONENT_ORDER) - 1,
                    allow_unused=True,
                )
                prediction_gradient = gradients[0]
                if prediction_gradient is None:
                    prediction_gradient = torch.zeros_like(prediction)
                prediction_gradients[name] = prediction_gradient
                parameter_gradients = gradients[1:]
                parameter_vectors[name] = flatten_optional_gradients(
                    parameter_gradients, parameters
                )
                module_norms[name] = parameter_module_norms(
                    parameter_gradients, named_parameters
                )

            summed = (
                parameter_vectors["base"]
                + parameter_vectors["h1_weighted"]
                + parameter_vectors["other"]
            )
            gradient_sum_relative_residual = torch.linalg.vector_norm(
                summed - parameter_vectors["total"]
            ) / torch.linalg.vector_norm(parameter_vectors["total"]).clamp_min(
                1.0e-30
            )
            if float(gradient_sum_relative_residual) > 1.0e-3:
                raise RuntimeError(
                    "Full component parameter gradients do not sum: "
                    f"relative residual={float(gradient_sum_relative_residual):.6g}"
                )
            pairs = {
                "base_h1_raw": serial_pair_metrics(
                    parameter_vectors["base"], parameter_vectors["h1_raw"]
                ),
                "base_h1_weighted": serial_pair_metrics(
                    parameter_vectors["base"], parameter_vectors["h1_weighted"]
                ),
                "base_other": serial_pair_metrics(
                    parameter_vectors["base"], parameter_vectors["other"]
                ),
                "base_total": serial_pair_metrics(
                    parameter_vectors["base"], parameter_vectors["total"]
                ),
                "h1_weighted_other": serial_pair_metrics(
                    parameter_vectors["h1_weighted"],
                    parameter_vectors["other"],
                ),
                "plain_total": serial_pair_metrics(
                    parameter_vectors["plain_l2"], parameter_vectors["total"]
                ),
            }
            clip = simulate_global_norm_clip(
                {
                    "base": parameter_vectors["base"],
                    "h1_weighted": parameter_vectors["h1_weighted"],
                    "other": parameter_vectors["other"],
                    "total": parameter_vectors["total"],
                },
                max_norm=1.0,
            )
            parameter_norms = {
                name: torch.linalg.vector_norm(vector)
                for name, vector in parameter_vectors.items()
            }
            record = {
                "model_state": spec.name,
                "model_mode": spec.mode,
                "checkpoint_epoch": epoch,
                "split": split,
                "source_snapshot": source,
                "target_snapshot": target,
                "component_values": components,
                "prediction_gradients": {
                    name: prediction_gradient_summary(
                        gradient, shells=processor.shells
                    )
                    for name, gradient in prediction_gradients.items()
                },
                "parameter_gradient_norms": parameter_norms,
                "parameter_module_norms": module_norms,
                "parameter_gradient_pairs": pairs,
                "h1_to_base_norm_ratio": parameter_norms["h1_weighted"]
                / parameter_norms["base"].clamp_min(1.0e-30),
                "weighted_h1_to_total_norm_fraction": parameter_norms[
                    "h1_weighted"
                ]
                / parameter_norms["total"].clamp_min(1.0e-30),
                "gradient_sum_relative_residual": gradient_sum_relative_residual,
                "clipping": clip,
            }
            records.append(record)
            if any(parameter.grad is not None for parameter in parameters):
                raise RuntimeError("Stage H populated parameter .grad tensors")
            del prediction, fields, result, components
        after_hash = tensor_state_sha256(model.state_dict())
        if after_hash != before_hash:
            raise RuntimeError(f"Gradient audit modified model state {spec.name}")
        state_hashes[spec.name] = {
            "before": before_hash,
            "after": after_hash,
            "unchanged": True,
            "epoch": epoch,
            "checkpoint_mode": metadata.get("mode"),
        }
        del processor, model
        torch.cuda.empty_cache()

    summary_rows = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[(record["model_state"], record["split"])].append(record)
    for (state, split), values in sorted(grouped.items()):
        row: dict[str, Any] = {
            "model_state": state,
            "split": split,
            "sample_count": len(values),
        }
        for component in COMPONENT_ORDER:
            row[f"{component}_parameter_norm_mean"] = mean(
                [
                    float(value["parameter_gradient_norms"][component])
                    for value in values
                ]
            )
            row[f"{component}_prediction_norm_mean"] = mean(
                [
                    float(value["prediction_gradients"][component]["norm"])
                    for value in values
                ]
            )
        for pair in (
            "base_h1_raw",
            "base_h1_weighted",
            "base_other",
            "base_total",
            "h1_weighted_other",
            "plain_total",
        ):
            for metric in (
                "cosine",
                "dot",
                "conflict_fraction",
                "opposite_sign_parameter_fraction",
            ):
                row[f"{pair}_{metric}_mean"] = mean(
                    [
                        float(value["parameter_gradient_pairs"][pair][metric])
                        for value in values
                    ]
                )
        row["h1_to_base_norm_ratio_mean"] = mean(
            [float(value["h1_to_base_norm_ratio"]) for value in values]
        )
        row["weighted_h1_to_total_norm_fraction_mean"] = mean(
            [
                float(value["weighted_h1_to_total_norm_fraction"])
                for value in values
            ]
        )
        row["clip_scale_mean"] = mean(
            [float(value["clipping"]["scale_factor"]) for value in values]
        )
        row["total_norm_before_clip_mean"] = mean(
            [float(value["clipping"]["total_norm_before"]) for value in values]
        )
        row["gradient_sum_relative_residual_max"] = max(
            float(value["gradient_sum_relative_residual"]) for value in values
        )
        row["base_projection_after_clip_mean"] = mean(
            [
                float(
                    value["clipping"]["component_projections"]["base"]["after"]
                )
                for value in values
            ]
        )
        summary_rows.append(row)

    overall = {
        key: mean([float(record[key]) for record in summary_rows])
        for key in (
            "h1_to_base_norm_ratio_mean",
            "weighted_h1_to_total_norm_fraction_mean",
            "clip_scale_mean",
            "base_h1_weighted_cosine_mean",
            "base_h1_weighted_conflict_fraction_mean",
            "total_norm_before_clip_mean",
        )
    }
    output = {
        "schema_version": "paper-stage-h-h1-gradient-audit-v1",
        "status": "completed",
        "frozen_inputs": frozen,
        "device": torch.cuda.get_device_name(device),
        "fixed_model_states": state_hashes,
        "records": records,
        "summary": summary_rows,
        "overall": overall,
        "safety": {
            "eval_mode": True,
            "mixed_precision": False,
            "batch_size": 1,
            "autograd_api": "torch.autograd.grad",
            "parameter_grad_tensors_remained_none": True,
            "optimizer_created": False,
            "optimizer_step": False,
            "scheduler_step": False,
            "checkpoint_write": False,
        },
    }
    write_json(output_root / "h1_gradient_audit.json", output)
    write_csv(output_root / "h1_gradient_audit.csv", summary_rows)
    markdown = f"""# Stage H H1 gradient audit

Across the five frozen model states and ten fixed pairs:

- weighted H1/base parameter-gradient norm ratio mean:
  {overall['h1_to_base_norm_ratio_mean']:.6g};
- weighted H1/total norm fraction mean:
  {overall['weighted_h1_to_total_norm_fraction_mean']:.6g};
- base-versus-weighted-H1 cosine mean:
  {overall['base_h1_weighted_cosine_mean']:.6g};
- base-versus-weighted-H1 conflict fraction mean:
  {overall['base_h1_weighted_conflict_fraction_mean']:.6g};
- Full total norm before clip mean:
  {overall['total_norm_before_clip_mean']:.6g};
- simulated norm-1 clip scale mean:
  {overall['clip_scale_mean']:.6g}.

The simulation scales the combined gradient exactly as global norm clipping
would, but does not write `.grad`, call `backward`, construct an optimizer, or
update parameters. Per-sample prediction-space/channel/shell/concentration
statistics, per-module parameter norms, dot products, cosines, and sign
conflicts are stored in `h1_gradient_audit.json`.
"""
    (output_root / "h1_gradient_audit.md").write_text(
        markdown, encoding="utf-8"
    )
    print(
        {
            "status": "completed",
            "records": len(records),
            "summary_rows": len(summary_rows),
            "overall": overall,
        }
    )


if __name__ == "__main__":
    main()
