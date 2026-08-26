#!/usr/bin/env python
"""Evaluate persistence, one-step forecasts, and autoregressive GRMHD rollout."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str((Path("outputs") / ".matplotlib").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import torch
import yaml

from grmhd import CHANNELS
from grmhd.dataset import GRMHDPairedDataset, make_temporal_datasets
from grmhd.losses import channel_relative_l2
from grmhd.models import apply_prediction_mode, build_model
from grmhd.normalizer import GRMHDNormalizer, validate_stats_bundle
from grmhd.priors import QuantileBounds, ResidualEnvelope
from grmhd.residual import validate_residual_wrapper_metadata
from grmhd.shells import radial_shells_tensor


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        result[key] = (
            deep_merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else value
        )
    return result


def load_config(path: Path) -> dict[str, Any]:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = values.pop("base_config", None)
    if base is None:
        return values
    base_path = Path(base)
    if not base_path.is_absolute() and not base_path.exists():
        base_path = path.parent / base_path
    return deep_merge(load_config(base_path.resolve()), values)


def source_regime(index: int, target_index: int) -> str:
    if target_index < 92:
        return "pre_change"
    if index <= 97 and target_index >= 92:
        return "change_92_97"
    return "post_change"


def prepare_state(
    raw: torch.Tensor,
    normalizer: GRMHDNormalizer,
    downsample: int,
    device: torch.device,
) -> torch.Tensor:
    state = normalizer.encode_tensor(raw.to(device), channel_axis=1)
    return state[..., ::downsample, ::downsample, ::downsample] if downsample > 1 else state


def model_input(state: torch.Tensor, shells: torch.Tensor | None) -> torch.Tensor:
    if shells is None:
        return state
    return torch.cat([state, shells.unsqueeze(0).expand(state.shape[0], -1, -1, -1, -1)], dim=1)


def apply_evaluation_clamps(
    prediction: torch.Tensor,
    input_state: torch.Tensor,
    quantile_bounds: QuantileBounds | None,
    residual_envelope: ResidualEnvelope | None,
    enabled: bool,
) -> torch.Tensor:
    if not enabled:
        return prediction
    output = prediction
    if residual_envelope is not None:
        output = residual_envelope.clamp(output, input_state)
    if quantile_bounds is not None:
        output = quantile_bounds.clamp(output)
    return output


def rel_l2_list(prediction: torch.Tensor, target: torch.Tensor) -> list[float]:
    return [float(value) for value in channel_relative_l2(prediction, target).detach().cpu()]


def global_relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm(prediction - target)
    denominator = torch.clamp_min(torch.linalg.vector_norm(target), 1.0e-12)
    return float((numerator / denominator).detach().cpu())


def finite_counts(values: torch.Tensor) -> dict[str, int]:
    return {
        "nan": int(torch.isnan(values).sum().item()),
        "inf": int(torch.isinf(values).sum().item()),
    }


def range_record(prediction: torch.Tensor, truth: torch.Tensor) -> dict[str, dict[str, float]]:
    result = {}
    for channel, name in enumerate(CHANNELS):
        result[name] = {
            "prediction_min": float(prediction[:, channel].min().cpu()),
            "prediction_max": float(prediction[:, channel].max().cpu()),
            "truth_min": float(truth[:, channel].min().cpu()),
            "truth_max": float(truth[:, channel].max().cpu()),
        }
    return result


def evaluate_pair(
    prediction_normalized: torch.Tensor,
    truth_normalized: torch.Tensor,
    normalizer: GRMHDNormalizer,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    prediction_physical = normalizer.decode_tensor(prediction_normalized, channel_axis=1)
    truth_physical = normalizer.decode_tensor(truth_normalized, channel_axis=1)
    counts_normalized = finite_counts(prediction_normalized)
    counts_physical = finite_counts(prediction_physical)
    result = {
        "relative_l2_normalized": rel_l2_list(prediction_normalized, truth_normalized),
        "relative_l2_physical": rel_l2_list(prediction_physical, truth_physical),
        "global_volumetric_relative_l2_normalized": global_relative_l2(
            prediction_normalized, truth_normalized
        ),
        "global_volumetric_relative_l2_physical": global_relative_l2(
            prediction_physical, truth_physical
        ),
        "prediction_nonfinite_normalized": counts_normalized,
        "prediction_nonfinite_physical": counts_physical,
        "positivity_violations": {
            "rho": int((prediction_physical[:, 3] <= 0).sum().item()),
            "press": int((prediction_physical[:, 4] <= 0).sum().item()),
        },
        "ranges": range_record(prediction_physical, truth_physical),
    }
    if any(counts_normalized.values()) or any(counts_physical.values()):
        raise FloatingPointError(f"Model prediction produced non-finite values: {result}")
    return result, prediction_physical, truth_physical


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"count": 0}
    normalized = np.asarray([record["relative_l2_normalized"] for record in records])
    physical = np.asarray([record["relative_l2_physical"] for record in records])
    global_normalized = np.asarray(
        [record["global_volumetric_relative_l2_normalized"] for record in records]
    )
    global_physical = np.asarray(
        [record["global_volumetric_relative_l2_physical"] for record in records]
    )
    return {
        "count": len(records),
        "channel_order": list(CHANNELS),
        "mean_relative_l2_normalized": normalized.mean(axis=0).tolist(),
        "mean_relative_l2_physical": physical.mean(axis=0).tolist(),
        "overall_mean_relative_l2_normalized": float(normalized.mean()),
        "overall_mean_relative_l2_physical": float(physical.mean()),
        "aggregation_definition": {
            "overall_mean": "arithmetic mean over channels and evaluated transitions",
            "global_volumetric": "relative L2 over all channels and voxels, then arithmetic mean over transitions",
        },
        "mean_global_volumetric_relative_l2_normalized": float(global_normalized.mean()),
        "mean_global_volumetric_relative_l2_physical": float(global_physical.mean()),
        "total_positivity_violations": {
            field: sum(record["positivity_violations"][field] for record in records)
            for field in ("rho", "press")
        },
        "total_prediction_nan": sum(
            record[domain]["nan"]
            for record in records
            for domain in ("prediction_nonfinite_normalized", "prediction_nonfinite_physical")
        ),
        "total_prediction_inf": sum(
            record[domain]["inf"]
            for record in records
            for domain in ("prediction_nonfinite_normalized", "prediction_nonfinite_physical")
        ),
    }


def physical_statistics(state: torch.Tensor, r: np.ndarray, n_shells: int = 8) -> dict[str, Any]:
    values = state[0].detach().cpu().numpy()
    quantities = {
        "rho": values[3],
        "press": values[4],
        "stored_B_coordinate_component_Euclidean_norm": component_magnitude(values, 0, 3),
        "stored_v_coordinate_component_Euclidean_norm": component_magnitude(values, 5, 8),
    }
    edges = np.geomspace(float(r.min()), float(r.max()) * (1.0 + 1.0e-12), n_shells + 1)
    shell_index = np.clip(np.searchsorted(edges, r, side="right") - 1, 0, n_shells - 1)
    result: dict[str, Any] = {"shell_edges": edges.tolist(), "quantiles": [0.01, 0.1, 0.5, 0.9, 0.99]}
    for name, quantity in quantities.items():
        shells = []
        for shell in range(n_shells):
            selected = quantity[..., shell_index == shell]
            shells.append({"mean": float(selected.mean()), "std": float(selected.std())})
        result[name] = {
            "quantile_values": np.quantile(quantity, result["quantiles"]).tolist(),
            "radial_mean": quantity.mean(axis=(0, 1)).tolist(),
            "radial_std": quantity.std(axis=(0, 1)).tolist(),
            "shells": shells,
        }
    return result


def artifact_diagnostics(
    prediction: torch.Tensor, truth: torch.Tensor, reference_input: torch.Tensor
) -> dict[str, Any]:
    diagnostics: dict[str, Any] = {"channels": {}, "flags": []}
    for channel, name in enumerate(CHANNELS):
        pred = prediction[0, channel]
        true = truth[0, channel]
        reference = reference_input[0, channel]
        pred_std = float(pred.std().cpu())
        true_std = float(true.std().cpu())
        std_ratio = pred_std / max(true_std, 1.0e-12)
        pred_diff = [float(torch.mean(torch.diff(pred, dim=axis).square()).cpu()) for axis in range(3)]
        true_diff = [float(torch.mean(torch.diff(true, dim=axis).square()).cpu()) for axis in range(3)]
        reference_diff = [
            float(torch.mean(torch.diff(reference, dim=axis).square()).cpu())
            for axis in range(3)
        ]
        roughness_ratio = sum(pred_diff) / max(sum(true_diff), 1.0e-12)
        roughness_creation_ratio = sum(pred_diff) / max(
            sum(true_diff), sum(reference_diff), 1.0e-12
        )
        anisotropy = max(pred_diff) / max(min(pred_diff), 1.0e-12)
        reference_anisotropy = max(reference_diff) / max(min(reference_diff), 1.0e-12)
        anisotropy_excess = anisotropy / max(reference_anisotropy, 1.0e-12)
        diagnostics["channels"][name] = {
            "std_ratio_prediction_over_truth": std_ratio,
            "first_difference_energy_ratio": roughness_ratio,
            "created_roughness_ratio_over_truth_or_input": roughness_creation_ratio,
            "prediction_axis_difference_energy": pred_diff,
            "axis_anisotropy_ratio": anisotropy,
            "input_axis_anisotropy_ratio": reference_anisotropy,
            "axis_anisotropy_excess_over_input": anisotropy_excess,
        }
        if std_ratio < 0.05:
            diagnostics["flags"].append(f"{name}:possible_field_collapse")
        if roughness_creation_ratio > 5.0:
            diagnostics["flags"].append(f"{name}:possible_high_frequency_ripple")
        if anisotropy_excess > 10.0:
            diagnostics["flags"].append(f"{name}:possible_stripe_anisotropy")
    return diagnostics


def aggregate_statistical_properties(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    if not records:
        return {}
    first = records[0][key]
    result: dict[str, Any] = {
        "shell_edges": first["shell_edges"],
        "quantiles": first["quantiles"],
        "interpretation": "Temporal summaries of sampled stored components; not metric/volume weighted.",
    }
    for quantity in (
        "rho",
        "press",
        "stored_B_coordinate_component_Euclidean_norm",
        "stored_v_coordinate_component_Euclidean_norm",
    ):
        radial = np.asarray([record[key][quantity]["radial_mean"] for record in records])
        quantiles = np.asarray([record[key][quantity]["quantile_values"] for record in records])
        shell_means = np.asarray([
            [shell["mean"] for shell in record[key][quantity]["shells"]] for record in records
        ])
        shell_stds = np.asarray([
            [shell["std"] for shell in record[key][quantity]["shells"]] for record in records
        ])
        result[quantity] = {
            "time_mean_radial_profile": radial.mean(axis=0).tolist(),
            "time_std_radial_profile": radial.std(axis=0).tolist(),
            "time_mean_quantiles": quantiles.mean(axis=0).tolist(),
            "shell_mean_over_time": shell_means.mean(axis=0).tolist(),
            "shell_temporal_std_of_mean": shell_means.std(axis=0).tolist(),
            "shell_mean_within_snapshot_std": shell_stds.mean(axis=0).tolist(),
        }
    return result


def statistical_error_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for quantity in (
        "rho",
        "press",
        "stored_B_coordinate_component_Euclidean_norm",
        "stored_v_coordinate_component_Euclidean_norm",
    ):
        radial_errors = []
        shell_mean_errors = []
        shell_std_errors = []
        quantile_errors = []
        for record in records:
            prediction = record["statistics_prediction"][quantity]
            truth = record["statistics_truth"][quantity]
            pred_radial = np.asarray(prediction["radial_mean"])
            true_radial = np.asarray(truth["radial_mean"])
            radial_errors.append(
                float(np.linalg.norm(pred_radial - true_radial) / max(np.linalg.norm(true_radial), 1e-30))
            )
            pred_shell_mean = np.asarray([shell["mean"] for shell in prediction["shells"]])
            true_shell_mean = np.asarray([shell["mean"] for shell in truth["shells"]])
            pred_shell_std = np.asarray([shell["std"] for shell in prediction["shells"]])
            true_shell_std = np.asarray([shell["std"] for shell in truth["shells"]])
            shell_mean_errors.append(
                float(np.linalg.norm(pred_shell_mean - true_shell_mean) / max(np.linalg.norm(true_shell_mean), 1e-30))
            )
            shell_std_errors.append(
                float(np.linalg.norm(pred_shell_std - true_shell_std) / max(np.linalg.norm(true_shell_std), 1e-30))
            )
            pred_quantiles = np.asarray(prediction["quantile_values"])
            true_quantiles = np.asarray(truth["quantile_values"])
            quantile_errors.append(
                float(np.linalg.norm(pred_quantiles - true_quantiles) / max(np.linalg.norm(true_quantiles), 1e-30))
            )
        result[quantity] = {
            "mean_radial_profile_relative_l2": float(np.mean(radial_errors)),
            "mean_shell_mean_relative_l2": float(np.mean(shell_mean_errors)),
            "mean_shell_std_relative_l2": float(np.mean(shell_std_errors)),
            "mean_quantile_relative_l2": float(np.mean(quantile_errors)),
        }
    return result


def regime_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    result = {}
    for regime in ("pre_change", "change_92_97", "post_change"):
        selected = [record for record in records if record.get("regime") == regime]
        result[regime] = aggregate(selected)
    return result


def write_regime_outputs(
    *,
    run_id: str,
    window: str,
    prediction_mode: str,
    model_records: list[dict[str, Any]],
    persistence_records: list[dict[str, Any]],
    definition_path: Path,
    csv_path: Path,
) -> None:
    definition = {
        "name": "source-index diagnostic segments",
        "pre_change": "target_index < 92",
        "change_92_97": "transition touches source indices 92..97 (target_index >= 92 and input index <= 97)",
        "post_change": "input index > 97",
        "warning": "A descriptive change interval only; no physical phase/regime transition is claimed.",
    }
    definition_path.parent.mkdir(parents=True, exist_ok=True)
    definition_path.write_text(json.dumps(definition, indent=2), encoding="utf-8")
    fields = [
        "run_id", "window", "prediction_mode", "method", "segment", "channel",
        "transition_count", "relative_l2_normalized", "relative_l2_physical",
        "global_volumetric_relative_l2_normalized", "global_volumetric_relative_l2_physical",
    ]
    existing: list[dict[str, str]] = []
    if csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as stream:
            existing = [row for row in csv.DictReader(stream) if row["run_id"] != run_id]
    new_rows: list[dict[str, Any]] = []
    for method, records in (("model", model_records), ("persistence", persistence_records)):
        metrics = regime_metrics(records)
        for segment, summary in metrics.items():
            if not summary.get("count"):
                continue
            for channel_index, channel in enumerate(CHANNELS):
                new_rows.append({
                    "run_id": run_id,
                    "window": window,
                    "prediction_mode": prediction_mode,
                    "method": method,
                    "segment": segment,
                    "channel": channel,
                    "transition_count": summary["count"],
                    "relative_l2_normalized": summary["mean_relative_l2_normalized"][channel_index],
                    "relative_l2_physical": summary["mean_relative_l2_physical"][channel_index],
                    "global_volumetric_relative_l2_normalized": summary["mean_global_volumetric_relative_l2_normalized"],
                    "global_volumetric_relative_l2_physical": summary["mean_global_volumetric_relative_l2_physical"],
                })
            new_rows.append({
                "run_id": run_id,
                "window": window,
                "prediction_mode": prediction_mode,
                "method": method,
                "segment": segment,
                "channel": "arithmetic_mean_channels",
                "transition_count": summary["count"],
                "relative_l2_normalized": summary["overall_mean_relative_l2_normalized"],
                "relative_l2_physical": summary["overall_mean_relative_l2_physical"],
                "global_volumetric_relative_l2_normalized": summary["mean_global_volumetric_relative_l2_normalized"],
                "global_volumetric_relative_l2_physical": summary["mean_global_volumetric_relative_l2_physical"],
            })
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(existing + new_rows)


def signed_norm(values: np.ndarray) -> TwoSlopeNorm:
    limit = float(np.percentile(np.abs(values), 99.5))
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def plot_prediction_triptychs(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    r: np.ndarray,
    theta: np.ndarray,
    out_dir: Path,
    label: str,
) -> None:
    prediction_np = prediction[0].detach().cpu().numpy()
    truth_np = truth[0].detach().cpu().numpy()
    phi_index = prediction_np.shape[1] // 2
    for channel, name in enumerate(CHANNELS):
        pred_slice = prediction_np[channel, phi_index]
        truth_slice = truth_np[channel, phi_index]
        error = pred_slice - truth_slice
        if name in {"rho", "press"}:
            positive = np.concatenate([pred_slice[pred_slice > 0], truth_slice[truth_slice > 0]])
            floor = float(positive.min()) if positive.size else np.finfo(np.float32).tiny
            pred_shown = np.log10(np.maximum(pred_slice, floor))
            truth_shown = np.log10(np.maximum(truth_slice, floor))
            shared_min = float(min(pred_shown.min(), truth_shown.min()))
            shared_max = float(max(pred_shown.max(), truth_shown.max()))
            shared_kwargs = {"vmin": shared_min, "vmax": shared_max}
            value_label = f"log10({name})"
        else:
            combined = np.stack([pred_slice, truth_slice])
            pred_shown, truth_shown = pred_slice, truth_slice
            shared_kwargs = {"norm": signed_norm(combined)}
            value_label = name
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
        for axis, values, title, kwargs, color_label in (
            (axes[0], truth_shown, "truth", shared_kwargs, value_label),
            (axes[1], pred_shown, "prediction", shared_kwargs, value_label),
            (axes[2], error, "physical error", {"norm": signed_norm(error)}, f"Δ{name}"),
        ):
            image = axis.pcolormesh(r, theta / np.pi, values, shading="auto", cmap="coolwarm", **kwargs)
            axis.set_xscale("log")
            axis.set_xlabel("r")
            axis.set_ylabel(r"$\theta/\pi$")
            axis.set_title(title)
            fig.colorbar(image, ax=axis, label=color_label)
        fig.suptitle(f"{label}: {name} (coordinate-basis component where applicable)")
        fig.savefig(out_dir / f"{label}_{name}_truth_prediction_error.png", dpi=150)
        plt.close(fig)


def component_magnitude(state: np.ndarray, start: int, stop: int) -> np.ndarray:
    # This is only the Euclidean magnitude of stored coordinate components, not
    # a metric-correct GRMHD invariant.
    return np.sqrt(np.sum(state[start:stop] ** 2, axis=0))


def plot_radial_profiles(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    r: np.ndarray,
    out_dir: Path,
    label: str,
) -> None:
    pred = prediction[0].detach().cpu().numpy()
    true = truth[0].detach().cpu().numpy()
    quantities = {
        "rho": (pred[3], true[3]),
        "press": (pred[4], true[4]),
        "coordinate_component_|B|": (component_magnitude(pred, 0, 3), component_magnitude(true, 0, 3)),
        "coordinate_component_|v|": (component_magnitude(pred, 5, 8), component_magnitude(true, 5, 8)),
    }
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, (name, (pred_values, true_values)) in zip(axes.flat, quantities.items(), strict=True):
        axis.plot(r, np.median(true_values, axis=(0, 1)), label="truth")
        axis.plot(r, np.median(pred_values, axis=(0, 1)), label="prediction")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("r")
        axis.set_title(name)
        axis.legend()
    fig.suptitle(f"{label}: radial median profiles (not metric-weighted)")
    fig.savefig(out_dir / f"{label}_radial_profiles.png", dpi=150)
    plt.close(fig)


def plot_error_curves(
    model_records: list[dict[str, Any]], persistence_records: list[dict[str, Any]], out_dir: Path
) -> None:
    steps = np.arange(1, len(model_records) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for axis, key, title in (
        (axes[0], "relative_l2_normalized", "normalized domain"),
        (axes[1], "relative_l2_physical", "decoded physical domain"),
    ):
        model_values = np.asarray([record[key] for record in model_records])
        persistence_values = np.asarray([record[key] for record in persistence_records])
        for channel, name in enumerate(CHANNELS):
            axis.plot(steps, model_values[:, channel], label=f"model {name}")
            axis.plot(steps, persistence_values[:, channel], linestyle="--", alpha=0.55, label=f"persist {name}")
        axis.set_yscale("log")
        axis.set_xlabel("rollout step")
        axis.set_ylabel("relative L2")
        axis.set_title(title)
        axis.legend(fontsize=6, ncol=2)
    fig.savefig(out_dir / "rollout_error_by_step.png", dpi=160)
    plt.close(fig)


def write_error_csv(
    model_records: list[dict[str, Any]], persistence_records: list[dict[str, Any]], path: Path
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "step", "index", "target_index", "regime", "method", "channel",
                "relative_l2_normalized", "relative_l2_physical",
                "global_volumetric_relative_l2_normalized",
                "global_volumetric_relative_l2_physical",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        for step, (model, persistence) in enumerate(
            zip(model_records, persistence_records, strict=True), start=1
        ):
            for method, record in (("model", model), ("persistence", persistence)):
                for channel, name in enumerate(CHANNELS):
                    writer.writerow(
                        {
                            "step": step,
                            "index": record["index"],
                            "target_index": record["truth_index"],
                            "regime": record["regime"],
                            "method": method,
                            "channel": name,
                            "relative_l2_normalized": record["relative_l2_normalized"][channel],
                            "relative_l2_physical": record["relative_l2_physical"][channel],
                            "global_volumetric_relative_l2_normalized": record[
                                "global_volumetric_relative_l2_normalized"
                            ],
                            "global_volumetric_relative_l2_physical": record[
                                "global_volumetric_relative_l2_physical"
                            ],
                        }
                    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--baseline", choices=["persistence"])
    parser.add_argument("--data", type=Path)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--out_dir", "--output-dir", dest="out_dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--regime-definition",
        type=Path,
        default=Path("outputs/experiment_round1/regime_segment_definition.json"),
    )
    parser.add_argument(
        "--regime-metrics-csv",
        type=Path,
        default=Path("outputs/experiment_round1/regime_segment_metrics.csv"),
    )
    args = parser.parse_args()

    if (args.checkpoint is None) == (args.baseline is None):
        parser.error("Specify exactly one of --checkpoint or --baseline persistence")
    if args.baseline and args.config is None:
        parser.error("--baseline persistence requires --config")
    checkpoint = (
        torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if args.checkpoint is not None
        else None
    )
    config = checkpoint["config"] if checkpoint is not None else load_config(args.config.resolve())
    data_path = args.data or Path(config["data"]["path"])
    out_dir = args.out_dir or (
        args.checkpoint.parent / f"eval_{args.split}"
        if args.checkpoint is not None
        else Path("outputs/experiment_round1")
        / f"persistence_{config['data'].get('window_name', 'window')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    data_config = config["data"]
    datasets = make_temporal_datasets(
        data_path,
        stride=int(data_config["stride"]),
        train_fraction=float(data_config["train_fraction"]),
        val_fraction=float(data_config["val_fraction"]),
        snapshot_start=int(data_config.get("snapshot_start", 0)),
        snapshot_end=(None if data_config.get("snapshot_end") is None else int(data_config["snapshot_end"])),
        train_snapshot_count=(None if data_config.get("train_snapshot_count") is None else int(data_config["train_snapshot_count"])),
        val_snapshot_count=(None if data_config.get("val_snapshot_count") is None else int(data_config["val_snapshot_count"])),
        test_snapshot_count=(None if data_config.get("test_snapshot_count") is None else int(data_config["test_snapshot_count"])),
    )
    dataset: GRMHDPairedDataset = datasets[args.split]
    if args.baseline:
        normalizer_path = Path(data_config["normalizer_stats_path"])
        validate_stats_bundle(
            normalizer_path.parent,
            data_path,
            expected_training_indices=datasets["train"].owned_snapshot_indices,
            expected_window_name=str(data_config["window_name"]),
        )
    else:
        normalizer_path = args.checkpoint.parent / "normalizer_stats.npz"
    normalizer = GRMHDNormalizer.load(
        normalizer_path,
        h5_path=data_path,
        expected_training_indices=datasets["train"].owned_snapshot_indices,
    )
    prior_stats = config.get("resolved", {}).get("prior_stats", {})
    quantile_bounds = (
        QuantileBounds.from_dict(prior_stats["quantile_bounds"])
        if "quantile_bounds" in prior_stats
        else None
    )
    residual_envelope = (
        ResidualEnvelope.from_dict(prior_stats["residual_envelope"])
        if "residual_envelope" in prior_stats
        else None
    )
    evaluation_clamp = bool(config.get("priors", {}).get("evaluation_clamp", False))
    downsample = int(data_config.get("downsample", 1))
    shape = tuple(size // downsample for size in dataset.snapshot_shape[1:])
    shells = None
    if data_config.get("with_shells", True):
        shells, _ = radial_shells_tensor(
            dataset.coords["r"][::downsample],
            shape[0],
            shape[1],
            n_shells=int(data_config.get("n_shells", 8)),
            device=device,
        )
    model_config = config["model"]
    predict_residual = bool(model_config.get("predict_residual", False)) if not args.baseline else False
    bounded_residual = bool(model_config.get("bounded_residual", False)) if not args.baseline else False
    residual_scale = None
    residual_wrapper_validation = None
    if bounded_residual:
        expected_wrapper_config = {
            "predict_residual": True,
            "zero_init_residual_head": bool(model_config.get("zero_init_residual_head", False)),
            "bounded_residual": True,
            "residual_scale_source": str(model_config["residual_scale_source"]),
            "residual_scale_quantile": float(model_config["residual_scale_quantile"]),
            "residual_scale_multiplier": float(model_config["residual_scale_multiplier"]),
        }
        validated_stats = validate_residual_wrapper_metadata(
            checkpoint["residual_wrapper"],
            h5_path=data_path,
            expected_training_indices=datasets["train"].owned_snapshot_indices,
            expected_config=expected_wrapper_config,
        )
        residual_scale = validated_stats.alpha
        residual_wrapper_validation = {
            "status": "passed",
            "config": expected_wrapper_config,
            "alpha": list(residual_scale),
            "source_hdf5_checksum": validated_stats.source_hdf5_checksum,
        }
    kwargs = (
        {"conv_padding_mode": model_config.get("conv_padding_mode", "zeros")}
        if str(model_config["type"]).startswith("localno")
        else {}
    )
    model = None
    if not args.baseline:
        model = build_model(
            str(model_config["type"]),
            in_channels=8 + (shells.shape[0] if shells is not None else 0),
            out_channels=8,
            default_in_shape=shape,
            n_modes=tuple(model_config["n_modes"]),
            hidden_channels=int(model_config["hidden_channels"]),
            n_layers=int(model_config["n_layers"]),
            positional_embedding=model_config.get("positional_embedding"),
            **kwargs,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.eval()

    one_step_model: list[dict[str, Any]] = []
    one_step_persistence: list[dict[str, Any]] = []
    with torch.no_grad():
        for item in range(len(dataset)):
            sample = dataset[item]
            x = prepare_state(sample["x"].unsqueeze(0), normalizer, downsample, device)
            y = prepare_state(sample["y"].unsqueeze(0), normalizer, downsample, device)
            operator_input = model_input(x, shells)
            prediction = (
                x
                if args.baseline
                else apply_prediction_mode(
                    model(operator_input),
                    operator_input,
                    predict_residual=predict_residual,
                    bounded_residual=bounded_residual,
                    residual_scale=residual_scale,
                )
            )
            prediction = apply_evaluation_clamps(
                prediction, x, quantile_bounds, residual_envelope, evaluation_clamp
            )
            model_record, _, _ = evaluate_pair(prediction, y, normalizer)
            persistence_record, _, _ = evaluate_pair(x, y, normalizer)
            record_metadata = {
                "index": sample["index"],
                "target_index": sample["target_index"],
                "source_file": sample["source_file"],
                "target_source_file": sample["target_source_file"],
                "dt": sample["dt"],
                "regime": source_regime(sample["index"], sample["target_index"]),
            }
            model_record.update(record_metadata)
            persistence_record.update(record_metadata)
            one_step_model.append(model_record)
            one_step_persistence.append(persistence_record)

    trajectory = dataset.trajectory_indices()
    available_steps = max(len(trajectory) - 1, 0)
    rollout_steps = min(args.max_steps, available_steps)
    if rollout_steps == 0:
        raise RuntimeError(f"{args.split} split has no rollout transition")
    initial_raw = dataset.load_snapshot(trajectory[0]).unsqueeze(0)
    current_model = prepare_state(initial_raw, normalizer, downsample, device)
    initial_normalized = current_model.clone()
    model_rollout: list[dict[str, Any]] = []
    persistence_rollout: list[dict[str, Any]] = []
    last_model_physical = last_truth_physical = None
    start = time.perf_counter()
    with torch.no_grad():
        for step in range(1, rollout_steps + 1):
            truth_index = trajectory[step]
            truth_normalized = prepare_state(
                dataset.load_snapshot(truth_index).unsqueeze(0), normalizer, downsample, device
            )
            operator_input = model_input(current_model, shells)
            prediction_normalized = (
                current_model
                if args.baseline
                else apply_prediction_mode(
                    model(operator_input),
                    operator_input,
                    predict_residual=predict_residual,
                    bounded_residual=bounded_residual,
                    residual_scale=residual_scale,
                )
            )
            prediction_normalized = apply_evaluation_clamps(
                prediction_normalized,
                current_model,
                quantile_bounds,
                residual_envelope,
                evaluation_clamp,
            )
            model_record, model_physical, truth_physical = evaluate_pair(
                prediction_normalized, truth_normalized, normalizer
            )
            persistence_record, persistence_physical, _ = evaluate_pair(
                initial_normalized, truth_normalized, normalizer
            )
            rollout_metadata = {
                "step": step,
                "index": trajectory[step - 1],
                "truth_index": truth_index,
                "truth_time": float(dataset.times[truth_index]),
                "regime": source_regime(trajectory[step - 1], truth_index),
            }
            model_record.update(rollout_metadata)
            persistence_record.update(rollout_metadata)
            radial_coordinate = dataset.coords["r"][::downsample]
            model_record["statistics_prediction"] = physical_statistics(
                model_physical, radial_coordinate
            )
            model_record["statistics_truth"] = physical_statistics(
                truth_physical, radial_coordinate
            )
            model_record["artifact_diagnostics"] = artifact_diagnostics(
                prediction_normalized, truth_normalized, current_model
            )
            persistence_record["statistics_prediction"] = physical_statistics(
                persistence_physical, radial_coordinate
            )
            persistence_record["statistics_truth"] = model_record["statistics_truth"]
            model_rollout.append(model_record)
            persistence_rollout.append(persistence_record)
            if step == 1:
                plot_prediction_triptychs(
                    model_physical,
                    truth_physical,
                    dataset.coords["r"][::downsample],
                    dataset.coords["theta"][::downsample],
                    out_dir,
                    "rollout_step_0001",
                )
            current_model = prediction_normalized
            last_model_physical, last_truth_physical = model_physical, truth_physical
    elapsed = time.perf_counter() - start

    assert last_model_physical is not None and last_truth_physical is not None
    plot_prediction_triptychs(
        last_model_physical,
        last_truth_physical,
        dataset.coords["r"][::downsample],
        dataset.coords["theta"][::downsample],
        out_dir,
        f"rollout_step_{rollout_steps:04d}",
    )
    plot_radial_profiles(
        last_model_physical,
        last_truth_physical,
        dataset.coords["r"][::downsample],
        out_dir,
        f"rollout_step_{rollout_steps:04d}",
    )
    plot_error_curves(model_rollout, persistence_rollout, out_dir)
    write_error_csv(model_rollout, persistence_rollout, out_dir / "rollout_error.csv")

    requested_horizons = {}
    for horizon in (1, 5, 10, 19, 20):
        available = rollout_steps >= horizon
        model_at_horizon = model_rollout[horizon - 1] if available else None
        persistence_at_horizon = persistence_rollout[horizon - 1] if available else None
        requested_horizons[str(horizon)] = {
            "available": available,
            "reason": None if available else f"ground truth provides only {available_steps} rollout steps",
            "model_metrics": (
                {
                    key: model_at_horizon[key]
                    for key in (
                        "relative_l2_normalized",
                        "relative_l2_physical",
                        "global_volumetric_relative_l2_normalized",
                        "global_volumetric_relative_l2_physical",
                        "positivity_violations",
                        "ranges",
                        "artifact_diagnostics",
                    )
                }
                if available
                else None
            ),
            "persistence_metrics": (
                {
                    key: persistence_at_horizon[key]
                    for key in (
                        "relative_l2_normalized",
                        "relative_l2_physical",
                        "global_volumetric_relative_l2_normalized",
                        "global_volumetric_relative_l2_physical",
                        "positivity_violations",
                        "ranges",
                    )
                }
                if available
                else None
            ),
        }
    one_step_model_summary = aggregate(one_step_model)
    one_step_persistence_summary = aggregate(one_step_persistence)
    rollout_model_summary = aggregate(model_rollout)
    rollout_persistence_summary = aggregate(persistence_rollout)
    comparison = {
        "definition": "error_ratio = model_error / persistence_error; improvement = 1 - error_ratio; positive improvement favors model",
    }
    for label, model_value, persistence_value in (
        ("one_step_normalized", one_step_model_summary["overall_mean_relative_l2_normalized"], one_step_persistence_summary["overall_mean_relative_l2_normalized"]),
        ("one_step_physical", one_step_model_summary["overall_mean_relative_l2_physical"], one_step_persistence_summary["overall_mean_relative_l2_physical"]),
        ("rollout_normalized", rollout_model_summary["overall_mean_relative_l2_normalized"], rollout_persistence_summary["overall_mean_relative_l2_normalized"]),
        ("rollout_physical", rollout_model_summary["overall_mean_relative_l2_physical"], rollout_persistence_summary["overall_mean_relative_l2_physical"]),
    ):
        ratio = model_value / max(persistence_value, 1.0e-30)
        comparison[label] = {"error_ratio": ratio, "improvement": 1.0 - ratio}
    metrics = {
        "checkpoint": str(args.checkpoint.resolve()) if args.checkpoint is not None else None,
        "baseline": args.baseline,
        "run_id": out_dir.name,
        "data": str(Path(data_path).resolve()),
        "data_window": data_config.get("window_name"),
        "split": args.split,
        "prediction_mode": "residual" if predict_residual else "direct",
        "bounded_residual": bounded_residual,
        "residual_wrapper_validation": residual_wrapper_validation,
        "channel_order": list(CHANNELS),
        "coordinate_component_warning": (
            "Bcc1/2/3 and vel1/2/3 remain spherical Kerr-Schild coordinate-basis components. "
            "Reported |B| and |v| profile labels are unweighted component norms, not Cartesian or metric invariants."
        ),
        "divergence_warning": "No physical div(B) metric is claimed without confirmed metric and B convention.",
        "evaluation_clamp": {
            "enabled": evaluation_clamp,
            "quantile_bounds_available": quantile_bounds is not None,
            "residual_envelope_available": residual_envelope is not None,
        },
        "one_step": {
            "model": one_step_model_summary,
            "persistence": one_step_persistence_summary,
            "records_model": one_step_model,
            "records_persistence": one_step_persistence,
            "regime_segments": {
                "model": regime_metrics(one_step_model),
                "persistence": regime_metrics(one_step_persistence),
            },
        },
        "rollout": {
            "initial_index": trajectory[0],
            "available_ground_truth_steps": available_steps,
            "evaluated_steps": rollout_steps,
            "seconds": elapsed,
            "steps_per_second": rollout_steps / elapsed,
            "model": rollout_model_summary,
            "persistence": rollout_persistence_summary,
            "records_model": model_rollout,
            "records_persistence": persistence_rollout,
            "requested_horizons": requested_horizons,
            "statistical_properties": {
                "model": aggregate_statistical_properties(model_rollout, "statistics_prediction"),
                "truth": aggregate_statistical_properties(model_rollout, "statistics_truth"),
                "persistence": aggregate_statistical_properties(
                    persistence_rollout, "statistics_prediction"
                ),
            },
            "statistical_errors": {
                "model": statistical_error_summary(model_rollout),
                "persistence": statistical_error_summary(persistence_rollout),
            },
            "regime_segments": {
                "model": regime_metrics(model_rollout),
                "persistence": regime_metrics(persistence_rollout),
            },
            "artifact_flags": sorted({
                flag
                for record in model_rollout
                for flag in record["artifact_diagnostics"]["flags"]
            }),
        },
        "persistence_comparison": comparison,
    }
    write_regime_outputs(
        run_id=out_dir.name,
        window=str(data_config.get("window_name")),
        prediction_mode="persistence" if args.baseline else ("residual" if predict_residual else "direct"),
        model_records=one_step_model,
        persistence_records=one_step_persistence,
        definition_path=args.regime_definition,
        csv_path=args.regime_metrics_csv,
    )
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "one_step_model": metrics["one_step"]["model"],
                "one_step_persistence": metrics["one_step"]["persistence"],
                "rollout_model": metrics["rollout"]["model"],
                "rollout_persistence": metrics["rollout"]["persistence"],
                "requested_horizons": {
                    horizon: {
                        "available": record["available"],
                        "model_global_physical": (
                            record["model_metrics"]["global_volumetric_relative_l2_physical"]
                            if record["available"]
                            else None
                        ),
                        "persistence_global_physical": (
                            record["persistence_metrics"]["global_volumetric_relative_l2_physical"]
                            if record["available"]
                            else None
                        ),
                    }
                    for horizon, record in metrics["rollout"]["requested_horizons"].items()
                },
                "evaluation_clamp": metrics["evaluation_clamp"],
                "persistence_comparison": metrics["persistence_comparison"],
                "one_step_regime_channel_mean_physical": {
                    method: {
                        segment: values.get("overall_mean_relative_l2_physical")
                        for segment, values in segments.items()
                    }
                    for method, segments in metrics["one_step"]["regime_segments"].items()
                },
                "artifact_flags": metrics["rollout"]["artifact_flags"],
            },
            indent=2,
        )
    )
    print(f"wrote evaluation to {out_dir}")


if __name__ == "__main__":
    main()
