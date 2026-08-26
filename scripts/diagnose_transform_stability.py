#!/usr/bin/env python
"""Diagnose normalizer saturation and decoded range growth for frozen checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str((Path("outputs") / ".matplotlib").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from grmhd import CHANNELS
from grmhd.dataset import GRMHDPairedDataset, make_temporal_datasets
from grmhd.models import apply_prediction_mode, build_model
from grmhd.normalizer import GRMHDNormalizer, TRANSFORMS, validate_stats_bundle
from grmhd.shells import radial_shells_tensor


QUANTILES = (0.001, 0.01, 0.5, 0.99, 0.999)
Z_THRESHOLDS = (4.0, 5.0, 5.5, 5.9)
CLIP_FRACTIONS = (0.90, 0.95, 0.99)
AMPLIFICATION_QUANTILES = (0.5, 0.9, 0.99, 0.999)


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


class ChannelAccumulator:
    """Exact moments/tails plus deterministic bounded samples for quantiles."""

    def __init__(self, seed: int, samples_per_update: int = 2048) -> None:
        self.rng = np.random.default_rng(seed)
        self.samples_per_update = samples_per_update
        self.count = 0
        self.total = 0.0
        self.total_squares = 0.0
        self.minimum = math.inf
        self.maximum = -math.inf
        self.z_counts = {threshold: 0 for threshold in Z_THRESHOLDS}
        self.clip_counts = {fraction: 0 for fraction in CLIP_FRACTIONS}
        self.samples: list[np.ndarray] = []

    def update(self, values: np.ndarray, gamma: float) -> None:
        flat = np.asarray(values, dtype=np.float64).reshape(-1)
        if not np.isfinite(flat).all():
            raise FloatingPointError("Transform diagnostics received non-finite values")
        self.count += flat.size
        self.total += float(flat.sum(dtype=np.float64))
        self.total_squares += float(np.dot(flat, flat))
        self.minimum = min(self.minimum, float(flat.min()))
        self.maximum = max(self.maximum, float(flat.max()))
        absolute = np.abs(flat)
        for threshold in Z_THRESHOLDS:
            self.z_counts[threshold] += int(np.count_nonzero(absolute > threshold))
        for fraction in CLIP_FRACTIONS:
            self.clip_counts[fraction] += int(
                np.count_nonzero(absolute > fraction * gamma)
            )
        take = min(self.samples_per_update, flat.size)
        selected = self.rng.choice(flat.size, size=take, replace=False)
        self.samples.append(flat[selected].copy())

    def finalize(self) -> dict[str, Any]:
        if self.count == 0:
            raise RuntimeError("Cannot finalize an empty accumulator")
        sample = np.concatenate(self.samples)
        mean = self.total / self.count
        variance = max(self.total_squares / self.count - mean * mean, 0.0)
        quantiles = np.quantile(sample, QUANTILES)
        return {
            "count": self.count,
            "min": self.minimum,
            "max": self.maximum,
            "mean": mean,
            "std": math.sqrt(variance),
            **{
                f"q{quantile:g}": float(value)
                for quantile, value in zip(QUANTILES, quantiles, strict=True)
            },
            **{
                f"abs_gt_{threshold:g}_fraction": self.z_counts[threshold] / self.count
                for threshold in Z_THRESHOLDS
            },
            **{
                f"abs_gt_{fraction:.2f}_gamma_fraction": self.clip_counts[fraction]
                / self.count
                for fraction in CLIP_FRACTIONS
            },
            "quantile_sample_count": int(sample.size),
        }


def make_accumulators(label: str) -> list[ChannelAccumulator]:
    base = sum(ord(character) for character in label)
    return [ChannelAccumulator(42 + base + 104729 * channel) for channel in range(8)]


def update_stage(
    stages: dict[str, list[ChannelAccumulator]],
    stage: str,
    values: np.ndarray,
    gamma: float,
) -> None:
    if stage not in stages:
        stages[stage] = make_accumulators(stage)
    if values.shape[0] != 8:
        raise ValueError(f"Expected channel-first values, found {values.shape}")
    for channel in range(8):
        stages[stage][channel].update(values[channel], gamma)


def finalize_stages(
    stages: dict[str, list[ChannelAccumulator]],
) -> dict[str, dict[str, Any]]:
    return {
        stage: {
            channel: accumulator.finalize()
            for channel, accumulator in zip(CHANNELS, accumulators, strict=True)
        }
        for stage, accumulators in stages.items()
    }


def transform_stages(
    raw: np.ndarray, normalizer: GRMHDNormalizer
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    transformed = np.empty_like(raw, dtype=np.float64)
    robust_z = np.empty_like(raw, dtype=np.float64)
    clipped = np.empty_like(raw, dtype=np.float64)
    for channel in range(8):
        transformed[channel] = normalizer._pretransform_numpy(
            raw[channel], channel, normalizer.epsilon
        )
        baseline = normalizer._radial_baseline_numpy(channel, raw.shape[-1])
        if baseline is not None:
            transformed[channel] -= baseline.reshape((1, 1, -1))
        robust_z[channel] = (
            transformed[channel] - normalizer.median[channel]
        ) / normalizer.scale[channel]
        clipped[channel] = normalizer.gamma * np.tanh(
            robust_z[channel] / normalizer.gamma
        )
    return transformed, robust_z, clipped


def inverse_stages(
    normalized: np.ndarray, normalizer: GRMHDNormalizer
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    effective = np.clip(
        np.asarray(normalized, dtype=np.float64),
        -0.99 * normalizer.gamma,
        0.99 * normalizer.gamma,
    )
    robust_z = normalizer.gamma * np.arctanh(effective / normalizer.gamma)
    transformed = np.empty_like(robust_z)
    decoded = np.empty_like(robust_z)
    for channel in range(8):
        transformed[channel] = (
            robust_z[channel] * normalizer.scale[channel]
            + normalizer.median[channel]
        )
        baseline = normalizer._radial_baseline_numpy(channel, normalized.shape[-1])
        if baseline is not None:
            transformed[channel] += baseline.reshape((1, 1, -1))
        kind = TRANSFORMS[channel]
        if kind == "signed_log":
            decoded[channel] = (
                np.sign(transformed[channel])
                * normalizer.epsilon[channel]
                * (np.power(10.0, np.abs(transformed[channel])) - 1.0)
            )
        elif kind == "positive_log":
            decoded[channel] = np.maximum(
                np.power(10.0, transformed[channel]) - normalizer.epsilon[channel],
                np.finfo(np.float64).tiny,
            )
        else:
            decoded[channel] = transformed[channel]
    if not np.isfinite(decoded).all():
        raise FloatingPointError("High-precision inverse transform produced non-finite values")
    return robust_z, transformed, decoded


def inverse_amplification(
    normalized: np.ndarray, normalizer: GRMHDNormalizer
) -> np.ndarray:
    """Return analytic |d physical / d normalized| for the current decoder."""
    effective = np.clip(
        np.asarray(normalized, dtype=np.float64),
        -0.99 * normalizer.gamma,
        0.99 * normalizer.gamma,
    )
    robust_z, transformed, _ = inverse_stages(effective, normalizer)
    del robust_z
    denominator = 1.0 - np.square(effective / normalizer.gamma)
    result = np.empty_like(effective)
    for channel in range(8):
        base = normalizer.scale[channel] / denominator[channel]
        kind = TRANSFORMS[channel]
        if kind == "signed_log":
            result[channel] = (
                math.log(10.0)
                * normalizer.epsilon[channel]
                * np.power(10.0, np.abs(transformed[channel]))
                * base
            )
        elif kind == "positive_log":
            power = np.power(10.0, transformed[channel])
            active = (
                power - normalizer.epsilon[channel]
                > np.finfo(np.float64).tiny
            )
            result[channel] = np.where(
                active, math.log(10.0) * power * base, 0.0
            )
        else:
            result[channel] = base
    return result


def inverse_amplification_factors(
    normalized: np.ndarray, normalizer: GRMHDNormalizer
) -> tuple[np.ndarray, np.ndarray]:
    """Separate atanh saturation from the inverse channel-transform derivative."""
    effective = np.clip(
        np.asarray(normalized, dtype=np.float64),
        -0.99 * normalizer.gamma,
        0.99 * normalizer.gamma,
    )
    _, transformed, _ = inverse_stages(effective, normalizer)
    atanh_factor = 1.0 / (1.0 - np.square(effective / normalizer.gamma))
    channel_inverse_factor = np.ones_like(effective)
    for channel in range(8):
        kind = TRANSFORMS[channel]
        if kind == "signed_log":
            channel_inverse_factor[channel] = (
                math.log(10.0)
                * normalizer.epsilon[channel]
                * np.power(10.0, np.abs(transformed[channel]))
            )
        elif kind == "positive_log":
            power = np.power(10.0, transformed[channel])
            active = (
                power - normalizer.epsilon[channel]
                > np.finfo(np.float64).tiny
            )
            channel_inverse_factor[channel] = np.where(
                active, math.log(10.0) * power, 0.0
            )
    return atanh_factor, channel_inverse_factor


def scalar_inverse(value: float, channel: int, normalizer: GRMHDNormalizer) -> float:
    effective = float(np.clip(value, -0.99 * normalizer.gamma, 0.99 * normalizer.gamma))
    robust_z = normalizer.gamma * math.atanh(effective / normalizer.gamma)
    transformed = robust_z * normalizer.scale[channel] + normalizer.median[channel]
    kind = TRANSFORMS[channel]
    if kind == "signed_log":
        return math.copysign(
            normalizer.epsilon[channel]
            * (math.pow(10.0, abs(transformed)) - 1.0),
            transformed,
        )
    if kind == "positive_log":
        return max(
            math.pow(10.0, transformed) - normalizer.epsilon[channel],
            np.finfo(np.float64).tiny,
        )
    return transformed


def derivative_validation(normalizer: GRMHDNormalizer) -> dict[str, Any]:
    result = {}
    for channel, name in enumerate(CHANNELS):
        records = []
        for normalized in (-0.9 * normalizer.gamma, 0.0, 0.9 * normalizer.gamma):
            h = 1.0e-6
            finite = (
                scalar_inverse(normalized + h, channel, normalizer)
                - scalar_inverse(normalized - h, channel, normalizer)
            ) / (2.0 * h)
            analytic = inverse_amplification(
                np.full((8, 1, 1, 1), normalized, dtype=np.float64), normalizer
            )[channel, 0, 0, 0]
            relative_error = abs(abs(finite) - analytic) / max(analytic, 1.0e-300)
            records.append(
                {
                    "normalized": normalized,
                    "analytic_abs_derivative": float(analytic),
                    "finite_difference_abs_derivative": float(abs(finite)),
                    "relative_error": float(relative_error),
                }
            )
        result[name] = records
    return result


def make_datasets(config: dict[str, Any]) -> dict[str, GRMHDPairedDataset]:
    data = config["data"]
    return make_temporal_datasets(
        data["path"],
        stride=int(data["stride"]),
        train_fraction=float(data["train_fraction"]),
        val_fraction=float(data["val_fraction"]),
        snapshot_start=int(data.get("snapshot_start", 0)),
        snapshot_end=None if data.get("snapshot_end") is None else int(data["snapshot_end"]),
        train_snapshot_count=int(data["train_snapshot_count"]),
        val_snapshot_count=int(data["val_snapshot_count"]),
        test_snapshot_count=int(data["test_snapshot_count"]),
    )


def build_checkpoint_model(
    checkpoint_path: Path,
    shape: tuple[int, int, int],
    shells: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model_config = config["model"]
    kwargs = (
        {"conv_padding_mode": model_config.get("conv_padding_mode", "zeros")}
        if str(model_config["type"]).startswith("localno")
        else {}
    )
    model = build_model(
        str(model_config["type"]),
        in_channels=8 + (shells.shape[0] if shells is not None else 0),
        out_channels=8,
        default_in_shape=shape,
        n_modes=tuple(int(value) for value in model_config["n_modes"]),
        hidden_channels=int(model_config["hidden_channels"]),
        n_layers=int(model_config["n_layers"]),
        positional_embedding=model_config.get("positional_embedding"),
        **kwargs,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, config


def relative_l2(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(
        np.linalg.norm((prediction - truth).reshape(-1))
        / max(np.linalg.norm(truth.reshape(-1)), 1.0e-30)
    )


def per_channel_relative_l2(prediction: np.ndarray, truth: np.ndarray) -> list[float]:
    return [relative_l2(prediction[channel], truth[channel]) for channel in range(8)]


def run_rollout(
    *,
    name: str,
    model: torch.nn.Module | None,
    config: dict[str, Any] | None,
    dataset: GRMHDPairedDataset,
    normalizer: GRMHDNormalizer,
    shells: torch.Tensor | None,
    device: torch.device,
    train_bounds: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    stages: dict[str, list[ChannelAccumulator]] = {}
    trajectory = dataset.trajectory_indices()
    current = normalizer.encode_tensor(dataset.load_snapshot(trajectory[0]).unsqueeze(0), channel_axis=1).to(device)
    predict_residual = bool(config["model"].get("predict_residual", False)) if config else False
    range_rows = []
    first_explosion = None
    out_of_train = {
        channel: {"below": 0, "above": 0, "count": 0} for channel in CHANNELS
    }
    with torch.no_grad():
        for step, truth_index in enumerate(trajectory[1:], start=1):
            operator_input = (
                current
                if shells is None
                else torch.cat(
                    [current, shells.unsqueeze(0).expand(current.shape[0], -1, -1, -1, -1)],
                    dim=1,
                )
            )
            if model is None:
                raw_output = torch.zeros_like(current)
                prediction = current
            else:
                raw_output = model(operator_input)
                prediction = apply_prediction_mode(
                    raw_output,
                    operator_input,
                    predict_residual=predict_residual,
                )
            prediction_np = prediction[0].detach().cpu().numpy().astype(np.float64)
            raw_output_np = raw_output[0].detach().cpu().numpy().astype(np.float64)
            inverse_z, inverse_transformed, decoded = inverse_stages(
                prediction_np, normalizer
            )
            amplification = inverse_amplification(prediction_np, normalizer)
            atanh_factor, channel_inverse_factor = inverse_amplification_factors(
                prediction_np, normalizer
            )
            truth_raw = dataset.load_snapshot(truth_index).numpy().astype(np.float64)
            truth_normalized = normalizer.encode_numpy(truth_raw)
            update_stage(stages, "network_output", raw_output_np, normalizer.gamma)
            update_stage(stages, "model_prediction", prediction_np, normalizer.gamma)
            update_stage(stages, "inverse_soft_clip", inverse_z, normalizer.gamma)
            update_stage(
                stages, "inverse_channel_transform", inverse_transformed, normalizer.gamma
            )
            update_stage(stages, "decoded_value", decoded, normalizer.gamma)
            update_stage(stages, "inverse_amplification", amplification, normalizer.gamma)
            update_stage(
                stages, "inverse_soft_clip_atanh_factor", atanh_factor, normalizer.gamma
            )
            update_stage(
                stages,
                "inverse_channel_transform_derivative",
                channel_inverse_factor,
                normalizer.gamma,
            )

            channel_ranges = {}
            for channel, channel_name in enumerate(CHANNELS):
                prediction_range = float(np.ptp(decoded[channel]))
                truth_range = float(np.ptp(truth_raw[channel]))
                prediction_absmax = float(np.max(np.abs(decoded[channel])))
                truth_absmax = float(np.max(np.abs(truth_raw[channel])))
                range_ratio = prediction_range / max(truth_range, 1.0e-30)
                absmax_ratio = prediction_absmax / max(truth_absmax, 1.0e-30)
                lower, upper = train_bounds[channel_name]
                out_of_train[channel_name]["below"] += int(
                    np.count_nonzero(decoded[channel] < lower)
                )
                out_of_train[channel_name]["above"] += int(
                    np.count_nonzero(decoded[channel] > upper)
                )
                out_of_train[channel_name]["count"] += decoded[channel].size
                channel_ranges[channel_name] = {
                    "prediction_min": float(decoded[channel].min()),
                    "prediction_max": float(decoded[channel].max()),
                    "prediction_range": prediction_range,
                    "truth_min": float(truth_raw[channel].min()),
                    "truth_max": float(truth_raw[channel].max()),
                    "truth_range": truth_range,
                    "range_ratio_prediction_over_truth": range_ratio,
                    "absmax_ratio_prediction_over_truth": absmax_ratio,
                    "normalized_relative_l2": relative_l2(
                        prediction_np[channel], truth_normalized[channel]
                    ),
                    "decoded_relative_l2": relative_l2(
                        decoded[channel], truth_raw[channel]
                    ),
                }
                explosion_score = max(range_ratio, absmax_ratio)
                if first_explosion is None and explosion_score > 5.0:
                    first_explosion = {
                        "definition": "first channel with decoded range or absmax > 5x concurrent truth",
                        "step": step,
                        "channel": channel_name,
                        "score": explosion_score,
                        "range_ratio": range_ratio,
                        "absmax_ratio": absmax_ratio,
                        "global_normalized_relative_l2": relative_l2(
                            prediction_np, truth_normalized
                        ),
                        "channel_normalized_relative_l2": channel_ranges[channel_name][
                            "normalized_relative_l2"
                        ],
                        "global_decoded_relative_l2": relative_l2(decoded, truth_raw),
                    }
            range_rows.append(
                {
                    "step": step,
                    "truth_index": truth_index,
                    "global_normalized_relative_l2": relative_l2(
                        prediction_np, truth_normalized
                    ),
                    "global_decoded_relative_l2": relative_l2(decoded, truth_raw),
                    "per_channel_normalized_relative_l2": per_channel_relative_l2(
                        prediction_np, truth_normalized
                    ),
                    "per_channel_decoded_relative_l2": per_channel_relative_l2(
                        decoded, truth_raw
                    ),
                    "channels": channel_ranges,
                }
            )
            current = prediction
    out_fraction = {
        channel: {
            "below_train_q0.001_fraction": values["below"] / values["count"],
            "above_train_q0.999_fraction": values["above"] / values["count"],
            "outside_train_quantiles_fraction": (
                values["below"] + values["above"]
            )
            / values["count"],
        }
        for channel, values in out_of_train.items()
    }
    return {
        "stages": finalize_stages(stages),
        "out_of_train_physical_quantiles": out_fraction,
        "range_drift_by_step": range_rows,
        "first_decoded_range_explosion": first_explosion,
    }


def plot_amplification(methods: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(1, len(methods), figsize=(6 * len(methods), 5), sharey=True)
    if len(methods) == 1:
        axes = [axes]
    x = np.arange(len(CHANNELS))
    for axis, (method, payload) in zip(axes, methods.items(), strict=True):
        stage = payload["stages"]["inverse_amplification"]
        for quantile, marker in (("q0.5", "o"), ("q0.99", "s"), ("q0.999", "^"), ("max", "x")):
            axis.plot(
                x,
                [stage[channel][quantile] for channel in CHANNELS],
                marker=marker,
                label=quantile,
            )
        axis.set_yscale("log")
        axis.set_xticks(x, CHANNELS, rotation=45, ha="right")
        axis.set_title(method)
        axis.set_ylabel("|d physical / d normalized|")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_range_drift(methods: dict[str, Any], path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    for method, payload in methods.items():
        rows = payload["range_drift_by_step"]
        steps = [row["step"] for row in rows]
        axes[0].plot(
            steps,
            [row["global_decoded_relative_l2"] for row in rows],
            marker="o",
            label=method,
        )
        for channel in CHANNELS[:3]:
            axes[1].plot(
                steps,
                [row["channels"][channel]["range_ratio_prediction_over_truth"] for row in rows],
                label=f"{method}:{channel}",
            )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("decoded global relative L2")
    axes[1].set_yscale("log")
    axes[1].axhline(5.0, color="black", linestyle="--", linewidth=1, label="explosion threshold")
    axes[1].set_ylabel("decoded range / truth range")
    axes[1].set_xlabel("rollout step")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_csv(payload: dict[str, Any], path: Path) -> None:
    fields = [
        "scope", "stage", "channel", "count", "min", "max", "mean", "std",
        "q0.001", "q0.01", "q0.5", "q0.99", "q0.999",
        "abs_gt_4_fraction", "abs_gt_5_fraction", "abs_gt_5.5_fraction",
        "abs_gt_5.9_fraction", "abs_gt_0.90_gamma_fraction",
        "abs_gt_0.95_gamma_fraction", "abs_gt_0.99_gamma_fraction",
        "outside_train_quantiles_fraction",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        scopes = {
            "train_reference": payload["reference"]["train"]["stages"],
            "test_truth": payload["reference"]["test_truth"]["stages"],
            **{
                method: values["stages"] for method, values in payload["methods"].items()
            },
        }
        for scope, stages in scopes.items():
            for stage, channels in stages.items():
                for channel, values in channels.items():
                    row = {field: None for field in fields}
                    row.update({"scope": scope, "stage": stage, "channel": channel})
                    row.update({key: value for key, value in values.items() if key in row})
                    if scope in payload["methods"] and stage == "decoded_value":
                        row["outside_train_quantiles_fraction"] = payload["methods"][scope][
                            "out_of_train_physical_quantiles"
                        ][channel]["outside_train_quantiles_fraction"]
                    writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/data/all111.yaml"))
    parser.add_argument(
        "--fno-checkpoint",
        type=Path,
        default=Path("outputs/experiment_round1/pilot20_fno_residual_all111/best.pt"),
    )
    parser.add_argument(
        "--localno-checkpoint",
        type=Path,
        default=Path("outputs/experiment_round1/pilot20_localno_residual_all111/best.pt"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/experiment_round2")
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    datasets = make_datasets(config)
    data = config["data"]
    validation = validate_stats_bundle(
        Path(data["normalizer_stats_path"]).parent,
        data["path"],
        expected_training_indices=datasets["train"].owned_snapshot_indices,
        expected_window_name=str(data["window_name"]),
    )
    normalizer = GRMHDNormalizer.load(
        data["normalizer_stats_path"],
        h5_path=data["path"],
        expected_training_indices=datasets["train"].owned_snapshot_indices,
    )
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    shape = tuple(int(value) for value in datasets["train"].snapshot_shape[1:])
    shells = None
    if data.get("with_shells", True):
        shells, _ = radial_shells_tensor(
            datasets["train"].coords["r"],
            shape[0],
            shape[1],
            n_shells=int(data.get("n_shells", 8)),
            device=device,
        )

    reference: dict[str, Any] = {}
    train_stages: dict[str, list[ChannelAccumulator]] = {}
    for index in datasets["train"].owned_snapshot_indices:
        raw = datasets["train"].load_snapshot(index).numpy().astype(np.float64)
        transformed, robust_z, clipped = transform_stages(raw, normalizer)
        update_stage(train_stages, "raw_physical_value", raw, normalizer.gamma)
        update_stage(train_stages, "channel_transform_x_hat", transformed, normalizer.gamma)
        update_stage(train_stages, "robust_z_score", robust_z, normalizer.gamma)
        update_stage(train_stages, "soft_clipped_z", clipped, normalizer.gamma)
        update_stage(
            train_stages,
            "inverse_amplification",
            inverse_amplification(clipped, normalizer),
            normalizer.gamma,
        )
        atanh_factor, channel_inverse_factor = inverse_amplification_factors(
            clipped, normalizer
        )
        update_stage(
            train_stages,
            "inverse_soft_clip_atanh_factor",
            atanh_factor,
            normalizer.gamma,
        )
        update_stage(
            train_stages,
            "inverse_channel_transform_derivative",
            channel_inverse_factor,
            normalizer.gamma,
        )
    reference["train"] = {"stages": finalize_stages(train_stages)}
    train_bounds = {
        channel: (
            reference["train"]["stages"]["raw_physical_value"][channel]["q0.001"],
            reference["train"]["stages"]["raw_physical_value"][channel]["q0.999"],
        )
        for channel in CHANNELS
    }

    test_stages: dict[str, list[ChannelAccumulator]] = {}
    for index in datasets["test"].owned_snapshot_indices:
        raw = datasets["test"].load_snapshot(index).numpy().astype(np.float64)
        transformed, robust_z, clipped = transform_stages(raw, normalizer)
        update_stage(test_stages, "raw_physical_value", raw, normalizer.gamma)
        update_stage(test_stages, "channel_transform_x_hat", transformed, normalizer.gamma)
        update_stage(test_stages, "robust_z_score", robust_z, normalizer.gamma)
        update_stage(test_stages, "soft_clipped_z", clipped, normalizer.gamma)
        update_stage(
            test_stages,
            "inverse_amplification",
            inverse_amplification(clipped, normalizer),
            normalizer.gamma,
        )
        atanh_factor, channel_inverse_factor = inverse_amplification_factors(
            clipped, normalizer
        )
        update_stage(
            test_stages,
            "inverse_soft_clip_atanh_factor",
            atanh_factor,
            normalizer.gamma,
        )
        update_stage(
            test_stages,
            "inverse_channel_transform_derivative",
            channel_inverse_factor,
            normalizer.gamma,
        )
    reference["test_truth"] = {"stages": finalize_stages(test_stages)}

    methods = {
        "persistence": run_rollout(
            name="persistence",
            model=None,
            config=None,
            dataset=datasets["test"],
            normalizer=normalizer,
            shells=shells,
            device=device,
            train_bounds=train_bounds,
        )
    }
    for name, checkpoint_path in (
        ("fno_residual", args.fno_checkpoint),
        ("localno_residual", args.localno_checkpoint),
    ):
        model, checkpoint_config = build_checkpoint_model(
            checkpoint_path, shape, shells, device
        )
        methods[name] = run_rollout(
            name=name,
            model=model,
            config=checkpoint_config,
            dataset=datasets["test"],
            normalizer=normalizer,
            shells=shells,
            device=device,
            train_bounds=train_bounds,
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    payload = {
        "definition": {
            "pipeline": (
                "physical -> channel transform x_hat -> robust z -> gamma*tanh(z/gamma); "
                "decoder hard-clips normalized values to +/-0.99gamma, applies gamma*atanh, "
                "then inverse channel transform"
            ),
            "inverse_amplification": "analytic absolute derivative d(physical)/d(normalized soft-clipped state)",
            "range_explosion": "first channel whose decoded range or absmax exceeds concurrent truth by >5x",
            "quantile_levels": list(QUANTILES),
            "moments_and_threshold_fractions": "exact over every included voxel",
            "quantile_method": "deterministic bounded sample; sample count recorded per statistic",
        },
        "data": {
            "window": data["window_name"],
            "hdf5": str(Path(data["path"]).resolve()),
            "stats_validation": validation,
            "train_snapshot_indices": list(datasets["train"].owned_snapshot_indices),
            "test_snapshot_indices": list(datasets["test"].owned_snapshot_indices),
            "train_physical_quantile_bounds": train_bounds,
        },
        "normalizer": normalizer.as_dict(),
        "analytic_derivative_finite_difference_validation": derivative_validation(normalizer),
        "reference": reference,
        "methods": methods,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "transform_diagnostics.json"
    csv_path = args.output_dir / "transform_diagnostics.csv"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_csv(payload, csv_path)
    plot_amplification(methods, args.output_dir / "transform_amplification.png")
    plot_range_drift(methods, args.output_dir / "range_drift_by_step.png")
    print(
        json.dumps(
            {
                "device": str(device),
                "first_decoded_range_explosion": {
                    method: values["first_decoded_range_explosion"]
                    for method, values in methods.items()
                },
                "prediction_saturation_abs_gt_0.95_gamma": {
                    method: {
                        channel: values["stages"]["model_prediction"][channel][
                            "abs_gt_0.95_gamma_fraction"
                        ]
                        for channel in CHANNELS
                    }
                    for method, values in methods.items()
                },
                "amplification_q0.99": {
                    method: {
                        channel: values["stages"]["inverse_amplification"][channel]["q0.99"]
                        for channel in CHANNELS
                    }
                    for method, values in methods.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
