#!/usr/bin/env python
"""Measure normalized one-step residual targets without refitting statistics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from grmhd import CHANNELS
from grmhd.dataset import GRMHDPairedDataset, make_temporal_datasets
from grmhd.models import apply_prediction_mode
from grmhd.normalizer import GRMHDNormalizer, validate_stats_bundle


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


def make_datasets(data: dict[str, Any]) -> dict[str, GRMHDPairedDataset]:
    return make_temporal_datasets(
        data["path"],
        stride=int(data["stride"]),
        train_fraction=float(data["train_fraction"]),
        val_fraction=float(data["val_fraction"]),
        snapshot_start=int(data.get("snapshot_start", 0)),
        snapshot_end=None if data.get("snapshot_end") is None else int(data["snapshot_end"]),
        train_snapshot_count=None
        if data.get("train_snapshot_count") is None
        else int(data["train_snapshot_count"]),
        val_snapshot_count=None
        if data.get("val_snapshot_count") is None
        else int(data["val_snapshot_count"]),
        test_snapshot_count=None
        if data.get("test_snapshot_count") is None
        else int(data["test_snapshot_count"]),
    )


def analyze(
    train: GRMHDPairedDataset,
    normalizer: GRMHDNormalizer,
    *,
    max_quantile_samples: int,
    seed: int,
) -> dict[str, Any]:
    spatial_voxels = int(np.prod(train.snapshot_shape[1:]))
    population_per_channel = len(train) * spatial_voxels
    sample_count = min(max_quantile_samples, population_per_channel)
    choices = [
        np.sort(
            np.random.default_rng(seed + 104729 * channel).choice(
                population_per_channel, size=sample_count, replace=False
            )
        )
        for channel in range(len(CHANNELS))
    ]
    samples = np.empty((len(CHANNELS), sample_count), dtype=np.float32)
    sample_offsets = np.zeros(len(CHANNELS), dtype=np.int64)
    residual_sum = np.zeros(len(CHANNELS), dtype=np.float64)
    residual_sum_squares = np.zeros(len(CHANNELS), dtype=np.float64)
    target_sum_squares = np.zeros(len(CHANNELS), dtype=np.float64)
    input_sum_squares = np.zeros(len(CHANNELS), dtype=np.float64)
    maxabs = np.zeros(len(CHANNELS), dtype=np.float64)
    zero_residual_max_difference = 0.0

    for pair_slot in range(len(train)):
        pair = train[pair_slot]
        x = normalizer.encode_tensor(pair["x"])
        y = normalizer.encode_tensor(pair["y"])
        delta = y - x
        if not torch.isfinite(delta).all():
            raise FloatingPointError(f"Non-finite residual at training pair {pair_slot}")
        shell_probe = torch.zeros((1, 10, 2, 2, 2), dtype=torch.float32)
        shell_probe[:, 8:] = 17.0
        zero_residual = apply_prediction_mode(
            torch.zeros((1, 8, 2, 2, 2)), shell_probe, predict_residual=True
        )
        zero_residual_max_difference = max(
            zero_residual_max_difference,
            float(torch.max(torch.abs(zero_residual - shell_probe[:, :8])).item()),
        )

        delta_np = delta.numpy()
        x_np = x.numpy()
        y_np = y.numpy()
        for channel in range(len(CHANNELS)):
            residual64 = np.asarray(delta_np[channel], dtype=np.float64).reshape(-1)
            target64 = np.asarray(y_np[channel], dtype=np.float64).reshape(-1)
            input64 = np.asarray(x_np[channel], dtype=np.float64).reshape(-1)
            residual_sum[channel] += residual64.sum(dtype=np.float64)
            residual_sum_squares[channel] += np.dot(residual64, residual64)
            target_sum_squares[channel] += np.dot(target64, target64)
            input_sum_squares[channel] += np.dot(input64, input64)
            maxabs[channel] = max(maxabs[channel], float(np.max(np.abs(residual64))))

            lower = pair_slot * spatial_voxels
            upper = lower + spatial_voxels
            selected = choices[channel]
            mask = (selected >= lower) & (selected < upper)
            local = selected[mask] - lower
            count = int(mask.sum())
            if count:
                offset = int(sample_offsets[channel])
                samples[channel, offset : offset + count] = delta_np[channel].reshape(-1)[local]
                sample_offsets[channel] += count

    if not np.all(sample_offsets == sample_count):
        raise RuntimeError(
            f"Residual quantile sample fill mismatch: {sample_offsets.tolist()} != {sample_count}"
        )
    total_count = population_per_channel
    channels: list[dict[str, Any]] = []
    for channel, name in enumerate(CHANNELS):
        mean = residual_sum[channel] / total_count
        variance = max(residual_sum_squares[channel] / total_count - mean * mean, 0.0)
        median = float(np.median(samples[channel]))
        quantiles = np.quantile(samples[channel], [0.001, 0.01, 0.5, 0.99, 0.999])
        target_ratio = float(
            np.sqrt(residual_sum_squares[channel] / target_sum_squares[channel])
        )
        input_ratio = float(
            np.sqrt(residual_sum_squares[channel] / input_sum_squares[channel])
        )
        channels.append(
            {
                "channel": name,
                "mean": float(mean),
                "std": float(np.sqrt(variance)),
                "median": median,
                "mad": float(np.median(np.abs(samples[channel] - median))),
                "q0.001": float(quantiles[0]),
                "q0.01": float(quantiles[1]),
                "q0.5": float(quantiles[2]),
                "q0.99": float(quantiles[3]),
                "q0.999": float(quantiles[4]),
                "maxabs": float(maxabs[channel]),
                "residual_norm_over_target_state_norm": target_ratio,
                "residual_norm_over_input_state_norm": input_ratio,
                "persistence_one_step_normalized_relative_l2": target_ratio,
            }
        )

    aggregate_ratio = float(
        np.sqrt(residual_sum_squares.sum() / target_sum_squares.sum())
    )
    return {
        "definition": {
            "state_space": "saved-normalizer encoded state space",
            "residual_target": "delta = encode(y) - encode(x)",
            "persistence_prediction": "encode(x)",
            "relative_l2": "sqrt(sum(delta^2) / sum(encode(y)^2))",
            "channel_aggregation": "per-channel values plus arithmetic mean and global volumetric ratio",
        },
        "coverage": {
            "train_pair_count": len(train),
            "train_pair_source_indices": [int(index) for index in train.pair_starts],
            "population_per_channel": population_per_channel,
            "exact_statistics": ["mean", "std", "maxabs", "L2 norm ratios"],
            "sampled_statistics": [
                "median",
                "MAD (unscaled median absolute deviation)",
                "q0.001",
                "q0.01",
                "q0.5",
                "q0.99",
                "q0.999",
            ],
            "quantile_sample_count_per_channel": sample_count,
            "quantile_sampling": "uniform without replacement over all train-pair voxels",
            "seed": seed,
        },
        "implementation_checks": {
            "normalizer_reused_without_refit": True,
            "residual_computed_after_single_state_normalization": True,
            "residual_shortcut_space": "normalized state space",
            "residual_channels": list(CHANNELS),
            "shell_channels_in_delta": False,
            "direct_target": "encode(y)",
            "residual_model_target": "encode(y) via prediction encode(x) + model_delta",
            "zero_residual_equals_persistence": zero_residual_max_difference == 0.0,
            "zero_residual_max_absolute_difference": zero_residual_max_difference,
        },
        "channels": channels,
        "persistence_one_step_normalized_relative_l2": {
            "global_volumetric": aggregate_ratio,
            "arithmetic_mean_over_channels": float(
                np.mean(
                    [
                        item["persistence_one_step_normalized_relative_l2"]
                        for item in channels
                    ]
                )
            ),
            "per_channel": {
                item["channel"]: item["persistence_one_step_normalized_relative_l2"]
                for item in channels
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--max-quantile-samples", type=int, default=200_000)
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    data = config["data"]
    datasets = make_datasets(data)
    train = datasets["train"]
    stats_dir = Path(data["normalizer_stats_path"]).parent
    validation = validate_stats_bundle(
        stats_dir,
        data["path"],
        expected_training_indices=train.owned_snapshot_indices,
        expected_window_name=str(data["window_name"]),
    )
    normalizer = GRMHDNormalizer.load(
        data["normalizer_stats_path"],
        h5_path=data["path"],
        expected_training_indices=train.owned_snapshot_indices,
    )
    result = analyze(
        train,
        normalizer,
        max_quantile_samples=args.max_quantile_samples,
        seed=int(config["seed"]),
    )
    result["data_window"] = data["window_name"]
    result["stats_validation"] = validation
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with args.output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(result["channels"][0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(result["channels"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
