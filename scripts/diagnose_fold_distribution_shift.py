#!/usr/bin/env python
"""Rolling-origin, train-only-normalized distribution-shift diagnosis."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from grmhd import CHANNELS
from grmhd.dataset import GRMHDPairedDataset, make_temporal_datasets
from grmhd.models import apply_prediction_mode
from grmhd.normalizer import GRMHDNormalizer
from grmhd.residual import ResidualScaleStats


FOLDS = {
    "A": {"snapshot_end": 90, "train": 70, "val": 10, "test": 10},
    "B": {"snapshot_end": 100, "train": 80, "val": 10, "test": 10},
    "C": {"snapshot_end": 111, "train": 90, "val": 10, "test": 11},
}
QUANTILES = (0.001, 0.01, 0.5, 0.99, 0.999)


def sample_choices(
    item_count: int, voxels_per_item: int, sample_count: int, seed: int
) -> list[np.ndarray]:
    total = item_count * voxels_per_item
    count = min(sample_count, total)
    return [
        np.sort(
            np.random.default_rng(seed + channel * 104729).choice(
                total, size=count, replace=False
            )
        )
        for channel in range(len(CHANNELS))
    ]


def snapshot_distribution(
    dataset: GRMHDPairedDataset,
    normalizer: GRMHDNormalizer,
    *,
    max_samples_per_channel: int,
    seed: int,
) -> dict[str, Any]:
    indices = dataset.owned_snapshot_indices
    spatial_shape = dataset.snapshot_shape[1:]
    voxels = int(np.prod(spatial_shape))
    choices = sample_choices(len(indices), voxels, max_samples_per_channel, seed)
    sample_count = len(choices[0])
    samples = {
        "physical": np.empty((len(CHANNELS), sample_count), dtype=np.float32),
        "robust_z": np.empty((len(CHANNELS), sample_count), dtype=np.float32),
        "normalized": np.empty((len(CHANNELS), sample_count), dtype=np.float32),
    }
    counts = {
        "robust_z_abs_gt_0.95gamma": np.zeros(len(CHANNELS), dtype=np.int64),
        "robust_z_abs_gt_0.99gamma": np.zeros(len(CHANNELS), dtype=np.int64),
        "normalized_abs_gt_0.95gamma": np.zeros(len(CHANNELS), dtype=np.int64),
        "normalized_abs_gt_0.99gamma": np.zeros(len(CHANNELS), dtype=np.int64),
    }
    for slot, snapshot_index in enumerate(indices):
        raw = dataset.load_snapshot(snapshot_index).numpy()
        normalized = normalizer.encode_numpy(raw, channel_axis=0)
        robust_z = normalizer.gamma * np.arctanh(
            np.clip(
                normalized.astype(np.float64) / normalizer.gamma,
                -1 + np.finfo(np.float64).eps,
                1 - np.finfo(np.float64).eps,
            )
        )
        counts["robust_z_abs_gt_0.95gamma"] += np.count_nonzero(
            np.abs(robust_z) > 0.95 * normalizer.gamma, axis=(1, 2, 3)
        )
        counts["robust_z_abs_gt_0.99gamma"] += np.count_nonzero(
            np.abs(robust_z) > 0.99 * normalizer.gamma, axis=(1, 2, 3)
        )
        counts["normalized_abs_gt_0.95gamma"] += np.count_nonzero(
            np.abs(normalized) > 0.95 * normalizer.gamma, axis=(1, 2, 3)
        )
        counts["normalized_abs_gt_0.99gamma"] += np.count_nonzero(
            np.abs(normalized) > 0.99 * normalizer.gamma, axis=(1, 2, 3)
        )
        lower = slot * voxels
        upper = lower + voxels
        for channel, channel_choices in enumerate(choices):
            start = int(np.searchsorted(channel_choices, lower, side="left"))
            stop = int(np.searchsorted(channel_choices, upper, side="left"))
            local = channel_choices[start:stop] - lower
            samples["physical"][channel, start:stop] = raw[channel].reshape(-1)[local]
            samples["robust_z"][channel, start:stop] = robust_z[channel].reshape(-1)[local]
            samples["normalized"][channel, start:stop] = normalized[channel].reshape(-1)[local]
    total_values = len(indices) * voxels
    quantiles = {
        space: np.quantile(values, QUANTILES, axis=1).T for space, values in samples.items()
    }
    return {
        "sample_count_per_channel": sample_count,
        "included_value_count_per_channel": total_values,
        "quantiles": quantiles,
        "fractions": {name: values / total_values for name, values in counts.items()},
    }


def adjacent_residual_distribution(
    dataset: GRMHDPairedDataset,
    normalizer: GRMHDNormalizer,
    *,
    max_samples_per_channel: int,
    seed: int,
) -> dict[str, Any]:
    spatial_shape = dataset.snapshot_shape[1:]
    voxels = int(np.prod(spatial_shape))
    choices = sample_choices(len(dataset), voxels, max_samples_per_channel, seed)
    sample_count = len(choices[0])
    absolute_samples = np.empty((len(CHANNELS), sample_count), dtype=np.float32)
    normalized_error = np.zeros(len(CHANNELS), dtype=np.float64)
    normalized_truth = np.zeros(len(CHANNELS), dtype=np.float64)
    decoded_error = np.zeros(len(CHANNELS), dtype=np.float64)
    decoded_truth = np.zeros(len(CHANNELS), dtype=np.float64)
    for slot in range(len(dataset)):
        pair = dataset[slot]
        raw_x = pair["x"].numpy()
        raw_y = pair["y"].numpy()
        encoded_x = normalizer.encode_numpy(raw_x, channel_axis=0)
        encoded_y = normalizer.encode_numpy(raw_y, channel_axis=0)
        residual = encoded_y - encoded_x
        normalized_error += np.square(residual.astype(np.float64)).sum(axis=(1, 2, 3))
        normalized_truth += np.square(encoded_y.astype(np.float64)).sum(axis=(1, 2, 3))
        decoded_error += np.square((raw_y - raw_x).astype(np.float64)).sum(axis=(1, 2, 3))
        decoded_truth += np.square(raw_y.astype(np.float64)).sum(axis=(1, 2, 3))
        lower = slot * voxels
        upper = lower + voxels
        for channel, channel_choices in enumerate(choices):
            start = int(np.searchsorted(channel_choices, lower, side="left"))
            stop = int(np.searchsorted(channel_choices, upper, side="left"))
            local = channel_choices[start:stop] - lower
            absolute_samples[channel, start:stop] = np.abs(residual[channel].reshape(-1)[local])
    return {
        "sample_count_per_channel": sample_count,
        "absolute_residual_quantiles": np.quantile(
            absolute_samples, QUANTILES, axis=1
        ).T,
        "normalized_relative_l2_per_channel": np.sqrt(
            normalized_error / np.maximum(normalized_truth, np.finfo(np.float64).tiny)
        ),
        "normalized_relative_l2_global": math.sqrt(
            normalized_error.sum()
            / max(normalized_truth.sum(), np.finfo(np.float64).tiny)
        ),
        "decoded_relative_l2_per_channel": np.sqrt(
            decoded_error / np.maximum(decoded_truth, np.finfo(np.float64).tiny)
        ),
        "decoded_relative_l2_global": math.sqrt(
            decoded_error.sum() / max(decoded_truth.sum(), np.finfo(np.float64).tiny)
        ),
    }


def append_row(
    rows: list[dict[str, Any]],
    fold: str,
    scope: str,
    split: str,
    channel: str,
    metric: str,
    value: float | int,
    *,
    method: str = "",
    reference: str = "",
) -> None:
    rows.append(
        {
            "fold": fold,
            "scope": scope,
            "method": method,
            "split": split,
            "channel": channel,
            "metric": metric,
            "value": value,
            "reference": reference,
        }
    )


def fold_rows(
    fold: str,
    datasets: dict[str, GRMHDPairedDataset],
    normalizer: GRMHDNormalizer,
    residual_scale: ResidualScaleStats,
    train_distribution: dict[str, Any],
    test_distribution: dict[str, Any],
    train_residual: dict[str, Any],
    test_residual: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for channel_index, channel in enumerate(CHANNELS):
        append_row(
            rows,
            fold,
            "residual_scale",
            "train",
            channel,
            "alpha_q0.999",
            residual_scale.alpha[channel_index],
            reference="train-only normalized |y-x|",
        )
        for split, distribution in (
            ("train", train_distribution),
            ("test", test_distribution),
        ):
            append_row(
                rows,
                fold,
                "distribution",
                split,
                channel,
                "snapshot_sample_count",
                distribution["sample_count_per_channel"],
            )
            for fraction_name, fractions in distribution["fractions"].items():
                append_row(
                    rows,
                    fold,
                    "transform_saturation",
                    split,
                    channel,
                    fraction_name,
                    fractions[channel_index],
                )
            for space, quantile_values in distribution["quantiles"].items():
                for quantile_index, quantile in enumerate(QUANTILES):
                    append_row(
                        rows,
                        fold,
                        "quantiles",
                        split,
                        channel,
                        f"{space}_q{quantile:g}",
                        quantile_values[channel_index, quantile_index],
                    )
        encoded_train = train_distribution["quantiles"]["normalized"][channel_index]
        encoded_test = test_distribution["quantiles"]["normalized"][channel_index]
        append_row(
            rows,
            fold,
            "distribution_shift",
            "test_vs_train",
            channel,
            "normalized_quantile_shift_l1",
            float(np.mean(np.abs(encoded_test - encoded_train))),
            reference="mean absolute q0.001/q0.01/q0.5/q0.99/q0.999 difference",
        )
        for split, residual in (("train", train_residual), ("test", test_residual)):
            for quantile_index, quantile in enumerate(QUANTILES):
                append_row(
                    rows,
                    fold,
                    "adjacent_residual",
                    split,
                    channel,
                    f"normalized_abs_residual_q{quantile:g}",
                    residual["absolute_residual_quantiles"][channel_index, quantile_index],
                )
            append_row(
                rows,
                fold,
                "persistence",
                split,
                channel,
                "normalized_relative_l2",
                residual["normalized_relative_l2_per_channel"][channel_index],
                method="persistence",
            )
            append_row(
                rows,
                fold,
                "persistence",
                split,
                channel,
                "decoded_relative_l2",
                residual["decoded_relative_l2_per_channel"][channel_index],
                method="persistence",
            )
        train_q999 = train_residual["absolute_residual_quantiles"][channel_index, -1]
        test_q999 = test_residual["absolute_residual_quantiles"][channel_index, -1]
        append_row(
            rows,
            fold,
            "adjacent_residual_shift",
            "test_vs_train",
            channel,
            "normalized_abs_residual_q0.999_ratio",
            test_q999 / max(train_q999, np.finfo(np.float64).tiny),
        )

    for split, residual in (("train", train_residual), ("test", test_residual)):
        for metric in ("normalized_relative_l2_global", "decoded_relative_l2_global"):
            append_row(
                rows,
                fold,
                "persistence",
                split,
                "global",
                metric,
                residual[metric],
                method="persistence",
            )
            append_row(
                rows,
                fold,
                "zero_output_bounded_residual",
                split,
                "global",
                metric,
                residual[metric],
                method="zero_output_bounded_residual",
                reference="analytically identical to persistence",
            )
    sample = datasets["test"][0]["x"].unsqueeze(0)
    encoded = normalizer.encode_tensor(sample, channel_axis=1)
    zero_prediction = apply_prediction_mode(
        torch.zeros_like(encoded),
        encoded,
        predict_residual=True,
        bounded_residual=True,
        residual_scale=residual_scale.alpha,
    )
    append_row(
        rows,
        fold,
        "zero_output_bounded_residual",
        "test",
        "global",
        "max_abs_difference_from_persistence",
        float(torch.max(torch.abs(zero_prediction - encoded))),
        method="zero_output_bounded_residual",
    )
    return rows


def metric_value(
    rows: list[dict[str, Any]],
    fold: str,
    metric: str,
    channel: str,
    split: str,
    *,
    scope: str | None = None,
) -> float:
    matches = [
        row
        for row in rows
        if row["fold"] == fold
        and row["metric"] == metric
        and row["channel"] == channel
        and row["split"] == split
        and (scope is None or row["scope"] == scope)
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one row for {fold}/{metric}/{channel}/{split}")
    return float(matches[0]["value"])


def markdown_report(rows: list[dict[str, Any]], fold_metadata: dict[str, Any]) -> str:
    lines = [
        "# Rolling-Origin Distribution Shift",
        "",
        "No fold model was trained. Each fold independently fits its normalizer and q0.999 "
        "bounded-residual alpha from only that fold's training snapshots/transitions. Cross-split "
        "transitions are omitted.",
        "",
        "| fold | train / val / test snapshots | test persistence normalized L2 | test "
        "persistence decoded L2 | max channel q0.999 residual ratio |",
        "|---|---|---:|---:|---:|",
    ]
    for fold, metadata in fold_metadata.items():
        normalized = metric_value(
            rows,
            fold,
            "normalized_relative_l2_global",
            "global",
            "test",
            scope="persistence",
        )
        decoded = metric_value(
            rows,
            fold,
            "decoded_relative_l2_global",
            "global",
            "test",
            scope="persistence",
        )
        ratios = [
            metric_value(
                rows,
                fold,
                "normalized_abs_residual_q0.999_ratio",
                channel,
                "test_vs_train",
            )
            for channel in CHANNELS
        ]
        lines.append(
            f"| {fold} | {metadata['ranges']} | {normalized:.6g} | {decoded:.6g} | "
            f"{max(ratios):.6g} ({CHANNELS[int(np.argmax(ratios))]}) |"
        )
    lines.extend(
        [
            "",
            "## Channel diagnostics",
            "",
            "The saturation columns below use the post-transform, post-robust-normalization "
            "soft-clipped state. The quantile-shift score is the mean absolute difference of "
            "five normalized-state quantiles between test and train.",
            "",
            "| fold | channel | alpha q0.999 | train saturation >0.95γ | test saturation "
            ">0.95γ | normalized quantile shift | test/train residual q0.999 |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for fold in FOLDS:
        for channel in CHANNELS:
            values = (
                metric_value(rows, fold, "alpha_q0.999", channel, "train"),
                metric_value(
                    rows,
                    fold,
                    "normalized_abs_gt_0.95gamma",
                    channel,
                    "train",
                ),
                metric_value(
                    rows,
                    fold,
                    "normalized_abs_gt_0.95gamma",
                    channel,
                    "test",
                ),
                metric_value(
                    rows,
                    fold,
                    "normalized_quantile_shift_l1",
                    channel,
                    "test_vs_train",
                ),
                metric_value(
                    rows,
                    fold,
                    "normalized_abs_residual_q0.999_ratio",
                    channel,
                    "test_vs_train",
                ),
            )
            lines.append(
                f"| {fold} | {channel} | {values[0]:.6g} | {values[1]:.6g} | "
                f"{values[2]:.6g} | {values[3]:.6g} | {values[4]:.6g} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The zero-output bounded residual is exactly persistence in every fold "
            "(`max_abs_difference_from_persistence = 0`). Therefore its untrained errors are "
            "not a separate model result. The fold-to-fold and train-to-test changes quantify "
            "data/transform drift before any learned dynamics can contribute. Full sampled "
            "physical, robust-z, normalized quantiles and saturation fractions are in "
            "`fold_distribution_shift.csv`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=Path("data_proc/grmhd_regrid_inner_r200_64.h5")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/experiment_round2")
    )
    parser.add_argument("--samples-per-channel", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stats_root = args.output_dir / "fold_stats"
    all_rows: list[dict[str, Any]] = []
    fold_metadata: dict[str, Any] = {}

    for fold, specification in FOLDS.items():
        datasets = make_temporal_datasets(
            args.data,
            snapshot_start=0,
            snapshot_end=specification["snapshot_end"],
            train_snapshot_count=specification["train"],
            val_snapshot_count=specification["val"],
            test_snapshot_count=specification["test"],
        )
        fold_dir = stats_root / f"fold_{fold.lower()}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        normalizer_path = fold_dir / "normalizer_stats.npz"
        residual_scale_path = fold_dir / "residual_scale.json"
        if normalizer_path.exists() and residual_scale_path.exists():
            normalizer = GRMHDNormalizer.load(
                normalizer_path,
                h5_path=args.data,
                expected_training_indices=datasets["train"].owned_snapshot_indices,
            )
            residual_scale = ResidualScaleStats.load(
                residual_scale_path,
                h5_path=args.data,
                expected_training_indices=datasets["train"].owned_snapshot_indices,
                expected_quantile=0.999,
                expected_multiplier=1.0,
            )
        else:
            normalizer = GRMHDNormalizer.fit(
                datasets["train"],
                max_samples_per_channel=args.samples_per_channel,
                seed=args.seed,
            )
            residual_scale = ResidualScaleStats.fit(
                datasets["train"],
                normalizer,
                quantile=0.999,
                multiplier=1.0,
                max_samples_per_channel=args.samples_per_channel,
                seed=args.seed,
            )
            normalizer.save(normalizer_path)
            residual_scale.save(residual_scale_path)
        train_distribution = snapshot_distribution(
            datasets["train"],
            normalizer,
            max_samples_per_channel=args.samples_per_channel,
            seed=args.seed + 1000,
        )
        test_distribution = snapshot_distribution(
            datasets["test"],
            normalizer,
            max_samples_per_channel=args.samples_per_channel,
            seed=args.seed + 2000,
        )
        train_residual = adjacent_residual_distribution(
            datasets["train"],
            normalizer,
            max_samples_per_channel=args.samples_per_channel,
            seed=args.seed + 3000,
        )
        test_residual = adjacent_residual_distribution(
            datasets["test"],
            normalizer,
            max_samples_per_channel=args.samples_per_channel,
            seed=args.seed + 4000,
        )
        all_rows.extend(
            fold_rows(
                fold,
                datasets,
                normalizer,
                residual_scale,
                train_distribution,
                test_distribution,
                train_residual,
                test_residual,
            )
        )
        fold_metadata[fold] = {
            "ranges": (
                f"0..{specification['train'] - 1} / "
                f"{specification['train']}..{specification['train'] + specification['val'] - 1} / "
                f"{specification['train'] + specification['val']}.."
                f"{specification['snapshot_end'] - 1}"
            ),
            "manifests": {name: dataset.manifest() for name, dataset in datasets.items()},
            "normalizer": normalizer.as_dict(),
            "residual_scale": residual_scale.as_dict(),
        }
        for dataset in datasets.values():
            dataset.close()
        print(f"completed fold {fold}", flush=True)

    csv_path = args.output_dir / "fold_distribution_shift.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    (args.output_dir / "fold_distribution_shift.md").write_text(
        markdown_report(all_rows, fold_metadata), encoding="utf-8"
    )
    (stats_root / "fold_metadata.json").write_text(
        json.dumps(fold_metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps({"rows": len(all_rows), "folds": list(FOLDS)}, indent=2))


if __name__ == "__main__":
    main()
