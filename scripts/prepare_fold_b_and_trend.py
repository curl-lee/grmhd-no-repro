#!/usr/bin/env python
"""Fit independent Fold B statistics and evaluate non-neural trend baselines."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch

from grmhd import CHANNELS
from grmhd.dataset import make_temporal_datasets, sha256_file
from grmhd.fold_b import fit_trend_alpha_streaming
from grmhd.hybrid import HybridTargetStats
from grmhd.normalizer import GRMHDNormalizer


def _evaluate(dataset, global_alpha, channel_alpha):
    variants = {
        "persistence": torch.zeros(8, dtype=torch.float64),
        "alpha_1": torch.ones(8, dtype=torch.float64),
        "train_fitted_global": torch.full((8,), global_alpha, dtype=torch.float64),
        "train_fitted_channel": torch.as_tensor(channel_alpha, dtype=torch.float64),
    }
    totals = {
        name: {
            "error": torch.zeros(8, dtype=torch.float64),
            "truth": torch.zeros(8, dtype=torch.float64),
            "rho_press_violations": 0,
        }
        for name in variants
    }
    # Forecast each in-split transition t -> t+1 using t-1 as trend history.
    # Start at split.start+1 so no history snapshot crosses a split boundary.
    for source_index in range(dataset.split.start + 1, dataset.split.stop - 1):
        previous = dataset.load_snapshot(source_index - 1).double()
        current = dataset.load_snapshot(source_index).double()
        truth = dataset.load_snapshot(source_index + 1).double()
        for name, alpha in variants.items():
            prediction = current + alpha.reshape(8, 1, 1, 1) * (current - previous)
            reduction = tuple(range(1, prediction.ndim))
            totals[name]["error"] += (prediction - truth).square().sum(dim=reduction)
            totals[name]["truth"] += truth.square().sum(dim=reduction)
            totals[name]["rho_press_violations"] += int((prediction[3:5] <= 0).sum())
    results = {}
    for name, values in totals.items():
        per_channel = torch.sqrt(values["error"] / values["truth"].clamp_min(1e-30))
        global_l2 = torch.sqrt(values["error"].sum() / values["truth"].sum().clamp_min(1e-30))
        results[name] = {
            "decoded_global_relative_l2": float(global_l2),
            "decoded_per_channel_relative_l2": dict(
                zip(CHANNELS, (float(value) for value in per_channel))
            ),
            "rho_press_violations": values["rho_press_violations"],
        }
    persistence = results["persistence"]["decoded_global_relative_l2"]
    for values in results.values():
        values["global_improvement_over_persistence_fraction"] = 1.0 - (
            values["decoded_global_relative_l2"] / persistence
        )
    return results


def main() -> None:
    data_path = Path("data_proc/grmhd_regrid_inner_r200_64.h5")
    datasets = make_temporal_datasets(
        data_path,
        snapshot_start=0,
        snapshot_end=100,
        train_snapshot_count=80,
        val_snapshot_count=10,
        test_snapshot_count=10,
    )
    output_dir = Path("outputs/experiment_round3/fold_b_stats")
    output_dir.mkdir(parents=True, exist_ok=True)
    normalizer = GRMHDNormalizer.fit(
        datasets["train"], max_samples_per_channel=200_000, seed=42
    )
    normalizer.save(output_dir / "normalizer_stats.npz")
    hybrid = HybridTargetStats.fit(
        datasets["train"], max_samples_per_channel=200_000, seed=42
    )
    hybrid.save(output_dir / "hybrid_target_stats.json")
    ranges = {channel: [float("inf"), float("-inf")] for channel in CHANNELS}
    for index in datasets["train"].owned_snapshot_indices:
        snapshot = datasets["train"].load_snapshot(index)
        for channel_index, channel in enumerate(CHANNELS):
            ranges[channel][0] = min(ranges[channel][0], float(snapshot[channel_index].min()))
            ranges[channel][1] = max(ranges[channel][1], float(snapshot[channel_index].max()))
    (output_dir / "physical_ranges.json").write_text(
        json.dumps(ranges, indent=2), encoding="utf-8"
    )
    global_alpha, channel_alpha = fit_trend_alpha_streaming(datasets["train"])
    validation = _evaluate(datasets["val"], global_alpha, channel_alpha)
    best_name = min(
        validation,
        key=lambda name: validation[name]["decoded_global_relative_l2"],
    )
    trend_gate_passed = (
        validation[best_name]["global_improvement_over_persistence_fraction"] >= 0.02
    )
    test = _evaluate(datasets["test"], global_alpha, channel_alpha)
    report = {
        "status": "completed",
        "fold": "B",
        "split": {"train": [0, 80], "validation": [80, 90], "test": [90, 100]},
        "selection_uses_test": False,
        "source_hdf5_checksum": sha256_file(data_path),
        "normalizer_stats_checksum": sha256_file(output_dir / "normalizer_stats.npz"),
        "hybrid_target_stats_checksum": hybrid.stats_checksum,
        "train_indices": list(datasets["train"].owned_snapshot_indices),
        "global_alpha": global_alpha,
        "channel_alpha": dict(zip(CHANNELS, channel_alpha)),
        "validation": validation,
        "validation_selected_variant": best_name,
        "trend_gate_required_improvement": 0.02,
        "trend_gate_passed": trend_gate_passed,
        "dual_frame_fno_allowed": trend_gate_passed,
        "test_reported_after_validation_selection": test,
        "hybrid_stats": hybrid.as_dict(),
        "manifests": {name: dataset.manifest() for name, dataset in datasets.items()},
    }
    root = Path("outputs/experiment_round3")
    (root / "fold_b_trend_baseline.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    rows = []
    for split_name, split_values in (("validation", validation), ("test", test)):
        for name, values in split_values.items():
            rows.append(
                {
                    "split": split_name,
                    "variant": name,
                    "decoded_global_relative_l2": values["decoded_global_relative_l2"],
                    "improvement_over_persistence_fraction": values[
                        "global_improvement_over_persistence_fraction"
                    ],
                    "rho_press_violations": values["rho_press_violations"],
                }
            )
    with (root / "fold_b_trend_baseline.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Fold B trend baseline",
        "",
        f"Validation selection: `{best_name}`; 2% gate: `{'passed' if trend_gate_passed else 'failed'}`.",
        "",
        "| Split | Variant | decoded global relative L2 | improvement vs persistence |",
        "|---|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['split']} | {row['variant']} | {row['decoded_global_relative_l2']:.6g} | "
            f"{row['improvement_over_persistence_fraction']:.3%} |"
        )
    (root / "fold_b_trend_baseline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
