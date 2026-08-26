#!/usr/bin/env python
"""Build the small, checkpoint-free Round-3 result summaries."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch

from grmhd import CHANNELS
from grmhd.dataset import make_temporal_datasets
from grmhd.upstream_adapters import PINNED_NEURALOP_COMMIT


ROOT = Path("outputs/experiment_round3")
RUN_ROOT = ROOT / "hybrid_experiments"
RUNS = (
    ("R1", "uniform_history"),
    ("R2", "uniform_history"),
    ("R3", "uniform_history"),
    ("R4", "uniform_history"),
    ("R3", "recency_weighted"),
    ("R3", "recent_window"),
)


def _late_persistence(dataset) -> float:
    error = torch.zeros(8, dtype=torch.float64)
    reference = torch.zeros(8, dtype=torch.float64)
    midpoint = dataset.split.start + dataset.split.snapshot_count // 2
    for item in range(len(dataset)):
        sample = dataset[item]
        if int(sample["index"]) < midpoint:
            continue
        x = sample["x"].double()
        y = sample["y"].double()
        error += (x - y).square().sum(dim=(1, 2, 3))
        reference += y.square().sum(dim=(1, 2, 3))
    return float(torch.sqrt(error.sum() / reference.sum().clamp_min(1e-30)))


def _row(variant: str, strategy: str, report: dict, split_name: str) -> dict:
    metrics = report[split_name]
    split = "validation" if split_name.startswith("validation") else "test"
    row = {
        "variant": variant,
        "strategy": strategy,
        "split": split,
        "epochs": report["epochs_completed"],
        "best_composite_epoch": report["best_epochs"]["composite"],
        "wall_seconds": report["wall_seconds"],
        "peak_cuda_memory_mib": report["peak_cuda_memory_mib"],
        "decoded_one_step": metrics["decoded_one_step_global"],
        "persistence_one_step": metrics["persistence_one_step_global"],
        "one_step_improvement_fraction": metrics["one_step_improvement_fraction"],
        "late_half_decoded_one_step": metrics["late_half_decoded_one_step_global"],
        "magnetic_channels_better": sum(
            metrics["decoded_one_step_per_channel"][channel]
            < metrics["persistence_one_step_per_channel"][channel]
            for channel in CHANNELS[:3]
        ),
        "maximum_magnetic_range_ratio": metrics["maximum_magnetic_range_ratio"],
        "maximum_consecutive_magnetic_range_growth": metrics[
            "maximum_consecutive_magnetic_range_growth"
        ],
        "prediction_nonfinite": metrics["prediction_nonfinite"],
        "rho_press_positivity_violations": metrics["rho_press_positivity_violations"],
        "artifact_flags": ",".join(
            name for name, flagged in metrics["artifact_flags"].items() if flagged
        ),
    }
    for horizon in (1, 3, 5, 9):
        horizon_metrics = metrics["horizons"][str(horizon)]
        row[f"step{horizon}"] = horizon_metrics["model_global"]
        row[f"persistence_step{horizon}"] = horizon_metrics["persistence_global"]
        row[f"step{horizon}_ratio"] = (
            horizon_metrics["model_global"] / horizon_metrics["persistence_global"]
        )
    return row


def main() -> None:
    rows = []
    reports = {}
    for variant, strategy in RUNS:
        name = f"{variant}_{strategy}"
        report = json.loads((RUN_ROOT / name / "metrics.json").read_text(encoding="utf-8"))
        reports[name] = report
        rows.extend(
            _row(variant, strategy, report, split_name)
            for split_name in ("validation_best_composite", "test_best_composite")
        )

    # Strategy selection is validation-only and fixes R3 before revealing test.
    strategy_names = (
        "R3_uniform_history", "R3_recency_weighted", "R3_recent_window"
    )
    selected_name = min(
        strategy_names,
        key=lambda name: reports[name]["validation_best_composite"]["composite"],
    )
    selected = reports[selected_name]
    validation = selected["validation_best_composite"]
    test = selected["test_best_composite"]
    datasets = make_temporal_datasets(
        "data_proc/grmhd_regrid_inner_r200_64.h5", snapshot_start=0, snapshot_end=100,
        train_snapshot_count=80, val_snapshot_count=10, test_snapshot_count=10,
    )
    late_persistence = {
        "validation": _late_persistence(datasets["val"]),
        "test": _late_persistence(datasets["test"]),
    }
    for dataset in datasets.values():
        dataset.close()

    parity = json.loads((ROOT / "autoregression_parity.json").read_text(encoding="utf-8"))
    magnetic_wins = {
        channel: test["decoded_one_step_per_channel"][channel]
        < test["persistence_one_step_per_channel"][channel]
        for channel in CHANNELS[:3]
    }
    gates = {
        "one_step_at_least_5pct_better": test["one_step_improvement_fraction"] >= 0.05,
        "at_least_two_magnetic_channels_better": sum(magnetic_wins.values()) >= 2,
        "step3_better": test["horizons"]["3"]["model_global"]
        < test["horizons"]["3"]["persistence_global"],
        "step5_not_worse": test["horizons"]["5"]["model_global"]
        <= test["horizons"]["5"]["persistence_global"],
        "step9_not_over_1_2x": test["horizons"]["9"]["model_global"]
        <= 1.2 * test["horizons"]["9"]["persistence_global"],
        "magnetic_range_not_continuously_growing": test[
            "maximum_consecutive_magnetic_range_growth"
        ] <= 1.05,
        "upstream_local_ar_match": parity["status"] == "passed",
        "no_nan_inf": test["prediction_nonfinite"] == 0,
        "rho_press_positive": test["rho_press_positivity_violations"] == 0,
        "no_artifact_flags": not any(test["artifact_flags"].values()),
        "best_not_near_initialization": selected["best_epochs"]["composite"] > 3,
        "late_half_not_sacrificed": test["late_half_decoded_one_step_global"]
        <= late_persistence["test"],
    }
    trend = json.loads((ROOT / "fold_b_trend_baseline.json").read_text(encoding="utf-8"))
    normalizer = json.loads((ROOT / "normalizer_ablation.json").read_text(encoding="utf-8"))
    round2 = json.loads(Path("outputs/experiment_round2/summary.json").read_text(encoding="utf-8"))
    summary = {
        "status": "completed",
        "pinned_neuraloperator_commit": PINNED_NEURALOP_COMMIT,
        "fold": {"train": [0, 79], "validation": [80, 89], "test": [90, 99]},
        "selection_uses_test": False,
        "selected_candidate": selected_name,
        "selected_checkpoint": str(
            (RUN_ROOT / selected_name / "best_composite.pt").resolve()
        ),
        "selected_best_epoch": selected["best_epochs"]["composite"],
        "selected_validation": validation,
        "selected_test": test,
        "late_half_persistence": late_persistence,
        "magnetic_channel_wins_on_test": magnetic_wins,
        "acceptance_gates": gates,
        "acceptance_passed": all(gates.values()),
        "long_training_allowed": False,
        "dual_frame_allowed": trend["dual_frame_fno_allowed"],
        "r0_frozen_round2_reference_not_fold_b_comparable": round2["bounded_fno_pilot"],
        "fold_b_trend": trend,
        "normalizer_ablation": normalizer,
        "autoregression_parity": parity,
        "rows": rows,
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Round 3 summary", "",
        f"Validation-only selected candidate: `{selected_name}` at epoch "
        f"{selected['best_epochs']['composite']}.", "",
        "| variant | strategy | split | one-step / persistence | improvement | step 3 / 5 / 9 | flags |",
        "|---|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['strategy']} | {row['split']} | "
            f"{row['decoded_one_step']:.6f} / {row['persistence_one_step']:.6f} | "
            f"{100 * row['one_step_improvement_fraction']:.4f}% | "
            f"{row['step3']:.6f} / {row['step5']:.6f} / {row['step9']:.6f} | "
            f"{row['artifact_flags'] or 'none'} |"
        )
    lines.extend(["", "## Acceptance gates", ""])
    lines.extend(f"- {name}: **{'PASS' if passed else 'FAIL'}**" for name, passed in gates.items())
    lines.extend([
        "", f"Overall: **{'PASS' if all(gates.values()) else 'FAIL'}**.",
        "Long training and model-capacity expansion remain disabled.", "",
    ])
    (ROOT / "summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
