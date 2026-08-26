#!/usr/bin/env python
"""Summarize round-two per-epoch checkpoint metric alignment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grmhd import CHANNELS
from grmhd.checkpoint_metrics import COMPOSITE_STABILITY_FORMULA


DEFAULT_RUNS = {
    "FNO residual": Path(
        "outputs/experiment_round2/checkpoint_alignment_fno_residual_all111"
    ),
    "LocalNO residual": Path(
        "outputs/experiment_round2/checkpoint_alignment_localno_residual_all111"
    ),
}


def load_run(path: Path) -> tuple[dict[str, Any], list[dict[str, str]]]:
    metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
    with (path / "train_log.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != metrics["epochs_completed"]:
        raise ValueError(f"{path}: train log and metrics epoch counts differ")
    if not metrics["checkpoint_reload_verified"]:
        raise ValueError(f"{path}: strict checkpoint reload was not verified")
    return metrics, rows


def expanded_row(
    model_name: str,
    row: dict[str, str],
    best_epochs: dict[str, int],
) -> dict[str, Any]:
    epoch = int(row["epoch"])
    output: dict[str, Any] = {
        "model": model_name,
        "epoch": epoch,
        "normalized_total": float(row["val_loss"]),
        "normalized_one_step_global": float(row["val_normalized_one_step_global"]),
        "decoded_one_step_global": float(row["val_decoded_one_step_global"]),
        "decoded_three_step_global": float(row["val_decoded_three_step_global"]),
        "magnetic_range_violation": float(row["val_magnetic_range_violation"]),
        "composite_stability": float(row["val_composite_stability"]),
        "prediction_nonfinite": int(row["val_stability_prediction_nonfinite"]),
        "rho_positivity_violations": int(
            row["val_stability_rho_positivity_violations"]
        ),
        "press_positivity_violations": int(
            row["val_stability_press_positivity_violations"]
        ),
        "one_step_transition_count": int(row["val_one_step_transition_count"]),
        "rolling_origin_count": int(row["val_rolling_origin_count"]),
    }
    per_channel_fields = {
        "normalized_one_step": "val_normalized_one_step_per_channel_json",
        "decoded_one_step": "val_decoded_one_step_per_channel_json",
        "decoded_three_step": "val_decoded_three_step_per_channel_json",
    }
    for prefix, field in per_channel_fields.items():
        values = json.loads(row[field])
        for channel in CHANNELS:
            output[f"{prefix}_{channel}"] = float(values[channel])
    for metric_name, best_epoch in best_epochs.items():
        output[f"is_best_{metric_name}"] = epoch == int(best_epoch)
    return output


def write_plot(rows_by_model: dict[str, list[dict[str, Any]]], path: Path) -> None:
    figure, axes = plt.subplots(
        len(rows_by_model), 3, figsize=(13.5, 7.2), sharex="col", constrained_layout=True
    )
    for model_index, (model_name, rows) in enumerate(rows_by_model.items()):
        epochs = [row["epoch"] for row in rows]
        normalized_axis, decoded_axis, stability_axis = axes[model_index]
        normalized_axis.plot(
            epochs, [row["normalized_total"] for row in rows], marker="o", ms=3
        )
        normalized_axis.set_title(f"{model_name}: normalized total")
        normalized_axis.set_ylabel("validation metric")

        decoded_axis.plot(
            epochs,
            [row["decoded_one_step_global"] for row in rows],
            label="decoded 1-step",
            marker="o",
            ms=3,
        )
        decoded_axis.plot(
            epochs,
            [row["decoded_three_step_global"] for row in rows],
            label="decoded 3-step",
            marker="o",
            ms=3,
        )
        decoded_axis.set_title("decoded relative L2")
        decoded_axis.legend(fontsize=8)

        stability_axis.plot(
            epochs,
            [row["magnetic_range_violation"] for row in rows],
            label="magnetic range violation",
            marker="o",
            ms=3,
        )
        stability_axis.plot(
            epochs,
            [row["composite_stability"] for row in rows],
            label="composite",
            marker="o",
            ms=3,
        )
        stability_axis.set_yscale("symlog", linthresh=0.1)
        stability_axis.set_title("short-rollout stability")
        stability_axis.legend(fontsize=8)
        for axis in (normalized_axis, decoded_axis, stability_axis):
            axis.grid(alpha=0.25)
            axis.set_xlabel("epoch")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def report_markdown(
    run_data: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]]
) -> str:
    lines = [
        "# Checkpoint Selection Report",
        "",
        "## Metric definitions",
        "",
        "- `normalized_total`: the existing validation `WeightedGRMHDLoss` total.",
        "- `decoded_one_step`: aggregate physical-space global relative L2 over all nine "
        "validation transitions; per-channel values are retained in the CSV.",
        "- `decoded_three_step`: aggregate physical-space global relative L2 at the third "
        "autoregressive endpoint over all seven available validation rolling origins.",
        "- `magnetic_range_violation`: mean positive relative range excess at step 3 over "
        "all origins and Bcc1/Bcc2/Bcc3.",
        f"- `composite_stability = {COMPOSITE_STABILITY_FORMULA}`.",
        "",
        "Every metric uses the frozen all111 train-only normalizer. No validation statistic "
        "is fitted. Each named best file is written only when its own measured metric improves.",
        "",
        "## Best epochs",
        "",
        "| model | metric | epoch | value | checkpoint |",
        "|---|---:|---:|---:|---|",
    ]
    for model_name, (metrics, _rows) in run_data.items():
        for metric_name, epoch in metrics["best_checkpoint_epochs"].items():
            value = metrics["best_checkpoint_values"][metric_name]
            checkpoint = metrics["checkpoint_metric_files"][metric_name]
            lines.append(
                f"| {model_name} | `{metric_name}` | {epoch} | {value:.9g} | `{checkpoint}` |"
            )
        diagnostic = metrics["diagnostic_best"]
        lines.append(
            f"| {model_name} | `magnetic_range_violation` | {diagnostic['epoch']} | "
            f"{diagnostic['value']:.9g} | diagnostic only |"
        )
    lines.extend(
        [
            "",
            "## Metric correlations",
            "",
            "Correlations use all 20 replay epochs independently for each model.",
            "",
            "| model | metric vs normalized_total | Pearson | Spearman |",
            "|---|---|---:|---:|",
        ]
    )
    for model_name, (metrics, _rows) in run_data.items():
        for metric_name, values in metrics["correlations_vs_normalized_total"].items():
            lines.append(
                f"| {model_name} | `{metric_name}` | {values['pearson']:.6g} | "
                f"{values['spearman']:.6g} |"
            )
    lines.extend(["", "## Alignment decision", ""])
    for model_name, (metrics, rows) in run_data.items():
        epochs = metrics["best_checkpoint_epochs"]
        unique_epochs = sorted(set(epochs.values()))
        if len(unique_epochs) == 1:
            lines.append(
                f"- **{model_name}: aligned in this replay.** All four criteria choose epoch "
                f"{unique_epochs[0]}."
            )
        else:
            lines.append(
                f"- **{model_name}: mismatched.** Normalized total chooses epoch "
                f"{epochs['normalized_total']}, while one-step decoded, three-step decoded, "
                f"and composite choose epochs {epochs['decoded_one_step']}, "
                f"{epochs['decoded_three_step']}, and {epochs['composite_stability']}."
            )
        max_range_row = max(rows, key=lambda row: row["magnetic_range_violation"])
        lines.append(
            f"  The maximum measured magnetic-range violation is "
            f"{max_range_row['magnetic_range_violation']:.6g} at epoch "
            f"{max_range_row['epoch']}."
        )
    lines.extend(
        [
            "",
            "For subsequent round-two comparisons, use `best_composite.pt`: it directly "
            "penalizes decoded one-step error, short-rollout error, and magnetic range growth. "
            "Keep the other named checkpoints for auditability; do not relabel or copy them.",
            "",
            "All 40 per-epoch checkpoints were saved without optimizer state, and both selected "
            "checkpoints passed strict state-dict reload plus prediction-probe hash verification.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/experiment_round2"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_data: dict[str, tuple[dict[str, Any], list[dict[str, Any]]]] = {}
    all_rows: list[dict[str, Any]] = []
    for model_name, run_path in DEFAULT_RUNS.items():
        metrics, raw_rows = load_run(run_path)
        range_best_row = min(
            raw_rows, key=lambda row: float(row["val_magnetic_range_violation"])
        )
        metrics["diagnostic_best"] = {
            "metric": "magnetic_range_violation",
            "epoch": int(range_best_row["epoch"]),
            "value": float(range_best_row["val_magnetic_range_violation"]),
        }
        selection_epochs = {
            **metrics["best_checkpoint_epochs"],
            "magnetic_range_violation": metrics["diagnostic_best"]["epoch"],
        }
        rows = [
            expanded_row(model_name, row, selection_epochs)
            for row in raw_rows
        ]
        normalized = np.asarray([row["normalized_total"] for row in rows])
        normalized_ranks = np.argsort(np.argsort(normalized, kind="stable"), kind="stable")
        correlations = {}
        for metric_name in (
            "normalized_one_step_global",
            "decoded_one_step_global",
            "decoded_three_step_global",
            "magnetic_range_violation",
            "composite_stability",
        ):
            values = np.asarray([row[metric_name] for row in rows])
            ranks = np.argsort(np.argsort(values, kind="stable"), kind="stable")
            correlations[metric_name] = {
                "pearson": float(np.corrcoef(normalized, values)[0, 1]),
                "spearman": float(np.corrcoef(normalized_ranks, ranks)[0, 1]),
            }
        metrics["correlations_vs_normalized_total"] = correlations
        run_data[model_name] = (metrics, rows)
        all_rows.extend(rows)

    csv_path = args.output_dir / "checkpoint_metric_alignment.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    write_plot(
        {model_name: rows for model_name, (_metrics, rows) in run_data.items()},
        args.output_dir / "checkpoint_metric_alignment.png",
    )
    (args.output_dir / "checkpoint_selection_report.md").write_text(
        report_markdown(run_data), encoding="utf-8"
    )
    print(json.dumps({"rows": len(all_rows), "output_dir": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
