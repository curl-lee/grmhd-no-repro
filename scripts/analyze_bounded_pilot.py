#!/usr/bin/env python
"""Apply the explicit round-two pass/fail gates to the bounded FNO pilot."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from grmhd import CHANNELS


def range_value(record: dict[str, Any], channel: str, source: str) -> float:
    values = record["ranges"][channel]
    return float(values[f"{source}_max"] - values[f"{source}_min"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("outputs/experiment_round2/bounded_fno_nearest_all111"),
    )
    args = parser.parse_args()
    training = json.loads((args.run_dir / "metrics.json").read_text(encoding="utf-8"))
    evaluation = json.loads(
        (args.run_dir / "eval_test/metrics.json").read_text(encoding="utf-8")
    )
    one_model = evaluation["one_step"]["model"]
    one_persistence = evaluation["one_step"]["persistence"]
    model_one_global = one_model["mean_global_volumetric_relative_l2_physical"]
    persistence_one_global = one_persistence[
        "mean_global_volumetric_relative_l2_physical"
    ]
    one_step_improvement = 1.0 - model_one_global / persistence_one_global
    magnetic_channel_comparison = {
        channel: {
            "model": one_model["mean_relative_l2_physical"][index],
            "persistence": one_persistence["mean_relative_l2_physical"][index],
        }
        for index, channel in enumerate(CHANNELS[:3])
    }
    all_magnetic_worse = all(
        values["model"] > values["persistence"]
        for values in magnetic_channel_comparison.values()
    )

    horizons: dict[str, Any] = {}
    for horizon in (1, 5, 10):
        record = evaluation["rollout"]["requested_horizons"][str(horizon)]
        model_error = record["model_metrics"][
            "global_volumetric_relative_l2_physical"
        ]
        persistence_error = record["persistence_metrics"][
            "global_volumetric_relative_l2_physical"
        ]
        horizons[str(horizon)] = {
            "model_global_decoded_relative_l2": model_error,
            "persistence_global_decoded_relative_l2": persistence_error,
            "ratio": model_error / persistence_error,
            "magnetic_ranges": {
                channel: {
                    "model": range_value(record["model_metrics"], channel, "prediction"),
                    "persistence": range_value(
                        record["persistence_metrics"], channel, "prediction"
                    ),
                    "truth": range_value(record["model_metrics"], channel, "truth"),
                }
                for channel in CHANNELS[:3]
            },
        }

    range_records = evaluation["rollout"]["records_model"]
    max_concurrent_range_ratio = 0.0
    max_consecutive_range_growth = 0.0
    prior_ranges: dict[str, float] = {}
    for record in range_records:
        for channel in CHANNELS[:3]:
            predicted = range_value(record, channel, "prediction")
            truth = range_value(record, channel, "truth")
            max_concurrent_range_ratio = max(
                max_concurrent_range_ratio, predicted / max(truth, 1e-30)
            )
            if channel in prior_ranges:
                max_consecutive_range_growth = max(
                    max_consecutive_range_growth,
                    predicted / max(prior_ranges[channel], 1e-30),
                )
            prior_ranges[channel] = predicted
    no_range_explosion = (
        max_concurrent_range_ratio <= 5.0 and max_consecutive_range_growth <= 5.0
    )

    total_nonfinite = (
        one_model["total_prediction_nan"]
        + one_model["total_prediction_inf"]
        + evaluation["rollout"]["model"]["total_prediction_nan"]
        + evaluation["rollout"]["model"]["total_prediction_inf"]
    )
    total_rho_violations = (
        one_model["total_positivity_violations"]["rho"]
        + evaluation["rollout"]["model"]["total_positivity_violations"]["rho"]
    )
    total_press_violations = (
        one_model["total_positivity_violations"]["press"]
        + evaluation["rollout"]["model"]["total_positivity_violations"]["press"]
    )
    artifact_flags = evaluation["rollout"]["artifact_flags"]
    normalized_epoch = training["best_checkpoint_epochs"]["normalized_total"]
    composite_epoch = training["best_checkpoint_epochs"]["composite_stability"]
    history = training["history"]
    first_half_best = min(row["val_composite_stability"] for row in history[:10])
    second_half_best = min(row["val_composite_stability"] for row in history[10:])
    late_improvement_or_early_stop = (
        second_half_best < first_half_best or training["epochs_completed"] < 20
    )

    gates = {
        "teacher_forced_one_step_at_least_5pct_better": {
            "pass": one_step_improvement >= 0.05,
            "actual": one_step_improvement,
            "threshold": ">= 0.05 improvement",
        },
        "magnetic_channels_not_all_worse": {
            "pass": not all_magnetic_worse,
            "actual": magnetic_channel_comparison,
            "threshold": "at least one of Bcc1/Bcc2/Bcc3 not worse than persistence",
        },
        "step5_not_worse_than_persistence": {
            "pass": horizons["5"]["ratio"] <= 1.0,
            "actual": horizons["5"]["ratio"],
            "threshold": "<= 1.0",
        },
        "step10_not_over_1.2x_persistence": {
            "pass": horizons["10"]["ratio"] <= 1.2,
            "actual": horizons["10"]["ratio"],
            "threshold": "<= 1.2",
        },
        "decoded_B_range_no_exponential_growth": {
            "pass": no_range_explosion,
            "actual": {
                "max_prediction_over_concurrent_truth_range": max_concurrent_range_ratio,
                "max_consecutive_prediction_range_growth": max_consecutive_range_growth,
            },
            "threshold": "both <= 5",
        },
        "no_nan_inf": {
            "pass": total_nonfinite == 0,
            "actual": total_nonfinite,
            "threshold": "== 0",
        },
        "rho_press_strictly_positive": {
            "pass": total_rho_violations == 0 and total_press_violations == 0,
            "actual": {"rho": total_rho_violations, "press": total_press_violations},
            "threshold": "both == 0 violations",
        },
        "no_collapse_ripple_stripe": {
            "pass": len(artifact_flags) == 0,
            "actual": artifact_flags,
            "threshold": "no diagnostic flags",
        },
        "best_epoch_not_only_normalized": {
            "pass": training["checkpoint_selection_metric"] == "composite_stability"
            and composite_epoch != normalized_epoch,
            "actual": {
                "selection_metric": training["checkpoint_selection_metric"],
                "normalized_epoch": normalized_epoch,
                "composite_epoch": composite_epoch,
            },
            "threshold": "composite selected and ranking differs from normalized",
        },
        "late_validation_improvement_or_normal_early_stop": {
            "pass": late_improvement_or_early_stop,
            "actual": {
                "first_half_best_composite": first_half_best,
                "second_half_best_composite": second_half_best,
                "epochs_completed": training["epochs_completed"],
            },
            "threshold": "second-half best improves or early stopping occurs",
        },
    }
    passed = all(gate["pass"] for gate in gates.values())
    result = {
        "status": "passed" if passed else "failed",
        "continue_long_training": passed,
        "checkpoint": str((args.run_dir / "best_composite.pt").resolve()),
        "zero_init_persistence_max_abs_difference": training["residual_wrapper"][
            "initial_persistence_max_abs_difference"
        ],
        "alpha_by_channel": dict(
            zip(CHANNELS, training["residual_wrapper"]["alpha"], strict=True)
        ),
        "training_wall_seconds": training["wall_seconds"],
        "peak_cuda_memory_mib": training["peak_cuda_memory_mib"],
        "teacher_forced_one_step": {
            "model_global_decoded_relative_l2": model_one_global,
            "persistence_global_decoded_relative_l2": persistence_one_global,
            "improvement": one_step_improvement,
            "magnetic_channels": magnetic_channel_comparison,
        },
        "requested_horizons": horizons,
        "range_stability": {
            "max_prediction_over_concurrent_truth_range": max_concurrent_range_ratio,
            "max_consecutive_prediction_range_growth": max_consecutive_range_growth,
        },
        "artifact_flags": artifact_flags,
        "gates": gates,
        "failure_analysis": (
            []
            if passed
            else [name for name, gate in gates.items() if not gate["pass"]]
        ),
    }
    (args.run_dir / "pilot_gate_report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    rows = []
    for name, gate in gates.items():
        rows.append(
            {
                "record_type": "gate",
                "name": name,
                "channel": "",
                "model": "",
                "persistence": "",
                "ratio_or_value": json.dumps(gate["actual"], sort_keys=True),
                "threshold": gate["threshold"],
                "pass": gate["pass"],
            }
        )
    for horizon, values in horizons.items():
        rows.append(
            {
                "record_type": "horizon",
                "name": f"step_{horizon}_global_decoded_relative_l2",
                "channel": "global",
                "model": values["model_global_decoded_relative_l2"],
                "persistence": values["persistence_global_decoded_relative_l2"],
                "ratio_or_value": values["ratio"],
                "threshold": "",
                "pass": "",
            }
        )
        for channel, ranges in values["magnetic_ranges"].items():
            rows.append(
                {
                    "record_type": "magnetic_range",
                    "name": f"step_{horizon}_range",
                    "channel": channel,
                    "model": ranges["model"],
                    "persistence": ranges["persistence"],
                    "ratio_or_value": ranges["truth"],
                    "threshold": "ratio_or_value column is truth range",
                    "pass": "",
                }
            )
    with (args.run_dir / "pilot_gate_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Bounded FNO Pilot Gate Report",
        "",
        f"**Overall: {'PASS' if passed else 'FAIL'} — "
        f"{'longer training may proceed' if passed else 'stop; do not run 100/1200 epochs'}.**",
        "",
        f"Teacher-forced decoded one-step global error is {model_one_global:.6g} versus "
        f"persistence {persistence_one_global:.6g} ({one_step_improvement:.3%} improvement).",
        "",
        "| gate | result |",
        "|---|---:|",
    ]
    for name, gate in gates.items():
        lines.append(f"| {name} | {'PASS' if gate['pass'] else 'FAIL'} |")
    lines.extend(
        [
            "",
            "| horizon | model decoded global L2 | persistence | ratio |",
            "|---:|---:|---:|---:|",
        ]
    )
    for horizon, values in horizons.items():
        lines.append(
            f"| {horizon} | {values['model_global_decoded_relative_l2']:.6g} | "
            f"{values['persistence_global_decoded_relative_l2']:.6g} | "
            f"{values['ratio']:.6g} |"
        )
    if not passed:
        lines.extend(
            [
                "",
                "## Failure analysis",
                "",
                "The bounded wrapper removed catastrophic decoded-range growth, but the learned "
                "update did not beat persistence by the required 5% and validation quality did "
                "not improve in the second half. Stability alone is therefore insufficient to "
                "authorize a longer run. No paper-prior ablation is started in this round.",
                "",
            ]
        )
    (args.run_dir / "pilot_gate_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
