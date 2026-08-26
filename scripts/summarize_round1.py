#!/usr/bin/env python
"""Create compact, comparable summaries for the bounded round-one experiments."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import yaml


EXPERIMENTS = (
    ("persistence_all111", "all111", "persistence", "identity", None, "persistence_all111"),
    ("persistence_late100", "late100", "persistence", "identity", None, "persistence_late100"),
    ("smoke_fno_direct_111", "all111", "FNO", "direct", "smoke_fno_direct_111", "smoke_fno_direct_111/eval_smoke3"),
    ("smoke_fno_residual_111", "all111", "FNO", "residual", "smoke_fno_residual_111", "smoke_fno_residual_111/eval_smoke3"),
    ("smoke_localno_direct_111", "all111", "LocalNO diff-only", "direct", "smoke_localno_direct_111", "smoke_localno_direct_111/eval_smoke3"),
    ("smoke_localno_residual_111", "all111", "LocalNO diff-only", "residual", "smoke_localno_residual_111", "smoke_localno_residual_111/eval_smoke3"),
    ("pilot20_fno_residual_all111", "all111", "FNO", "residual", "pilot20_fno_residual_all111", "pilot20_fno_residual_all111/eval_test20"),
    ("pilot20_localno_residual_all111", "all111", "LocalNO diff-only", "residual", "pilot20_localno_residual_all111", "pilot20_localno_residual_all111/eval_test20"),
    ("pilot20_fno_residual_late100", "late100", "FNO", "residual", "pilot20_fno_residual_late100", "pilot20_fno_residual_late100/eval_test19"),
)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def regime_value(metrics: dict[str, Any], segment: str) -> float | None:
    value = metrics["one_step"]["regime_segments"]["model"][segment]
    return value.get("overall_mean_relative_l2_physical") if value.get("count") else None


def horizon_values(metrics: dict[str, Any], horizon: int) -> tuple[Any, Any, Any, Any]:
    record = metrics["rollout"]["requested_horizons"][str(horizon)]
    if not record["available"]:
        return None, None, None, None
    model = record["model_metrics"]["global_volumetric_relative_l2_physical"]
    persistence = record["persistence_metrics"]["global_volumetric_relative_l2_physical"]
    ratio = model / max(persistence, 1.0e-30)
    return model, persistence, ratio, 1.0 - ratio


def summarize(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_id, window, model, mode, train_subdir, eval_subdir in EXPERIMENTS:
        evaluation = load_json(root / eval_subdir / "metrics.json")
        training = load_json(root / train_subdir / "metrics.json") if train_subdir else None
        config = (
            yaml.safe_load((root / train_subdir / "config_resolved.yaml").read_text(encoding="utf-8"))
            if train_subdir
            else None
        )
        one_step = evaluation["one_step"]["model"]
        persistence = evaluation["one_step"]["persistence"]
        one_step_ratio = (
            one_step["overall_mean_relative_l2_physical"]
            / max(persistence["overall_mean_relative_l2_physical"], 1.0e-30)
        )
        horizon_fields: dict[str, Any] = {}
        for horizon in (1, 5, 10, 19, 20):
            model_error, persistence_error, ratio, improvement = horizon_values(
                evaluation, horizon
            )
            horizon_fields.update(
                {
                    f"step{horizon}_global_physical": model_error,
                    f"step{horizon}_persistence_global_physical": persistence_error,
                    f"step{horizon}_error_ratio": ratio,
                    f"step{horizon}_improvement": improvement,
                }
            )
        rollout = evaluation["rollout"]["model"]
        stability_flags = list(evaluation["rollout"]["artifact_flags"])
        step10_ratio = horizon_fields["step10_error_ratio"]
        last_horizon = 20 if horizon_fields["step20_error_ratio"] is not None else 19
        last_ratio = horizon_fields[f"step{last_horizon}_error_ratio"]
        if step10_ratio is not None and step10_ratio > 5:
            stability_flags.append("decoded_range_explosion_by_step10")
        if last_ratio is not None and last_ratio > 5:
            stability_flags.append(f"decoded_range_explosion_by_step{last_horizon}")
        row = {
            "run_id": run_id,
            "dataset_window": window,
            "model": model,
            "prediction_mode": mode,
            "train_epochs": training["epochs_completed"] if training else 0,
            "best_epoch": training["best_epoch"] if training else None,
            "parameter_count": config["resolved"]["trainable_parameters"] if config else 0,
            "training_wall_seconds": training["wall_seconds"] if training else 0.0,
            "peak_gpu_mib": training["peak_cuda_memory_mib"] if training else None,
            "best_validation_metric": training["best_validation_loss"] if training else None,
            "checkpoint_selection_metric": training["checkpoint_selection_metric"] if training else None,
            "one_step_channel_mean_normalized": one_step["overall_mean_relative_l2_normalized"],
            "one_step_channel_mean_physical": one_step["overall_mean_relative_l2_physical"],
            "one_step_global_normalized": one_step["mean_global_volumetric_relative_l2_normalized"],
            "one_step_global_physical": one_step["mean_global_volumetric_relative_l2_physical"],
            "one_step_per_channel_normalized": json.dumps(one_step["mean_relative_l2_normalized"]),
            "one_step_per_channel_physical": json.dumps(one_step["mean_relative_l2_physical"]),
            "one_step_persistence_channel_mean_physical": persistence["overall_mean_relative_l2_physical"],
            "one_step_error_ratio": one_step_ratio,
            "one_step_improvement": 1.0 - one_step_ratio,
            **horizon_fields,
            "rho_positivity_violations_rollout": rollout["total_positivity_violations"]["rho"],
            "press_positivity_violations_rollout": rollout["total_positivity_violations"]["press"],
            "prediction_nan_rollout": rollout["total_prediction_nan"],
            "prediction_inf_rollout": rollout["total_prediction_inf"],
            "artifact_flags": json.dumps(evaluation["rollout"]["artifact_flags"]),
            "stability_flags": json.dumps(sorted(set(stability_flags))),
            "pre_change_one_step_channel_mean_physical": regime_value(evaluation, "pre_change"),
            "change_92_97_one_step_channel_mean_physical": regime_value(evaluation, "change_92_97"),
            "post_change_one_step_channel_mean_physical": regime_value(evaluation, "post_change"),
            "evaluated_rollout_steps": evaluation["rollout"]["evaluated_steps"],
            "aggregation_note": "one_step_channel_mean is arithmetic mean over channels/transitions; one_step_global is volumetric relative L2 then mean over transitions; horizon values are global volumetric physical relative L2",
        }
        rows.append(row)

    decision = {
        "continue_to_100_epochs": False,
        "all111_relative_best_architecture": "FNO residual",
        "selection_reason": (
            "Neither residual model passes. FNO is only the relative-best architecture because its "
            "decoded 10/20-step rollout is less catastrophic than LocalNO; this is not a claim that FNO is adequate."
        ),
        "failed_criteria": [
            "validation best occurred at epoch 4 (FNO) or epoch 2 (LocalNO), with no late improvement",
            "one-step aggregate decoded errors are worse than persistence",
            "decoded magnetic-component rollout errors explode by 10-20 steps",
            "10-step global physical relative L2 is catastrophic for both all111 pilots",
        ],
        "conclusion": (
            "No bounded model currently outperforms persistence sufficiently to justify long training."
        ),
        "extra_priors_enabled": False,
        "long_training_started": False,
    }
    return rows, decision


def fmt(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("outputs/experiment_round1"))
    args = parser.parse_args()
    rows, decision = summarize(args.root)
    args.root.mkdir(parents=True, exist_ok=True)
    with (args.root / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "metric_definitions": {
            "error_ratio": "model_error / persistence_error",
            "improvement": "1 - error_ratio; positive means better than persistence",
            "one_step_channel_mean": "arithmetic mean over channels and teacher-forced transitions",
            "one_step_global": "global volumetric relative L2, then arithmetic mean over transitions",
            "horizon_metrics": "global volumetric decoded/physical relative L2 at that autoregressive horizon",
        },
        "channel_order": ["Bcc1", "Bcc2", "Bcc3", "rho", "press", "vel1", "vel2", "vel3"],
        "decision": decision,
        "experiments": rows,
    }
    (args.root / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pilot_rows = [row for row in rows if row["run_id"].startswith("pilot20")]
    lines = [
        "# Round-one bounded GRMHD experiments",
        "",
        "Errors in the table are decoded global volumetric relative L2 at the stated horizon. "
        "One-step `channel mean` is separately the arithmetic mean over channels and transitions.",
        "",
        "| run | window | epochs/best | val loss | one-step channel mean | 5-step | 10-step | 19/20-step | one-step improvement | flags |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        terminal = row["step20_global_physical"]
        terminal_label = "20"
        if terminal is None:
            terminal = row["step19_global_physical"]
            terminal_label = "19"
        lines.append(
            "| {run_id} | {dataset_window} | {train_epochs}/{best_epoch} | {best_validation_metric} | "
            "{one_step_channel_mean_physical} | {step5_global_physical} | {step10_global_physical} | "
            "{terminal_label}:{terminal} | {one_step_improvement} | {flags} |".format(
                **{key: fmt(value) for key, value in row.items()},
                terminal_label=terminal_label,
                terminal=fmt(terminal),
                flags=", ".join(json.loads(row["stability_flags"])) or "none",
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            decision["conclusion"],
            "",
            "No run continues to 100 epochs. Both all111 pilots selected an early best checkpoint and "
            "developed catastrophic decoded magnetic-component errors in autoregressive rollout. "
            "The late100 FNO sensitivity run also remains worse than persistence and has only 19 honest test transitions.",
            "",
            "Pilot resource summary: "
            + "; ".join(
                f"{row['run_id']} best={row['best_epoch']}, val={fmt(row['best_validation_metric'])}, "
                f"wall={fmt(row['training_wall_seconds'])} s, peak={fmt(row['peak_gpu_mib'])} MiB"
                for row in pilot_rows
            ),
            "",
        ]
    )
    (args.root / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "decision": decision}, indent=2))


if __name__ == "__main__":
    main()
