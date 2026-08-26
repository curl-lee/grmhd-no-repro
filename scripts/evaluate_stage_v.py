#!/usr/bin/env python3
"""Apply the frozen Stage-V selector and unified one-step/rollout/OOD evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
import torch
import yaml

from grmhd import CHANNELS
from grmhd.paper_stage_g_evaluation import radial_shell_indices
from grmhd.stage_t_training import validation_thirds

from audit_stage_v import pair_metrics
from evaluate_stage_s import build_train_reference
from evaluate_stage_t import evaluate_one_step, model_from_checkpoint, rollout
from train_stage_s import StageSBatchPath, tensor_sha256
from train_stage_t import verify_frozen_contract


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("direction", "transport", "direction_transport")
DISPLAY = {
    "stage_t_plain": "Stage-T Plain",
    "direction": "Direction",
    "transport": "Transport",
    "direction_transport": "Direction+Transport",
}
PRIMARY = ("Bcc2", "Bcc3", "vel3")
TAIL = ("rho", "press")
SELECTED_STEPS = (1, 2, 3, 5, 10, 19, 25, 42, 50, 75, 100)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty Stage V table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def score_row(epoch: int, metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "optimizer_updates": epoch * 42,
        "state_relative_l2": metrics["model"]["normalized_relative_l2"]["arithmetic_average"],
        "state_global_relative_l2": metrics["model"]["normalized_relative_l2"]["global"],
        "persistence_ratio": metrics["model_error_over_persistence_error"],
        "residual_relative_l2": metrics["model"]["residual"]["arithmetic_average_relative_l2"],
        "residual_global_cosine": metrics["model"]["residual"]["global_cosine"],
        "shell_skill": metrics["transport"]["shell_skill_median"],
        "radial_skill": metrics["transport"]["radial_skill_median"],
        "physical_relative_l2": metrics["model"]["physical_relative_l2"]["arithmetic_average"],
        "finite": metrics["finite"],
        "rho_press_positive": metrics["rho_press_positive"],
    }


def selector(history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    definitions = {
        "formal_best_state_l2": ("state_relative_l2", min),
        "diagnostic_best_residual_l2": ("residual_relative_l2", min),
        "diagnostic_best_residual_cosine": ("residual_global_cosine", max),
        "diagnostic_best_shell_skill": ("shell_skill", max),
        "diagnostic_best_radial_skill": ("radial_skill", max),
    }
    result: dict[str, Any] = {
        "formal_selector": "validation normalized per-channel state relative-L2 arithmetic average",
        "diagnostic_selectors_do_not_select_formal_checkpoint": True,
    }
    for name, (key, operation) in definitions.items():
        finite = [row for row in history if row[key] is not None and math.isfinite(float(row[key]))]
        chosen_value = operation(float(row[key]) for row in finite)
        chosen = next(row for row in finite if float(row[key]) == chosen_value)
        result[name] = {"epoch": int(chosen["epoch"]), "metric": key, "value": chosen_value}
    return result


def temporal_rows(
    variant: str,
    epoch: int,
    model: torch.nn.Module,
    data: StageSBatchPath,
    thirds: Mapping[str, Sequence[int]],
    shell_index: np.ndarray,
    reference: Mapping[str, Any],
) -> list[dict[str, Any]]:
    output = []
    for stratum, pairs in thirds.items():
        metrics = evaluate_one_step(model, data, list(pairs), shell_index, reference, include_transport_rows=False)
        output.append({
            "variant": variant,
            "display_name": DISPLAY[variant],
            "formal_best_epoch": epoch,
            "stratum": stratum,
            "pair_count": len(pairs),
            "state_relative_l2": metrics["model"]["normalized_relative_l2"]["arithmetic_average"],
            "residual_relative_l2": metrics["model"]["residual"]["arithmetic_average_relative_l2"],
            "residual_global_cosine": metrics["model"]["residual"]["global_cosine"],
            "shell_skill": metrics["transport"]["shell_skill_median"],
            "radial_skill": metrics["transport"]["radial_skill_median"],
        })
    early = next(row for row in output if row["stratum"] == "early")
    late = next(row for row in output if row["stratum"] == "late")
    for row in output:
        for metric in ("state_relative_l2", "residual_relative_l2"):
            row[f"late_over_early_{metric}"] = float(late[metric]) / max(float(early[metric]), 1e-300)
    return output


def ood_correlation_rows(
    variant: str,
    pairs: Sequence[Mapping[str, Any]],
    ood_by_snapshot: Mapping[int, float],
) -> list[dict[str, Any]]:
    output = []
    x = np.asarray([ood_by_snapshot[int(row["source_snapshot"])] for row in pairs], dtype=np.float64)
    for metric in ("state_relative_l2", "residual_relative_l2", "shell_error", "radial_error"):
        y = np.asarray([float(row[metric]) for row in pairs], dtype=np.float64)
        value = spearmanr(x, y)
        output.append({
            "variant": variant,
            "display_name": DISPLAY[variant],
            "x": "train_derived_source_snapshot_ood_score",
            "y": metric,
            "pair_count": len(pairs),
            "spearman_r": float(value.statistic),
            "spearman_p": float(value.pvalue),
            "causal_claim": False,
        })
    return output


def comparison_rows(
    variant: str, epoch: int, metrics: Mapping[str, Any], baseline: Mapping[str, Any]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    state = float(metrics["model"]["normalized_relative_l2"]["arithmetic_average"])
    residual = float(metrics["model"]["residual"]["arithmetic_average_relative_l2"])
    cosine = float(metrics["model"]["residual"]["global_cosine"])
    shell = metrics["transport"]["shell_skill_median"]
    radial = metrics["transport"]["radial_skill_median"]
    row = {
        "variant": variant,
        "display_name": DISPLAY[variant],
        "formal_best_epoch": epoch,
        "state_relative_l2": state,
        "state_global_relative_l2": metrics["model"]["normalized_relative_l2"]["global"],
        "persistence_state_relative_l2": metrics["persistence"]["normalized_relative_l2"]["arithmetic_average"],
        "persistence_ratio": metrics["model_error_over_persistence_error"],
        "residual_relative_l2": residual,
        "residual_global_cosine": cosine,
        "shell_skill": shell,
        "radial_skill": radial,
        "delta_shell_skill_vs_stage_t": float(shell) - float(baseline["transport"]["shell_skill_median"]),
        "delta_radial_skill_vs_stage_t": float(radial) - float(baseline["transport"]["radial_skill_median"]),
        "physical_relative_l2": metrics["model"]["physical_relative_l2"]["arithmetic_average"],
        "finite": metrics["finite"],
        "rho_press_positive": metrics["rho_press_positive"],
        "O1_state_retention": state <= 1.05 * 0.266711,
        "O2_residual_below_one": residual < 1.0,
        "O3_cosine_above_0_7": cosine > 0.7,
        "O4_shell_positive": shell is not None and float(shell) > 0,
        "O5_radial_positive": radial is not None and float(radial) > 0,
        "O7_nontrivial_not_persistence": state < float(metrics["persistence"]["normalized_relative_l2"]["arithmetic_average"]),
    }
    per_channel = []
    for channel in CHANNELS:
        transport = metrics["transport"]["per_channel"][channel]
        tail = metrics["preprocessing_tail"].get(channel, {})
        per_channel.append({
            "variant": variant,
            "display_name": DISPLAY[variant],
            "formal_best_epoch": epoch,
            "channel": channel,
            "state_relative_l2": metrics["model"]["normalized_relative_l2"]["per_channel"][channel],
            "residual_relative_l2": metrics["model"]["residual"]["per_channel_relative_l2"][channel],
            "residual_cosine": metrics["model"]["residual"]["per_channel_cosine"][channel],
            "shell_skill_median": transport["shell_skill_median"],
            "radial_skill_median": transport["radial_skill_median"],
            "physical_relative_l2": metrics["model"]["physical_relative_l2"]["per_channel"][channel],
            "extreme_decoder_tail_fraction": tail.get("extreme_decoder_tail_fraction"),
            "physical_squared_error_contribution_fraction": tail.get("physical_squared_error_contribution_fraction"),
        })
    return row, per_channel


def rollout_table_rows(variant: str, payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    selected = set(SELECTED_STEPS)
    for record in payload["records"]:
        step = int(record["step"])
        if step not in selected:
            continue
        transport = record.get("transport")
        output.append({
            "variant": variant,
            "display_name": DISPLAY[variant],
            "formal_best_epoch": payload.get("checkpoint_epoch", 150 if variant == "stage_t_plain" else None),
            "step": step,
            "ground_truth_available": record["ground_truth_available"],
            "normalized_state_error": record.get("normalized_relative_l2_average"),
            "residual_relative_l2": record.get("residual_relative_l2"),
            "residual_cosine": record.get("residual_cosine"),
            "shell_skill": None if transport is None else transport["shell_skill_median"],
            "radial_skill": None if transport is None else transport["radial_skill_median"],
            "normalized_global_norm": record["normalized_global_norm"],
            "above_Rout": record["above_Rout"],
            "physical_range_explosion": record["physical_range_explosion"],
            "finite": record["finite"],
            "rho_press_positive": record["rho_press_positive"],
            "rho_physical_q001": record["physical_range"]["rho"].get("q001"),
            "rho_physical_q999": record["physical_range"]["rho"].get("q999"),
            "press_physical_q001": record["physical_range"]["press"].get("q001"),
            "press_physical_q999": record["physical_range"]["press"].get("q999"),
            "FIRST_NEGATIVE_RESIDUAL_COSINE_STEP": payload["FIRST_NEGATIVE_RESIDUAL_COSINE_STEP"],
            "FIRST_NEGATIVE_SHELL_SKILL_STEP": payload["FIRST_NEGATIVE_SHELL_SKILL_STEP"],
            "FIRST_NEGATIVE_RADIAL_SKILL_STEP": payload["FIRST_NEGATIVE_RADIAL_SKILL_STEP"],
            "FIRST_10X_PHYSICAL_RANGE_STEP": payload["FIRST_10X_PHYSICAL_RANGE_STEP"],
        })
    return output


def baseline_artifacts(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metrics = json.loads((ROOT / "artifacts/stage_t/full_long/checkpoint_metrics_epoch_0150.json").read_text())
    rollout_metrics = json.loads((ROOT / "artifacts/stage_t/full_long/rollout/epoch_0150_rollout.json").read_text())
    return metrics, rollout_metrics


def make_figures(
    root: Path,
    one_step: Sequence[Mapping[str, Any]],
    rollout_rows: Sequence[Mapping[str, Any]],
    ood_scores: Sequence[Mapping[str, Any]],
    validation_pair_rows: Sequence[Mapping[str, Any]],
) -> None:
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    labels = [row["display_name"] for row in one_step]

    fig, axis = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(labels))
    axis.bar(x - 0.18, [float(row["state_relative_l2"]) for row in one_step], width=0.36, label="state L2")
    axis.bar(x + 0.18, [max(0.0, -float(row["shell_skill"])) for row in one_step], width=0.36, label="-shell skill (negative part)")
    axis.set_xticks(x, labels, rotation=15)
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "state_vs_transport.png", dpi=160)
    plt.close(fig)

    for metric, filename in (("shell_skill", "shell_skill_comparison.png"), ("radial_skill", "radial_skill_comparison.png")):
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.bar(labels, [float(row[metric]) for row in one_step])
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_ylabel(metric.replace("_", " "))
        axis.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        fig.savefig(figures / filename, dpi=160)
        plt.close(fig)

    gt = [row for row in rollout_rows if row["normalized_state_error"] not in (None, "")]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    for variant in DISPLAY:
        selected = [row for row in gt if row["variant"] == variant]
        if selected:
            axis.plot([int(row["step"]) for row in selected], [float(row["normalized_state_error"]) for row in selected], marker="o", label=DISPLAY[variant])
    axis.set_xlabel("rollout step")
    axis.set_ylabel("normalized state error")
    axis.set_yscale("log")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "rollout_error.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for variant in DISPLAY:
        selected = [row for row in gt if row["variant"] == variant]
        if selected:
            steps = [int(row["step"]) for row in selected]
            axes[0].plot(steps, [float(row["shell_skill"]) for row in selected], marker="o", label=DISPLAY[variant])
            axes[1].plot(steps, [float(row["radial_skill"]) for row in selected], marker="o", label=DISPLAY[variant])
    for axis, label in zip(axes, ("shell skill", "radial skill"), strict=True):
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_xlabel("rollout step")
        axis.set_ylabel(label)
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figures / "rollout_transport.png", dpi=160)
    plt.close(fig)

    ood = {int(row["snapshot"]): float(row["ood_score"]) for row in ood_scores}
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
    for axis, metric in zip(axes, ("state_relative_l2", "residual_relative_l2", "shell_error", "radial_error"), strict=True):
        for variant in DISPLAY:
            selected = [row for row in validation_pair_rows if row["variant"] == variant]
            if selected:
                axis.scatter([ood[int(row["source_snapshot"])] for row in selected], [float(row[metric]) for row in selected], s=10, label=DISPLAY[variant])
        axis.set_xlabel("OOD score")
        axis.set_ylabel(metric.replace("_", " "))
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "error_vs_ood.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.5))
    for axis, metric in zip(axes, ("state_relative_l2", "residual_relative_l2", "shell_error", "radial_error"), strict=True):
        for variant in DISPLAY:
            selected = [row for row in validation_pair_rows if row["variant"] == variant]
            if selected:
                axis.plot(
                    [int(row["source_snapshot"]) for row in selected],
                    [float(row[metric]) for row in selected],
                    marker=".", linewidth=0.8, label=DISPLAY[variant],
                )
        axis.set_xlabel("validation source snapshot")
        axis.set_ylabel(metric.replace("_", " "))
    axes[-1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "error_vs_validation_time.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/stage_v/objective_alignment.yaml"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/stage_v"))
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="all")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    stage_t, stage_s, _, _, _, validation_pairs = verify_frozen_contract(ROOT / config["frozen_stage_t_config"])
    if not torch.cuda.is_available():
        raise RuntimeError("Stage V evaluation requires CUDA; refusing silent CPU fallback")
    device = torch.device("cuda:0")
    data = StageSBatchPath(ROOT / stage_s["data"]["dataset"], ROOT / stage_s["preprocessing"]["artifact"], device)
    r = np.asarray(data.handle["coords/r"], dtype=np.float64)
    _, shell_index = radial_shell_indices(r, int(stage_s["representation"]["shell_count"]))
    reference_path = ROOT / "artifacts/stage_s/diagnostic_reference_train_only.json"
    reference = json.loads(reference_path.read_text()) if reference_path.exists() else build_train_reference(data, ROOT / "artifacts/stage_s")
    thirds = validation_thirds(validation_pairs)
    ood_rows = read_csv(args.root / "shift/snapshot_ood_scores.csv")
    ood_by_snapshot = {int(row["snapshot"]): float(row["ood_score"]) for row in ood_rows}
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    baseline_metrics, baseline_rollout = baseline_artifacts(args.root)
    baseline_model, _, _ = model_from_checkpoint(
        ROOT / config["baseline"]["checkpoint"], stage_s, device,
        expected_updates=int(config["baseline"]["optimizer_updates"]),
    )
    all_shift: list[dict[str, Any]] = temporal_rows(
        "stage_t_plain", 150, baseline_model, data, thirds, shell_index, reference
    )
    baseline_pair_rows = [
        row for row in read_csv(args.root / "alignment/pair_metrics.csv")
        if row["split"] == "validation"
    ]
    for row in baseline_pair_rows:
        row["variant"] = "stage_t_plain"
        row["display_name"] = DISPLAY["stage_t_plain"]
        row["formal_best_epoch"] = 150
    all_validation_pair_rows: list[dict[str, Any]] = list(baseline_pair_rows)
    all_ood: list[dict[str, Any]] = ood_correlation_rows(
        "stage_t_plain", baseline_pair_rows, ood_by_snapshot
    )
    baseline_row, baseline_channels = comparison_rows(
        "stage_t_plain", 150, baseline_metrics, baseline_metrics
    )
    all_one_step = [baseline_row]
    all_channels = list(baseline_channels)
    all_rollout = rollout_table_rows("stage_t_plain", baseline_rollout)
    del baseline_model
    torch.cuda.empty_cache()

    for variant in variants:
        output = args.root / "variants" / variant
        history: list[dict[str, Any]] = []
        for epoch in [int(value) for value in config["pilots"]["checkpoint_epochs"]]:
            checkpoint = output / "checkpoints" / f"epoch_{epoch:04d}.pt"
            model, _, reload_status = model_from_checkpoint(checkpoint, stage_s, device, expected_updates=epoch * 42)
            metrics = evaluate_one_step(model, data, validation_pairs, shell_index, reference, include_transport_rows=False)
            metrics["checkpoint_epoch"] = epoch
            metrics["checkpoint_reload"] = reload_status
            write_json(output / "validation" / f"epoch_{epoch:04d}.json", metrics)
            history.append(score_row(epoch, metrics))
            print(json.dumps({"variant": variant, "validation_epoch": epoch, "state_l2": history[-1]["state_relative_l2"], "shell": history[-1]["shell_skill"], "radial": history[-1]["radial_skill"]}), flush=True)
            del model
            torch.cuda.empty_cache()
        write_csv(output / "validation_history.csv", history)
        selection = selector(history)
        write_json(output / "checkpoint_selection.json", selection)
        formal_epoch = int(selection["formal_best_state_l2"]["epoch"])
        source_checkpoint = output / "checkpoints" / f"epoch_{formal_epoch:04d}.pt"
        formal_checkpoint = output / "formal_best.pt"
        shutil.copy2(source_checkpoint, formal_checkpoint)
        model, payload, reload_status = model_from_checkpoint(
            formal_checkpoint, stage_s, device, expected_updates=formal_epoch * 42
        )
        with torch.no_grad():
            probe_a = tensor_sha256(data.predict(model, 169)["z_prediction"])
            probe_b = tensor_sha256(data.predict(model, 169)["z_prediction"])
        if probe_a != probe_b:
            raise RuntimeError("Stage V formal checkpoint prediction is nondeterministic")
        formal_metrics = evaluate_one_step(model, data, validation_pairs, shell_index, reference)
        first_metrics = json.loads((output / "validation" / f"epoch_{formal_epoch:04d}.json").read_text())
        recomputed = float(formal_metrics["model"]["normalized_relative_l2"]["arithmetic_average"])
        recorded = float(first_metrics["model"]["normalized_relative_l2"]["arithmetic_average"])
        if not math.isclose(recomputed, recorded, rel_tol=1e-7, abs_tol=1e-9):
            raise RuntimeError("Stage V formal validation failed deterministic recomputation")
        formal_metrics.update({
            "checkpoint_epoch": formal_epoch,
            "optimizer_updates": formal_epoch * 42,
            "checkpoint_reload": reload_status,
            "deterministic_probe_prediction": True,
            "probe_prediction_sha256": probe_a,
            "validation_recomputed_matches_recorded": True,
        })
        write_json(output / "formal_best_one_step.json", formal_metrics)
        write_json(output / "formal_best_reload.json", {
            **reload_status,
            "epoch": formal_epoch,
            "optimizer_updates": int(payload["training_state"]["optimizer_updates"]),
            "deterministic_probe_prediction": True,
            "probe_prediction_sha256": probe_a,
            "validation_recomputed_matches_recorded": True,
        })
        strata = temporal_rows(variant, formal_epoch, model, data, thirds, shell_index, reference)
        write_csv(output / "temporal_thirds.csv", strata)
        all_shift.extend(strata)
        rollout_metrics = rollout(model, data, shell_index, reference, epoch=formal_epoch, steps=100)
        write_json(output / "rollout/formal_best_rollout.json", rollout_metrics)
        validation_metrics = [pair_metrics(model, data, int(source), shell_index, "validation") for source in validation_pairs]
        for row in validation_metrics:
            row["variant"] = variant
            row["display_name"] = DISPLAY[variant]
            row["formal_best_epoch"] = formal_epoch
        write_csv(output / "formal_best_pair_metrics.csv", validation_metrics)
        all_validation_pair_rows.extend(validation_metrics)
        all_ood.extend(ood_correlation_rows(variant, validation_metrics, ood_by_snapshot))
        row, channels = comparison_rows(variant, formal_epoch, formal_metrics, baseline_metrics)
        all_one_step.append(row)
        all_channels.extend(channels)
        all_rollout.extend(rollout_table_rows(variant, rollout_metrics))
        print(json.dumps({"variant": variant, "formal_epoch": formal_epoch, "first_10x": rollout_metrics["FIRST_10X_PHYSICAL_RANGE_STEP"], "completed_steps": rollout_metrics["completed_steps"]}), flush=True)
        del model
        torch.cuda.empty_cache()
    write_csv(args.root / "comparison/one_step_comparison.csv", all_one_step)
    write_csv(args.root / "comparison/per_channel_comparison.csv", all_channels)
    write_csv(args.root / "comparison/rollout_comparison.csv", all_rollout)
    write_csv(args.root / "comparison/shift_comparison.csv", all_shift)
    write_csv(args.root / "shift/error_ood_correlations.csv", all_ood)
    make_figures(args.root, all_one_step, all_rollout, ood_rows, all_validation_pair_rows)
    data.close()


if __name__ == "__main__":
    main()
