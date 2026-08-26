#!/usr/bin/env python3
"""Build the fixed Stage T learning curves, figures, report, and decisions."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grmhd import CHANNELS
from grmhd.stage_t_training import data_effect_at_1260


ROOT = Path(__file__).resolve().parents[1]
STAGE_T = ROOT / "artifacts/stage_t"
STAGE_S = ROOT / "artifacts/stage_s"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def stage_s_row(label: str, directory: str, train_pairs: int, updates: int) -> dict[str, Any]:
    metrics = read_json(STAGE_S / directory / "one_step_metrics.json")
    rollout = read_json(STAGE_S / directory / "rollout_metrics.json")
    return {
        "label": label, "source": "Stage S formal best checkpoint",
        "train_pairs": train_pairs, "epoch": None, "optimizer_updates": updates,
        "normalized_relative_l2": metrics["model"]["normalized_relative_l2"]["arithmetic_average"],
        "persistence_ratio": metrics["model_error_over_persistence_error"],
        "residual_relative_l2": metrics["model"]["residual"]["arithmetic_average_relative_l2"],
        "residual_global_cosine": metrics["model"]["residual"]["global_cosine"],
        "shell_skill": metrics["transport"]["shell_skill_median"],
        "radial_skill": metrics["transport"]["radial_skill_median"],
        "physical_relative_l2": metrics["model"]["physical_relative_l2"]["arithmetic_average"],
        "Rout_fraction": metrics["normalized_global_norm"]["above_Rout_fraction"],
        "first_10x_step": rollout["first_physical_range_explosion_step"],
    }


def metric_row(epoch: int, metrics: Mapping[str, Any], rollout: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        "label": f"full-{epoch}epoch", "source": "Stage T fixed checkpoint",
        "train_pairs": 168, "epoch": epoch, "optimizer_updates": epoch * 42,
        "normalized_relative_l2": metrics["model"]["normalized_relative_l2"]["arithmetic_average"],
        "persistence_ratio": metrics["model_error_over_persistence_error"],
        "residual_relative_l2": metrics["model"]["residual"]["arithmetic_average_relative_l2"],
        "residual_global_cosine": metrics["model"]["residual"]["global_cosine"],
        "shell_skill": metrics["transport"]["shell_skill_median"],
        "radial_skill": metrics["transport"]["radial_skill_median"],
        "physical_relative_l2": metrics["model"]["physical_relative_l2"]["arithmetic_average"],
        "Rout_fraction": metrics["normalized_global_norm"]["above_Rout_fraction"],
        "first_10x_step": None if rollout is None else rollout["FIRST_10X_PHYSICAL_RANGE_STEP"],
    }


def fmt(value: Any) -> str:
    if value is None or value == "":
        return "not evaluated"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6g}"
    return str(value)


def plot_metric(rows: list[dict[str, Any]], key: str, title: str, ylabel: str, filename: str, *, log: bool = False, reference: float | None = None) -> None:
    epochs = [int(row["epoch"]) for row in rows]
    values = [float(row[key]) for row in rows]
    fig, axis = plt.subplots(figsize=(7.0, 4.3))
    axis.plot(epochs, values, marker="o", label="full-data LocalNO")
    if reference is not None:
        axis.axhline(reference, color="black", linestyle="--", linewidth=1, label="persistence")
    if log and all(value > 0 for value in values):
        axis.set_yscale("log")
    axis.set_xlabel("epoch")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(STAGE_T / "figures" / filename, dpi=150)
    plt.close(fig)


def main() -> None:
    (STAGE_T / "metrics").mkdir(parents=True, exist_ok=True)
    (STAGE_T / "figures").mkdir(parents=True, exist_ok=True)
    epochs = (2, 10, 30, 75, 150, 300, 600, 900, 1200)
    rollout_epochs = {30, 75, 150, 300, 600, 1200}
    checkpoint_metrics = {
        epoch: read_json(STAGE_T / "full_long" / f"checkpoint_metrics_epoch_{epoch:04d}.json")
        for epoch in epochs
    }
    rollouts = {
        epoch: read_json(STAGE_T / "full_long/rollout" / f"epoch_{epoch:04d}_rollout.json")
        for epoch in rollout_epochs
    }
    persistence = checkpoint_metrics[2]["persistence"]
    persistence_norm = float(persistence["normalized_relative_l2"]["arithmetic_average"])
    persistence_physical = float(persistence["physical_relative_l2"]["arithmetic_average"])
    long_rows = [
        metric_row(epoch, checkpoint_metrics[epoch], rollouts.get(epoch)) for epoch in epochs
    ]
    write_csv(STAGE_T / "metrics/learning_curve.csv", long_rows)

    per_channel_rows: list[dict[str, Any]] = []
    for epoch in epochs:
        metrics = checkpoint_metrics[epoch]
        for channel in CHANNELS:
            model_norm = metrics["model"]["normalized_relative_l2"]["per_channel"][channel]
            persistence_channel = metrics["persistence"]["normalized_relative_l2"]["per_channel"][channel]
            transport = metrics["transport"]["per_channel"][channel]
            per_channel_rows.append({
                "epoch": epoch, "optimizer_updates": epoch * 42, "channel": channel,
                "state_relative_l2": model_norm,
                "persistence_relative_l2": persistence_channel,
                "persistence_ratio": float(model_norm) / max(float(persistence_channel), 1e-300),
                "residual_relative_l2": metrics["model"]["residual"]["per_channel_relative_l2"][channel],
                "residual_cosine": metrics["model"]["residual"]["per_channel_cosine"][channel],
                "shell_skill": transport["shell_skill_median"],
                "radial_skill": transport["radial_skill_median"],
                "physical_relative_l2": metrics["model"]["physical_relative_l2"]["per_channel"][channel],
            })
    write_csv(STAGE_T / "metrics/per_channel_learning_curve.csv", per_channel_rows)

    persistence_row = {
        "label": "Persistence", "source": "frozen expanded validation baseline",
        "train_pairs": None, "epoch": None, "optimizer_updates": None,
        "normalized_relative_l2": persistence_norm, "persistence_ratio": 1.0,
        "residual_relative_l2": 1.0, "residual_global_cosine": 0.0,
        "shell_skill": 0.0, "radial_skill": 0.0,
        "physical_relative_l2": persistence_physical, "Rout_fraction": None,
        "first_10x_step": None,
    }
    small600 = stage_s_row("small-600", "s_small_matched", 79, 600)
    full600 = stage_s_row("full-600", "s_full_matched", 168, 600)
    full1260 = stage_s_row("full-1260", "s_full_30epoch", 168, 1260)
    small_metrics = read_json(STAGE_T / "causal_control/small_1260_metrics.json")
    small1260 = {
        "label": "small-1260", "source": "Stage T final matched-update checkpoint",
        "train_pairs": 79, "epoch": None, "optimizer_updates": 1260,
        "normalized_relative_l2": small_metrics["model"]["normalized_relative_l2"]["arithmetic_average"],
        "persistence_ratio": small_metrics["model_error_over_persistence_error"],
        "residual_relative_l2": small_metrics["model"]["residual"]["arithmetic_average_relative_l2"],
        "residual_global_cosine": small_metrics["model"]["residual"]["global_cosine"],
        "shell_skill": small_metrics["transport"]["shell_skill_median"],
        "radial_skill": small_metrics["transport"]["radial_skill_median"],
        "physical_relative_l2": small_metrics["model"]["physical_relative_l2"]["arithmetic_average"],
        "Rout_fraction": small_metrics["normalized_global_norm"]["above_Rout_fraction"],
        "first_10x_step": None,
    }
    comparison = [persistence_row, small600, full600, small1260, full1260, *long_rows]
    write_csv(STAGE_T / "metrics/stage_s_stage_t_comparison.csv", comparison)

    train_log = read_csv(STAGE_T / "full_long/train_log.csv")
    train_by_epoch = {int(row["epoch"]): row for row in train_log}
    checkpoint_rows = read_csv(STAGE_T / "full_long/checkpoint_metrics.csv")
    for row in checkpoint_rows:
        epoch = int(row["epoch"])
        row["train_loss_mean"] = train_by_epoch[epoch]["train_loss_mean"]
        row["learning_rate"] = train_by_epoch[epoch]["learning_rate_end"]
        row["clipping_fraction"] = train_by_epoch[epoch]["clipping_fraction"]
    write_csv(STAGE_T / "full_long/checkpoint_metrics.csv", checkpoint_rows)

    data_effect, data_relative_change = data_effect_at_1260(
        float(small1260["normalized_relative_l2"]), float(full1260["normalized_relative_l2"])
    )
    best_state = min(long_rows, key=lambda row: float(row["normalized_relative_l2"]))
    best_residual = min(long_rows, key=lambda row: float(row["residual_relative_l2"]))
    best_shell = max(long_rows, key=lambda row: float(row["shell_skill"]))
    best_radial = max(long_rows, key=lambda row: float(row["radial_skill"]))
    final = long_rows[-1]
    epoch30 = next(row for row in long_rows if row["epoch"] == 30)
    decoupled_pairs = []
    for left, right in zip(long_rows, long_rows[1:]):
        if (
            float(right["normalized_relative_l2"]) < float(left["normalized_relative_l2"])
            and (
                float(right["shell_skill"]) < float(left["shell_skill"])
                or float(right["radial_skill"]) < float(left["radial_skill"])
            )
        ):
            decoupled_pairs.append([left["epoch"], right["epoch"]])
    objective_decoupling = bool(decoupled_pairs)

    strata = read_csv(STAGE_T / "metrics/validation_time_stratification.csv")
    best_epoch = int(best_state["epoch"])
    best_strata = [row for row in strata if int(row["epoch"]) == best_epoch]
    stratum_average = {
        name: float(np.mean([
            float(row["model_normalized_relative_l2"]) for row in best_strata if row["stratum"] == name
        ])) for name in ("early", "middle", "late")
    }
    late_early = stratum_average["late"] / max(stratum_average["early"], 1e-300)
    temporal_shift = "LOW" if late_early <= 1.10 else "MODERATE" if late_early <= 1.50 else "STRONG"

    best_rollout = rollouts.get(best_epoch)
    if best_rollout is None:
        eligible = min(rollout_epochs, key=lambda value: abs(value - best_epoch))
        best_rollout = rollouts[eligible]
        best_rollout_epoch = eligible
    else:
        best_rollout_epoch = best_epoch
    rescued = (
        float(best_state["normalized_relative_l2"]) < persistence_norm
        and float(best_state["residual_relative_l2"]) < 1.0
        and float(best_state["shell_skill"]) > 0.0
        and float(best_state["radial_skill"]) > 0.0
        and (best_rollout["FIRST_10X_PHYSICAL_RANGE_STEP"] is None or best_rollout["FIRST_10X_PHYSICAL_RANGE_STEP"] > 3)
    )
    overfit = (
        best_epoch < 1200
        and float(final["normalized_relative_l2"]) > 1.10 * float(best_state["normalized_relative_l2"])
    )
    if rescued:
        primary = "A"
        primary_label = "OPTIMIZATION_BUDGET_RESCUES_LOCALNO"
    elif overfit:
        primary = "D"
        primary_label = "OPTIMIZATION_CAUSES_OVERFIT_OR_TRANSPORT_DEGRADATION"
    elif float(best_state["normalized_relative_l2"]) < persistence_norm:
        primary = "B"
        primary_label = "OPTIMIZATION_IMPROVES_STATE_NOT_DYNAMICS"
    else:
        primary = "C"
        primary_label = "OPTIMIZATION_SATURATES_BELOW_PERSISTENCE"
    secondary = "INTERACTION" if data_effect == "POSITIVE" else data_effect
    authorize_u = (
        float(final["shell_skill"]) <= 0.0
        or float(final["radial_skill"]) <= 0.0
        or rollouts[1200]["FIRST_10X_PHYSICAL_RANGE_STEP"] in (1, 2, 3)
        or rollouts[1200]["FIRST_NEGATIVE_RESIDUAL_COSINE_STEP"] is not None
    )

    plot_metric(long_rows, "normalized_relative_l2", "Normalized state error", "average relative L2", "normalized_l2_vs_epoch.png", reference=persistence_norm)
    plot_metric(long_rows, "residual_relative_l2", "Residual error", "residual relative L2", "residual_l2_vs_epoch.png", reference=1.0)
    plot_metric(long_rows, "residual_global_cosine", "Residual direction", "global residual cosine", "residual_cosine_vs_epoch.png", reference=0.0)
    plot_metric(long_rows, "shell_skill", "Shell transport skill", "persistence-relative skill", "shell_skill_vs_epoch.png", reference=0.0)
    plot_metric(long_rows, "radial_skill", "Radial transport skill", "persistence-relative skill", "radial_skill_vs_epoch.png", reference=0.0)
    plot_metric(long_rows, "physical_relative_l2", "Physical-space error", "average relative L2", "physical_error_vs_epoch.png", log=True, reference=persistence_physical)

    fig, axes = plt.subplots(4, 2, figsize=(10, 12), sharex=True)
    for axis, channel in zip(axes.ravel(), CHANNELS, strict=True):
        selected = [row for row in per_channel_rows if row["channel"] == channel]
        axis.plot([int(row["epoch"]) for row in selected], [float(row["state_relative_l2"]) for row in selected], marker="o", label="state L2")
        axis.plot([int(row["epoch"]) for row in selected], [float(row["residual_relative_l2"]) for row in selected], marker="s", label="residual L2")
        axis.set_title(channel)
        axis.grid(alpha=0.2)
    axes[0, 0].legend()
    axes[-1, 0].set_xlabel("epoch")
    axes[-1, 1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(STAGE_T / "figures/per_channel_learning_curves.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4.3))
    for epoch in sorted(rollout_epochs):
        records = rollouts[epoch]["records"]
        gt = [row for row in records if row["ground_truth_available"]]
        axis.plot([row["step"] for row in gt], [row["normalized_relative_l2_average"] for row in gt], label=f"epoch {epoch}")
    axis.set_yscale("log")
    axis.set_xlabel("closed-loop step")
    axis.set_ylabel("normalized GT relative L2")
    axis.set_title("Closed-loop error")
    axis.grid(alpha=0.2)
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(STAGE_T / "figures/rollout_error_vs_step.png", dpi=150)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4.3))
    for epoch in sorted(rollout_epochs):
        records = rollouts[epoch]["records"]
        rout = checkpoint_metrics[epoch]["normalized_global_norm"]["Rout"]
        axis.plot([row["step"] for row in records], [row["normalized_global_norm"] / rout for row in records], label=f"epoch {epoch}")
    axis.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Rout")
    axis.set_yscale("log")
    axis.set_xlabel("closed-loop step")
    axis.set_ylabel("normalized norm / Rout")
    axis.set_title("Closed-loop range")
    axis.grid(alpha=0.2)
    axis.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(STAGE_T / "figures/rollout_range_vs_step.png", dpi=150)
    plt.close(fig)

    table_order = [persistence_row, small600, full600, small1260, full1260] + [
        row for row in long_rows if row["epoch"] in (75, 150, 300, 600, 1200)
    ]
    table = [
        "| checkpoint | updates | norm L2 | persistence ratio | residual L2 | residual cosine | shell skill | radial skill | first 10x step |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in table_order:
        table.append("| " + " | ".join(fmt(row[key]) for key in (
            "label", "optimizer_updates", "normalized_relative_l2", "persistence_ratio",
            "residual_relative_l2", "residual_global_cosine", "shell_skill",
            "radial_skill", "first_10x_step",
        )) + " |")

    final_metrics = checkpoint_metrics[1200]
    focus_rows = []
    for channel in ("Bcc2", "Bcc3", "vel3", "rho", "press"):
        row = next(item for item in per_channel_rows if item["epoch"] == best_epoch and item["channel"] == channel)
        tail = checkpoint_metrics[best_epoch]["preprocessing_tail"][channel]
        focus_rows.append(
            f"| {channel} | {fmt(row['state_relative_l2'])} | {fmt(row['residual_relative_l2'])} | "
            f"{fmt(row['residual_cosine'])} | {fmt(row['shell_skill'])} | {fmt(row['radial_skill'])} | "
            f"{fmt(row['physical_relative_l2'])} | {fmt(tail['extreme_decoder_tail_fraction'])} | "
            f"{fmt(tail['physical_squared_error_contribution_fraction'])} |"
        )

    report = "\n".join([
        "# Stage T — Optimization Convergence and Paper-Budget Audit",
        "",
        *table,
        "",
        "## Frozen contract and executed budget",
        "",
        "`DATASET_FROZEN=true`, `SPLIT_FROZEN=true`, `PREPROCESSING_FROZEN=true`, and `MODEL_FROZEN=true`. "
        "The expanded HDF5, 0..168/169..211 split, P3 normalizer, eight radial shells, 358,296-parameter LocalNO, Plain L2 residual contract, Adam settings, and shared initial tensor hash were checksum-verified before training.",
        "",
        "The long run used 168 microbatches and 42 optimizer updates per epoch. It completed 1,200 epochs / 50,400 updates; warmup was 75 epochs / 3,150 updates, followed by cosine decay to 1e-6. No validation metric was used to tune or stop training.",
        "",
        "## Causal control",
        "",
        f"T-small-1260 scored {fmt(small1260['normalized_relative_l2'])}; S-full-1260 scored {fmt(full1260['normalized_relative_l2'])}. The full-minus-small benefit is {data_relative_change:.3%} under the sign convention `(small-full)/small`. With the predeclared ±2% neutral band, `DATA_EFFECT_AT_1260={data_effect}`.",
        "",
        "## Checkpoint selection and convergence",
        "",
        f"Formal best-state-L2: epoch {best_state['epoch']} ({fmt(best_state['normalized_relative_l2'])}). Diagnostic best-residual-L2: epoch {best_residual['epoch']} ({fmt(best_residual['residual_relative_l2'])}); best-shell: epoch {best_shell['epoch']} ({fmt(best_shell['shell_skill'])}); best-radial: epoch {best_radial['epoch']} ({fmt(best_radial['radial_skill'])}). These diagnostic selectors do not replace the formal selector.",
        "",
        f"`OBJECTIVE_METRIC_DECOUPLING={str(objective_decoupling).lower()}` using the frozen consecutive-checkpoint rule. Triggering intervals: {decoupled_pairs or 'none'}.",
        "",
        "## Focus-channel dynamics at formal best checkpoint",
        "",
        "| channel | state L2 | residual L2 | residual cosine | shell skill | radial skill | physical L2 | decoder-tail fraction | physical-error contribution |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *focus_rows,
        "",
        "## Validation-time distribution shift",
        "",
        f"At the formal best checkpoint, early/middle/late normalized averages are {fmt(stratum_average['early'])}, {fmt(stratum_average['middle'])}, and {fmt(stratum_average['late'])}. Late/early={fmt(late_early)}; `TEMPORAL_SHIFT_SENSITIVITY={temporal_shift}`. This stratification is reporting-only.",
        "",
        "## Direct answers",
        "",
        f"1. **Q1 — independent data-volume effect?** `DATA_EFFECT_AT_1260={data_effect}`; the matched small/full difference is {data_relative_change:.3%}.",
        f"2. **Q2 — were 30 epochs under-trained?** {'Yes' if best_epoch > 30 and float(best_state['normalized_relative_l2']) < 0.98 * float(epoch30['normalized_relative_l2']) else 'No clear evidence'}; formal best is epoch {best_epoch}.",
        f"3. **Q3 — did normalized one-step error improve?** Epoch 30={fmt(epoch30['normalized_relative_l2'])}, best={fmt(best_state['normalized_relative_l2'])}, epoch 1200={fmt(final['normalized_relative_l2'])}.",
        f"4. **Q4 — residual rel-L2 below 1?** {'Yes' if float(best_residual['residual_relative_l2']) < 1 else 'No'}; minimum={fmt(best_residual['residual_relative_l2'])} at epoch {best_residual['epoch']}.",
        f"5. **Q5 — residual cosine sustained improvement?** Epoch 30={fmt(epoch30['residual_global_cosine'])}; epoch 1200={fmt(final['residual_global_cosine'])}.",
        f"6. **Q6 — shell/radial skill positive?** Best shell={fmt(best_shell['shell_skill'])}; best radial={fmt(best_radial['radial_skill'])}; epoch-1200 values={fmt(final['shell_skill'])}/{fmt(final['radial_skill'])}.",
        f"7. **Q7 — did rho/press inverse-tail explosion improve?** Physical average: epoch 30={fmt(epoch30['physical_relative_l2'])}, best-state checkpoint={fmt(best_state['physical_relative_l2'])}, epoch 1200={fmt(final['physical_relative_l2'])}, persistence={fmt(persistence_physical)}. Frozen catastrophic-tail gates are in each checkpoint JSON.",
        f"8. **Q8 — was step-1 range failure delayed?** Epoch-1200 first 10x step={fmt(rollouts[1200]['FIRST_10X_PHYSICAL_RANGE_STEP'])}; formal-best rollout proxy uses epoch {best_rollout_epoch} and first 10x={fmt(best_rollout['FIRST_10X_PHYSICAL_RANGE_STEP'])}.",
        f"9. **Q9 — state/transport decoupling?** `OBJECTIVE_METRIC_DECOUPLING={str(objective_decoupling).lower()}`.",
        f"10. **Q10 — did paper-like optimization rescue this adapted LocalNO?** {primary_label}.",
        f"11. **Q11 — enter Stage U?** `AUTHORIZE_STAGE_U={str(authorize_u).lower()}`.",
        "",
        "## Scientific scope",
        "",
        "This is a spherical-Kerr-Schild, 64^3, expanded212, P3, radial-shell adapted reproduction with a differential LocalNO proxy—not the paper's exact geometry/operator. A small normalized one-step advantage is not treated as operator success unless residual dynamics, transport, physical tails, and closed-loop stability agree.",
        "",
        f"`PRIMARY_DECISION = {primary}`",
        f"`PRIMARY_DECISION_LABEL = {primary_label}`",
        f"`SECONDARY_DATA_FINDING = {secondary}`",
        f"`AUTHORIZE_STAGE_U = {str(authorize_u).lower()}`",
    ])
    write_text(STAGE_T / "STAGE_T_REPORT.md", report)
    handoff = (
        "Optimization budget has been sufficiently tested; the next controlled variable should be "
        "spherical-grid/operator geometry."
    ) if authorize_u else "The Stage U geometry/operator audit is not yet authorized."
    decision = "\n".join([
        "# Stage T Decision",
        "",
        f"PRIMARY_DECISION = {primary}",
        f"PRIMARY_DECISION_LABEL = {primary_label}",
        f"SECONDARY_DATA_FINDING = {secondary}",
        f"DATA_EFFECT_AT_1260 = {data_effect}",
        f"OBJECTIVE_METRIC_DECOUPLING = {str(objective_decoupling).lower()}",
        f"TEMPORAL_SHIFT_SENSITIVITY = {temporal_shift}",
        f"AUTHORIZE_STAGE_U = {str(authorize_u).lower()}",
        "",
        handoff,
    ])
    write_text(STAGE_T / "STAGE_T_DECISION.md", decision)


if __name__ == "__main__":
    main()
