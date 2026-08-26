#!/usr/bin/env python3
"""Build Stage S comparison tables, figures, report, and frozen decision."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grmhd import CHANNELS


EXPERIMENTS = ("s_small_matched", "s_full_matched", "s_full_30epoch")
LABELS = {
    "s_small_matched": "S-small matched",
    "s_full_matched": "S-full matched",
    "s_full_30epoch": "S-full 30epoch",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = list(rows[0])
    for row in rows[1:]:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def value(metrics: Mapping[str, Any], path: str) -> Any:
    current: Any = metrics
    for key in path.split("."):
        current = current[key]
    return current


def selected_rollout(rollout: Mapping[str, Any], step: int, key: str) -> Any:
    return rollout["records"][step - 1].get(key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/stage_s"))
    args = parser.parse_args()
    root = args.root
    audit = read_json(root / "expanded_data_audit.json")
    split = read_json(root / "train_val_split.json")
    metrics = {name: read_json(root / name / "one_step_metrics.json") for name in EXPERIMENTS}
    rollouts = {name: read_json(root / name / "rollout_metrics.json") for name in EXPERIMENTS}
    training = {name: read_json(root / name / "training_summary.json") for name in EXPERIMENTS}
    persistence = metrics[EXPERIMENTS[0]]["persistence"]

    core_rows = [{
        "model": "Persistence", "train_pairs": "", "updates": "", "effective_samples": "",
        "epochs_equivalent": "", "avg_norm_rel_l2": value(persistence, "normalized_relative_l2.arithmetic_average"),
        "global_norm_rel_l2": value(persistence, "normalized_relative_l2.global"),
        "persistence_ratio": 1.0,
        "avg_physical_rel_l2": value(persistence, "physical_relative_l2.arithmetic_average"),
        "residual_rel_l2": 1.0, "residual_global_cosine": "",
        "shell_skill": 0.0, "radial_skill": 0.0, "first_unstable_step": "stable baseline",
        "step100_finite": True,
    }]
    for name in EXPERIMENTS:
        item, rollout, train = metrics[name], rollouts[name], training[name]
        core_rows.append({
            "model": LABELS[name], "train_pairs": train["train_pairs"],
            "updates": train["optimizer_updates"], "effective_samples": train["effective_samples_seen"],
            "epochs_equivalent": train["population_epochs_equivalent"],
            "avg_norm_rel_l2": value(item, "model.normalized_relative_l2.arithmetic_average"),
            "global_norm_rel_l2": value(item, "model.normalized_relative_l2.global"),
            "persistence_ratio": item["model_error_over_persistence_error"],
            "avg_physical_rel_l2": value(item, "model.physical_relative_l2.arithmetic_average"),
            "residual_rel_l2": value(item, "model.residual.arithmetic_average_relative_l2"),
            "residual_global_cosine": value(item, "model.residual.global_cosine"),
            "shell_skill": value(item, "transport.shell_skill_median"),
            "radial_skill": value(item, "transport.radial_skill_median"),
            "first_unstable_step": rollout["first_physical_range_explosion_step"],
            "step100_finite": rollout["records"][99]["finite"],
        })
    write_csv(root / "core_comparison.csv", core_rows)

    requested_metrics = {
        "avg_normalized_relative_l2": "model.normalized_relative_l2.arithmetic_average",
        "avg_physical_relative_l2": "model.physical_relative_l2.arithmetic_average",
        "residual_relative_l2": "model.residual.arithmetic_average_relative_l2",
        "Bcc2_normalized_relative_l2": "model.normalized_relative_l2.per_channel.Bcc2",
        "Bcc3_normalized_relative_l2": "model.normalized_relative_l2.per_channel.Bcc3",
        "vel3_normalized_relative_l2": "model.normalized_relative_l2.per_channel.vel3",
        "shell_skill": "transport.shell_skill_median",
        "radial_skill": "transport.radial_skill_median",
    }
    volume_rows = []
    for metric_name, path in requested_metrics.items():
        small = float(value(metrics["s_small_matched"], path))
        full = float(value(metrics["s_full_matched"], path))
        natural = float(value(metrics["s_full_30epoch"], path))
        volume_rows.append({
            "metric": metric_name, "s_small_matched": small, "s_full_matched": full,
            "improvement_data_formula": (small - full) / small if small != 0 else "undefined",
            "full_minus_small_for_skill_interpretation": full - small if "skill" in metric_name else "",
            "s_full_30epoch": natural,
            "interpretation": (
                "higher_is_better; formula ratio is not meaningful when signed skill crosses/approaches zero"
                if "skill" in metric_name else "lower_is_better"
            ),
        })
    write_csv(root / "data_volume_comparison.csv", volume_rows)

    stage_r_root = Path("outputs/paper_reduced100/stage_r")
    stage_r_validation = read_json(stage_r_root / "validation_metrics.json")
    stage_r_residual = read_json(stage_r_root / "residual_metrics.json")
    stage_r_decision = read_json(stage_r_root / "stage_r_decision.json")
    stage_r_gt = read_json(stage_r_root / "rollout_gt.json")
    stage_r_gate = read_json(stage_r_root / "stage_m_v1_gate.json")
    stage_r_shell = float(np.median([
        stage_r_gate["channels"][name]["gate"]["gate_3_transport_skill"]["shell_skill"]
        for name in TARGET_CHANNELS
    ]))
    stage_r_radial = float(np.median([
        stage_r_gate["channels"][name]["gate"]["gate_3_transport_skill"]["radial_skill"]
        for name in TARGET_CHANNELS
    ]))
    stage_r_values = {
        "one_step_avg_normalized_relative_l2": stage_r_validation["residual_reconstructed_model"]["metrics"]["E_norm"]["arithmetic_average"],
        "persistence_ratio": stage_r_validation["model_over_persistence_normalized_average"],
        "residual_relative_l2": stage_r_residual["arithmetic_average_relative_l2"],
        "residual_cosine": stage_r_residual["residual_cosine"]["mean"],
        "Rout_failure_fraction": stage_r_validation["constraints"]["prediction_above_rout_fraction"],
        "step1_normalized_error": stage_r_gt["records"][0]["oracle_aware"]["metrics"]["E_norm"]["arithmetic_average"],
        "first_physical_range_explosion_step": stage_r_decision["instability_evidence"]["range_explosion_by_gt_step"],
        "shell_skill": stage_r_shell, "radial_skill": stage_r_radial,
        "step19_normalized_error": stage_r_gt["records"][18]["oracle_aware"]["metrics"]["E_norm"]["arithmetic_average"],
        "step100_finite": stage_r_decision["engineering_evidence"]["no_gt_rollout_finite_positive"],
    }
    stage_s_paths = {
        "one_step_avg_normalized_relative_l2": lambda m, r: value(m, "model.normalized_relative_l2.arithmetic_average"),
        "persistence_ratio": lambda m, r: m["model_error_over_persistence_error"],
        "residual_relative_l2": lambda m, r: value(m, "model.residual.arithmetic_average_relative_l2"),
        "residual_cosine": lambda m, r: value(m, "model.residual.global_cosine"),
        "Rout_failure_fraction": lambda m, r: value(m, "normalized_global_norm.above_Rout_fraction"),
        "step1_normalized_error": lambda m, r: selected_rollout(r, 1, "normalized_relative_l2_average"),
        "first_physical_range_explosion_step": lambda m, r: r["first_physical_range_explosion_step"],
        "shell_skill": lambda m, r: value(m, "transport.shell_skill_median"),
        "radial_skill": lambda m, r: value(m, "transport.radial_skill_median"),
        "step19_normalized_error": lambda m, r: selected_rollout(r, 19, "normalized_relative_l2_average"),
        "step100_finite": lambda m, r: r["records"][99]["finite"],
    }
    stage_r_rows = []
    for metric_name, stage_r_value in stage_r_values.items():
        row = {"metric": metric_name, "stage_r_reduced100": stage_r_value}
        for name in EXPERIMENTS:
            row[name] = stage_s_paths[metric_name](metrics[name], rollouts[name])
        row["comparability_note"] = (
            "contextual only: Stage R used snapshots 91..110 and old P3 fit; Stage S uses snapshots 169..211 and expanded-train P3 refit"
        )
        stage_r_rows.append(row)
    write_csv(root / "stage_r_vs_stage_s.csv", stage_r_rows)

    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    names = ["Persistence", *(LABELS[name] for name in EXPERIMENTS)]
    colors = ["0.5", "#4c78a8", "#f58518", "#54a24b"]
    plt.figure(figsize=(8, 4.5))
    values = [core_rows[index]["avg_norm_rel_l2"] for index in range(4)]
    plt.bar(names, values, color=colors); plt.ylabel("avg normalized relative L2")
    plt.xticks(rotation=15, ha="right"); plt.tight_layout(); plt.savefig(figures / "one_step_error_comparison.png", dpi=160); plt.close()

    x = np.arange(len(CHANNELS)); width = 0.2
    plt.figure(figsize=(10, 5))
    series = [persistence["normalized_relative_l2"]["per_channel"]] + [metrics[name]["model"]["normalized_relative_l2"]["per_channel"] for name in EXPERIMENTS]
    for index, (label, values_by_channel) in enumerate(zip(names, series, strict=True)):
        plt.bar(x + (index - 1.5) * width, [values_by_channel[ch] for ch in CHANNELS], width, label=label, color=colors[index])
    plt.xticks(x, CHANNELS, rotation=30); plt.ylabel("normalized relative L2"); plt.legend(fontsize=8); plt.tight_layout(); plt.savefig(figures / "per_channel_error.png", dpi=160); plt.close()

    plt.figure(figsize=(8, 4.5))
    for name, color in zip(EXPERIMENTS, colors[1:], strict=True):
        gt = rollouts[name]["records"][:42]
        plt.plot([row["step"] for row in gt], [row["residual_cosine"] for row in gt], label=LABELS[name], color=color)
    plt.axhline(0, color="black", lw=0.8); plt.xlabel("rollout step"); plt.ylabel("residual cosine"); plt.legend(); plt.tight_layout(); plt.savefig(figures / "residual_cosine_rollout.png", dpi=160); plt.close()

    plt.figure(figsize=(8, 4.5))
    for name, color in zip(EXPERIMENTS, colors[1:], strict=True):
        gt = rollouts[name]["records"][:42]
        plt.plot([row["step"] for row in gt], [row["normalized_relative_l2_average"] for row in gt], label=LABELS[name], color=color)
    plt.yscale("log"); plt.xlabel("rollout step"); plt.ylabel("avg normalized relative L2"); plt.legend(); plt.tight_layout(); plt.savefig(figures / "rollout_error_vs_step.png", dpi=160); plt.close()

    for key, filename, ylabel in (
        ("shell_skill", "shell_skill_comparison.png", "persistence-relative shell skill"),
        ("radial_skill", "radial_skill_comparison.png", "persistence-relative radial skill"),
    ):
        plt.figure(figsize=(8, 4.5))
        vals = [0.0] + [float(metrics[name]["transport"][f"{key}_median"]) for name in EXPERIMENTS]
        plt.bar(names, vals, color=colors); plt.axhline(0, color="black", lw=0.8); plt.ylabel(ylabel)
        plt.xticks(rotation=15, ha="right"); plt.tight_layout(); plt.savefig(figures / filename, dpi=160); plt.close()

    plt.figure(figsize=(8, 4.5))
    for name, color in zip(EXPERIMENTS, colors[1:], strict=True):
        rows = rollouts[name]["records"]
        max_abs = [max(max(abs(v["minimum"]), abs(v["maximum"])) for v in row["physical_range"].values()) for row in rows]
        plt.plot([row["step"] for row in rows], max_abs, label=LABELS[name], color=color)
    plt.yscale("log"); plt.xlabel("rollout step"); plt.ylabel("maximum absolute physical stored-component value"); plt.legend(); plt.tight_layout(); plt.savefig(figures / "physical_range_vs_step.png", dpi=160); plt.close()

    small_norm = float(value(metrics["s_small_matched"], "model.normalized_relative_l2.arithmetic_average"))
    full_norm = float(value(metrics["s_full_matched"], "model.normalized_relative_l2.arithmetic_average"))
    natural_norm = float(value(metrics["s_full_30epoch"], "model.normalized_relative_l2.arithmetic_average"))
    persistence_norm = float(value(persistence, "normalized_relative_l2.arithmetic_average"))
    matched_improvement = (small_norm - full_norm) / small_norm
    natural_vs_full = (full_norm - natural_norm) / full_norm
    decision = "D. MORE_OPTIMIZATION_NOT_MORE_DATA"

    table_lines = [
        "| model | train pairs | updates | avg norm rel-L2 | persistence ratio | residual rel-L2 | shell skill | radial skill | first unstable step |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in core_rows:
        table_lines.append(
            f"| {row['model']} | {row['train_pairs'] or '-'} | {row['updates'] or '-'} | {float(row['avg_norm_rel_l2']):.6g} | {float(row['persistence_ratio']):.6g} | {float(row['residual_rel_l2']):.6g} | {float(row['shell_skill']):.6g} | {float(row['radial_skill']):.6g} | {row['first_unstable_step']} |"
        )
    report = f"""# Stage S — Expanded-Data LocalNO Reproduction and Data-Scarcity Audit

## 1. Expanded data summary

Total unique snapshots: 212
Train snapshots: 169 (0..168)
Train pairs: 168
Validation snapshots: 43 (169..211)
Validation pairs: 42
Time range: 0.0 .. 2110.000838137111
Median dt: 10.000659066036974
Grid: static Kerr-Schild AMR source; processed tensor `(212,8,64,64,64)` in `(N,C,phi,theta,r)` order
Channels: Bcc1, Bcc2, Bcc3, rho, press, vel1, vel2, vel3

The 101 new active files are a continuous extension of the 111 old files. All 212 have the same schema/grid signature, strictly increasing time, finite fields, and positive rho/press. Recycle-bin copies were hash-audited and excluded. The processed old prefix is bitwise identical to the frozen 111-snapshot artifact. `DISTRIBUTION_SHIFT=STRONG`; no difficult samples were removed.

## 2. Model

3D differential LocalNO, P3 preprocessing refit on snapshots 0..168 only, normalized-residual target, 358,296 trainable parameters. Inputs are 8 P3 state channels plus 8 frozen spherical-r shell channels; output is 8 residual channels. The model, Plain L2 loss, Adam settings, clip=1, seed=42, and initial tensor hash are frozen from Stage R. This is an adapted method reproduction, not exact volumetric 3D DISCO.

## 3. Core comparison table

{chr(10).join(table_lines)}

All learned rows use their best checkpoint under the same 42-pair held-out normalized-average selection metric. Matched runs both use exactly 2,400 microbatches and 600 optimizer updates. S-full 30epoch uses 5,040 microbatches and 1,260 updates.

The matched scheduler has exactly 600 update calls for both populations and a 40-update warmup (two Stage-R 20-update epochs). The natural run has an 84-update warmup (two complete 42-update full-data epochs). Physical-space average relative L2 is `0.571584` for persistence, `6.41753e21` for S-small, `15.0064` for S-full matched, and `1.14409e22` for S-full 30epoch. These extreme learned values are dominated by rho/press inverse-tail excursions; q001/q999 and saturation evidence is preserved in each `one_step_metrics.json`.

## 4. Direct answers

**Q1. Are the 101 new snapshots compatible?** Yes operationally: exact channel/schema/grid/layout agreement, continuous indices/times, and a bitwise-identical processed old prefix. Component basis and physical units are not explicit in ATHDF metadata; compatibility is supported by the continuous `mad98.prim` series and identical stored spherical-KS contract, not by undocumented unit metadata.

**Q2. What is the actual total?** 212 unique active snapshots: 111 old + 101 new. The chronological split is 169 train snapshots/168 pairs, one dropped boundary pair 168→169, and 43 held-out validation snapshots/42 pairs.

**Q3. Does more data reduce one-step error at matched updates?** No. S-small is `{small_norm:.6g}` and S-full is `{full_norm:.6g}`; `Improvement_data={matched_improvement:.6g}` ({matched_improvement*100:.3f}%). Full is worse, not better.

**Q4. Does it improve residual direction?** No at matched updates. Global residual cosine changes from `{value(metrics['s_small_matched'], 'model.residual.global_cosine'):.6g}` to `{value(metrics['s_full_matched'], 'model.residual.global_cosine'):.6g}`. The 30-epoch full run improves it to `{value(metrics['s_full_30epoch'], 'model.residual.global_cosine'):.6g}`, which is extra-optimization evidence.

**Q5. Does it improve shell/radial transport?** No. Matched-full shell/radial skills are `{value(metrics['s_full_matched'], 'transport.shell_skill_median'):.6g}`/`{value(metrics['s_full_matched'], 'transport.radial_skill_median'):.6g}` versus small `{value(metrics['s_small_matched'], 'transport.shell_skill_median'):.6g}`/`{value(metrics['s_small_matched'], 'transport.radial_skill_median'):.6g}`. All are below persistence (skill <= 0); the 30-epoch run is more negative.

**Q6. Is the Stage-R step~3 physical-range explosion delayed?** No. Under the explicit GT-aware Stage-S rule, all three Stage-S rollouts first exceed the 10× physical range criterion at step 1. S-full 30epoch normalized rollout error grows from `{selected_rollout(rollouts['s_full_30epoch'], 1, 'normalized_relative_l2_average'):.6g}` at step 1 to `{selected_rollout(rollouts['s_full_30epoch'], 5, 'normalized_relative_l2_average'):.6g}` at step 5 and `{selected_rollout(rollouts['s_full_30epoch'], 19, 'normalized_relative_l2_average'):.6g}` at step 19.

**Q7. Does any model beat persistence?** Only S-full 30epoch in normalized one-step average: `{natural_norm:.6g}` versus persistence `{persistence_norm:.6g}` (ratio `{natural_norm/persistence_norm:.6g}`). It does not beat persistence in physical-space average or shell/radial transport, and its rollout rapidly diverges.

**Q8. Is improvement from more data or more optimizer updates?** Evidence supports more optimization, not more data. Matched-full is 2.92% worse than matched-small, while 30epoch-full is `{natural_vs_full*100:.3f}%` better than matched-full after 1,260 rather than 600 updates.

**Q9. Is geometry/operator mismatch still the likely bottleneck?** Data scarcity is not supported as the primary explanation. The remaining evidence is consistent with operator/geometry mismatch plus P3 inverse-tail amplification: normalized one-step improves with updates, but physical errors are tail-dominated, transport skill remains negative, and closed-loop residual direction reverses after early steps. This audit does not isolate those mechanisms causally.

## 5. Engineering and provenance

- Both 2-epoch CUDA smokes passed; training/backward gradients were finite and non-zero.
- All formal runs began from tensor SHA256 `7e10895cb283be8532e4c036e8a526c536a632cce880c500080ea1d14fd89311`.
- Best and last checkpoints strictly reload for all runs.
- RTX 5070 peak allocated memory was 562.816 MiB; no AMP was used.
- All 100-step rollouts remained numerically finite with positive decoded rho/press, but this is not scientific stability: decoded ranges and normalized norms grow severely.
- Stage R numbers in `stage_r_vs_stage_s.csv` are contextual only because its validation time block and P3 fit population differ.

## 6. Decision

**{decision}**

The matched-update experiment does not show a benefit from the larger pair population. The natural 30-epoch full-data run improves normalized one-step error and residual direction only after receiving more optimizer updates, even slightly beating P3 persistence in normalized one-step average, while physical-space tails, shell/radial transport, and autoregressive stability remain poor. Therefore the evidence attributes the limited improvement to optimization exposure rather than data volume.
"""
    (root / "STAGE_S_REPORT.md").write_text(report, encoding="utf-8")
    decision_md = f"""# Stage S Decision

## {decision}

Matched update budget: S-full (`{full_norm:.6g}`) did not improve over S-small (`{small_norm:.6g}`) and had worse residual cosine and transport skill. The S-full 30-epoch run improved normalized one-step error to `{natural_norm:.6g}` only with 1,260 updates and beat P3 persistence (`{persistence_norm:.6g}`) by `{(1-natural_norm/persistence_norm)*100:.3f}%`, but it retained catastrophic physical-tail errors, negative shell/radial skill, and step-1 range failure. This supports more optimization rather than more data; it does not support data scarcity as the primary Stage-R failure mechanism.
"""
    (root / "STAGE_S_DECISION.md").write_text(decision_md, encoding="utf-8")
    print(json.dumps({
        "decision": decision,
        "matched_improvement": matched_improvement,
        "natural_vs_full_matched_improvement": natural_vs_full,
        "persistence_normalized_average": persistence_norm,
        "figures": sorted(path.name for path in figures.glob("*.png")),
    }, indent=2, sort_keys=True))


TARGET_CHANNELS = ("Bcc2", "Bcc3", "vel3")


if __name__ == "__main__":
    main()
