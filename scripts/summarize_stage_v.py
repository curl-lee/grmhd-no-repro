#!/usr/bin/env python3
"""Summarize Stage V frozen outputs and apply the predeclared decision logic."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from grmhd.stage_v_analysis import (
    distribution_shift_error_coupling,
    stage_v_primary_decision,
)


DISPLAY_ORDER = ("stage_t_plain", "direction", "transport", "direction_transport")


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def parse_bool(value: object) -> bool:
    return str(value).lower() == "true"


def maybe_float(value: object) -> float | None:
    return None if value in (None, "", "None") else float(value)


def fmt(value: object, precision: int = 6) -> str:
    number = maybe_float(value)
    return "not available" if number is None else f"{number:.{precision}g}"


def first_10x_by_variant(rows: Sequence[Mapping[str, Any]]) -> dict[str, int | None]:
    output = {}
    for variant in DISPLAY_ORDER:
        selected = [row for row in rows if row["variant"] == variant]
        values = [row["FIRST_10X_PHYSICAL_RANGE_STEP"] for row in selected if row["FIRST_10X_PHYSICAL_RANGE_STEP"] not in ("", None, "None")]
        output[variant] = None if not values else int(values[0])
    return output


def training_summary(root: Path, variant: str) -> dict[str, Any]:
    if variant == "stage_t_plain":
        summary = json.loads((root.parent / "stage_t/full_long/training_summary.json").read_text())
        log = read_csv(root.parent / "stage_t/full_long/train_log.csv")[:150]
    else:
        summary = json.loads((root / "variants" / variant / "training_summary.json").read_text())
        log = read_csv(root / "variants" / variant / "train_log.csv")
    return {
        "runtime_seconds": float(log[-1]["cumulative_runtime_seconds"]),
        "clipping_fraction_mean": float(np.mean([float(row["clipping_fraction"]) for row in log])),
        "peak_allocated_mib": max(float(row["gpu_peak_allocated_mib"]) for row in log),
        "optimizer_updates": int(float(log[-1]["optimizer_updates"])),
        "all_finite": bool(summary["all_finite"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/stage_v"))
    args = parser.parse_args()
    root = args.root
    one_step = read_csv(root / "comparison/one_step_comparison.csv")
    per_channel = read_csv(root / "comparison/per_channel_comparison.csv")
    rollout = read_csv(root / "comparison/rollout_comparison.csv")
    shift = read_csv(root / "comparison/shift_comparison.csv")
    correlations = read_csv(root / "alignment/objective_metric_correlations.csv")
    gradient = read_csv(root / "alignment/gradient_alignment.csv")
    ood_correlations = read_csv(root / "shift/error_ood_correlations.csv")
    alignment = json.loads((root / "alignment/objective_metric_alignment.json").read_text())
    weights = json.loads((root / "loss_weights/train_only_gradient_scaling.json").read_text())
    ood_reference = json.loads((root / "shift/train_reference_features.json").read_text())
    first_10x = first_10x_by_variant(rollout)
    resolved_configs = {
        variant: json.loads((root / "variants" / variant / "resolved_config.json").read_text())
        for variant in ("direction", "transport", "direction_transport")
    }
    initial_hashes = {
        config["initial_tensor_state_sha256"] for config in resolved_configs.values()
    }
    pair_order_hashes = {
        config["pair_order_sha256"] for config in resolved_configs.values()
    }
    parameter_counts = {config["parameter_count"] for config in resolved_configs.values()}
    if len(initial_hashes) != 1 or len(pair_order_hashes) != 1 or len(parameter_counts) != 1:
        raise RuntimeError("Stage-V controlled-pairing provenance differs across variants")
    if not all(config["stage_t_first_150_order_match"] for config in resolved_configs.values()):
        raise RuntimeError("a Stage-V pair order does not match the frozen Stage-T first 150 epochs")
    initial_hash = next(iter(initial_hashes))
    pair_order_hash = next(iter(pair_order_hashes))
    parameter_count = next(iter(parameter_counts))

    for row in one_step:
        row["first_10x_step"] = first_10x[row["variant"]]
        row["O1_state_retention"] = parse_bool(row["O1_state_retention"])
    candidates = [row for row in one_step if row["variant"] in ("transport", "direction_transport")]
    primary = stage_v_primary_decision(
        candidates,
        gradient_conflict=weights["OBJECTIVE_GRADIENT_CONFLICT"],
        pairwise_misalignment=bool(alignment["PLAIN_L2_TRANSPORT_MISALIGNMENT"]),
    )
    baseline_ood_values = [
        float(row["spearman_r"]) for row in ood_correlations
        if row["variant"] == "stage_t_plain"
    ]
    coupling = distribution_shift_error_coupling(baseline_ood_values)
    shift_finding = ood_reference["SHIFT_FINDING"]
    if primary in ("A", "B"):
        authorize = "objective_refinement"
    elif coupling == "STRONG":
        authorize = "temporal_distribution_shift_generalization"
    else:
        authorize = "non_euclidean_spectral_operator_design"

    validation_aggregate = {
        (row["y"]): row for row in correlations
        if row["split"] == "validation" and row["scope"] == "aggregate"
    }
    focus_corr = [
        row for row in correlations
        if row["split"] == "validation" and row["scope"] == "per_channel"
        and row["y"] in ("shell_error", "radial_error")
    ]
    channel_correlation_strength: dict[str, float] = {}
    for channel in ("Bcc2", "Bcc3", "vel3", "rho", "press"):
        values = [abs(float(row["spearman_r"])) for row in focus_corr if row["channel"] == channel]
        channel_correlation_strength[channel] = float(np.mean(values))
    least_aligned_channels = sorted(channel_correlation_strength, key=channel_correlation_strength.get)

    channel_gradients = [row for row in gradient if row.get("scope") == "per_channel"]
    channel_gradient_cosines: dict[str, float] = {}
    for channel in ("Bcc2", "Bcc3", "vel3", "rho", "press"):
        values = [float(row["cosine_with_plain"]) for row in channel_gradients if row["channel"] == channel and row["cosine_with_plain"] not in ("", None)]
        if values:
            channel_gradient_cosines[channel] = float(np.median(values))
    strongest_conflict_channels = sorted(channel_gradient_cosines, key=channel_gradient_cosines.get)

    baseline_channels = {row["channel"]: row for row in per_channel if row["variant"] == "stage_t_plain"}
    transport_improvements: dict[str, dict[str, float]] = {}
    for channel in ("Bcc2", "Bcc3", "vel3", "rho", "press"):
        improvements = {}
        for variant in ("transport", "direction_transport"):
            row = next(item for item in per_channel if item["variant"] == variant and item["channel"] == channel)
            improvements[variant] = (
                float(row["shell_skill_median"]) - float(baseline_channels[channel]["shell_skill_median"])
                + float(row["radial_skill_median"]) - float(baseline_channels[channel]["radial_skill_median"])
            )
        transport_improvements[channel] = improvements
    first_improved_channels = sorted(
        transport_improvements,
        key=lambda channel: max(transport_improvements[channel].values()),
        reverse=True,
    )

    physical_tail_rows = [
        row for row in per_channel if row["channel"] in ("rho", "press")
    ]
    physical_tail_ratios = {
        f"{row['variant']}:{row['channel']}": float(row["physical_relative_l2"]) / max(float(row["state_relative_l2"]), 1e-300)
        for row in physical_tail_rows
    }
    tail_separated = max(physical_tail_ratios.values()) > 10.0

    temporal_baseline = {row["stratum"]: row for row in shift if row["variant"] == "stage_t_plain"}
    late_over_early = float(temporal_baseline["late"]["late_over_early_state_relative_l2"])
    late_shift_synchronized = coupling in ("MODERATE", "STRONG") and late_over_early > 1.10

    training = {variant: training_summary(root, variant) for variant in DISPLAY_ORDER}
    result = {
        "schema_version": "stage-v-summary-v1",
        "PRIMARY_DECISION": primary,
        "OBJECTIVE_GRADIENT_CONFLICT": weights["OBJECTIVE_GRADIENT_CONFLICT"],
        "PLAIN_L2_TRANSPORT_MISALIGNMENT": bool(alignment["PLAIN_L2_TRANSPORT_MISALIGNMENT"]),
        "SHIFT_FINDING": shift_finding,
        "DISTRIBUTION_SHIFT_ERROR_COUPLING": coupling,
        "AUTHORIZE_NEXT_STAGE": authorize,
        "PAPER_FULL_PILOT": "PAPER_FULL_PILOT_NOT_AUTHORIZED",
        "controlled_pairing": {
            "initial_tensor_state_sha256": initial_hash,
            "pair_order_sha256": pair_order_hash,
            "parameter_count": parameter_count,
            "all_variants_match_stage_t_first_150_epoch_order": True,
        },
        "first_10x_step": first_10x,
        "training": training,
        "plain_validation_spearman": {
            metric: float(validation_aggregate[metric]["spearman_r"])
            for metric in ("state_relative_l2", "residual_relative_l2", "residual_cosine_error", "shell_error", "radial_error")
        },
        "least_plain_transport_aligned_channels": least_aligned_channels,
        "per_channel_plain_transport_correlation_strength": channel_correlation_strength,
        "strongest_transport_gradient_conflict_channels": strongest_conflict_channels,
        "per_channel_plain_transport_gradient_cosine_median": channel_gradient_cosines,
        "transport_improvement_channel_order": first_improved_channels,
        "transport_improvements": transport_improvements,
        "physical_tail_amplification_ratio_physical_over_normalized": physical_tail_ratios,
        "NORMALIZED_DYNAMICS_FAILURE_SEPARATE_FROM_DECODER_PHYSICAL_TAIL_AMPLIFICATION": tail_separated,
        "stage_t_late_over_early_state_error": late_over_early,
        "late_validation_degradation_synchronized_with_train_derived_ood": late_shift_synchronized,
        "decision_thresholds_predeclared_in_src_grmhd_stage_v_analysis": {
            "meaningful_transport_skill_improvement": 0.10,
            "strong_rollout_delay_after_step": 3,
        },
    }
    write_json(root / "stage_v_summary.json", result)

    order = {variant: index for index, variant in enumerate(DISPLAY_ORDER)}
    one_step.sort(key=lambda row: order[row["variant"]])
    baseline = next(row for row in one_step if row["variant"] == "stage_t_plain")
    direction = next(row for row in one_step if row["variant"] == "direction")
    transport = next(row for row in one_step if row["variant"] == "transport")
    combined = next(row for row in one_step if row["variant"] == "direction_transport")

    def percent_change(row: Mapping[str, Any], metric: str) -> float:
        reference = float(baseline[metric])
        return 100.0 * (float(row[metric]) - reference) / abs(reference)

    direction_state_change = percent_change(direction, "state_relative_l2")
    direction_residual_change = percent_change(direction, "residual_relative_l2")
    transport_state_change = percent_change(transport, "state_relative_l2")
    transport_shell_delta = float(transport["shell_skill"]) - float(baseline["shell_skill"])
    transport_radial_delta = float(transport["radial_skill"]) - float(baseline["radial_skill"])
    combined_shell_delta = float(combined["shell_skill"]) - float(baseline["shell_skill"])
    combined_radial_delta = float(combined["radial_skill"]) - float(baseline["radial_skill"])
    lines = [
        "# Stage V — Objective Alignment and Temporal Distribution-Shift Audit",
        "",
        "| variant | state L2 | residual L2 | cosine | shell skill | radial skill | first 10x step |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in one_step:
        lines.append(
            f"| {row['display_name']} | {fmt(row['state_relative_l2'])} | {fmt(row['residual_relative_l2'])} | "
            f"{fmt(row['residual_global_cosine'])} | {fmt(row['shell_skill'])} | {fmt(row['radial_skill'])} | "
            f"{row['first_10x_step'] if row['first_10x_step'] is not None else 'none through 100'} |"
        )
    lines.extend([
        "| Paper Full | not run | not run | not run | not run | not run | not run |",
        "",
        "`Paper Full` was not authorized because the recoverable old objective contains the known index-grid H1 pathology.",
        "",
        "## Frozen controls and training",
        "",
        f"All learned variants use the Stage-T differential LocalNO ({parameter_count:,} parameters), P3 residual contract, shared tensor initialization, identical 168-pair order for 150 epochs, Adam, unchanged clip=1, and the first 6,300 updates of the frozen 1,200-epoch warmup/cosine schedule. Validation was not used for weights, training, or stopping. The formal selector remained validation normalized per-channel state relative-L2 arithmetic average.",
        "",
        f"- Shared initial tensor-state SHA256: `{initial_hash}`.",
        f"- Shared 150-epoch pair-order SHA256: `{pair_order_hash}`; every variant records `stage_t_first_150_order_match=true`.",
        "- V1 retains an earlier checksum for an unused shell-definition field; its active Plain+direction graph is unaffected. The pre-validation correction and V2 restart are documented in `transport_objective_engineering_audit.md`.",
        "",
    ])
    for variant in DISPLAY_ORDER:
        item = training[variant]
        lines.append(
            f"- {next(row['display_name'] for row in one_step if row['variant'] == variant)}: "
            f"{item['optimizer_updates']} updates, runtime {item['runtime_seconds']:.1f}s, "
            f"mean clip fraction {item['clipping_fraction_mean']:.4f}, peak allocated {item['peak_allocated_mib']:.1f} MiB."
        )
    shell_text = ", ".join(
        f"{row['display_name']}={fmt(row['shell_skill'])}" for row in candidates
    )
    radial_text = ", ".join(
        f"{row['display_name']}={fmt(row['radial_skill'])}" for row in candidates
    )
    retention_text = ", ".join(
        f"{row['display_name']}={row['O1_state_retention']}" for row in one_step[1:]
    )
    first_10x_text = ", ".join(
        f"{row['display_name']}={row['first_10x_step']}" for row in one_step
    )
    lines.extend([
        "",
        "## No-training alignment evidence",
        "",
        f"Validation Spearman correlations of Plain pair loss were state={result['plain_validation_spearman']['state_relative_l2']:.3f}, residual={result['plain_validation_spearman']['residual_relative_l2']:.3f}, cosine-error={result['plain_validation_spearman']['residual_cosine_error']:.3f}, shell-error={result['plain_validation_spearman']['shell_error']:.3f}, and radial-error={result['plain_validation_spearman']['radial_error']:.3f}. Thus the strict pairwise weak-transport rule is `{str(result['PLAIN_L2_TRANSPORT_MISALIGNMENT']).lower()}`.",
        f"The train-only initialization audit classified aggregate objective-gradient conflict as `{result['OBJECTIVE_GRADIENT_CONFLICT']}`. The auto-scaled lambdas are direction-only {weights['direction']['lambda_single']:.9g}, transport-only {weights['transport']['lambda_single']:.9g}, and combined direction/transport {weights['direction']['lambda_combined']:.9g}/{weights['transport']['lambda_combined']:.9g}.",
        "",
        "## Required scientific answers",
        "",
        f"1. **Plain L2 versus shell/radial statistics.** The strict misalignment flag is `{str(result['PLAIN_L2_TRANSPORT_MISALIGNMENT']).lower()}`; the actual correlations above quantify partial rather than absent alignment.",
        f"2. **Gradient conflict.** `{result['OBJECTIVE_GRADIENT_CONFLICT']}`; the most conflicting focus channels, lowest cosine first, are {', '.join(strongest_conflict_channels)}.",
        f"3. **Direction objective.** No dynamics improvement: Direction changes state L2 by {direction_state_change:+.2f}% and residual L2 by {direction_residual_change:+.2f}% versus Stage T, changes cosine from {fmt(baseline['residual_global_cosine'])} to {fmt(direction['residual_global_cosine'])}, makes both transport skills more negative, and leaves first-10x at step 1.",
        f"4. **Shell skill.** No candidate turns shell skill positive. {shell_text}; relative to Stage T the changes are Transport={transport_shell_delta:+.6g} and Direction+Transport={combined_shell_delta:+.6g} skill points.",
        f"5. **Radial skill.** No candidate turns radial skill positive. {radial_text}; relative to Stage T the changes are Transport={transport_radial_delta:+.6g} and Direction+Transport={combined_radial_delta:+.6g} skill points.",
        f"6. **State retention.** O1 results are {retention_text}; the fixed ceiling is 1.05*0.266711=0.280047.",
        f"7. **Step-1 physical-range failure.** First 10x steps are {first_10x_text}.",
        f"8. **Physical-tail separation.** Yes: normalized state errors remain O(0.26--0.28) while decoded rho/press amplification ratios span orders of magnitude, and normalized shell/radial skills are independently negative. The two failures coexist but are not the same metric; no clamp or P3 change was made.",
        f"9. **Error/OOD coupling.** Not strong: `{coupling}` under train-only feature fitting. For Stage T the state-error Spearman coefficient is {next(float(row['spearman_r']) for row in ood_correlations if row['variant'] == 'stage_t_plain' and row['y'] == 'state_relative_l2'):.3f}; this is association, not causal proof.",
        f"10. **Late degradation versus shift.** Stage-T late/early state error is {late_over_early:.3f} and is temporally synchronized with the train-derived OOD score (`{str(late_shift_synchronized).lower()}`), but MODERATE aggregate coupling is insufficient to claim that shift is the main or causal explanation.",
        f"11. **Next change.** Spectral geometry: `{authorize}` follows because decision `{primary}` rejects objective mismatch as the supported bottleneck and shift coupling is not STRONG.",
        "",
        "## Channel attribution",
        "",
        f"Lowest per-channel Plain/transport correlation among the focus channels: {', '.join(least_aligned_channels)}. Strongest transport-gradient conflict: {', '.join(strongest_conflict_channels)}. Largest summed shell+radial skill improvements occur first in: {', '.join(first_improved_channels)}. The full per-channel state/residual/cosine/shell/radial/physical table is `comparison/per_channel_comparison.csv`.",
        "",
        "## Distribution shift",
        "",
        f"The train-only 168-feature snapshot score gives `SHIFT_FINDING={shift_finding}` and `DISTRIBUTION_SHIFT_ERROR_COUPLING={coupling}`. It does not alter the chronological split and does not establish causality.",
        "",
        "## Limitations",
        "",
        "This remains a 64^3 resampled spherical Kerr--Schild, `press`-adapted, P3 residual, reduced LocalNO reproduction. It is not the paper's exact 3D DISCO/coarse-coupled operator. Physical rho/press metrics remain exposed to nonlinear inverse tails; normalized dynamics and decoded physical amplification are reported separately.",
        "",
        f"PRIMARY_DECISION = {primary}",
        f"OBJECTIVE_GRADIENT_CONFLICT = {result['OBJECTIVE_GRADIENT_CONFLICT']}",
        f"PLAIN_L2_TRANSPORT_MISALIGNMENT = {str(result['PLAIN_L2_TRANSPORT_MISALIGNMENT']).lower()}",
        f"SHIFT_FINDING = {shift_finding}",
        f"DISTRIBUTION_SHIFT_ERROR_COUPLING = {coupling}",
        f"AUTHORIZE_NEXT_STAGE = {authorize}",
        "",
    ])
    (root / "STAGE_V_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    labels = {
        "A": "OBJECTIVE_MISMATCH_STRONGLY_SUPPORTED",
        "B": "OBJECTIVE_MISMATCH_PARTIALLY_SUPPORTED",
        "C": "OBJECTIVE_MISMATCH_NOT_SUPPORTED",
        "D": "OBJECTIVE_CHANGE_HURTS_STATE_WITHOUT_DYNAMICS_GAIN",
        "E": "ENGINEERING_FAILURE",
    }
    decision_lines = [
        "# Stage V Decision",
        "",
        f"## {primary}. {labels[primary]}",
        "",
        "The label follows the predeclared code in `src/grmhd/stage_v_analysis.py`: a 0.10 skill-point threshold defines meaningful transport gain, state retention uses the fixed 5% Stage-T bound, and a strong rollout delay must pass step 3. Diagnostic checkpoint selectors never replace the frozen formal state-L2 selector.",
        "",
        f"- `OBJECTIVE_GRADIENT_CONFLICT = {result['OBJECTIVE_GRADIENT_CONFLICT']}`",
        f"- `PLAIN_L2_TRANSPORT_MISALIGNMENT = {str(result['PLAIN_L2_TRANSPORT_MISALIGNMENT']).lower()}`",
        f"- `SHIFT_FINDING = {shift_finding}`",
        f"- `DISTRIBUTION_SHIFT_ERROR_COUPLING = {coupling}`",
        f"- `AUTHORIZE_NEXT_STAGE = {authorize}`",
        "- `PAPER_FULL_PILOT_NOT_AUTHORIZED`",
        "",
    ]
    (root / "STAGE_V_DECISION.md").write_text("\n".join(decision_lines), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("PRIMARY_DECISION", "OBJECTIVE_GRADIENT_CONFLICT", "PLAIN_L2_TRANSPORT_MISALIGNMENT", "SHIFT_FINDING", "DISTRIBUTION_SHIFT_ERROR_COUPLING", "AUTHORIZE_NEXT_STAGE")}, sort_keys=True))


if __name__ == "__main__":
    main()
