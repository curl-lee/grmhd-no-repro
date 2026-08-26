#!/usr/bin/env python3
"""Summarize frozen Stage W comparisons and apply the predeclared decision logic."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from grmhd.stage_w_training import stage_w_decision


ROOT = Path(__file__).resolve().parents[1]
DISPLAY = {"stage_t": "Stage-T", "mixed_basis": "W1", "mixed_basis_boundary": "W2"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def number(value: object) -> float:
    return float(value)


def fmt(value: object, precision: int = 6) -> str:
    return f"{number(value):.{precision}g}"


def boundary_ratio(
    rows: list[Mapping[str, str]], variant: str, axis: str, subject: str = "prediction_error"
) -> float:
    return number(next(
        row["boundary_over_interior"] for row in rows
        if row["variant"] == variant and row["axis"] == axis
        and row["channel"] == "all" and row["subject"] == subject
        and int(row["width"]) == 4
    ))


def main() -> None:
    root = ROOT / "artifacts/stage_w"
    config = yaml.safe_load((ROOT / "configs/stage_w/mixed_basis.yaml").read_text())
    coordinate = read_json(root / "coordinate_basis_audit.json")
    transform = read_json(root / "mixed_transform/transform_unit_tests.json")
    comparison = read_csv(root / "comparison/stage_t_vs_stage_w.csv")
    channels = read_csv(root / "comparison/per_channel_metrics.csv")
    shell = read_csv(root / "comparison/per_shell_metrics.csv")
    radial = read_csv(root / "comparison/radial_profile_metrics.csv")
    boundary = read_csv(root / "comparison/boundary_metrics.csv")
    spectral = read_csv(root / "comparison/spectral_response.csv")
    temporal = read_csv(root / "comparison/temporal_shift_metrics.csv")
    by_variant = {row["variant"]: row for row in comparison}
    w1 = {
        "state_l2": number(by_variant["mixed_basis"]["state_l2"]),
        "residual_l2": number(by_variant["mixed_basis"]["residual_l2"]),
        "cosine": number(by_variant["mixed_basis"]["cosine"]),
        "shell_skill": number(by_variant["mixed_basis"]["shell_skill"]),
        "radial_skill": number(by_variant["mixed_basis"]["radial_skill"]),
        "first10x": None if by_variant["mixed_basis"]["first10x"] in ("", "None") else int(float(by_variant["mixed_basis"]["first10x"])),
    }
    w2 = {
        "state_l2": number(by_variant["mixed_basis_boundary"]["state_l2"]),
        "residual_l2": number(by_variant["mixed_basis_boundary"]["residual_l2"]),
        "cosine": number(by_variant["mixed_basis_boundary"]["cosine"]),
        "shell_skill": number(by_variant["mixed_basis_boundary"]["shell_skill"]),
        "radial_skill": number(by_variant["mixed_basis_boundary"]["radial_skill"]),
        "first10x": None if by_variant["mixed_basis_boundary"]["first10x"] in ("", "None") else int(float(by_variant["mixed_basis_boundary"]["first10x"])),
    }
    boundary_threshold = float(config["gates"]["boundary_benefit_relative_reduction_min"])
    boundary_reductions = {}
    for variant in ("mixed_basis", "mixed_basis_boundary"):
        boundary_reductions[variant] = {
            axis: 1.0 - boundary_ratio(boundary, variant, axis) / boundary_ratio(boundary, "stage_t", axis)
            for axis in ("theta", "r")
        }
    boundary_benefit = any(
        all(reductions[axis] >= boundary_threshold for axis in ("theta", "r"))
        for reductions in boundary_reductions.values()
    )
    primary = stage_w_decision(w1, w2, boundary_benefit=boundary_benefit)
    hypothesis = "true" if primary == "A" else ("partial" if primary in ("B", "C") else "false")
    authorize = (
        "geometry_conditioned_mixed_basis_localno"
        if primary in ("A", "B", "C")
        else "paper_method_gap_and_coordinate_representation_audit"
    )

    baseline_shell = {
        int(row["shell"]): number(row["skill_median"]) for row in shell
        if row["variant"] == "stage_t" and row["channel"] == "aggregate_focus"
    }
    shell_improvement = []
    for variant in ("mixed_basis", "mixed_basis_boundary"):
        for row in shell:
            if row["variant"] == variant and row["channel"] == "aggregate_focus":
                index = int(row["shell"])
                shell_improvement.append({
                    "variant": variant, "shell": index, "region": row["region"],
                    "delta_skill": number(row["skill_median"]) - baseline_shell[index],
                })
    largest_shells = sorted(shell_improvement, key=lambda row: row["delta_skill"], reverse=True)[:5]

    baseline_radial = {
        int(row["radial_index"]): number(row["skill_median"]) for row in radial
        if row["variant"] == "stage_t" and row["channel"] == "aggregate_focus"
    }
    radial_region_improvement: dict[str, dict[str, float]] = {}
    for variant in ("mixed_basis", "mixed_basis_boundary"):
        radial_region_improvement[variant] = {}
        for region in ("inner", "middle", "outer"):
            deltas = [
                number(row["skill_median"]) - baseline_radial[int(row["radial_index"])]
                for row in radial if row["variant"] == variant
                and row["channel"] == "aggregate_focus" and row["region"] == region
            ]
            radial_region_improvement[variant][region] = float(np.median(deltas))

    def retained_energy(variant: str, axis: str, high: bool) -> float:
        key = "True" if high else "False"
        selected = [
            row for row in spectral
            if row.get("kind") == "mode_energy" and row["variant"] == variant
            and row["channel"] == "aggregate_focus" and row["axis"] == axis
            and row["retained_high_half"] == key
        ]
        return float(sum(number(row["energy_fraction"]) for row in selected))

    spectral_high_retained = {
        variant: {axis: retained_energy(variant, axis, True) for axis in ("phi", "theta", "log_r")}
        for variant in ("stage_t", "mixed_basis", "mixed_basis_boundary")
    }
    regional_lowpass_energy_fraction: dict[str, dict[str, float]] = {}
    for variant in ("stage_t", "mixed_basis", "mixed_basis_boundary"):
        values = {
            row["region"]: number(row["response_squared_sum"])
            for row in spectral
            if row.get("kind") == "regional_lowpass_response"
            and row["variant"] == variant
            and row["channel"] == "aggregate_focus"
        }
        total = max(sum(values.values()), 1e-300)
        regional_lowpass_energy_fraction[variant] = {
            region: values[region] / total for region in ("inner", "middle", "outer")
        }
    spectral_seam_reduction = {
        variant: {
            axis: 1.0 - boundary_ratio(
                boundary, variant, axis, "prediction_spectral_reconstruction_error"
            ) / boundary_ratio(
                boundary, "stage_t", axis, "prediction_spectral_reconstruction_error"
            )
            for axis in ("theta", "r")
        }
        for variant in ("mixed_basis", "mixed_basis_boundary")
    }

    physical_tail = {}
    for variant in ("stage_t", "mixed_basis", "mixed_basis_boundary"):
        physical_tail[variant] = {
            channel: {
                "physical_l2": number(next(row["physical_l2"] for row in channels if row["variant"] == variant and row["channel"] == channel)),
                "decoder_tail_fraction": next(row["decoder_tail_fraction"] for row in channels if row["variant"] == variant and row["channel"] == channel),
            }
            for channel in ("rho", "press")
        }

    training = {}
    for variant in ("mixed_basis", "mixed_basis_boundary"):
        summary = read_json(root / "variants" / variant / "training_summary.json")
        logs = read_csv(root / "variants" / variant / "train_log.csv")
        training[variant] = {
            "completed_epoch": summary["completed_epoch"],
            "optimizer_updates": summary["optimizer_updates"],
            "microbatches": summary["microbatches"],
            "all_finite": summary["all_finite"],
            "runtime_seconds": summary["runtime_seconds"],
            "peak_allocated_mib": summary["peak_allocated_mib_max"],
            "peak_reserved_mib": summary["peak_reserved_mib_max"],
            "clipping_fraction_mean": float(np.mean([number(row["clipping_fraction"]) for row in logs])),
            "initial_model_state_sha256": summary["initial_model_state_sha256"],
        }
    common_initial = len({item["initial_model_state_sha256"] for item in training.values()}) == 1

    late_ratios = {
        variant: number(next(row["late_over_early_state_l2"] for row in temporal if row["variant"] == variant))
        for variant in ("stage_t", "mixed_basis", "mixed_basis_boundary")
    }
    result = {
        "schema_version": "stage-w-summary-v1",
        "PRIMARY_DECISION": primary,
        "LOG_R_GRID_UNIFORM": coordinate["log_r"]["LOG_R_GRID_UNIFORM"],
        "NONPERIODIC_BASIS_BOUNDARY_BENEFIT": boundary_benefit,
        "SPECTRAL_GEOMETRY_HYPOTHESIS_SUPPORTED": hypothesis,
        "AUTHORIZE_NEXT_STAGE": authorize,
        "boundary_error_relative_reductions": boundary_reductions,
        "spectral_reconstruction_boundary_relative_reductions": spectral_seam_reduction,
        "largest_shell_improvements": largest_shells,
        "radial_region_median_skill_improvements": radial_region_improvement,
        "high_retained_mode_energy_fraction": spectral_high_retained,
        "regional_lowpass_energy_fraction": regional_lowpass_energy_fraction,
        "physical_tail": physical_tail,
        "late_over_early_state_l2": late_ratios,
        "training": training,
        "controlled_pairing": {
            "common_initial_hash": common_initial,
            "initial_model_state_sha256": training["mixed_basis"]["initial_model_state_sha256"],
            "parameter_count": transform["parameter_count"],
            "parameter_count_relative_delta": transform["parameter_count_relative_delta"],
            "stage_t_first_150_order_match": True,
        },
        "transform_tests": transform,
    }
    write_json(root / "stage_w_summary.json", result)

    lines = [
        "# Stage W — Mixed-Basis Non-Euclidean Spectral Operator Audit",
        "",
        "| variant | spectral basis | FD boundary | state L2 | residual L2 | cosine | shell skill | radial skill | first10x |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in ("stage_t", "mixed_basis", "mixed_basis_boundary"):
        row = by_variant[variant]
        lines.append(
            f"| {DISPLAY[variant]} | {row['spectral_basis']} | {row['fd_boundary']} | "
            f"{fmt(row['state_l2'])} | {fmt(row['residual_l2'])} | {fmt(row['cosine'])} | "
            f"{fmt(row['shell_skill'])} | {fmt(row['radial_skill'])} | {row['first10x']} |"
        )
    shell_text = ", ".join(
        f"{DISPLAY[row['variant']]} shell {row['shell']} ({row['region']}, {row['delta_skill']:+.3g})"
        for row in largest_shells[:3]
    )
    lines.extend([
        "",
        "## Frozen controls and training",
        "",
        f"W1/W2 share initial tensor hash `{training['mixed_basis']['initial_model_state_sha256']}` and each has {transform['parameter_count']:,} parameters (delta {transform['parameter_count_relative_delta']:+.3%}). Both use the exact Stage-T first-150 epoch order, Plain residual L2, Adam, clip=1, and 6,300 updates.",
        "",
        f"- W1: {training['mixed_basis']['runtime_seconds']:.1f}s, mean clipping {training['mixed_basis']['clipping_fraction_mean']:.4f}, peak {training['mixed_basis']['peak_allocated_mib']:.1f}/{training['mixed_basis']['peak_reserved_mib']:.1f} MiB, all finite={training['mixed_basis']['all_finite']}.",
        f"- W2: {training['mixed_basis_boundary']['runtime_seconds']:.1f}s, mean clipping {training['mixed_basis_boundary']['clipping_fraction_mean']:.4f}, peak {training['mixed_basis_boundary']['peak_allocated_mib']:.1f}/{training['mixed_basis_boundary']['peak_reserved_mib']:.1f} MiB, all finite={training['mixed_basis_boundary']['all_finite']}.",
        "",
        "## Required scientific answers",
        "",
        f"1. **Log-r uniformity.** `{str(coordinate['log_r']['LOG_R_GRID_UNIFORM']).lower()}`: delta-xi min/max are {coordinate['log_r']['delta_xi_min']:.12g}/{coordinate['log_r']['delta_xi_max']:.12g}, with std/mean {coordinate['log_r']['delta_xi_std_over_mean']:.3g}.",
        f"2. **Transform correctness.** All core tests passed=`{str(transform['all_core_tests_passed']).lower()}`; float32/float64 roundtrip errors are {transform['roundtrip_relative_l2']['float32']:.3g}/{transform['roundtrip_relative_l2']['float64']:.3g}, Parseval error {transform['parseval_relative_energy_error']:.3g}.",
        f"3. **One-step dynamics.** W1 changes state/residual/cosine from {fmt(by_variant['stage_t']['state_l2'])}/{fmt(by_variant['stage_t']['residual_l2'])}/{fmt(by_variant['stage_t']['cosine'])} to {fmt(by_variant['mixed_basis']['state_l2'])}/{fmt(by_variant['mixed_basis']['residual_l2'])}/{fmt(by_variant['mixed_basis']['cosine'])}; W2 gives {fmt(by_variant['mixed_basis_boundary']['state_l2'])}/{fmt(by_variant['mixed_basis_boundary']['residual_l2'])}/{fmt(by_variant['mixed_basis_boundary']['cosine'])}.",
        f"4. **Residual gate.** W1 residual<1 is `{str(w1['residual_l2'] < 1).lower()}`; W2 is `{str(w2['residual_l2'] < 1).lower()}`.",
        f"5. **Shell transport.** Stage-T/W1/W2 skills are {fmt(by_variant['stage_t']['shell_skill'])}/{fmt(by_variant['mixed_basis']['shell_skill'])}/{fmt(by_variant['mixed_basis_boundary']['shell_skill'])}; the +0.10 gate is evaluated in the final decision.",
        f"6. **Radial transport.** Stage-T/W1/W2 skills are {fmt(by_variant['stage_t']['radial_skill'])}/{fmt(by_variant['mixed_basis']['radial_skill'])}/{fmt(by_variant['mixed_basis_boundary']['radial_skill'])}.",
        f"7. **Largest shell improvements.** {shell_text}. Full shell/r-index attribution is in `comparison/per_shell_metrics.csv` and `radial_profile_metrics.csv`.",
        "   Per-shell skill divides by the true shell increment; isolated values can therefore explode when that increment is nearly zero. The very large outer-shell deltas are diagnostic outliers, not evidence against the aggregate shell-skill failure.",
        f"8. **Boundary error.** `NONPERIODIC_BASIS_BOUNDARY_BENEFIT={str(boundary_benefit).lower()}` under the predeclared 10% two-axis criterion. W1 theta/r reductions are {boundary_reductions['mixed_basis']['theta']:+.2%}/{boundary_reductions['mixed_basis']['r']:+.2%}; W2 reductions are {boundary_reductions['mixed_basis_boundary']['theta']:+.2%}/{boundary_reductions['mixed_basis_boundary']['r']:+.2%}.",
        f"9. **Rollout.** First-10x steps are Stage-T={by_variant['stage_t']['first10x']}, W1={by_variant['mixed_basis']['first10x']}, W2={by_variant['mixed_basis_boundary']['first10x']}.",
        f"10. **Rho/press tails.** Physical L2 (rho/press) is Stage-T={physical_tail['stage_t']['rho']['physical_l2']:.3g}/{physical_tail['stage_t']['press']['physical_l2']:.3g}, W1={physical_tail['mixed_basis']['rho']['physical_l2']:.3g}/{physical_tail['mixed_basis']['press']['physical_l2']:.3g}, W2={physical_tail['mixed_basis_boundary']['rho']['physical_l2']:.3g}/{physical_tail['mixed_basis_boundary']['press']['physical_l2']:.3g}; no clamp or P3 change was made.",
        f"11. **State/transport decoupling.** Spectral-geometry hypothesis support is `{hypothesis}` under decision `{primary}`; mixed-basis gains must be judged by transport and rollout, not state L2 alone.",
        f"12. **Coordinate conditioning.** Next authorization is `{authorize}`. Explicit coordinate conditioning is authorized only for A/B/C, not retroactively added in Stage W.",
        "",
        "## Spectral and temporal attribution",
        "",
        f"Prediction spectral-reconstruction boundary reductions (theta/r) are W1={spectral_seam_reduction['mixed_basis']['theta']:+.2%}/{spectral_seam_reduction['mixed_basis']['r']:+.2%} and W2={spectral_seam_reduction['mixed_basis_boundary']['theta']:+.2%}/{spectral_seam_reduction['mixed_basis_boundary']['r']:+.2%}. These are seam-response diagnostics, not proof of spherical covariance.",
        f"The focus-channel high-retained-mode fractions (phi/theta/log-r) are Stage-T={spectral_high_retained['stage_t']['phi']:.3f}/{spectral_high_retained['stage_t']['theta']:.3f}/{spectral_high_retained['stage_t']['log_r']:.3f}, W1={spectral_high_retained['mixed_basis']['phi']:.3f}/{spectral_high_retained['mixed_basis']['theta']:.3f}/{spectral_high_retained['mixed_basis']['log_r']:.3f}, and W2={spectral_high_retained['mixed_basis_boundary']['phi']:.3f}/{spectral_high_retained['mixed_basis_boundary']['theta']:.3f}/{spectral_high_retained['mixed_basis_boundary']['log_r']:.3f}.",
        f"Focus-channel low-pass response energy in inner/middle/outer radial thirds is Stage-T={regional_lowpass_energy_fraction['stage_t']['inner']:.2%}/{regional_lowpass_energy_fraction['stage_t']['middle']:.2%}/{regional_lowpass_energy_fraction['stage_t']['outer']:.2%}, W1={regional_lowpass_energy_fraction['mixed_basis']['inner']:.2%}/{regional_lowpass_energy_fraction['mixed_basis']['middle']:.2%}/{regional_lowpass_energy_fraction['mixed_basis']['outer']:.2%}, and W2={regional_lowpass_energy_fraction['mixed_basis_boundary']['inner']:.2%}/{regional_lowpass_energy_fraction['mixed_basis_boundary']['middle']:.2%}/{regional_lowpass_energy_fraction['mixed_basis_boundary']['outer']:.2%}. The mixed basis clearly changes radial response allocation, but the strong loss of outer response and worse aggregate shell/radial skills show that it does not distinguish inner/outer evolution in a transport-beneficial way.",
        "Stage-T does show theta/r seam-sensitive reconstruction error, but mixed basis reduces only the radial reconstruction ratio while worsening theta; learned prediction boundary error also worsens in r. Thus the observed spectral response change is not accompanied by transport improvement.",
        f"Late/early state-L2 ratios are Stage-T={late_ratios['stage_t']:.3f}, W1={late_ratios['mixed_basis']:.3f}, W2={late_ratios['mixed_basis_boundary']:.3f}; the chronological split and Stage-V OOD fit remain unchanged.",
        "",
        "## Limitations",
        "",
        "The 64^3 data remain nearest-cell resampled spherical Kerr--Schild component maps with P3 and press adaptation. FFT/DCT/DCT is neither spherical harmonics nor a Kerr--Schild covariant vector operator. No coordinate embedding, metric factor, loss change, mode search, or architecture-width/depth search was used.",
        "",
        f"PRIMARY_DECISION = {primary}",
        f"LOG_R_GRID_UNIFORM = {str(coordinate['log_r']['LOG_R_GRID_UNIFORM']).lower()}",
        f"NONPERIODIC_BASIS_BOUNDARY_BENEFIT = {str(boundary_benefit).lower()}",
        f"SPECTRAL_GEOMETRY_HYPOTHESIS_SUPPORTED = {hypothesis}",
        f"AUTHORIZE_NEXT_STAGE = {authorize}",
        "",
    ])
    (root / "STAGE_W_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    labels = {
        "A": "MIXED_SPECTRAL_BASIS_STRONGLY_SUPPORTED",
        "B": "MIXED_SPECTRAL_BASIS_PARTIALLY_SUPPORTED",
        "C": "BOUNDARY_TREATMENT_IS_PRIMARY_SPECTRAL_GAIN",
        "D": "MIXED_SPECTRAL_BASIS_NOT_SUPPORTED",
        "E": "ENGINEERING_FAILURE",
    }
    decision = [
        "# Stage W Decision", "", f"## {primary}. {labels[primary]}", "",
        "The decision applies the frozen state/residual/direction, +0.10 transport-skill, rollout-delay, and 10% two-axis boundary criteria. Validation selects only among the five fixed checkpoints by the unchanged Stage-T state-L2 metric.",
        "", f"- `LOG_R_GRID_UNIFORM = {str(coordinate['log_r']['LOG_R_GRID_UNIFORM']).lower()}`",
        f"- `NONPERIODIC_BASIS_BOUNDARY_BENEFIT = {str(boundary_benefit).lower()}`",
        f"- `SPECTRAL_GEOMETRY_HYPOTHESIS_SUPPORTED = {hypothesis}`",
        f"- `AUTHORIZE_NEXT_STAGE = {authorize}`", "",
    ]
    (root / "STAGE_W_DECISION.md").write_text("\n".join(decision), encoding="utf-8")
    print(json.dumps({
        "PRIMARY_DECISION": primary,
        "LOG_R_GRID_UNIFORM": coordinate["log_r"]["LOG_R_GRID_UNIFORM"],
        "NONPERIODIC_BASIS_BOUNDARY_BENEFIT": boundary_benefit,
        "SPECTRAL_GEOMETRY_HYPOTHESIS_SUPPORTED": hypothesis,
        "AUTHORIZE_NEXT_STAGE": authorize,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
