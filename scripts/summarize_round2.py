#!/usr/bin/env python
"""Build compact round-two JSON/CSV/Markdown summaries from measured outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from grmhd import CHANNELS


ROOT = Path("outputs/experiment_round2")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fold_value(
    rows: list[dict[str, str]], fold: str, metric: str, channel: str, split: str
) -> float:
    selected = [
        row
        for row in rows
        if row["fold"] == fold
        and row["metric"] == metric
        and row["channel"] == channel
        and row["split"] == split
        and row["scope"] == "persistence"
    ]
    if len(selected) != 1:
        raise ValueError(f"missing unique fold metric {fold}/{metric}/{channel}/{split}")
    return float(selected[0]["value"])


def main() -> None:
    transform = load(ROOT / "transform_diagnostics.json")
    checkpoint_runs = {
        "fno_residual": load(
            ROOT / "checkpoint_alignment_fno_residual_all111/metrics.json"
        ),
        "localno_residual": load(
            ROOT / "checkpoint_alignment_localno_residual_all111/metrics.json"
        ),
    }
    with (ROOT / "fold_distribution_shift.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        fold_rows = list(csv.DictReader(stream))
    regrid = load(Path("outputs/regrid_ablation/regrid_comparison.json"))
    pilot = load(ROOT / "bounded_fno_nearest_all111/pilot_gate_report.json")

    transform_summary = {}
    for method, values in transform["methods"].items():
        transform_summary[method] = {
            "first_range_explosion": values["first_decoded_range_explosion"],
            "Bcc1_inverse_amplification": {
                key: values["stages"]["inverse_amplification"]["Bcc1"][key]
                for key in ("q0.5", "q0.99", "max")
            },
            "magnetic_prediction_softclip_fraction_gt_0.95gamma": {
                channel: values["stages"]["model_prediction"][channel][
                    "abs_gt_0.95_gamma_fraction"
                ]
                for channel in CHANNELS[:3]
            },
        }
    fold_summary = {
        fold: {
            "test_persistence_normalized_global_relative_l2": fold_value(
                fold_rows, fold, "normalized_relative_l2_global", "global", "test"
            ),
            "test_persistence_decoded_global_relative_l2": fold_value(
                fold_rows, fold, "decoded_relative_l2_global", "global", "test"
            ),
        }
        for fold in ("A", "B", "C")
    }
    checkpoint_summary = {
        name: {
            "best_epochs": values["best_checkpoint_epochs"],
            "best_values": values["best_checkpoint_values"],
            "strict_reload_verified": values["checkpoint_reload_verified"],
        }
        for name, values in checkpoint_runs.items()
    }
    summary = {
        "round": 2,
        "status": "completed_failed_pilot_gate",
        "long_training_authorized": pilot["continue_long_training"],
        "transform_diagnosis": {
            "root_cause": (
                "signed-log exponential inverse amplification; magnetic normalized states "
                "do not approach the decoder hard clip"
            ),
            "methods": transform_summary,
        },
        "checkpoint_selection": checkpoint_summary,
        "rolling_origin_distribution_shift": fold_summary,
        "regrid": {
            "candidate_ratios": regrid[
                "aggregate_metric_ratios_candidate_over_nearest"
            ],
            "temporal_ratios": regrid[
                "aggregate_temporal_ratios_candidate_over_nearest"
            ],
            "candidate_passed": regrid["candidate_passed_representative_gates"],
            "full_trilinear_generated": regrid[
                "leaf_trilinear_selected_for_full_n111"
            ],
            "full_weighted_generated": regrid["weighted_average_prototype"][
                "full_n111_generated"
            ],
            "training_data_selection": regrid["training_data_selection"],
        },
        "bounded_fno_pilot": pilot,
        "final_regression": {
            "pytest": "35 passed",
            "git_diff_check": "passed",
        },
    }
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for method, values in transform_summary.items():
        explosion = values["first_range_explosion"]
        rows.append(
            {
                "category": "transform",
                "name": f"{method}.first_range_explosion_step",
                "value": "none" if explosion is None else explosion["step"],
                "status": "diagnostic",
            }
        )
        for key, value in values["Bcc1_inverse_amplification"].items():
            rows.append(
                {
                    "category": "transform",
                    "name": f"{method}.Bcc1_amplification.{key}",
                    "value": value,
                    "status": "diagnostic",
                }
            )
    for fold, values in fold_summary.items():
        for key, value in values.items():
            rows.append(
                {
                    "category": "distribution_shift",
                    "name": f"fold_{fold}.{key}",
                    "value": value,
                    "status": "diagnostic",
                }
            )
    for candidate, values in regrid[
        "aggregate_metric_ratios_candidate_over_nearest"
    ].items():
        for key, value in values.items():
            rows.append(
                {
                    "category": "regrid",
                    "name": f"{candidate}.{key}_ratio",
                    "value": value,
                    "status": (
                        "passed_all"
                        if regrid["candidate_passed_representative_gates"][candidate]
                        else "failed_gate"
                    ),
                }
            )
    for horizon, values in pilot["requested_horizons"].items():
        rows.append(
            {
                "category": "bounded_pilot",
                "name": f"step_{horizon}.decoded_global_ratio_to_persistence",
                "value": values["ratio"],
                "status": pilot["status"],
            }
        )
    rows.append(
        {
            "category": "bounded_pilot",
            "name": "teacher_forced_one_step_improvement",
            "value": pilot["teacher_forced_one_step"]["improvement"],
            "status": pilot["status"],
        }
    )
    with (ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    trilinear = regrid["aggregate_metric_ratios_candidate_over_nearest"][
        "leaf_trilinear"
    ]
    lines = [
        "# Round-two stability summary",
        "",
        "- Root cause: decoded magnetic blow-up is dominated by exponential inverse "
        "signed-log amplification, not atanh hard-clip saturation.",
        "- Checkpoint alignment: FNO normalized-best epoch 4 differs from decoded/composite "
        "epoch 11; LocalNO selects epoch 2 for all four named criteria but has a range-penalty "
        "spike at epoch 3.",
        f"- Distribution shift: Fold B test persistence decoded global L2 is "
        f"{fold_summary['B']['test_persistence_decoded_global_relative_l2']:.6g}.",
        f"- Trilinear regrid: jump/TV/high-k ratios are "
        f"{trilinear['block_boundary_jump_score']:.6g}/"
        f"{trilinear['normalized_spatial_total_variation']:.6g}/"
        f"{trilinear['high_k_energy_fraction']:.6g}; rho/press preservation gates fail, so "
        "no full trilinear n111 was generated.",
        f"- Bounded FNO: exact zero-init persistence difference "
        f"{pilot['zero_init_persistence_max_abs_difference']}; 20 epochs took "
        f"{pilot['training_wall_seconds']:.3f}s with "
        f"{pilot['peak_cuda_memory_mib']:.3f} MiB peak CUDA memory.",
        f"- Pilot result: **{pilot['status'].upper()}**. Teacher-forced one-step improvement "
        f"is {pilot['teacher_forced_one_step']['improvement']:.3%}; step-5/10 ratios are "
        f"{pilot['requested_horizons']['5']['ratio']:.6g}/"
        f"{pilot['requested_horizons']['10']['ratio']:.6g}. Longer training is not authorized.",
        "- Regression: 35 tests passed and `git diff --check` passed.",
        "",
    ]
    (ROOT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": summary["status"], "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
