#!/usr/bin/env python
"""Complete the frozen seven-way Stage I extension comparison and decision."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch

from compare_paper_stage_g import (
    GT_STEPS,
    NO_GT_STEPS,
    load_selected,
    model_only_saturation,
    state_category_scores,
    table_row,
)
from compare_paper_stage_i_unit_index import (
    gradient_summary,
    numeric_column,
    render_morphology,
    selected,
    summary,
    write_csv,
)
from grmhd import CHANNELS
from grmhd.dataset import sha256_file
from grmhd.paper_config import load_paper_experiment_config
from grmhd.paper_data_processor import PaperDataProcessor


MODEL_NAMES = ("full", "plain", "no_h1", "unit_index", "stored_coordinate")
LABELS = {
    "canonical_oracle": "canonical oracle",
    "persistence": "persistence",
    "full": "Stage G paper-adapted Full",
    "plain": "Stage G Plain L2",
    "no_h1": "Stage I Run A no-H1",
    "unit_index": "Stage I Run B unit-index H1",
    "stored_coordinate": "Stage I Run C stored-coordinate/volume H1",
}
RUN_FILES = (
    "best_validation_l2/manifest.pt",
    "best_validation_l2/optimizer.pt",
    "best_validation_l2/paper_grmhd_metadata.json",
    "best_validation_l2/paper_metadata.pkl",
    "best_validation_l2/paper_state_dict.pt",
    "best_validation_l2/scheduler.pt",
    "last/manifest.pt",
    "last/optimizer.pt",
    "last/paper_grmhd_metadata.json",
    "last/paper_metadata.pkl",
    "last/paper_state_dict.pt",
    "last/scheduler.pt",
    "metrics.json",
    "evaluation_summary.json",
    "gt_rollout.json",
    "no_gt_rollout.json",
)


def flatten_metric(model: str, metric: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        **{channel: metric["per_channel"][channel] for channel in CHANNELS},
        "arithmetic_average": metric["arithmetic_average"],
        "global_relative_l2": metric["global_relative_l2"],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    stage_g = root / "outputs/paper_reduced100/stage_g"
    stage_i = root / "outputs/paper_reduced100/stage_i"
    dirs = {
        "full": stage_g / "pilot30_full_fno",
        "plain": stage_g / "pilot30_plain_l2",
        "no_h1": stage_i / "no_h1",
        "unit_index": stage_i / "unit_index_h1",
        "stored_coordinate": stage_i / "stored_coordinate_volume_h1",
    }
    metrics = {
        name: json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        for name, path in dirs.items()
    }
    evaluations = {
        name: json.loads(
            (path / "evaluation_summary.json").read_text(encoding="utf-8")
        )
        for name, path in dirs.items()
    }
    gt = {
        name: json.loads((path / "gt_rollout.json").read_text(encoding="utf-8"))
        for name, path in dirs.items()
    }
    no_gt = {
        name: json.loads((path / "no_gt_rollout.json").read_text(encoding="utf-8"))
        for name, path in dirs.items()
    }
    states = {
        name: load_selected(path / "selected_states.pt")
        for name, path in dirs.items()
    }

    config = load_paper_experiment_config(
        root / "configs/paper_reduced100/full_fno_proxy.yaml",
        project_root=root,
    )
    processor = PaperDataProcessor.from_config(config)
    raw_targets: dict[int, torch.Tensor] = {}
    oracle_targets: dict[int, torch.Tensor] = {}
    with h5py.File(config.resolve_path(config.values["protocol"]["dataset"]), "r") as handle:
        phi = np.asarray(handle["coords/phi"][...], dtype=np.float64)
        theta = np.asarray(handle["coords/theta"][...], dtype=np.float64)
        radius = np.asarray(handle["coords/r"][...], dtype=np.float64)
        initial = torch.from_numpy(
            np.asarray(handle["snapshots"][91], dtype=np.float32)
        ).unsqueeze(0)
        for step in GT_STEPS:
            raw = torch.from_numpy(
                np.asarray(handle["snapshots"][91 + step], dtype=np.float32)
            ).unsqueeze(0)
            raw_targets[step] = raw
            oracle_targets[step] = processor.preprocessor.decode(
                processor.preprocessor.encode(raw, channel_axis=1),
                channel_axis=1,
            )
    persistence = processor.preprocessor.decode(
        processor.preprocessor.encode(initial, channel_axis=1), channel_axis=1
    )

    validation = {
        name: evaluations[name]["best"]["validation"]["model"]["metrics"]
        for name in MODEL_NAMES
    }
    persistence_metrics = evaluations["full"]["best"]["validation"][
        "persistence"
    ]["metrics"]
    zero = {
        "per_channel": {channel: 0.0 for channel in CHANNELS},
        "arithmetic_average": 0.0,
        "global_relative_l2": 0.0,
    }
    table = [
        table_row("canonical_oracle", zero),
        table_row("persistence", persistence_metrics["E_norm"]),
        *[table_row(name, validation[name]["E_norm"]) for name in MODEL_NAMES],
    ]
    write_csv(stage_i / "table2_style.csv", table)

    oracle_rows = []
    for name in MODEL_NAMES:
        row: dict[str, Any] = {"model": name}
        for metric_name in ("E_model_oracle", "E_model_raw", "E_oracle_raw"):
            value = validation[name][metric_name]
            row[f"{metric_name}_average"] = value["arithmetic_average"]
            row[f"{metric_name}_global"] = value["global_relative_l2"]
        oracle_rows.append(row)
    write_csv(stage_i / "oracle_comparison.csv", oracle_rows)

    gradients = {
        name: gradient_summary(dirs[name] / "optimizer_log.csv")
        for name in MODEL_NAMES
    }
    gradient_rows = []
    for name in MODEL_NAMES:
        row: dict[str, Any] = {
            "model": name,
            "clipping_fraction": metrics[name]["runtime"]["clipping_fraction"],
            "parameter_displacement": metrics[name]["epoch_summaries"][-1].get(
                "parameter_displacement_from_initial"
            ),
        }
        for field, values in gradients[name].items():
            row[f"{field}_mean"] = values["mean"]
        gradient_rows.append(row)
    write_csv(stage_i / "gradient_comparison.csv", gradient_rows)

    rollout_rows = []
    for step in (*GT_STEPS, *NO_GT_STEPS):
        for name in MODEL_NAMES:
            record = selected(evaluations[name], step)
            rollout_rows.append(
                {
                    "step": step,
                    "model": name,
                    "ground_truth_available": step <= 19,
                    "E_norm": record.get("E_norm") if step <= 19 else None,
                    "E_model_oracle": (
                        record.get("E_model_oracle") if step <= 19 else None
                    ),
                    "E_model_raw": record.get("E_model_raw") if step <= 19 else None,
                    "oracle_floor": record.get("E_oracle_raw") if step <= 19 else None,
                    "prediction_global_norm": record["prediction_global_norm"],
                    "model_only_saturation": (
                        record.get("model_only_saturation_mean") if step <= 19 else None
                    ),
                    "artifact_flag_count": record["artifact_flag_count"],
                    "evaluation_bound_clamp_fraction": record[
                        "evaluation_bound_clamp_fraction"
                    ],
                    "finite": record["finite"],
                    "rho_press_positive": record["rho_press_positive"],
                    "above_rin": record["above_rin"],
                    "above_rout": record["above_rout"],
                }
            )
    write_csv(stage_i / "rollout_comparison.csv", rollout_rows)

    category_records: dict[str, list[dict[str, float]]] = {
        name: [] for name in (*MODEL_NAMES, "persistence", "canonical_oracle")
    }
    for step in GT_STEPS:
        for name in MODEL_NAMES:
            scores = state_category_scores(
                prediction=states[name][step], oracle=oracle_targets[step], r=radius
            )
            scores["model_only_saturation"] = model_only_saturation(
                gt[name]["records"][step - 1]
            )
            category_records[name].append(scores)
        persistence_scores = state_category_scores(
            prediction=persistence, oracle=oracle_targets[step], r=radius
        )
        persistence_scores["model_only_saturation"] = 0.0
        category_records["persistence"].append(persistence_scores)
        oracle_scores = state_category_scores(
            prediction=oracle_targets[step], oracle=oracle_targets[step], r=radius
        )
        oracle_scores["model_only_saturation"] = 0.0
        category_records["canonical_oracle"].append(oracle_scores)

    category_names = (
        "center_morphology",
        "polar_morphology",
        "magnetic_texture",
        "radial_statistics",
        "outer_shell_statistics",
        "model_only_saturation",
    )
    categories: dict[str, dict[str, float]] = {}
    for category in category_names:
        categories[category] = {
            name: float(np.mean([record[category] for record in records]))
            for name, records in category_records.items()
        }
    morphology_rows = [
        {"category": category, **categories[category]}
        for category in (
            "center_morphology",
            "polar_morphology",
            "magnetic_texture",
            "model_only_saturation",
        )
    ]
    write_csv(stage_i / "morphology_comparison.csv", morphology_rows)

    boundary_rows = []
    for category in ("radial_statistics", "outer_shell_statistics"):
        boundary_rows.append({"category": category, **categories[category]})
    for step in NO_GT_STEPS:
        boundary_rows.append(
            {
                "category": f"step_{step}_artifact_count",
                "canonical_oracle": None,
                "persistence": None,
                **{
                    name: selected(evaluations[name], step)["artifact_flag_count"]
                    for name in MODEL_NAMES
                },
            }
        )
    write_csv(stage_i / "boundary_comparison.csv", boundary_rows)

    figures = render_morphology(
        output_dir=stage_i / "figures",
        raw_targets=raw_targets,
        oracle_targets=oracle_targets,
        persistence=persistence,
        states=states,
        phi=phi,
        theta=theta,
        r=radius,
        title_prefix="Final Stage I frozen comparison",
        file_prefix="stage_i_final",
    )
    figures = [str(Path(path).relative_to(root)) for path in figures]

    comparisons = {}
    for parent, child in (
        ("full", "no_h1"),
        ("full", "unit_index"),
        ("no_h1", "unit_index"),
        ("unit_index", "stored_coordinate"),
        ("plain", "no_h1"),
        ("plain", "unit_index"),
        ("plain", "stored_coordinate"),
        ("full", "stored_coordinate"),
    ):
        comparisons[f"{parent}_vs_{child}"] = {
            "child_validation_average_minus_parent": (
                validation[child]["E_norm"]["arithmetic_average"]
                - validation[parent]["E_norm"]["arithmetic_average"]
            ),
            "child_validation_global_minus_parent": (
                validation[child]["E_norm"]["global_relative_l2"]
                - validation[parent]["E_norm"]["global_relative_l2"]
            ),
            "child_better_categories": [
                category
                for category in category_names
                if categories[category][child] < categories[category][parent]
            ],
            "child_better_gt_steps": [
                step
                for step in GT_STEPS
                if selected(evaluations[child], step)["E_norm"]
                < selected(evaluations[parent], step)["E_norm"]
            ],
            "child_fewer_no_gt_artifacts": [
                step
                for step in NO_GT_STEPS
                if selected(evaluations[child], step)["artifact_flag_count"]
                < selected(evaluations[parent], step)["artifact_flag_count"]
            ],
        }

    run_c_gate = {
        "training_complete": metrics["stored_coordinate"]["epochs"] == 30,
        "microbatches_exact": metrics["stored_coordinate"]["train_batches"] == 2370,
        "optimizer_updates_exact": (
            metrics["stored_coordinate"]["runtime"]["optimizer_updates"] == 600
        ),
        "strict_reload": all(
            metrics["stored_coordinate"]["checkpoint_reload"][name][
                "metadata_validated"
            ]
            for name in ("best", "last")
        ),
        "nonfinite_zero": metrics["stored_coordinate"]["runtime"][
            "nonfinite_count"
        ] == 0,
        "validation_finite_positive": (
            evaluations["stored_coordinate"]["best"]["validation"]["constraints"][
                "finite"
            ]
            and evaluations["stored_coordinate"]["best"]["validation"][
                "constraints"
            ]["rho_press_positive"]
        ),
        "gt_finite_positive": all(
            selected(evaluations["stored_coordinate"], step)["finite"]
            and selected(evaluations["stored_coordinate"], step)["rho_press_positive"]
            for step in GT_STEPS
        ),
        "no_gt_finite_positive": all(
            selected(evaluations["stored_coordinate"], step)["finite"]
            and selected(evaluations["stored_coordinate"], step)["rho_press_positive"]
            for step in NO_GT_STEPS
        ),
    }
    stored_vs_full = comparisons["full_vs_stored_coordinate"]
    stored_vs_plain = comparisons["plain_vs_stored_coordinate"]
    extension_vs_full_counts = {
        name: len(comparisons[f"full_vs_{name}"]["child_better_categories"])
        for name in ("no_h1", "unit_index", "stored_coordinate")
    }
    extension_vs_plain_counts = {
        name: len(comparisons[f"plain_vs_{name}"]["child_better_categories"])
        for name in ("no_h1", "unit_index", "stored_coordinate")
    }
    if not all(run_c_gate.values()):
        decision = "E"
        decision_name = "Inconclusive or technical failure"
    elif (
        all(count >= 2 for count in extension_vs_full_counts.values())
        and all(count < 2 for count in extension_vs_plain_counts.values())
    ):
        decision = "D"
        decision_name = "No extension rescues Full"
    elif (
        len(stored_vs_full["child_better_categories"]) >= 2
        and validation["stored_coordinate"]["E_norm"]["arithmetic_average"]
        < 10.0 * validation["full"]["E_norm"]["arithmetic_average"]
    ):
        decision = "A"
        decision_name = "Stored-coordinate H1 gives controlled rescue"
    elif len(comparisons["full_vs_unit_index"]["child_better_categories"]) >= 2:
        decision = "B"
        decision_name = "Unit-index H1 is sufficient"
    elif (
        len(comparisons["full_vs_no_h1"]["child_better_categories"]) >= 2
        and len(comparisons["no_h1_vs_unit_index"]["child_better_categories"]) < 3
    ):
        decision = "C"
        decision_name = "Removing H1 is best"
    else:
        decision = "D"
        decision_name = "No extension rescues Full"

    questions = {
        "full_vs_no_h1": {
            "observed": comparisons["full_vs_no_h1"],
            "inference": "Removing current upstream H1 changes the frozen morphology/rollout tradeoff.",
            "unsupported_claim": "This does not isolate all differences from the paper's unavailable LocalNO geometry.",
        },
        "full_vs_unit_index": {
            "observed": comparisons["full_vs_unit_index"],
            "inference": "Removing the 4096 spacing amplification substantially reduces H1 gradient influence.",
            "unsupported_claim": "The unit-index proxy is not a covariant physical H1.",
        },
        "no_h1_vs_unit_index": {
            "observed": comparisons["no_h1_vs_unit_index"],
            "inference": "The paired scores show whether a weak H1 provides repeatable benefit over removal.",
            "unsupported_claim": "A 30-epoch pilot cannot establish asymptotic advantage.",
        },
        "unit_index_vs_stored_coordinate": {
            "observed": comparisons["unit_index_vs_stored_coordinate"],
            "inference": "Any multi-category gain is attributable to the frozen coordinate/volume adaptation within this proxy.",
            "unsupported_claim": "The proxy is not Kerr-Schild proper-volume or covariant GRMHD H1.",
        },
        "stored_coordinate_vs_plain": {
            "observed": stored_vs_plain,
            "inference": "The comparison tests whether adapted H1 plus Full priors has multi-category advantages over Plain.",
            "unsupported_claim": "It cannot establish equivalence to paper Table 2 or 1200-epoch behavior.",
        },
    }
    extension_rows = []
    for name in ("canonical_oracle", "persistence", *MODEL_NAMES):
        metric = zero if name == "canonical_oracle" else (
            persistence_metrics["E_norm"] if name == "persistence" else validation[name]["E_norm"]
        )
        extension_rows.append(
            {
                "model": name,
                "validation_average": metric["arithmetic_average"],
                "validation_global": metric["global_relative_l2"],
                "best_epoch": metrics[name]["best_epoch"] if name in metrics else None,
                "runtime_seconds": (
                    metrics[name]["runtime"]["wall_seconds"] if name in metrics else None
                ),
                "finite": (
                    evaluations[name]["best"]["validation"]["constraints"]["finite"]
                    if name in evaluations else True
                ),
            }
        )
    write_csv(stage_i / "extension_comparison.csv", extension_rows)

    result = {
        "schema_version": "paper-stage-i-final-extension-comparison-v1",
        "status": "completed",
        "frozen_scoring": {
            "implementation": "src/grmhd/paper_stage_g_evaluation.py and scripts/compare_paper_stage_g.py",
            "validation_pairs": 19,
            "selected_gt_steps": list(GT_STEPS),
            "selected_no_gt_steps": list(NO_GT_STEPS),
            "same_aggregation": True,
            "same_preprocessing_oracle": True,
            "same_artifact_thresholds": True,
        },
        "table2_style": table,
        "oracle_aware": {name: validation[name] for name in MODEL_NAMES},
        "gradient_diagnostics": gradients,
        "morphology_and_boundary": categories,
        "comparisons": comparisons,
        "five_questions": questions,
        "run_c_gate": run_c_gate,
        "figures": figures,
        "decision": {"choice": decision, "name": decision_name},
    }
    (stage_i / "extension_comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Final Stage I extension comparison",
        "",
        "All rows use the frozen reduced100 validation, preprocessing oracle, selected rollout steps, aggregation, and artifact thresholds.",
        "",
        "| model | validation average | global |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| {row['model']} | {row['validation_average']:.6g} | {row['validation_global']:.6g} |"
        for row in extension_rows
    )
    lines.extend(["", "## Decision", "", f"`{decision} — {decision_name}`"])
    (stage_i / "extension_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    decision_payload = {
        "schema_version": "paper-stage-i-final-decision-v1",
        "choice": decision,
        "name": decision_name,
        "run_c_gate": run_c_gate,
        "stored_better_than_full_categories": stored_vs_full[
            "child_better_categories"
        ],
        "stored_better_than_plain_categories": stored_vs_plain[
            "child_better_categories"
        ],
        "extension_better_category_counts_vs_full": extension_vs_full_counts,
        "extension_better_category_counts_vs_plain": extension_vs_plain_counts,
        "paper_main_path": "Stage G Full with current upstream H1 weight 0.05 remains frozen",
        "h1_adaptation_conclusion": "Stage I extensions diagnose adaptation mismatch and do not replace Stage G Full.",
        "paper_ablation_conclusion": "Do not start further training automatically; review this decision first.",
        "recommended_next_direction": "Improve upstream-to-paper geometry/operator fidelity before interpreting more loss ablations.",
    }
    (stage_i / "stage_i_decision.json").write_text(
        json.dumps(decision_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    decision_lines = [
        "# Stage I final decision",
        "",
        f"## {decision} — {decision_name}",
        "",
        f"- Run C engineering gate: `{all(run_c_gate.values())}`",
        "- Better than Full categories: "
        + (", ".join(stored_vs_full["child_better_categories"]) or "none"),
        "- Better than Plain categories: "
        + (", ".join(stored_vs_plain["child_better_categories"]) or "none"),
        "- Stage G Full remains the paper-adapted main path; Run A/B/C remain diagnostic extensions.",
        "- No Stage J or additional training was started.",
    ]
    (stage_i / "stage_i_decision.md").write_text(
        "\n".join(decision_lines) + "\n", encoding="utf-8"
    )

    manifest_path = stage_i / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stored_metric = validation["stored_coordinate"]["E_norm"]
    manifest["status"] = "stage_i_completed"
    manifest["runs"]["stored_coordinate_volume_proxy"].update(
        {
            "status": "completed_and_evaluated",
            "best_epoch": metrics["stored_coordinate"]["best_epoch"],
            "best_validation_average": stored_metric["arithmetic_average"],
            "best_validation_global": stored_metric["global_relative_l2"],
            "runtime_seconds": metrics["stored_coordinate"]["runtime"]["wall_seconds"],
            "optimizer_updates": metrics["stored_coordinate"]["runtime"]["optimizer_updates"],
            "nonfinite_count": metrics["stored_coordinate"]["runtime"]["nonfinite_count"],
            "strict_reload": metrics["stored_coordinate"]["checkpoint_reload"],
            "file_sha256": {
                name: sha256_file(dirs["stored_coordinate"] / name)
                for name in RUN_FILES
            },
        }
    )
    manifest["checksums"].update(
        {
            "extension_comparison": sha256_file(stage_i / "extension_comparison.json"),
            "stage_i_decision": sha256_file(stage_i / "stage_i_decision.json"),
        }
    )
    manifest["final_decision"] = decision_payload
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (stage_i / "run_manifest.md").write_text(
        "# Stage I completed manifest\n\n"
        f"- Status: `{manifest['status']}`\n"
        f"- Run C best epoch: `{metrics['stored_coordinate']['best_epoch']}`\n"
        f"- Run C validation average/global: `{stored_metric['arithmetic_average']:.12g}` / `{stored_metric['global_relative_l2']:.12g}`\n"
        "- Run C: `30/30 epochs`, `2370` microbatches, `600` optimizer updates\n"
        "- Best/last strict reload, GT19, no-GT100: `passed`\n"
        f"- Final decision: `{decision} — {decision_name}`\n",
        encoding="utf-8",
    )
    print(json.dumps(decision_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
