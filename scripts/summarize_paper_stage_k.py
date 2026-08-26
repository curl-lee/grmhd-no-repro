#!/usr/bin/env python
"""Summarize Stage K validation, rollouts, morphology, and frozen decision gates."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/grmhd-matplotlib")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from compare_paper_stage_g import (
    load_selected,
    model_only_saturation,
    state_category_scores,
)
from grmhd import CHANNELS
from grmhd.paper_config import load_paper_experiment_config
from grmhd.paper_data_processor import PaperDataProcessor


MORPHOLOGY_GT_STEPS = (1, 5, 10, 19)
MORPHOLOGY_NO_GT_STEPS = (50, 100)
ROLLOUT_GT_STEPS = (1, 3, 5, 10, 19)
ROLLOUT_NO_GT_STEPS = (25, 50, 75, 100)
MODEL_LABELS = {
    "canonical_oracle": "canonical oracle",
    "persistence": "persistence",
    "fno_plain": "Stage G FNO Plain",
    "fno_full": "Stage G FNO Full",
    "localno_plain": "Stage K differential LocalNO Plain",
}


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def validation_row(name: str, metric: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": name,
        **{channel: metric["per_channel"][channel] for channel in CHANNELS},
        "arithmetic_average": metric["arithmetic_average"],
        "global_normalized_relative_l2": metric["global_relative_l2"],
    }


def model_only_for_persistence(
    processor: PaperDataProcessor,
    initial: torch.Tensor,
    raw_target: torch.Tensor,
) -> float:
    initial_normalized = processor.preprocessor.encode(initial, channel_axis=1)
    target_normalized = processor.preprocessor.encode(raw_target, channel_axis=1)
    limit = (
        processor.preprocessor.gamma
        * processor.preprocessor.inverse_clamp_fraction
    )
    model_mask = torch.abs(initial_normalized) > limit
    target_mask = torch.abs(target_normalized) > limit
    channel_values = [
        float((model_mask[:, channel] & ~target_mask[:, channel]).float().mean())
        for channel in range(len(CHANNELS))
    ]
    return float(np.mean(channel_values))


def render_morphology(
    *,
    output_dir: Path,
    raw_targets: Mapping[int, torch.Tensor],
    oracle_targets: Mapping[int, torch.Tensor],
    persistence: torch.Tensor,
    states: Mapping[str, Mapping[int, torch.Tensor]],
    phi: np.ndarray,
    theta: np.ndarray,
    r: np.ndarray,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed_phi_index = len(phi) // 2
    equatorial_theta_index = int(np.argmin(np.abs(theta - np.pi / 2)))
    files = []
    for step in (*MORPHOLOGY_GT_STEPS, *MORPHOLOGY_NO_GT_STEPS):
        panels = (
            [
                ("raw truth", raw_targets[step]),
                ("canonical oracle", oracle_targets[step]),
                ("persistence", persistence),
            ]
            if step in MORPHOLOGY_GT_STEPS
            else [("persistence", persistence)]
        )
        panels.extend(
            [
                ("Stage G FNO Full", states["fno_full"][step]),
                ("Stage G FNO Plain", states["fno_plain"][step]),
                ("Stage K LocalNO Plain", states["localno_plain"][step]),
            ]
        )
        for view in ("fixed_phi_theta_r", "equatorial_phi_r"):
            figure, axes = plt.subplots(
                len(CHANNELS),
                len(panels),
                figsize=(4.0 * len(panels), 2.7 * len(CHANNELS)),
                constrained_layout=True,
            )
            for channel, channel_name in enumerate(CHANNELS):
                slices = []
                for _, state in panels:
                    values = state[0, channel].detach().cpu().numpy()
                    slices.append(
                        values[fixed_phi_index]
                        if view == "fixed_phi_theta_r"
                        else values[:, equatorial_theta_index, :]
                    )
                combined = np.concatenate([value.reshape(-1) for value in slices])
                if channel_name in {"rho", "press"}:
                    vmin, vmax = np.quantile(combined, [0.01, 0.99])
                    cmap = "viridis"
                else:
                    maximum = float(np.quantile(np.abs(combined), 0.99))
                    vmin, vmax = -maximum, maximum
                    cmap = "coolwarm"
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                    vmin, vmax = float(combined.min()), float(combined.max() + 1e-12)
                for column, ((label, _), values) in enumerate(zip(panels, slices)):
                    axis = axes[channel, column]
                    extent = (
                        [float(r.min()), float(r.max()), float(theta.min()), float(theta.max())]
                        if view == "fixed_phi_theta_r"
                        else [float(r.min()), float(r.max()), float(phi.min()), float(phi.max())]
                    )
                    image = axis.imshow(
                        values,
                        origin="lower",
                        aspect="auto",
                        extent=extent,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                    )
                    if channel == 0:
                        axis.set_title(label)
                    if column == 0:
                        axis.set_ylabel(
                            f"{channel_name}\n"
                            + ("theta" if view == "fixed_phi_theta_r" else "phi")
                        )
                    if channel == len(CHANNELS) - 1:
                        axis.set_xlabel("r")
                    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
            availability = (
                "raw/oracle shown; preprocessing floor and clamp masks reported separately"
                if step in MORPHOLOGY_GT_STEPS
                else "no ground truth after step 19; raw/oracle omitted"
            )
            figure.suptitle(
                f"Stage K step {step}: {view.replace('_', ' ')}\n"
                f"{availability}; spherical-coordinate adapted views; "
                "spherical Kerr-Schild stored components, not Cartesian slices",
                fontsize=12,
            )
            path = output_dir / f"stage_k_step_{step:03d}_{view}.png"
            figure.savefig(path, dpi=120)
            plt.close(figure)
            files.append(str(path))
    return files


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    stage_k = root / "outputs/paper_reduced100/stage_k"
    local_dir = stage_k / "localno_differential_plain"
    smoke_dir = stage_k / "smoke_localno_differential_plain"
    stage_g = root / "outputs/paper_reduced100/stage_g"
    model_dirs = {
        "fno_plain": stage_g / "pilot30_plain_l2",
        "fno_full": stage_g / "pilot30_full_fno",
        "localno_plain": local_dir,
    }

    manifest = load_json(stage_k / "run_manifest.json")
    preflight = load_json(stage_k / "localno_preflight.json")
    smoke = load_json(smoke_dir / "metrics.json")
    metrics = {name: load_json(path / "metrics.json") for name, path in model_dirs.items()}
    evaluations = {
        name: load_json(path / "evaluation_summary.json")
        for name, path in model_dirs.items()
    }
    gt = {name: load_json(path / "gt_rollout.json") for name, path in model_dirs.items()}
    no_gt = {
        name: load_json(path / "no_gt_rollout.json")
        for name, path in model_dirs.items()
    }
    states = {name: load_selected(path / "selected_states.pt") for name, path in model_dirs.items()}

    config = load_paper_experiment_config(
        root / "configs/paper_reduced100/stage_k_localno_differential_plain.yaml",
        project_root=root,
    )
    processor = PaperDataProcessor.from_config(config)
    raw_targets: dict[int, torch.Tensor] = {}
    oracle_targets: dict[int, torch.Tensor] = {}
    with h5py.File(
        config.resolve_path(config.values["protocol"]["dataset"]), "r"
    ) as handle:
        phi = np.asarray(handle["coords/phi"][...], dtype=np.float64)
        theta = np.asarray(handle["coords/theta"][...], dtype=np.float64)
        r = np.asarray(handle["coords/r"][...], dtype=np.float64)
        initial = torch.from_numpy(
            np.asarray(handle["snapshots"][91], dtype=np.float32)
        ).unsqueeze(0)
        for step in MORPHOLOGY_GT_STEPS:
            raw = torch.from_numpy(
                np.asarray(handle["snapshots"][91 + step], dtype=np.float32)
            ).unsqueeze(0)
            raw_targets[step] = raw
            oracle_targets[step] = processor.preprocessor.decode(
                processor.preprocessor.encode(raw, channel_axis=1), channel_axis=1
            )
    persistence = processor.preprocessor.decode(
        processor.preprocessor.encode(initial, channel_axis=1), channel_axis=1
    )

    local_metrics = metrics["localno_plain"]
    training_rows = []
    for summary in local_metrics["epoch_summaries"]:
        training_rows.append(
            {
                "epoch": int(summary["epoch"]) + 1,
                "microbatches": summary["microbatches_seen"],
                "optimizer_steps": summary["optimizer_steps"],
                "final_accumulation_count": summary["final_accumulation_count"],
                "train_total_mean": summary["train_total"]["mean"],
                "train_total_median": summary["train_total"]["median"],
                "train_total_q95": summary["train_total"]["q95"],
                "train_total_maximum": summary["train_total"]["maximum"],
                "validation_arithmetic_average": summary[
                    "validation_normalized_arithmetic_average_relative_l2"
                ],
                "validation_global": summary[
                    "validation_normalized_global_relative_l2"
                ],
                "clipping_fraction": summary["clipping_fraction"],
                "clip_scale_mean": summary["clip_scale"]["mean"],
                "parameter_displacement": summary[
                    "parameter_displacement_from_initial"
                ],
                "wall_seconds": summary["wall_seconds"],
                "peak_allocated_mib": summary["peak_allocated_mib"],
                "peak_reserved_mib": summary["peak_reserved_mib"],
                "nonfinite_count": summary["nonfinite_count"],
            }
        )
    training_summary = {
        "schema_version": "paper-stage-k-training-summary-v1",
        "status": local_metrics["status"],
        "best_epoch": local_metrics["best_epoch"],
        "last_epoch": local_metrics["epochs"],
        "microbatches": local_metrics["train_batches"],
        "optimizer_updates": local_metrics["runtime"]["optimizer_updates"],
        "runtime": local_metrics["runtime"],
        "checkpoint_reload": local_metrics["checkpoint_reload"],
        "epochs": training_rows,
    }
    write_csv(stage_k / "training_summary.csv", training_rows)
    json_dump(stage_k / "training_summary.json", training_summary)

    oracle_zero = {
        "per_channel": {channel: 0.0 for channel in CHANNELS},
        "arithmetic_average": 0.0,
        "global_relative_l2": 0.0,
    }
    local_validation = evaluations["localno_plain"]["best"]["validation"]
    persistence_metric = local_validation["persistence"]["metrics"]["E_norm"]
    validation_metrics = {
        "canonical_oracle": oracle_zero,
        "persistence": persistence_metric,
        "fno_plain": evaluations["fno_plain"]["best"]["validation"]["model"]["metrics"]["E_norm"],
        "fno_full": evaluations["fno_full"]["best"]["validation"]["model"]["metrics"]["E_norm"],
        "localno_plain": local_validation["model"]["metrics"]["E_norm"],
    }
    validation_rows = [
        validation_row(name, value) for name, value in validation_metrics.items()
    ]
    write_csv(stage_k / "validation_metrics.csv", validation_rows)
    json_dump(
        stage_k / "validation_metrics.json",
        {
            "schema_version": "paper-stage-k-validation-summary-v1",
            "table": validation_rows,
            "localno_oracle_aware": local_validation["model"],
            "localno_constraints": local_validation["constraints"],
            "strict_recompute_parity": evaluations["localno_plain"]["best"][
                "training_time_metric_parity"
            ],
        },
    )

    local_norm = validation_metrics["localno_plain"]
    persistence_rows = [
        {
            "metric": channel,
            "localno": local_norm["per_channel"][channel],
            "persistence": persistence_metric["per_channel"][channel],
            "localno_over_persistence": local_norm["per_channel"][channel]
            / persistence_metric["per_channel"][channel],
        }
        for channel in CHANNELS
    ]
    persistence_rows.extend(
        [
            {
                "metric": "arithmetic_average",
                "localno": local_norm["arithmetic_average"],
                "persistence": persistence_metric["arithmetic_average"],
                "localno_over_persistence": local_norm["arithmetic_average"]
                / persistence_metric["arithmetic_average"],
            },
            {
                "metric": "global",
                "localno": local_norm["global_relative_l2"],
                "persistence": persistence_metric["global_relative_l2"],
                "localno_over_persistence": local_norm["global_relative_l2"]
                / persistence_metric["global_relative_l2"],
            },
        ]
    )
    write_csv(stage_k / "persistence_comparison.csv", persistence_rows)
    json_dump(
        stage_k / "persistence_comparison.json",
        {
            "schema_version": "paper-stage-k-persistence-comparison-v1",
            "beats_persistence_one_step_average": local_norm["arithmetic_average"]
            < persistence_metric["arithmetic_average"],
            "rows": persistence_rows,
        },
    )

    local_gt = gt["localno_plain"]
    gt_rows = []
    for step in ROLLOUT_GT_STEPS:
        record = local_gt["records"][step - 1]
        model_norm = record["oracle_aware"]["metrics"]["E_norm"]
        persistence_norm = record["persistence"]["metrics"]["E_norm"]
        gt_rows.append(
            {
                "step": step,
                "normalized_average": model_norm["arithmetic_average"],
                "normalized_global": model_norm["global_relative_l2"],
                "persistence_average": persistence_norm["arithmetic_average"],
                "model_over_persistence_average": model_norm["arithmetic_average"]
                / persistence_norm["arithmetic_average"],
                "model_to_oracle_average": record["oracle_aware"]["metrics"]["E_model_oracle"]["arithmetic_average"],
                "model_to_raw_average": record["oracle_aware"]["metrics"]["E_model_raw"]["arithmetic_average"],
                "oracle_floor_average": record["oracle_aware"]["metrics"]["E_oracle_raw"]["arithmetic_average"],
                "finite": record["finite"],
                "rho_press_positive": record["rho_press_positive"],
                "prediction_norm": record["prediction_global_norm"],
                "evaluation_clamp": record["evaluation_bound_clamp_fraction"],
                "artifact_count": len(record["artifacts"]["flags"]),
                "artifact_flags": ";".join(record["artifacts"]["flags"]),
                "rho_minimum": record["physical_range"]["rho"]["minimum"],
                "press_minimum": record["physical_range"]["press"]["minimum"],
            }
        )
    write_csv(stage_k / "rollout_gt.csv", gt_rows)
    json_dump(
        stage_k / "rollout_gt.json",
        {
            "schema_version": "paper-stage-k-selected-gt-rollout-v1",
            "selected_rows": gt_rows,
            "finite": local_gt["finite"],
            "rho_press_positive": local_gt["rho_press_positive"],
            "records": [local_gt["records"][step - 1] for step in ROLLOUT_GT_STEPS],
        },
    )

    local_no_gt = no_gt["localno_plain"]
    no_gt_rows = []
    for step in ROLLOUT_NO_GT_STEPS:
        record = local_no_gt["records"][step - 20]
        no_gt_rows.append(
            {
                "step": step,
                "ground_truth_available": False,
                "finite": record["finite"],
                "rho_press_positive": record["rho_press_positive"],
                "prediction_norm": record["prediction_global_norm"],
                "above_rin": record["above_rin"],
                "above_rout": record["above_rout"],
                "evaluation_clamp": record["evaluation_bound_clamp_fraction"],
                "artifact_count": len(record["artifacts"]["flags"]),
                "artifact_flags": ";".join(record["artifacts"]["flags"]),
                "rho_minimum": record["physical_range"]["rho"]["minimum"],
                "rho_maximum": record["physical_range"]["rho"]["maximum"],
                "press_minimum": record["physical_range"]["press"]["minimum"],
                "press_maximum": record["physical_range"]["press"]["maximum"],
            }
        )
    write_csv(stage_k / "rollout_no_gt.csv", no_gt_rows)
    json_dump(
        stage_k / "rollout_no_gt.json",
        {
            "schema_version": "paper-stage-k-selected-no-gt-rollout-v1",
            "selected_rows": no_gt_rows,
            "finite": local_no_gt["finite"],
            "rho_press_positive": local_no_gt["rho_press_positive"],
            "transform_count_delta": local_no_gt["transform_count_delta"],
            "temporal_channel_means": local_no_gt["temporal_channel_means"],
            "temporal_channel_stds": local_no_gt["temporal_channel_stds"],
            "records": [
                local_no_gt["records"][step - 20] for step in ROLLOUT_NO_GT_STEPS
            ],
        },
    )

    category_records: dict[str, list[dict[str, float]]] = {
        name: [] for name in MODEL_LABELS
    }
    for step in MORPHOLOGY_GT_STEPS:
        category_records["canonical_oracle"].append(
            {
                "step": float(step),
                "center_morphology": 0.0,
                "polar_morphology": 0.0,
                "magnetic_texture": 0.0,
                "radial_statistics": 0.0,
                "outer_shell_statistics": 0.0,
                "model_only_saturation": 0.0,
            }
        )
        persistence_scores = state_category_scores(
            prediction=persistence, oracle=oracle_targets[step], r=r
        )
        persistence_scores["step"] = float(step)
        persistence_scores["model_only_saturation"] = model_only_for_persistence(
            processor, initial, raw_targets[step]
        )
        category_records["persistence"].append(persistence_scores)
        for name in ("fno_plain", "fno_full", "localno_plain"):
            scores = state_category_scores(
                prediction=states[name][step], oracle=oracle_targets[step], r=r
            )
            scores["step"] = float(step)
            scores["model_only_saturation"] = model_only_saturation(
                gt[name]["records"][step - 1]
            )
            category_records[name].append(scores)

    categories = (
        "center_morphology",
        "polar_morphology",
        "magnetic_texture",
        "radial_statistics",
        "outer_shell_statistics",
        "model_only_saturation",
    )
    category_values: dict[str, dict[str, float | None]] = {}
    for category in categories:
        category_values[category] = {
            name: float(np.mean([record[category] for record in records]))
            for name, records in category_records.items()
        }
    category_values["step50_100_artifacts"] = {
        "canonical_oracle": None,
        "persistence": 0.0,
        **{
            name: float(
                sum(
                    len(no_gt[name]["records"][step - 20]["artifacts"]["flags"])
                    for step in MORPHOLOGY_NO_GT_STEPS
                )
            )
            for name in ("fno_plain", "fno_full", "localno_plain")
        },
    }
    morphology_rows = [
        {
            "category": category,
            "model": name,
            "value": value,
            "lower_is_better": True,
        }
        for category, values in category_values.items()
        for name, value in values.items()
    ]
    write_csv(stage_k / "morphology_comparison.csv", morphology_rows)
    figure_files = render_morphology(
        output_dir=stage_k / "figures",
        raw_targets=raw_targets,
        oracle_targets=oracle_targets,
        persistence=persistence,
        states=states,
        phi=phi,
        theta=theta,
        r=r,
    )
    json_dump(
        stage_k / "morphology_comparison.json",
        {
            "schema_version": "paper-stage-k-frozen-morphology-comparison-v1",
            "implementation": "Stage G frozen thresholds and category functions",
            "views": ["fixed-phi theta-r", "equatorial phi-r"],
            "view_classification": "spherical-coordinate adapted views",
            "not_cartesian_central_slices": True,
            "aggregate_categories": category_values,
            "per_step": category_records,
            "figure_files": figure_files,
        },
    )

    smoke_counts_ok = (
        smoke["epochs"] == 2
        and smoke["train_batches"] == 158
        and smoke["runtime"]["optimizer_updates"] == 40
        and all(
            record["microbatches_seen"] == 79
            and record["optimizer_steps"] == 20
            and record["final_accumulation_count"] == 3
            for record in smoke["epoch_summaries"]
        )
    )
    formal_counts_ok = (
        local_metrics["epochs"] == 30
        and local_metrics["train_batches"] == 2370
        and local_metrics["runtime"]["optimizer_updates"] == 600
        and all(
            record["microbatches_seen"] == 79
            and record["optimizer_steps"] == 20
            and record["final_accumulation_count"] == 3
            for record in local_metrics["epoch_summaries"]
        )
    )
    checkpoints_ok = all(
        record["strict_model_reload"]
        and record["optimizer_reload"]
        and record["scheduler_reload"]
        and record["prediction_parity"]
        for record in local_metrics["checkpoint_reload"].values()
    )
    transforms_ok = local_no_gt["transform_count_delta"] == {
        "input_encode": 100,
        "target_encode": 100,
        "oracle_decode": 100,
        "prediction_decode": 100,
    }
    persistent_collapse_flags = {
        str(step): [
            flag
            for flag in local_no_gt["records"][step - 20]["artifacts"]["flags"]
            if "possible_field_collapse" in flag
        ]
        for step in ROLLOUT_NO_GT_STEPS
    }
    persistent_collapse = all(persistent_collapse_flags[str(step)] for step in ROLLOUT_NO_GT_STEPS)
    range_explosion = any(
        record["above_rout"] for record in local_no_gt["records"]
    )
    engineering_gates = {
        "preflight_passed": preflight["status"] == "passed",
        "smoke_counts_and_reload_passed": smoke_counts_ok
        and all(
            item["strict_model_reload"]
            for item in smoke["checkpoint_reload"].values()
        ),
        "formal_counts_passed": formal_counts_ok,
        "formal_checkpoints_reload": checkpoints_ok,
        "no_training_nonfinite": local_metrics["runtime"]["nonfinite_count"] == 0,
        "gt_rollout_finite_positive": local_gt["finite"]
        and local_gt["rho_press_positive"],
        "no_gt_rollout_finite_positive": local_no_gt["finite"]
        and local_no_gt["rho_press_positive"],
        "transform_counters_exact": transforms_ok,
        "decoded_range_above_frozen_rout": range_explosion,
        "persistent_frozen_collapse_flags_25_50_75_100": persistent_collapse,
        "beats_persistence_one_step_average": local_norm["arithmetic_average"]
        < persistence_metric["arithmetic_average"],
    }
    if not all(
        engineering_gates[key]
        for key in (
            "preflight_passed",
            "smoke_counts_and_reload_passed",
            "formal_counts_passed",
            "formal_checkpoints_reload",
            "no_training_nonfinite",
        )
    ):
        choice = "D"
        decision = "ENGINEERING_FAILURE"
    elif (
        not engineering_gates["gt_rollout_finite_positive"]
        or not engineering_gates["no_gt_rollout_finite_positive"]
        or not engineering_gates["transform_counters_exact"]
        or range_explosion
        or persistent_collapse
    ):
        choice = "C"
        decision = "TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE"
    elif engineering_gates["beats_persistence_one_step_average"]:
        choice = "A"
        decision = "ENGINEERING_COMPLETE_AND_BEATS_PERSISTENCE"
    else:
        choice = "B"
        decision = "ENGINEERING_COMPLETE_BUT_NOT_BEAT_PERSISTENCE"
    decision_payload = {
        "schema_version": "paper-stage-k-decision-v1",
        "choice": choice,
        "decision": decision,
        "classification": "adapted_method_reproduction",
        "engineering_gates": engineering_gates,
        "persistent_collapse_flags": persistent_collapse_flags,
        "evidence": {
            "localno_validation_average": local_norm["arithmetic_average"],
            "persistence_validation_average": persistence_metric["arithmetic_average"],
            "localno_over_persistence": local_norm["arithmetic_average"]
            / persistence_metric["arithmetic_average"],
            "gt_step19_average": gt_rows[-1]["normalized_average"],
            "gt_step19_model_over_persistence": gt_rows[-1][
                "model_over_persistence_average"
            ],
            "step50_100_artifact_count": category_values[
                "step50_100_artifacts"
            ]["localno_plain"],
            "no_gt_above_rout_any": range_explosion,
            "preprocessing_floor_warning_channels": ["Bcc3", "vel3"],
        },
        "interpretation": (
            "Training, checkpoint, validation, and transform-count engineering gates "
            "passed, and decoded ranges stayed finite and positive. The frozen "
            "artifact detector nevertheless marked field collapse at every selected "
            "no-GT step 25/50/75/100, so the explicit Stage K C rule takes precedence."
        ),
        "not_claimed": [
            "3D_DISCO_reproduction",
            "exact_paper_reproduction",
            "physical_correctness_from_finite_rollout",
        ],
        "provenance": {
            "project_commit": manifest["project_git_commit"],
            "upstream_commit": manifest["upstream_commit"],
            "dataset_sha256": manifest["checksums"]["dataset"],
            "preprocessing_sha256": manifest["checksums"]["preprocessing"],
            "pair_order_sha256": manifest["checksums"]["pair_order_file"],
            "initial_state_sha256": manifest["initial_state"][
                "tensor_state_sha256"
            ],
        },
    }
    json_dump(stage_k / "stage_k_decision.json", decision_payload)
    lines = [
        "# Stage K decision",
        "",
        f"## {choice}. {decision}",
        "",
        "Classification: `adapted_method_reproduction` (3D differential LocalNO; "
        "DISCO integral disabled; not exact paper reproduction).",
        "",
        f"- LocalNO/persistence one-step average: "
        f"`{local_norm['arithmetic_average']:.6g}/"
        f"{persistence_metric['arithmetic_average']:.6g}` "
        f"(ratio `{local_norm['arithmetic_average'] / persistence_metric['arithmetic_average']:.6g}`)",
        f"- Training microbatches/updates: "
        f"`{local_metrics['train_batches']}/{local_metrics['runtime']['optimizer_updates']}`",
        f"- GT/no-GT finite and positive: "
        f"`{local_gt['finite'] and local_gt['rho_press_positive']}/"
        f"{local_no_gt['finite'] and local_no_gt['rho_press_positive']}`",
        f"- Transform counters exact: `{transforms_ok}`",
        f"- Any frozen Rout exceedance: `{range_explosion}`",
        f"- Persistent selected-step collapse flags: `{persistent_collapse}`",
        "",
        decision_payload["interpretation"],
        "Bcc3 and vel3 also carry a known preprocessing-oracle floor; that caveat is "
        "reported but does not change the frozen Stage K decision rule.",
    ]
    (stage_k / "stage_k_decision.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(decision_payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
