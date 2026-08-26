#!/usr/bin/env python
"""Build the controlled Stage R residual-pilot summaries and decision."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch

from grmhd import CHANNELS
from grmhd.paper_config import load_paper_experiment_config
from grmhd.paper_stage_g_evaluation import radial_shell_indices
from grmhd.paper_stage_l_attribution import (
    basic_field_metrics,
    radial_profile,
    spectrum_metrics,
)
from grmhd.paper_stage_m import (
    evaluate_oracle_conditioned_structure_gate,
    radial_profile_vector,
    transport_metrics,
    variance_vector,
)
from grmhd.paper_stage_o import load_frozen_p3


SELECTED_GT = (1, 3, 5, 10, 19)
SELECTED_NO_GT = (25, 50, 75, 100)
TARGET_CHANNELS = ("Bcc2", "Bcc3", "vel3")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([json_safe(row) for row in rows])


def record_map(payload: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {int(record["step"]): record for record in payload["records"]}


def metric_rows(metrics: Mapping[str, Any], source: str) -> list[dict[str, Any]]:
    rows = []
    for channel in CHANNELS:
        rows.append(
            {
                "source": source,
                "channel": channel,
                **{
                    name: values["per_channel"][channel]
                    for name, values in metrics.items()
                    if isinstance(values, Mapping) and "per_channel" in values
                },
            }
        )
    rows.extend(
        (
            {
                "source": source,
                "channel": "arithmetic_average",
                **{
                    name: values["arithmetic_average"]
                    for name, values in metrics.items()
                    if isinstance(values, Mapping) and "arithmetic_average" in values
                },
            },
            {
                "source": source,
                "channel": "global_relative_l2",
                **{
                    name: values["global_relative_l2"]
                    for name, values in metrics.items()
                    if isinstance(values, Mapping) and "global_relative_l2" in values
                },
            },
        )
    )
    return rows


def smoke_outputs(stage_r: Path) -> dict[str, Any]:
    metrics = read_json(stage_r / "smoke_localno_p3_residual_plain/metrics.json")
    epochs = metrics["epoch_summaries"]
    passed = bool(
        metrics["status"] == "passed"
        and metrics["train_batches"] == 158
        and metrics["runtime"]["optimizer_updates"] == 40
        and all(row["microbatches_seen"] == 79 for row in epochs)
        and all(row["optimizer_steps"] == 20 for row in epochs)
        and all(row["final_accumulation_count"] == 3 for row in epochs)
        and metrics["checkpoint_reload"]["best"]["prediction_parity"]
        and metrics["checkpoint_reload"]["last"]["prediction_parity"]
        and metrics["rollout3"]["finite"]
        and metrics["rollout3"]["rho_press_positive"]
        and metrics["rollout3"]["no_double_transform"]
        and metrics["rollout3"]["artifact_flags"]["evaluation_rho_press_clamp"]
        is False
    )
    payload = {
        "schema_version": "paper-stage-r-smoke-summary-v1",
        "status": metrics["status"],
        "passed": passed,
        "microbatches": metrics["train_batches"],
        "optimizer_updates": metrics["runtime"]["optimizer_updates"],
        "epoch_accumulation": [
            {
                key: row[key]
                for key in (
                    "epoch",
                    "microbatches_seen",
                    "optimizer_steps",
                    "final_accumulation_count",
                )
            }
            for row in epochs
        ],
        "best_epoch": metrics["best_epoch"],
        "checkpoint_reload": metrics["checkpoint_reload"],
        "rollout3": metrics["rollout3"],
        "runtime": metrics["runtime"],
    }
    write_json(stage_r / "smoke_summary.json", payload)
    (stage_r / "smoke_summary.md").write_text(
        "\n".join(
            [
                "# Stage R two-epoch engineering smoke",
                "",
                f"- Passed: `{passed}`.",
                "- Complete-data count: `158` microbatches and `40` updates.",
                "- Each epoch used 79 pairs, 20 updates, and a final accumulation of 3.",
                "- Best/last strict reload and deterministic prediction parity: passed.",
                "- Three-step physical rollout: finite, positive, exact counters, no output clamp.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError("Stage R smoke summary gate failed")
    return payload


def training_outputs(stage_r: Path, experiment: Path) -> dict[str, Any]:
    metrics = read_json(experiment / "metrics.json")
    rows = [
        {
            "epoch": row["epoch"] + 1,
            "microbatches": row["microbatches_seen"],
            "optimizer_steps": row["optimizer_steps"],
            "final_accumulation_count": row["final_accumulation_count"],
            "train_total_mean": row["train_total"]["mean"],
            "train_total_median": row["train_total"]["median"],
            "train_total_q95": row["train_total"]["q95"],
            "train_total_maximum": row["train_total"]["maximum"],
            "validation_arithmetic_average": row[
                "validation_normalized_arithmetic_average_relative_l2"
            ],
            "validation_global": row["validation_normalized_global_relative_l2"],
            "clipping_fraction": row["clipping_fraction"],
            "clip_scale_mean": row["clip_scale"]["mean"],
            "parameter_displacement": row["parameter_displacement_from_initial"],
            "wall_seconds": row["wall_seconds"],
            "peak_allocated_mib": row["peak_allocated_mib"],
            "peak_reserved_mib": row["peak_reserved_mib"],
            "nonfinite_count": row["nonfinite_count"],
        }
        for row in metrics["epoch_summaries"]
    ]
    payload = {
        "schema_version": "paper-stage-r-training-summary-v1",
        "classification": "adapted_residual_contract_model_pilot",
        "status": metrics["status"],
        "epochs": metrics["epochs"],
        "microbatches": metrics["train_batches"],
        "optimizer_updates": metrics["runtime"]["optimizer_updates"],
        "best_epoch": metrics["best_epoch"],
        "best_validation_normalized_arithmetic_average_relative_l2": metrics[
            "best_validation_normalized_arithmetic_average_relative_l2"
        ],
        "runtime": metrics["runtime"],
        "checkpoint_reload": metrics["checkpoint_reload"],
        "initial_state_sha256": metrics["shared_initial_state_sha256"],
        "pair_order_sha256": metrics["pair_order_sha256"],
        "rows": rows,
    }
    if not (
        payload["epochs"] == 30
        and payload["microbatches"] == 2370
        and payload["optimizer_updates"] == 600
        and all(row["final_accumulation_count"] == 3 for row in rows)
    ):
        raise RuntimeError("Stage R formal training counts changed")
    write_json(stage_r / "training_summary.json", payload)
    write_csv(stage_r / "training_summary.csv", rows)
    (stage_r / "training_summary.md").write_text(
        "\n".join(
            [
                "# Stage R 30-epoch residual-target training",
                "",
                f"- Status/best epoch: `{payload['status']}` / `{payload['best_epoch']}`.",
                "- Counts: `2,370` microbatches and `600` optimizer updates.",
                f"- Best normalized average: `{payload['best_validation_normalized_arithmetic_average_relative_l2']:.9g}`.",
                f"- Clipping fraction: `{payload['runtime']['clipping_fraction']:.6f}`.",
                f"- Wall time: `{payload['runtime']['wall_seconds']:.3f} s`; peak allocated/reserved: "
                f"`{payload['runtime']['peak_allocated_mib']:.2f}/"
                f"{payload['runtime']['peak_reserved_mib']:.2f} MiB`.",
                "- Strict Plain L2 was applied to the unscaled P3 normalized residual target.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def validation_outputs(
    stage_r: Path, experiment: Path, evaluation: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    validation = evaluation["best"]["validation"]
    payload = {
        "schema_version": "paper-stage-r-validation-metrics-v1",
        "checkpoint": "best_validation_l2",
        "epoch": evaluation["best"]["reload"]["epoch"],
        "residual_reconstructed_model": validation["model"],
        "p3_persistence": validation["persistence"],
        "model_over_persistence_normalized_average": validation["model"]["metrics"][
            "E_norm"
        ]["arithmetic_average"]
        / validation["persistence"]["metrics"]["E_norm"]["arithmetic_average"],
        "saturation": validation["saturation"],
        "shell_wise_saturation": validation["shell_wise_saturation"],
        "constraints": validation["constraints"],
        "training_time_metric_parity": evaluation["best"][
            "training_time_metric_parity"
        ],
        "evaluation_rho_press_clamp": False,
    }
    write_json(stage_r / "validation_metrics.json", payload)
    write_csv(
        stage_r / "validation_metrics.csv",
        metric_rows(validation["model"]["metrics"], "stage_r_residual_model")
        + metric_rows(validation["persistence"]["metrics"], "p3_persistence"),
    )

    training_metrics = read_json(experiment / "metrics.json")
    residual = {
        "schema_version": "paper-stage-r-residual-metrics-v1",
        "checkpoint": "best_validation_l2",
        "pairs": 19,
        "target": "delta_z_true = z_t1 - z_t",
        "prediction": "raw LocalNO output r_theta",
        "zero_residual_is_persistence": True,
        **training_metrics["one_step"]["residual_metrics"],
    }
    write_json(stage_r / "residual_metrics.json", residual)
    write_csv(
        stage_r / "residual_metrics.csv",
        [
            {
                "channel": channel,
                "residual_relative_l2": residual["per_channel_relative_l2"][channel],
                "zero_residual_relative_l2": 1.0,
            }
            for channel in CHANNELS
        ],
    )

    frozen_o = read_json(stage_r.parent / "stage_o/validation_metrics.json")
    persistence_parity = all(
        validation["persistence"]["metrics"][name]["arithmetic_average"]
        == frozen_o["p3_persistence"]["metrics"][name]["arithmetic_average"]
        for name in ("E_norm", "E_model_oracle", "E_model_raw", "E_oracle_raw")
    )
    if not persistence_parity:
        raise RuntimeError("Stage R P3 persistence pipeline differs from frozen Stage O")
    persistence = {
        "schema_version": "paper-stage-r-p3-persistence-v1",
        "definition": "P3 oracle input at t held unchanged and compared with P3 oracle at t+1",
        "pairs": 19,
        "metrics": validation["persistence"],
        "stage_o_pipeline_parity": persistence_parity,
        "residual_zero_baseline": True,
    }
    write_json(stage_r / "p3_persistence.json", persistence)
    write_csv(
        stage_r / "p3_persistence.csv",
        metric_rows(validation["persistence"]["metrics"], "p3_persistence"),
    )
    return payload, residual


def rollout_outputs(stage_r: Path, experiment: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    gt = read_json(experiment / "gt_rollout.json")
    no_gt = read_json(experiment / "no_gt_rollout.json")
    gt["undefined_diagnostic_ratios_encoded_as_null"] = True
    no_gt["undefined_diagnostic_ratios_encoded_as_null"] = True
    write_json(stage_r / "rollout_gt.json", gt)
    write_json(stage_r / "rollout_no_gt.json", no_gt)
    write_csv(
        stage_r / "rollout_gt.csv",
        [
            {
                "step": row["step"],
                "finite": row["finite"],
                "rho_press_positive": row["rho_press_positive"],
                "prediction_global_norm": row["prediction_global_norm"],
                "predicted_residual_norm": row["predicted_residual_norm"],
                "residual_cosine": row["residual_skill"]["cosine"],
                "residual_sign_agreement": row["residual_skill"]["sign_agreement"],
                "normalized_average": row["oracle_aware"]["metrics"]["E_norm"][
                    "arithmetic_average"
                ],
                "persistence_normalized_average": row["persistence"]["metrics"][
                    "E_norm"
                ]["arithmetic_average"],
                "model_to_oracle_average": row["oracle_aware"]["metrics"][
                    "E_model_oracle"
                ]["arithmetic_average"],
                "model_to_raw_average": row["oracle_aware"]["metrics"][
                    "E_model_raw"
                ]["arithmetic_average"],
                "oracle_floor_average": row["oracle_aware"]["metrics"][
                    "E_oracle_raw"
                ]["arithmetic_average"],
                "above_rout": row["above_rout"],
                "artifact_count": len(row["artifacts"]["flags"]),
            }
            for row in gt["records"]
        ],
    )
    write_csv(
        stage_r / "rollout_no_gt.csv",
        [
            {
                "step": row["step"],
                "finite": row["finite"],
                "rho_press_positive": row["rho_press_positive"],
                "prediction_global_norm": row["prediction_global_norm"],
                "predicted_residual_norm": row["predicted_residual_norm"],
                "normalized_state_change_norm": row[
                    "normalized_state_change_norm"
                ],
                "above_rout": row["above_rout"],
                "artifact_count": len(row["artifacts"]["flags"]),
                "ground_truth_error_omitted": True,
            }
            for row in no_gt["records"]
        ],
    )
    return gt, no_gt


def legacy_and_gate_outputs(
    *,
    stage_r: Path,
    config: Any,
    validation: Mapping[str, Any],
    gt: Mapping[str, Any],
    no_gt: Mapping[str, Any],
) -> dict[str, Any]:
    gt_by_step = record_map(gt)
    no_gt_by_step = record_map(no_gt)
    legacy_rows = []
    for ground_truth, records in ((True, gt["records"]), (False, no_gt["records"])):
        for record in records:
            flags = list(record["artifacts"]["flags"])
            legacy_rows.append(
                {
                    "step": record["step"],
                    "ground_truth_available": ground_truth,
                    "flag_count": len(flags),
                    "flags": ";".join(flags),
                    "collapse_count": sum(
                        flag.endswith("possible_field_collapse") for flag in flags
                    ),
                    "ripple_count": sum(
                        flag.endswith("possible_high_frequency_ripple") for flag in flags
                    ),
                    "stripe_count": sum(
                        flag.endswith("possible_stripe_anisotropy") for flag in flags
                    ),
                }
            )
    legacy = {
        "schema_version": "paper-stage-r-legacy-detector-v1",
        "implementation": "frozen_stage_g_artifact_diagnostics",
        "rows": legacy_rows,
        "selected_gt": {
            str(step): gt_by_step[step]["artifacts"] for step in SELECTED_GT
        },
        "selected_no_gt": {
            str(step): no_gt_by_step[step]["artifacts"] for step in SELECTED_NO_GT
        },
    }
    write_json(stage_r / "legacy_detector.json", legacy)
    write_csv(stage_r / "legacy_detector.csv", legacy_rows)

    selected = torch.load(
        stage_r / "localno_p3_residual_plain/selected_states.pt",
        map_location="cpu",
        weights_only=True,
    )["selected_steps"]
    p3 = load_frozen_p3(config)
    parity_rows = {
        row["channel"]: row
        for row in read_json(stage_r.parent / "stage_o/run_manifest.json")[
            "p3_roundtrip_reproduction"
        ]["validation_rows"]
    }
    gate_channels = {}
    with h5py.File(config.resolve_path(config.values["protocol"]["dataset"]), "r") as handle:
        snapshots = handle["snapshots"]
        _, shell_index = radial_shell_indices(np.asarray(handle["coords/r"][...]), 8)
        for channel_name in TARGET_CHANNELS:
            channel = CHANNELS.index(channel_name)
            severe_by_step = {}
            transport_rows = []
            shell_skills: list[float] = []
            radial_skills: list[float] = []
            shell_signs: list[float] = []
            radial_signs: list[float] = []
            for step in SELECTED_GT:
                raw_input = np.asarray(snapshots[90 + step], dtype=np.float32)
                raw_target = np.asarray(snapshots[91 + step], dtype=np.float32)
                _, input_oracle = p3.round_trip(raw_input, channel_axis=0)
                _, target_oracle = p3.round_trip(raw_target, channel_axis=0)
                model_field = selected[step][0, channel].numpy()
                input_field = input_oracle[channel]
                target_field = target_oracle[channel]
                raw_basic = basic_field_metrics(target_field)
                model_basic = basic_field_metrics(model_field)
                raw_shell = variance_vector(target_field, shell_index)
                model_shell = variance_vector(model_field, shell_index)
                raw_radial = radial_profile(target_field)["variance"]
                model_radial = radial_profile(model_field)["variance"]
                raw_high = spectrum_metrics(target_field, axis="combined", demean=True)[
                    "high_k_energy"
                ]
                model_high = spectrum_metrics(model_field, axis="combined", demean=True)[
                    "high_k_energy"
                ]
                ratios = {
                    "global_variance": model_basic["variance"]
                    / max(raw_basic["variance"], 1e-30),
                    "shell_radial_variance": min(
                        float(np.median(model_shell / np.maximum(raw_shell, 1e-30))),
                        model_radial / max(raw_radial, 1e-30),
                    ),
                    "dynamic_span": model_basic["dynamic_span_q99_q01"]
                    / max(raw_basic["dynamic_span_q99_q01"], 1e-30),
                    "high_k_energy": model_high / max(raw_high, 1e-30),
                }
                severe_by_step[step] = {
                    name: value < 0.5 for name, value in ratios.items()
                }
                shell = transport_metrics(
                    variance_vector(input_field, shell_index),
                    variance_vector(target_field, shell_index),
                    variance_vector(model_field, shell_index),
                    epsilon=1e-30,
                    sign_zero_tolerance=1e-12,
                )
                radial = transport_metrics(
                    radial_profile_vector(input_field),
                    radial_profile_vector(target_field),
                    radial_profile_vector(model_field),
                    epsilon=1e-30,
                    sign_zero_tolerance=1e-12,
                )
                for value, destination in (
                    (shell["persistence_relative_skill"], shell_skills),
                    (radial["persistence_relative_skill"], radial_skills),
                    (shell["signed_transport_agreement"], shell_signs),
                    (radial["signed_transport_agreement"], radial_signs),
                ):
                    if value is not None:
                        destination.append(float(value))
                transport_rows.append(
                    {"step": step, "ratios": ratios, "shell": shell, "radial": radial}
                )
            channel_flags = [
                flag
                for row in legacy_rows
                for flag in row["flags"].split(";")
                if flag.startswith(f"{channel_name}:")
            ]
            floor = parity_rows[channel_name]
            gate = evaluate_oracle_conditioned_structure_gate(
                legacy_detector={"flags": channel_flags},
                engineering_checks={
                    "finite": bool(gt["finite"] and no_gt["finite"]),
                    "rho_press_positive": bool(
                        gt["rho_press_positive"] and no_gt["rho_press_positive"]
                    ),
                    "transform_counters": no_gt["transform_count_delta"]
                    == {
                        "input_encode": 100,
                        "target_encode": 19,
                        "oracle_decode": 19,
                        "prediction_decode": 100,
                    },
                    "checkpoint_provenance": True,
                    "decoded_range": False,
                    "Rout": validation["constraints"][
                        "prediction_above_rout_fraction"
                    ]
                    == 0.0,
                    "shape_device": True,
                },
                floor_metrics={
                    "oracle_legacy_detector_triggered": bool(
                        floor["median_global_variance_retention"] < 0.0025
                    ),
                    "median_variance_retention": floor[
                        "median_global_variance_retention"
                    ],
                    "median_shell_radial_retention": floor[
                        "median_shell_radial_variance_retention"
                    ],
                    "median_high_k_retention": floor["median_high_k_retention"],
                },
                severe_by_step=severe_by_step,
                shell_transport_skill=(
                    None if not shell_skills else float(np.median(shell_skills))
                ),
                radial_transport_skill=(
                    None if not radial_skills else float(np.median(radial_skills))
                ),
                shell_sign_agreement=(
                    None if not shell_signs else float(np.median(shell_signs))
                ),
                radial_sign_agreement=(
                    None if not radial_signs else float(np.median(radial_signs))
                ),
            )
            gate_channels[channel_name] = {
                "gate": gate,
                "selected_gt_transport": transport_rows,
            }
    payload = {
        "schema_version": "paper-stage-r-stage-m-v1-gate-v1",
        "oracle_conditioned_gate_version": "stage_m_v1",
        "reporting_only": True,
        "channels": gate_channels,
        "no_gt_gate_3_omitted": True,
        "engineering_interpretation": (
            "finite/positivity/counters/checkpoint pass; decoded range and Rout fail"
        ),
    }
    rows = []
    for channel, item in gate_channels.items():
        gate = item["gate"]
        rows.append(
            {
                "channel": channel,
                "gate0_passed": gate["gate_0_engineering_validity"]["passed"],
                "gate1_floor_limited": gate["gate_1_floor_qualification"][
                    "floor_limited"
                ],
                "gate2_failed": gate["gate_2_model_added_degradation"]["failed"],
                "gate2_failure_steps": ";".join(
                    str(value)
                    for value in gate["gate_2_model_added_degradation"][
                        "failure_steps"
                    ]
                ),
                "gate3_passed": gate["gate_3_transport_skill"]["passed"],
                "shell_skill": gate["gate_3_transport_skill"]["shell_skill"],
                "radial_skill": gate["gate_3_transport_skill"]["radial_skill"],
                "shell_sign_agreement": gate["gate_3_transport_skill"][
                    "shell_sign_agreement"
                ],
                "radial_sign_agreement": gate["gate_3_transport_skill"][
                    "radial_sign_agreement"
                ],
            }
        )
    write_json(stage_r / "stage_m_v1_gate.json", payload)
    write_csv(stage_r / "stage_m_v1_gate.csv", rows)
    (stage_r / "stage_m_v1_gate.md").write_text(
        "\n".join(
            [
                "# Stage R legacy detector and stage_m_v1",
                "",
                "- Gate 0 retains finite/positivity/counter/checkpoint success but fails decoded range and frozen Rout.",
                "- Gate 1 uses the frozen P3 raw-to-oracle floor.",
                "- Gate 2 measures model-added structural degradation at GT steps.",
                "- Gate 3 measures shell/radial persistence-relative transport; no-GT Gate 3 is omitted.",
                "- Legacy and stage_m_v1 remain side by side and do not rewrite Stage K–Q.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def comparison_outputs(
    stage_r: Path,
    validation: Mapping[str, Any],
    residual: Mapping[str, Any],
    gt: Mapping[str, Any],
    no_gt: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    stage_o_validation = read_json(stage_r.parent / "stage_o/validation_metrics.json")
    stage_o_gt = read_json(stage_r.parent / "stage_o/rollout_gt.json")
    stage_o_no_gt = read_json(stage_r.parent / "stage_o/rollout_no_gt.json")
    stage_o_gate = read_json(stage_r.parent / "stage_o/stage_m_v1_gate.json")
    o = stage_o_validation["p3_model"]["metrics"]
    r = validation["residual_reconstructed_model"]["metrics"]
    p = validation["p3_persistence"]["metrics"]
    rows = [
        {
            "channel": channel,
            "stage_o_direct_normalized_l2": o["E_norm"]["per_channel"][channel],
            "stage_r_residual_normalized_l2": r["E_norm"]["per_channel"][channel],
            "p3_persistence_normalized_l2": p["E_norm"]["per_channel"][channel],
            "stage_r_over_stage_o": r["E_norm"]["per_channel"][channel]
            / max(o["E_norm"]["per_channel"][channel], 1e-30),
            "stage_r_over_persistence": r["E_norm"]["per_channel"][channel]
            / max(p["E_norm"]["per_channel"][channel], 1e-30),
            "stage_o_model_to_oracle": o["E_model_oracle"]["per_channel"][channel],
            "stage_r_model_to_oracle": r["E_model_oracle"]["per_channel"][channel],
            "stage_r_residual_relative_l2": residual["per_channel_relative_l2"][
                channel
            ],
        }
        for channel in CHANNELS
    ]
    o_gt = record_map(stage_o_gt)
    r_gt = record_map(gt)
    o_no = record_map(stage_o_no_gt)
    r_no = record_map(no_gt)
    selected_rollout = {
        str(step): {
            "stage_o_normalized_average": o_gt[step]["oracle_aware"]["metrics"][
                "E_norm"
            ]["arithmetic_average"],
            "stage_r_normalized_average": r_gt[step]["oracle_aware"]["metrics"][
                "E_norm"
            ]["arithmetic_average"],
            "stage_o_prediction_norm": o_gt[step]["prediction_global_norm"],
            "stage_r_prediction_norm": r_gt[step]["prediction_global_norm"],
            "stage_o_above_rout": o_gt[step]["above_rout"],
            "stage_r_above_rout": r_gt[step]["above_rout"],
            "stage_r_residual_cosine": r_gt[step]["residual_skill"]["cosine"],
        }
        for step in SELECTED_GT
    }
    selected_no_gt = {
        str(step): {
            "stage_o_prediction_norm": o_no[step]["prediction_global_norm"],
            "stage_r_prediction_norm": r_no[step]["prediction_global_norm"],
            "stage_o_artifact_flags": o_no[step]["artifacts"]["flags"],
            "stage_r_artifact_flags": r_no[step]["artifacts"]["flags"],
        }
        for step in SELECTED_NO_GT
    }
    gate_comparison = {
        channel: {
            "stage_o": stage_o_gate["channels"][channel]["gate"],
            "stage_r": gates["channels"][channel]["gate"],
        }
        for channel in TARGET_CHANNELS
    }
    model_persistence_ratio = r["E_norm"]["arithmetic_average"] / p["E_norm"][
        "arithmetic_average"
    ]
    payload = {
        "schema_version": "paper-stage-r-stage-o-comparison-v1",
        "paired_conditions": {
            "architecture": True,
            "initial_state": True,
            "pair_order": True,
            "training_budget": True,
            "optimizer_scheduler": True,
            "p3_transform_statistics": True,
            "only_training_factor": "direct state versus normalized residual target",
            "physical_output_repair": False,
        },
        "rows": rows,
        "aggregate": {
            "stage_o_normalized_average": o["E_norm"]["arithmetic_average"],
            "stage_r_normalized_average": r["E_norm"]["arithmetic_average"],
            "p3_persistence_normalized_average": p["E_norm"]["arithmetic_average"],
            "stage_r_over_stage_o": r["E_norm"]["arithmetic_average"]
            / o["E_norm"]["arithmetic_average"],
            "stage_r_over_persistence": model_persistence_ratio,
            "stage_o_model_to_oracle_average": o["E_model_oracle"][
                "arithmetic_average"
            ],
            "stage_r_model_to_oracle_average": r["E_model_oracle"][
                "arithmetic_average"
            ],
            "stage_r_validation_above_rout_fraction": validation["constraints"][
                "prediction_above_rout_fraction"
            ],
            "residual_cosine_mean": residual["residual_cosine"]["mean"],
            "residual_sign_agreement_mean": residual["residual_sign_agreement"][
                "mean"
            ],
            "residual_relative_l2_average": residual[
                "arithmetic_average_relative_l2"
            ],
        },
        "selected_gt_rollout": selected_rollout,
        "selected_no_gt_rollout": selected_no_gt,
        "gate_comparison": gate_comparison,
        "answers": [
            {"question": "step-1 normalized overshoot reduced", "status": "observed", "answer": r_gt[1]["oracle_aware"]["metrics"]["E_norm"]["arithmetic_average"] < o_gt[1]["oracle_aware"]["metrics"]["E_norm"]["arithmetic_average"]},
            {"question": "Rout failure reduced", "status": "observed", "answer": False, "evidence": "both Stage O and Stage R are above Rout"},
            {"question": "P3 decoder-tail exposure reduced", "status": "observed", "answer": "partially_at_step1_but_not_closed_loop", "evidence": "physical ranges explode by step 3 and reach transform limits"},
            {"question": "Gate 2 improved", "status": "observed", "answer": "see gate_comparison"},
            {"question": "Gate 3 improved", "status": "observed", "answer": "see gate_comparison"},
            {"question": "beats P3 persistence", "status": "observed", "answer": model_persistence_ratio < 1.0, "ratio": model_persistence_ratio},
            {"question": "correct residual direction", "status": "observed", "answer": "positive_on_average_but_reverses_after_step1", "evidence": {"validation_mean_cosine": residual["residual_cosine"]["mean"], "selected_gt_cosines": {str(step): r_gt[step]["residual_skill"]["cosine"] for step in SELECTED_GT}}},
            {"question": "range explosion remains", "status": "observed", "answer": True},
            {"question": "improvement is persistence-like conservatism", "status": "inference", "answer": True, "evidence": {"residual_over_true_mean": residual["residual_over_true_residual"]["mean"], "model_over_persistence": model_persistence_ratio}},
            {"question": "shell/radial transport genuinely improved", "status": "observed", "answer": "see Gate 3; positive skill is required for a yes"},
            {"question": "unique causal operator defect identified", "status": "unsupported", "answer": False},
        ],
        "comparison_limit": (
            "Stage O frozen evaluation used its authorized rho/press evaluation clamp; "
            "Stage R forbids all physical output repair. Normalized and structural comparisons "
            "remain directly paired; repaired physical ranges are not treated as exact peers."
        ),
    }
    write_json(stage_r / "stage_o_stage_r_comparison.json", payload)
    write_csv(stage_r / "stage_o_stage_r_comparison.csv", rows)
    (stage_r / "stage_o_stage_r_comparison.md").write_text(
        "\n".join(
            [
                "# Stage O direct-state versus Stage R residual-target LocalNO",
                "",
                f"- One-step normalized average: Stage O `{payload['aggregate']['stage_o_normalized_average']:.6g}`, Stage R `{payload['aggregate']['stage_r_normalized_average']:.6g}`, P3 persistence `{payload['aggregate']['p3_persistence_normalized_average']:.6g}`.",
                f"- Stage R / Stage O: `{payload['aggregate']['stage_r_over_stage_o']:.6g}`; Stage R / persistence: `{model_persistence_ratio:.6g}`.",
                f"- Mean residual cosine/sign agreement: `{payload['aggregate']['residual_cosine_mean']:.6g}` / `{payload['aggregate']['residual_sign_agreement_mean']:.6g}`.",
                "- Residual targeting sharply reduces direct-state normalized overshoot, but all validation pairs remain above Rout and physical ranges explode by GT step 3.",
                "- The long rollout remains finite and positive but carries ripple/stripe flags and extreme transform-tail exposure.",
                f"- Limitation: {payload['comparison_limit']}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def decision_output(
    stage_r: Path,
    training: Mapping[str, Any],
    validation: Mapping[str, Any],
    residual: Mapping[str, Any],
    comparison: Mapping[str, Any],
    gt: Mapping[str, Any],
    no_gt: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    gate_rows = {
        channel: item["gate"] for channel, item in gates["channels"].items()
    }
    payload = {
        "schema_version": "paper-stage-r-decision-v1",
        "choice": "C",
        "decision": "RESIDUAL_PILOT_COMPLETE_BUT_ROLLOUT_UNSTABLE",
        "engineering_evidence": {
            "preflight_passed": True,
            "smoke_passed": True,
            "epochs": training["epochs"],
            "microbatches": training["microbatches"],
            "optimizer_updates": training["optimizer_updates"],
            "strict_best_last_reload": True,
            "training_nonfinite": training["runtime"]["nonfinite_count"],
            "gt_rollout_finite_positive": bool(
                gt["finite"] and gt["rho_press_positive"]
            ),
            "no_gt_rollout_finite_positive": bool(
                no_gt["finite"] and no_gt["rho_press_positive"]
            ),
            "transform_counters": no_gt["transform_count_delta"],
            "loss_equivalence": True,
            "direct_residual_contract_mix": False,
        },
        "skill_evidence": {
            "stage_r_over_stage_o_one_step": comparison["aggregate"][
                "stage_r_over_stage_o"
            ],
            "stage_r_over_persistence_one_step": comparison["aggregate"][
                "stage_r_over_persistence"
            ],
            "residual_cosine_mean": residual["residual_cosine"]["mean"],
            "residual_relative_l2_average": residual[
                "arithmetic_average_relative_l2"
            ],
            "zero_residual_relative_l2": residual["zero_residual_relative_l2"],
            "validation_above_rout_fraction": validation["constraints"][
                "prediction_above_rout_fraction"
            ],
            "selected_gt_residual_cosines": {
                str(step): record_map(gt)[step]["residual_skill"]["cosine"]
                for step in SELECTED_GT
            },
        },
        "instability_evidence": {
            "range_explosion_by_gt_step": 3,
            "selected_gt_physical_ranges": {
                str(step): record_map(gt)[step]["physical_range"]
                for step in SELECTED_GT
            },
            "step100_prediction_norm": record_map(no_gt)[100][
                "prediction_global_norm"
            ],
            "step100_artifact_flags": record_map(no_gt)[100]["artifacts"][
                "flags"
            ],
            "stage_m_v1": gate_rows,
        },
        "interpretation": (
            "Residual targeting substantially reduces Stage O normalized overshoot and keeps "
            "the 100-step normalized norm far smaller, but it remains worse than zero-residual "
            "P3 persistence, residual direction reverses after step 1, every validation pair is "
            "above Rout, and decoded physical ranges explode by step 3. This is scientific/closed-"
            "loop instability, not a residual-contract engineering failure."
        ),
        "historical_status_unchanged": read_json(stage_r / "run_manifest.json")[
            "historical_status_unchanged"
        ],
        "next_authorized_action": (
            "propose_only: diagnose why unscaled normalized residuals enter P3 inverse tails "
            "and why residual transport direction reverses; no new training is authorized"
        ),
    }
    write_json(stage_r / "stage_r_decision.json", payload)
    (stage_r / "stage_r_decision.md").write_text(
        "\n".join(
            [
                "# Stage R decision",
                "",
                "## C. RESIDUAL_PILOT_COMPLETE_BUT_ROLLOUT_UNSTABLE",
                "",
                "The residual contract, preflight, complete-data smoke, 30 epochs, 2,370 microbatches, 600 updates, strict reload, and exact 19/100-step transform counters passed. All rollout states remained finite and rho/press positive.",
                "",
                f"Stage R reduced the Stage O normalized one-step average by a factor of `{comparison['aggregate']['stage_r_over_stage_o']:.6g}`, but remained `{comparison['aggregate']['stage_r_over_persistence']:.6g}x` P3 persistence. Mean validation residual cosine was `{residual['residual_cosine']['mean']:.6g}`, while selected closed-loop cosine turned negative after step 1.",
                "",
                "All validation predictions remained above frozen Rout. Physical ranges exploded by GT step 3, and step 100 retained widespread high-frequency artifact flags. Therefore the model is not stable enough for choice A or B; the engineering contract itself did not fail, so choice D is not applicable.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    stage_r = root / "outputs/paper_reduced100/stage_r"
    experiment = stage_r / "localno_p3_residual_plain"
    config = load_paper_experiment_config(
        root / "configs/paper_reduced100/stage_r_localno_p3_residual_plain.yaml",
        project_root=root,
    )
    smoke_outputs(stage_r)
    training = training_outputs(stage_r, experiment)
    evaluation = read_json(experiment / "evaluation_summary.json")
    validation, residual = validation_outputs(stage_r, experiment, evaluation)
    gt, no_gt = rollout_outputs(stage_r, experiment)
    gates = legacy_and_gate_outputs(
        stage_r=stage_r,
        config=config,
        validation=validation,
        gt=gt,
        no_gt=no_gt,
    )
    comparison = comparison_outputs(
        stage_r, validation, residual, gt, no_gt, gates
    )
    decision = decision_output(
        stage_r, training, validation, residual, comparison, gt, no_gt, gates
    )
    print(json.dumps({"decision": decision["decision"]}, indent=2))


if __name__ == "__main__":
    main()
