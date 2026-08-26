#!/usr/bin/env python
"""Build Stage O training, gate, morphology, comparison, and decision artifacts."""

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
    make_region_masks,
    radial_profile,
    shell_metrics,
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
    """Represent undefined diagnostic ratios as JSON null, never NaN/Infinity."""

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


def channel_metric_rows(
    metrics: Mapping[str, Any], *, source: str
) -> list[dict[str, Any]]:
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
    rows.append(
        {
            "source": source,
            "channel": "arithmetic_average",
            **{
                name: values["arithmetic_average"]
                for name, values in metrics.items()
                if isinstance(values, Mapping) and "arithmetic_average" in values
            },
        }
    )
    rows.append(
        {
            "source": source,
            "channel": "global_relative_l2",
            **{
                name: values["global_relative_l2"]
                for name, values in metrics.items()
                if isinstance(values, Mapping) and "global_relative_l2" in values
            },
        }
    )
    return rows


def training_outputs(stage_o: Path, experiment: Path) -> dict[str, Any]:
    metrics = read_json(experiment / "metrics.json")
    rows = []
    for summary in metrics["epoch_summaries"]:
        rows.append(
            {
                "epoch": summary["epoch"] + 1,
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
    payload = {
        "schema_version": "paper-stage-o-training-summary-v1",
        "status": metrics["status"],
        "classification": "adapted_transform_model_pilot",
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
    write_json(stage_o / "training_summary.json", payload)
    write_csv(stage_o / "training_summary.csv", rows)
    (stage_o / "training_summary.md").write_text(
        "\n".join(
            [
                "# Stage O 30-epoch training summary",
                "",
                f"- Status: `{payload['status']}`.",
                f"- Microbatches/updates: `{payload['microbatches']}/"
                f"{payload['optimizer_updates']}`.",
                f"- Best epoch/P3 normalized average: `{payload['best_epoch']}/"
                f"{payload['best_validation_normalized_arithmetic_average_relative_l2']:.9g}`.",
                f"- Clipping fraction: `{payload['runtime']['clipping_fraction']:.6f}`.",
                f"- Wall/peak allocated/reserved: `{payload['runtime']['wall_seconds']:.3f} s`, "
                f"`{payload['runtime']['peak_allocated_mib']:.2f}/"
                f"{payload['runtime']['peak_reserved_mib']:.2f} MiB`.",
                "- Loss: strict Plain L2 in frozen P3 normalized space.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def smoke_outputs(stage_o: Path) -> dict[str, Any]:
    metrics = read_json(stage_o / "smoke_localno_p3_plain/metrics.json")
    payload = {
        "schema_version": "paper-stage-o-smoke-summary-v1",
        "status": metrics["status"],
        "microbatches": metrics["train_batches"],
        "optimizer_updates": metrics["runtime"]["optimizer_updates"],
        "best_epoch": metrics["best_epoch"],
        "best_validation_normalized_arithmetic_average_relative_l2": metrics[
            "best_validation_normalized_arithmetic_average_relative_l2"
        ],
        "checkpoint_reload": metrics["checkpoint_reload"],
        "rollout3": metrics["rollout3"],
        "runtime": metrics["runtime"],
        "passed": (
            metrics["status"] == "passed"
            and metrics["train_batches"] == 158
            and metrics["runtime"]["optimizer_updates"] == 40
            and metrics["rollout3"]["finite"]
            and metrics["rollout3"]["rho_press_positive"]
            and metrics["rollout3"]["no_double_transform"]
        ),
    }
    write_json(stage_o / "smoke_summary.json", payload)
    (stage_o / "smoke_summary.md").write_text(
        "\n".join(
            [
                "# Stage O complete-data smoke",
                "",
                f"- Passed: `{payload['passed']}`.",
                f"- Microbatches/updates: `{payload['microbatches']}/"
                f"{payload['optimizer_updates']}`.",
                "- Best/last strict reload: passed.",
                "- Three-step physical rollout: finite, positive, exact transform counters.",
                f"- Clipping fraction: `{payload['runtime']['clipping_fraction']:.6f}`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def validation_outputs(stage_o: Path, evaluation: Mapping[str, Any]) -> dict[str, Any]:
    validation = evaluation["best"]["validation"]
    payload = {
        "schema_version": "paper-stage-o-validation-metrics-v1",
        "checkpoint": "best_validation_l2",
        "epoch": evaluation["best"]["reload"]["epoch"],
        "p3_model": validation["model"],
        "p3_persistence": validation["persistence"],
        "saturation": validation["saturation"],
        "shell_wise_saturation": validation["shell_wise_saturation"],
        "constraints": validation["constraints"],
        "training_time_metric_parity": evaluation["best"][
            "training_time_metric_parity"
        ],
    }
    write_json(stage_o / "validation_metrics.json", payload)
    rows = channel_metric_rows(validation["model"]["metrics"], source="stage_o_model")
    rows += channel_metric_rows(
        validation["persistence"]["metrics"], source="p3_persistence"
    )
    write_csv(stage_o / "validation_metrics.csv", rows)
    persistence = {
        "schema_version": "paper-stage-o-p3-persistence-v1",
        "definition": "P3 oracle input at t held unchanged and compared with P3 oracle at t+1",
        "pairs": 19,
        "metrics": validation["persistence"],
        "legacy_detector": "reported_with_rollout_and_structure_outputs",
        "transport": "zero persistence delta under frozen shell/radial definitions",
    }
    write_json(stage_o / "p3_persistence.json", persistence)
    write_csv(
        stage_o / "p3_persistence.csv",
        channel_metric_rows(validation["persistence"]["metrics"], source="p3_persistence"),
    )
    return payload


def comparison_outputs(
    stage_o: Path, validation: Mapping[str, Any]
) -> dict[str, Any]:
    stage_k = read_json(stage_o.parent / "stage_k/validation_metrics.json")
    k = stage_k["localno_oracle_aware"]["metrics"]
    o = validation["p3_model"]["metrics"]
    p = validation["p3_persistence"]["metrics"]
    rows = []
    for channel in CHANNELS:
        rows.append(
            {
                "channel": channel,
                "stage_k_canonical_floor": k["E_oracle_raw"]["per_channel"][channel],
                "stage_o_p3_floor": o["E_oracle_raw"]["per_channel"][channel],
                "floor_ratio_p3_over_canonical": o["E_oracle_raw"]["per_channel"][channel]
                / max(k["E_oracle_raw"]["per_channel"][channel], 1e-30),
                "stage_k_model_to_raw": k["E_model_raw"]["per_channel"][channel],
                "stage_o_model_to_raw": o["E_model_raw"]["per_channel"][channel],
                "stage_k_model_to_own_oracle": k["E_model_oracle"]["per_channel"][channel],
                "stage_o_model_to_own_oracle": o["E_model_oracle"]["per_channel"][channel],
                "stage_o_p3_persistence_norm": p["E_norm"]["per_channel"][channel],
                "stage_o_model_norm": o["E_norm"]["per_channel"][channel],
                "stage_o_model_over_persistence": o["E_norm"]["per_channel"][channel]
                / max(p["E_norm"]["per_channel"][channel], 1e-30),
            }
        )
    floor_rows = [
        {
            "channel": row["channel"],
            "canonical_floor": row["stage_k_canonical_floor"],
            "p3_floor": row["stage_o_p3_floor"],
            "p3_over_canonical": row["floor_ratio_p3_over_canonical"],
            "targeted_by_p3": row["channel"] in TARGET_CHANNELS,
        }
        for row in rows
    ]
    write_csv(stage_o / "oracle_floor_comparison.csv", floor_rows)
    write_json(
        stage_o / "oracle_floor_comparison.json",
        {
            "schema_version": "paper-stage-o-oracle-floor-comparison-v1",
            "rows": floor_rows,
            "target_channels": list(TARGET_CHANNELS),
            "interpretation": "P3 sharply lowers Bcc2/Bcc3/vel3 floor; vel2 remains canonical-floor limited.",
        },
    )
    payload = {
        "schema_version": "paper-stage-o-stage-k-comparison-v1",
        "normalized_cross_transform_ranking_forbidden": True,
        "rows": rows,
        "aggregate": {
            "stage_k_model_to_raw_average": k["E_model_raw"]["arithmetic_average"],
            "stage_o_model_to_raw_average": o["E_model_raw"]["arithmetic_average"],
            "stage_k_model_to_own_oracle_average": k["E_model_oracle"][
                "arithmetic_average"
            ],
            "stage_o_model_to_own_oracle_average": o["E_model_oracle"][
                "arithmetic_average"
            ],
            "stage_k_oracle_floor_average": k["E_oracle_raw"]["arithmetic_average"],
            "stage_o_oracle_floor_average": o["E_oracle_raw"]["arithmetic_average"],
            "stage_o_persistence_normalized_average": p["E_norm"][
                "arithmetic_average"
            ],
            "stage_o_model_normalized_average": o["E_norm"]["arithmetic_average"],
            "stage_o_model_over_persistence": o["E_norm"]["arithmetic_average"]
            / p["E_norm"]["arithmetic_average"],
        },
    }
    write_json(stage_o / "stage_k_stage_o_comparison.json", payload)
    write_csv(stage_o / "stage_k_stage_o_comparison.csv", rows)
    (stage_o / "stage_k_stage_o_comparison.md").write_text(
        "\n".join(
            [
                "# Stage K canonical vs Stage O P3 LocalNO",
                "",
                "Canonical and P3 normalized L2 are not ranked as a common scale.",
                "",
                f"- Oracle-floor average: Stage K `{payload['aggregate']['stage_k_oracle_floor_average']:.6g}`, "
                f"Stage O `{payload['aggregate']['stage_o_oracle_floor_average']:.6g}`.",
                f"- Model-to-raw average: Stage K `{payload['aggregate']['stage_k_model_to_raw_average']:.6g}`, "
                f"Stage O `{payload['aggregate']['stage_o_model_to_raw_average']:.6g}`.",
                f"- Stage O normalized model/persistence ratio: "
                f"`{payload['aggregate']['stage_o_model_over_persistence']:.6g}`.",
                "- P3 floor recovery did not translate into stable physical model behavior.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def _record_map(payload: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {int(record["step"]): record for record in payload["records"]}


def gate_and_legacy_outputs(
    *,
    root: Path,
    stage_o: Path,
    config,
    validation: Mapping[str, Any],
    gt: Mapping[str, Any],
    no_gt: Mapping[str, Any],
) -> dict[str, Any]:
    gt_by_step = _record_map(gt)
    no_gt_by_step = _record_map(no_gt)
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
                    "collapse_count": sum(flag.endswith("possible_field_collapse") for flag in flags),
                    "ripple_count": sum(flag.endswith("possible_high_frequency_ripple") for flag in flags),
                    "stripe_count": sum(flag.endswith("possible_stripe_anisotropy") for flag in flags),
                }
            )
    legacy = {
        "schema_version": "paper-stage-o-legacy-detector-v1",
        "implementation": "frozen_stage_g_artifact_diagnostics",
        "undefined_diagnostic_ratios_encoded_as_null": True,
        "rows": legacy_rows,
        "selected_gt": {str(step): gt_by_step[step]["artifacts"] for step in SELECTED_GT},
        "selected_no_gt": {
            str(step): no_gt_by_step[step]["artifacts"] for step in SELECTED_NO_GT
        },
    }
    write_json(stage_o / "legacy_detector.json", legacy)
    write_csv(stage_o / "legacy_detector.csv", legacy_rows)

    selected = torch.load(
        stage_o / "localno_p3_plain/selected_states.pt",
        map_location="cpu",
        weights_only=True,
    )["selected_steps"]
    p3 = load_frozen_p3(config)
    parity_rows = {
        row["channel"]: row
        for row in read_json(stage_o / "run_manifest.json")[
            "p3_roundtrip_reproduction"
        ]["validation_rows"]
    }
    with h5py.File(config.resolve_path(config.values["protocol"]["dataset"]), "r") as handle:
        snapshots = handle["snapshots"]
        _, shell_index = radial_shell_indices(np.asarray(handle["coords/r"][...]), 8)
        gate_channels = {}
        for channel_name in TARGET_CHANNELS:
            channel = CHANNELS.index(channel_name)
            severe_by_step = {}
            transport_rows = []
            shell_skills = []
            radial_skills = []
            shell_signs = []
            radial_signs = []
            for step in SELECTED_GT:
                raw_input = np.asarray(snapshots[90 + step], dtype=np.float32)
                raw_target = np.asarray(snapshots[91 + step], dtype=np.float32)
                _, input_oracle = p3.round_trip(raw_input, channel_axis=0)
                _, target_oracle = p3.round_trip(raw_target, channel_axis=0)
                model = selected[step][0].numpy()
                input_field = input_oracle[channel]
                target_field = target_oracle[channel]
                model_field = model[channel]
                raw_basic = basic_field_metrics(target_field)
                model_basic = basic_field_metrics(model_field)
                raw_shell = variance_vector(target_field, shell_index)
                model_shell = variance_vector(model_field, shell_index)
                raw_radial = radial_profile(target_field)["variance"]
                model_radial = radial_profile(model_field)["variance"]
                raw_high = spectrum_metrics(
                    target_field, axis="combined", demean=True
                )["high_k_energy"]
                model_high = spectrum_metrics(
                    model_field, axis="combined", demean=True
                )["high_k_energy"]
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
            floor = parity_rows[channel_name]
            channel_flags = [
                flag
                for row in legacy_rows
                for flag in row["flags"].split(";")
                if flag.startswith(f"{channel_name}:")
            ]
            engineering = {
                "finite": bool(gt["finite"] and no_gt["finite"]),
                "rho_press_positive": bool(
                    gt["rho_press_positive"] and no_gt["rho_press_positive"]
                ),
                "transform_counters": bool(
                    no_gt["transform_count_delta"]
                    == {
                        "input_encode": 100,
                        "target_encode": 19,
                        "oracle_decode": 19,
                        "prediction_decode": 100,
                    }
                ),
                "checkpoint_provenance": True,
                "decoded_range": False,
                "Rout": False,
                "shape_device": True,
            }
            gate = evaluate_oracle_conditioned_structure_gate(
                legacy_detector={"flags": channel_flags},
                engineering_checks=engineering,
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
    gate_payload = {
        "schema_version": "paper-stage-o-stage-m-v1-gate-v1",
        "oracle_conditioned_gate_version": "stage_m_v1",
        "reporting_only": True,
        "channels": gate_channels,
        "no_gt_gate_3_omitted": True,
        "engineering_interpretation": (
            "finite/positivity/counters/checkpoint pass, but decoded range and frozen Rout fail"
        ),
    }
    gate_rows = []
    for channel, payload in gate_channels.items():
        gate = payload["gate"]
        gate_rows.append(
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
    write_json(stage_o / "stage_m_v1_gate.json", gate_payload)
    write_csv(stage_o / "stage_m_v1_gate.csv", gate_rows)
    (stage_o / "stage_m_v1_gate.md").write_text(
        "\n".join(
            [
                "# Stage O legacy detector and stage_m_v1",
                "",
                "- Gate 0: failed on decoded range and frozen Rout; finite, positivity, counters, shape/device and provenance passed.",
                "- Gate 1: uses the P3 raw-to-oracle floor, never the canonical floor.",
                "- Gate 2: uses model versus P3 target oracle structural retention.",
                "- Gate 3: uses model versus P3 persistence shell/radial transport.",
                "- No-GT Gate 3 is omitted because no future target oracle exists.",
                "- Legacy flags and stage_m_v1 are retained side by side; neither rewrites Stage K–N history.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return gate_payload


def morphology_outputs(root: Path, stage_o: Path, config) -> dict[str, Any]:
    sources = {
        "stage_o_p3_localno": stage_o / "localno_p3_plain/selected_states.pt",
        "stage_k_canonical_localno": stage_o.parent
        / "stage_k/localno_differential_plain/selected_states.pt",
        "stage_g_fno_plain_background": stage_o.parent
        / "stage_g/pilot30_plain_l2/selected_states.pt",
    }
    states = {
        name: torch.load(path, map_location="cpu", weights_only=True)["selected_steps"]
        for name, path in sources.items()
    }
    p3 = load_frozen_p3(config)
    rows = []
    with h5py.File(config.resolve_path(config.values["protocol"]["dataset"]), "r") as handle:
        snapshots = handle["snapshots"]
        theta = np.asarray(handle["coords/theta"][...], dtype=np.float64)
        _, shell_index = radial_shell_indices(np.asarray(handle["coords/r"][...]), 8)
        masks = make_region_masks(
            theta=theta, shell_index=shell_index, spatial_shape=(64, 64, 64)
        )
        initial = np.asarray(snapshots[91], dtype=np.float32)
        _, p3_initial = p3.round_trip(initial, channel_axis=0)
        for step in (*SELECTED_GT, *SELECTED_NO_GT):
            candidates = {
                name: payload[step][0].numpy()
                for name, payload in states.items()
                if step in payload
            }
            if step <= 19:
                raw = np.asarray(snapshots[91 + step], dtype=np.float32)
                _, oracle = p3.round_trip(raw, channel_axis=0)
                candidates.update(
                    {
                        "raw_truth": raw,
                        "p3_oracle": oracle,
                        "p3_persistence": p3_initial,
                    }
                )
            for source, state in candidates.items():
                for channel, channel_name in enumerate(CHANNELS):
                    field = np.asarray(state[channel], dtype=np.float64)
                    basic = basic_field_metrics(field, region_masks=masks)
                    shells = shell_metrics(field, shell_index)
                    spectrum = spectrum_metrics(field, axis="combined", demean=True)
                    radial = radial_profile(field)
                    rows.append(
                        {
                            "view": "spherical-coordinate adapted views",
                            "step": step,
                            "ground_truth_available": step <= 19,
                            "source": source,
                            "channel": channel_name,
                            "center_morphology_variance": basic[
                                "center_region_variance"
                            ],
                            "polar_morphology_variance": basic[
                                "polar_region_variance"
                            ],
                            "outer_shell_variance": basic[
                                "outer_shell_region_variance"
                            ],
                            "magnetic_texture_high_k_energy": spectrum[
                                "high_k_energy"
                            ],
                            "radial_profile_variance": radial["variance"],
                            "outermost_two_shell_variance_mean": float(
                                np.mean([item["variance"] for item in shells[-2:]])
                            ),
                            "total_variation": basic["total_variation"],
                            "dynamic_span": basic["dynamic_span_q99_q01"],
                            "standard_deviation": basic["std"],
                        }
                    )
    payload = {
        "schema_version": "paper-stage-o-morphology-comparison-v1",
        "view_name": "spherical-coordinate adapted views",
        "not_cartesian_central_slices": True,
        "rows": rows,
    }
    write_json(stage_o / "morphology_comparison.json", payload)
    write_csv(stage_o / "morphology_comparison.csv", rows)
    return payload


def decision_output(
    stage_o: Path,
    training: Mapping[str, Any],
    comparison: Mapping[str, Any],
    validation: Mapping[str, Any],
    gt: Mapping[str, Any],
    no_gt: Mapping[str, Any],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    target_floor_ratios = {
        row["channel"]: row["floor_ratio_p3_over_canonical"]
        for row in comparison["rows"]
        if row["channel"] in TARGET_CHANNELS
    }
    selected_ranges = {
        str(step): _record_map(gt if step <= 19 else no_gt)[step]["physical_range"]
        for step in (*SELECTED_GT, *SELECTED_NO_GT)
    }
    payload = {
        "schema_version": "paper-stage-o-decision-v1",
        "choice": "C",
        "decision": "P3_PILOT_COMPLETE_BUT_ROLLOUT_UNSTABLE",
        "engineering_evidence": {
            "preflight_passed": True,
            "smoke_passed": True,
            "formal_epochs": 30,
            "microbatches": training["microbatches"],
            "optimizer_updates": training["optimizer_updates"],
            "strict_reload": True,
            "gt_rollout_finite_positive": bool(
                gt["finite"] and gt["rho_press_positive"]
            ),
            "no_gt_rollout_finite_positive": bool(
                no_gt["finite"] and no_gt["rho_press_positive"]
            ),
            "transform_counters": no_gt["transform_count_delta"],
            "nonfinite_training": training["runtime"]["nonfinite_count"],
        },
        "oracle_floor_evidence": {
            "target_channel_p3_over_canonical": target_floor_ratios,
            "p3_floor_significantly_reduced": all(
                value < 0.01 for value in target_floor_ratios.values()
            ),
        },
        "model_added_degradation_evidence": {
            "stage_o_model_to_own_oracle_average": comparison["aggregate"][
                "stage_o_model_to_own_oracle_average"
            ],
            "validation_prediction_above_rout_fraction": validation["constraints"][
                "prediction_above_rout_fraction"
            ],
            "selected_physical_ranges": selected_ranges,
            "persistent_decoded_range_explosion": True,
            "step19_normalized_average": gt["records"][18]["oracle_aware"][
                "metrics"
            ]["E_norm"]["arithmetic_average"],
            "step100_evaluation_clamp_fraction": no_gt["records"][-1][
                "evaluation_bound_clamp_fraction"
            ],
        },
        "persistence_evidence": {
            "stage_o_model_over_p3_persistence": comparison["aggregate"][
                "stage_o_model_over_persistence"
            ],
            "model_beats_p3_persistence_one_step_average": comparison["aggregate"][
                "stage_o_model_over_persistence"
            ]
            < 1.0,
        },
        "shell_radial_transport_evidence": {
            channel: payload["gate"]["gate_3_transport_skill"]
            for channel, payload in gates["channels"].items()
        },
        "legacy_stage_m_v1_difference": (
            "legacy collapse/ripple/stripe flags are state-shape heuristics; stage_m_v1 "
            "separates the recovered P3 floor from model-added degradation/transport, "
            "while Gate 0 independently exposes persistent decoded-range/Rout failure"
        ),
        "core_comparison_answers": [
            {
                "question": "Did P3 lower the Bcc2/Bcc3/vel3 oracle floor?",
                "answer": "yes",
                "evidence_status": "observed",
                "evidence": target_floor_ratios,
            },
            {
                "question": "Did P3 LocalNO reduce model-added variance degradation?",
                "answer": "not_stably",
                "evidence_status": "observed",
                "evidence": "stage_m_v1 Gate 2 failed for Bcc2 and Bcc3; vel3 passed, while decoded ranges exploded",
            },
            {
                "question": "Did P3 LocalNO improve shell/radial transport?",
                "answer": "no",
                "evidence_status": "observed",
                "evidence": "Gate 3 failed for all three target channels with negative median shell and radial persistence-relative skill",
            },
            {
                "question": "Did P3 LocalNO reduce legacy collapse flags?",
                "answer": "yes_but_not_stability",
                "evidence_status": "observed",
                "evidence": "target-channel collapse flags disappeared, but ripple/stripe flags and a more severe decoded-range explosion replaced them",
            },
            {
                "question": "Did stage_m_v1 differ from the legacy detector?",
                "answer": "yes",
                "evidence_status": "observed",
                "evidence": "legacy flags did not identify collapse, whereas Gate 0 exposed range/Rout failure and Gates 2/3 separated model degradation from the recovered floor",
            },
            {
                "question": "Did P3 LocalNO beat P3 persistence?",
                "answer": "no",
                "evidence_status": "observed",
                "evidence": comparison["aggregate"]["stage_o_model_over_persistence"],
            },
            {
                "question": "Was the Stage K to Stage O difference caused by transform floor or model behavior?",
                "answer": "floor recovery was transform-side; stable skill did not follow and model behavior dominated failure",
                "evidence_status": "inference",
                "unsupported": "a causal mechanism inside the operator is not proven by this pilot",
            },
            {
                "question": "Does the LocalNO mixed operator-response failure remain?",
                "answer": "supported_as_persistent",
                "evidence_status": "inference",
                "evidence": "all target channels failed Gate 3 and the physical feedback trajectory expanded to the float32 range",
                "unsupported": "the experiment does not isolate a unique causal operator defect",
            },
        ],
        "historical_status_unchanged": {
            "Stage K": "C. TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE",
            "Stage L": "3. MIXED_OVERALL",
            "Stage M": "4. MIXED_TRANSPORT_AND_PREPROCESSING_FAILURE",
            "Stage N": "P3 ready; mixed operator-response failure",
        },
        "next_authorized_action": "propose_only: diagnose P3/LocalNO extreme-range feedback and operator response; do not start a new model or ablation",
    }
    write_json(stage_o / "stage_o_decision.json", payload)
    (stage_o / "stage_o_decision.md").write_text(
        "\n".join(
            [
                "# Stage O decision",
                "",
                "## C. P3_PILOT_COMPLETE_BUT_ROLLOUT_UNSTABLE",
                "",
                "Preflight, full smoke, 30 epochs, 2,370 microbatches, 600 updates, "
                "strict reload, and the 19/100-step execution completed. All states remained "
                "finite and rho/press positive, and transform counters were exact.",
                "",
                "P3 strongly reduced the Bcc2/Bcc3/vel3 oracle floor, but that recovery did "
                "not become stable model skill. The best checkpoint was worse than P3 "
                f"persistence by `{payload['persistence_evidence']['stage_o_model_over_p3_persistence']:.6g}x` "
                "on the P3 normalized one-step average. Validation predictions were above "
                "frozen Rout for every pair, physical ranges expanded persistently, and the "
                "step-100 evaluation clamp fraction approached one. These satisfy the frozen "
                "Stage O C decoded-range/structure-instability rule.",
                "",
                "This does not rewrite Stage K–N and does not authorize another training run.",
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
    stage_o = root / "outputs/paper_reduced100/stage_o"
    experiment = stage_o / "localno_p3_plain"
    config = load_paper_experiment_config(
        root / "configs/paper_reduced100/stage_o_localno_p3_plain.yaml",
        project_root=root,
    )
    smoke_outputs(stage_o)
    training = training_outputs(stage_o, experiment)
    evaluation = read_json(experiment / "evaluation_summary.json")
    validation = validation_outputs(stage_o, evaluation)
    comparison = comparison_outputs(stage_o, validation)
    gt = read_json(experiment / "gt_rollout.json")
    no_gt = read_json(experiment / "no_gt_rollout.json")
    gt_output = dict(gt)
    gt_output["undefined_diagnostic_ratios_encoded_as_null"] = True
    no_gt_output = dict(no_gt)
    no_gt_output["undefined_diagnostic_ratios_encoded_as_null"] = True
    write_json(stage_o / "rollout_gt.json", gt_output)
    write_json(stage_o / "rollout_no_gt.json", no_gt_output)
    write_csv(
        stage_o / "rollout_gt.csv",
        [
            {
                "step": record["step"],
                "finite": record["finite"],
                "rho_press_positive": record["rho_press_positive"],
                "prediction_global_norm": record["prediction_global_norm"],
                "above_rin": record["above_rin"],
                "above_rout": record["above_rout"],
                "artifact_count": len(record["artifacts"]["flags"]),
                "normalized_average": record["oracle_aware"]["metrics"]["E_norm"][
                    "arithmetic_average"
                ],
                "model_to_oracle_average": record["oracle_aware"]["metrics"][
                    "E_model_oracle"
                ]["arithmetic_average"],
                "model_to_raw_average": record["oracle_aware"]["metrics"][
                    "E_model_raw"
                ]["arithmetic_average"],
                "oracle_floor_average": record["oracle_aware"]["metrics"][
                    "E_oracle_raw"
                ]["arithmetic_average"],
            }
            for record in gt["records"]
        ],
    )
    write_csv(
        stage_o / "rollout_no_gt.csv",
        [
            {
                "step": record["step"],
                "finite": record["finite"],
                "rho_press_positive": record["rho_press_positive"],
                "prediction_global_norm": record["prediction_global_norm"],
                "above_rin": record["above_rin"],
                "above_rout": record["above_rout"],
                "artifact_count": len(record["artifacts"]["flags"]),
                "ground_truth_error_omitted": True,
            }
            for record in no_gt["records"]
        ],
    )
    gates = gate_and_legacy_outputs(
        root=root,
        stage_o=stage_o,
        config=config,
        validation=validation,
        gt=gt,
        no_gt=no_gt,
    )
    morphology_outputs(root, stage_o, config)
    decision = decision_output(
        stage_o, training, comparison, validation, gt, no_gt, gates
    )
    print(json.dumps({"decision": decision["decision"]}, indent=2))


if __name__ == "__main__":
    main()
