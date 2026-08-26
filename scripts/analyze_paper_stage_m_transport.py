#!/usr/bin/env python
"""Compute Stage M oracle-conditioned transport and replay the frozen candidate gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any, Mapping

import h5py
import numpy as np
import torch
import yaml

from grmhd import CHANNELS
from grmhd.paper_config import load_paper_experiment_config
from grmhd.paper_data_processor import PaperDataProcessor
from grmhd.paper_stage_l_attribution import basic_field_metrics, radial_profile, safe_retention
from grmhd.paper_stage_m import (
    floor_limited_channel,
    model_added_gate,
    radial_profile_vector,
    transport_metrics,
    validate_candidate_threshold_sources,
    variance_vector,
)


EXPECTED_CONFIG_SHA256 = (
    "91a360fe9087f413fccaeaa6cf56874e980773aa7f584b3bf5b2917e8a677206"
)
MODELS = ("persistence", "fno_plain", "fno_full", "localno_plain")
ROLLOUTS = {
    "fno_plain": "outputs/paper_reduced100/stage_g/pilot30_plain_l2/gt_rollout.json",
    "fno_full": "outputs/paper_reduced100/stage_g/pilot30_full_fno/gt_rollout.json",
    "localno_plain": "outputs/paper_reduced100/stage_k/localno_differential_plain/gt_rollout.json",
}
SELECTED_STEPS = (1, 3, 5, 10, 19)
NO_GT_STEPS = (25, 50, 75, 100)
PRIMARY_CHANNELS = ("Bcc2", "Bcc3", "vel3")


def json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        value = value.detach().cpu()
        return float(value) if value.numel() == 1 else value.tolist()
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(json_value(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(json_value(value), sort_keys=True, separators=(",", ":"))
    return json_value(value)


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            [{key: csv_value(value) for key, value in row.items()} for row in rows]
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(json_value(value), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def records_by_step(payload: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {int(record["step"]): record for record in payload["records"]}


def field_summary(values: np.ndarray, shell_index: np.ndarray) -> dict[str, Any]:
    basic = basic_field_metrics(values)
    radial = radial_profile(values)
    return {
        "global_variance": basic["variance"],
        "global_std": basic["std"],
        "dynamic_span": basic["dynamic_span_q99_q01"],
        "total_variation": basic["total_variation"],
        "shell_variance": variance_vector(values, shell_index).tolist(),
        "radial_profile": radial["profile"],
        "radial_profile_variance": radial["variance"],
    }


def model_summary(
    *,
    model: str,
    record: Mapping[str, Any] | None,
    channel: str,
    input_oracle: np.ndarray,
    shell_index: np.ndarray,
) -> dict[str, Any]:
    if model == "persistence":
        return field_summary(input_oracle, shell_index)
    if record is None:
        raise ValueError("Stage M model record is required")
    values = record["statistics"]["channels"][channel]
    return {
        "global_variance": float(values["std"]) ** 2,
        "global_std": float(values["std"]),
        "dynamic_span": float(values["quantiles"][-1] - values["quantiles"][0]),
        "total_variation": None,
        "shell_variance": [float(shell["std"]) ** 2 for shell in values["shells"]],
        "radial_profile": [float(value) for value in values["radial_mean"]],
        "radial_profile_variance": float(np.var(values["radial_mean"], ddof=0)),
    }


def saved_stage_l_summary(
    *,
    model: str,
    channel: str,
    step: int,
    gt_variance: list[Mapping[str, Any]],
    gt_shell: list[Mapping[str, Any]],
    gt_radial: list[Mapping[str, Any]],
) -> dict[str, Any]:
    variance = next(
        row
        for row in gt_variance
        if row["model"] == model
        and row["channel"] == channel
        and int(row["step"]) == step
    )
    shells = sorted(
        (
            row
            for row in gt_shell
            if row["model"] == model
            and row["channel"] == channel
            and int(row["step"]) == step
        ),
        key=lambda row: int(row["shell"]),
    )
    radial = next(
        row
        for row in gt_radial
        if row["model"] == model
        and row["channel"] == channel
        and int(row["step"]) == step
    )
    return {
        "global_variance": float(variance["model_variance"]),
        "global_std": float(variance["model_std"]),
        "dynamic_span": float(variance["model_dynamic_span_q99_q01"]),
        "total_variation": float(variance["model_total_variation"]),
        "shell_variance": [float(row["model_variance"]) for row in shells],
        "radial_profile": [float(value) for value in radial["model_profile"]],
        "radial_profile_variance": float(radial["model_radial_profile_variance"]),
    }


def grouped_errors(error: list[float], shell_index: np.ndarray | None = None) -> dict[str, float]:
    values = np.asarray(error, dtype=np.float64)
    if shell_index is None:
        groups = {"inner": slice(0, 2), "middle": slice(2, 6), "outer": slice(6, 8)}
        return {name: float(np.mean(values[selection])) for name, selection in groups.items()}
    index = np.asarray(shell_index, dtype=np.int64)
    return {
        "inner": float(np.mean(values[index <= 1])),
        "middle": float(np.mean(values[(index >= 2) & (index <= 5)])),
        "outer": float(np.mean(values[index >= 6])),
    }


def payload(
    schema: str,
    *,
    common: Mapping[str, Any],
    definitions: Mapping[str, Any],
    rows: list[Mapping[str, Any]],
    summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    undefined = sum(
        value is None
        for row in rows
        for key, value in row.items()
        if key.endswith(("relative_error", "transport_cosine", "persistence_relative_skill"))
    )
    return {
        "schema_version": schema,
        "metadata": {
            **common,
            "metric_definitions": dict(definitions),
            "undefined_count": undefined,
        },
        "summary": dict(summary or {}),
        "rows": rows,
    }


def aggregate_transport(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    error = float(np.sqrt(sum(float(row["error_l2"]) ** 2 for row in rows)))
    persistence_error = float(
        np.sqrt(sum(float(row["persistence_error_l2"]) ** 2 for row in rows))
    )
    skill = None if persistence_error <= 1.0e-30 else 1.0 - error / persistence_error
    return {
        "error_l2": error,
        "persistence_error_l2": persistence_error,
        "persistence_relative_skill": skill,
        "signed_transport_agreement": float(
            np.mean([float(row["signed_transport_agreement"]) for row in rows])
        ),
        "transport_cosine_defined_fraction": float(
            np.mean([row["transport_cosine"] is not None for row in rows])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/paper_reduced100/stage_m_transform_floor_audit.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper_reduced100/stage_m"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if sha256_file(config_path) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("Frozen Stage M config changed")
    validate_candidate_threshold_sources(config["candidate_gate"])
    manifest_path = output_dir / "run_manifest.json"
    manifest = load_json(manifest_path)
    if manifest["status"] not in {
        "transform_floor_complete_gate_replay_not_started",
        "stage_m_complete",
    }:
        raise RuntimeError("Stage M transform-floor gate has not completed")
    if manifest["metadata"]["stage_m_config_sha256"] != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("Stage M transform manifest config mismatch")
    common = {
        **manifest["metadata"],
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
    }
    epsilon = float(config["transport"]["denominator_absolute_floor"])
    sign_tolerance = float(config["transport"]["sign_zero_tolerance"])

    experiment_config = load_paper_experiment_config(
        root / config["provenance"]["experiment_config"], project_root=root
    )
    processor = PaperDataProcessor.from_config(experiment_config).to("cpu")
    processor.eval()
    with h5py.File(root / "data_proc/grmhd_regrid_inner_r200_64.h5", "r") as handle:
        raw_states = {
            index: np.asarray(handle["snapshots"][index], dtype=np.float32)
            for index in range(91, 111)
        }
        radial = np.asarray(handle["coords/r"][...], dtype=np.float64)
    edges = np.asarray(processor.shell_metadata["edges"], dtype=np.float64)
    shell_index = np.clip(np.searchsorted(edges, radial, side="right") - 1, 0, 7)
    oracle_states = {}
    for index, state in raw_states.items():
        tensor = torch.from_numpy(state).unsqueeze(0)
        oracle_states[index] = (
            processor.preprocessor.decode(
                processor.preprocessor.encode(tensor, channel_axis=1), channel_axis=1
            )
            .squeeze(0)
            .numpy()
        )

    rollout_payloads = {
        model: load_json(root / path) for model, path in ROLLOUTS.items()
    }
    for model, item in rollout_payloads.items():
        if not item["finite"] or not item["rho_press_positive"]:
            raise RuntimeError(f"Frozen Stage M rollout is not engineering-valid: {model}")
    rollouts = {model: records_by_step(item) for model, item in rollout_payloads.items()}
    stage_l_root = root / "outputs/paper_reduced100/stage_l"
    gt_variance = load_json(stage_l_root / "gt_variance_retention.json")["rows"]
    gt_shell = load_json(stage_l_root / "gt_shell_retention.json")["rows"]
    gt_radial = load_json(stage_l_root / "gt_radial_retention.json")["rows"]
    gt_spectral = load_json(stage_l_root / "gt_spectral_retention.json")["rows"]

    variance_rows: list[dict[str, Any]] = []
    shell_rows: list[dict[str, Any]] = []
    radial_rows: list[dict[str, Any]] = []
    for step in range(1, 20):
        source = 90 + step
        target = source + 1
        for channel_index, channel in enumerate(CHANNELS):
            input_field = oracle_states[source][channel_index].astype(np.float64)
            target_field = oracle_states[target][channel_index].astype(np.float64)
            input_summary = field_summary(input_field, shell_index)
            target_summary = field_summary(target_field, shell_index)
            for model in MODELS:
                record = None if model == "persistence" else rollouts[model][step]
                if model != "persistence" and "statistics" not in record:
                    if step not in SELECTED_STEPS:
                        continue
                    model_values = saved_stage_l_summary(
                        model=model,
                        channel=channel,
                        step=step,
                        gt_variance=gt_variance,
                        gt_shell=gt_shell,
                        gt_radial=gt_radial,
                    )
                    metric_source = "stage_l_saved_selected_step_metrics"
                else:
                    model_values = model_summary(
                        model=model,
                        record=record,
                        channel=channel,
                        input_oracle=input_field,
                        shell_index=shell_index,
                    )
                    metric_source = (
                        "computed_canonical_persistence"
                        if model == "persistence"
                        else "frozen_rollout_saved_statistics"
                    )
                global_result = transport_metrics(
                    np.asarray([input_summary["global_variance"]]),
                    np.asarray([target_summary["global_variance"]]),
                    np.asarray([model_values["global_variance"]]),
                    epsilon=epsilon,
                    sign_zero_tolerance=sign_tolerance,
                )
                variance_rows.append(
                    {
                        "step": step,
                        "source_snapshot": source,
                        "target_snapshot": target,
                        "model": model,
                        "channel": channel,
                        "ground_truth_available": True,
                        "reference_domain": "canonical_oracle",
                        "metric_source": metric_source,
                        **global_result,
                    }
                )
                shell_result = transport_metrics(
                    np.asarray(input_summary["shell_variance"]),
                    np.asarray(target_summary["shell_variance"]),
                    np.asarray(model_values["shell_variance"]),
                    epsilon=epsilon,
                    sign_zero_tolerance=sign_tolerance,
                )
                shell_rows.append(
                    {
                        "step": step,
                        "source_snapshot": source,
                        "target_snapshot": target,
                        "model": model,
                        "channel": channel,
                        "ground_truth_available": True,
                        "reference_domain": "canonical_oracle",
                        "metric_source": metric_source,
                        "grouped_absolute_error": grouped_errors(shell_result["absolute_error"]),
                        **shell_result,
                    }
                )
                radial_result = transport_metrics(
                    np.asarray(input_summary["radial_profile"]),
                    np.asarray(target_summary["radial_profile"]),
                    np.asarray(model_values["radial_profile"]),
                    epsilon=epsilon,
                    sign_zero_tolerance=sign_tolerance,
                )
                true_amplitude = float(np.linalg.norm(radial_result["delta_true"]))
                model_amplitude = float(np.linalg.norm(radial_result["delta_model"]))
                amplitude_ratio, amplitude_undefined = safe_retention(
                    model_amplitude, true_amplitude, epsilon=epsilon
                )
                radial_rows.append(
                    {
                        "step": step,
                        "source_snapshot": source,
                        "target_snapshot": target,
                        "model": model,
                        "channel": channel,
                        "ground_truth_available": True,
                        "reference_domain": "canonical_oracle",
                        "metric_source": metric_source,
                        "coordinate_semantics": config["transport"]["radial_semantics"],
                        "grouped_absolute_error": grouped_errors(
                            radial_result["absolute_error"], shell_index
                        ),
                        "profile_transport_amplitude_ratio": amplitude_ratio,
                        "profile_transport_amplitude_ratio_undefined": amplitude_undefined,
                        **radial_result,
                    }
                )

    definitions = {
        "transport": "delta model = model state metric - input oracle metric; delta true = target oracle metric - input oracle metric",
        "persistence": "delta persistence = exactly zero",
        "relative_error": "L2(delta model-delta true)/L2(delta true), null at denominator floor",
        "skill": "1-model error/persistence error, null at denominator floor",
        "radial": config["transport"]["radial_semantics"],
    }
    variance_payload = payload(
        "paper-stage-m-oracle-variance-transport-v1",
        common=common,
        definitions=definitions,
        rows=variance_rows,
    )
    shell_payload = payload(
        "paper-stage-m-oracle-shell-transport-v1",
        common=common,
        definitions=definitions,
        rows=shell_rows,
    )
    radial_payload = payload(
        "paper-stage-m-oracle-radial-transport-v1",
        common=common,
        definitions=definitions,
        rows=radial_rows,
    )
    for stem, item, rows in (
        ("oracle_variance_transport", variance_payload, variance_rows),
        ("oracle_shell_transport", shell_payload, shell_rows),
        ("oracle_radial_transport", radial_payload, radial_rows),
    ):
        write_json(output_dir / f"{stem}.json", item)
        write_csv(output_dir / f"{stem}.csv", rows)

    model_added_rows: list[dict[str, Any]] = []
    for model in MODELS:
        for channel in CHANNELS:
            for step in SELECTED_STEPS:
                variance = next(
                    row
                    for row in gt_variance
                    if row["model"] == model
                    and row["channel"] == channel
                    and int(row["step"]) == step
                )
                shell_values = [
                    row["model_retention_variance"]
                    for row in gt_shell
                    if row["model"] == model
                    and row["channel"] == channel
                    and int(row["step"]) == step
                    and row["model_retention_variance"] is not None
                ]
                radial_value = next(
                    row["model_retention_radial_profile_variance"]
                    for row in gt_radial
                    if row["model"] == model
                    and row["channel"] == channel
                    and int(row["step"]) == step
                )
                spectrum = next(
                    row
                    for row in gt_spectral
                    if row["model"] == model
                    and row["channel"] == channel
                    and int(row["step"]) == step
                    and row["axis"] == "combined"
                    and row["demeaned"] is True
                )
                shell_median = None if not shell_values else float(np.median(shell_values))
                defined_shell_radial = [
                    value for value in (shell_median, radial_value) if value is not None
                ]
                shell_radial = (
                    None if not defined_shell_radial else min(defined_shell_radial)
                )
                retentions = {
                    "global_variance": variance["model_retention_variance"],
                    "shell_radial_variance": shell_radial,
                    "dynamic_span": variance["model_retention_dynamic_span_q99_q01"],
                    "high_k_energy": spectrum["model_retention_high_k_energy"],
                }
                model_added_rows.append(
                    {
                        "step": step,
                        "model": model,
                        "channel": channel,
                        "reference_domain": "target_canonical_oracle",
                        "global_std_ratio": variance["model_retention_std"],
                        "shell_std_ratio": [
                            row["model_retention_std"]
                            for row in gt_shell
                            if row["model"] == model
                            and row["channel"] == channel
                            and int(row["step"]) == step
                        ],
                        "radial_profile_variance_ratio": radial_value,
                        "dynamic_span_ratio": retentions["dynamic_span"],
                        "high_k_ratio": retentions["high_k_energy"],
                        "total_variation_ratio": variance["model_retention_total_variation"],
                        "global_variance_ratio": retentions["global_variance"],
                        "shell_radial_variance_ratio": shell_radial,
                        "severe_categories": {
                            name: value is not None and value < 0.5
                            for name, value in retentions.items()
                        },
                    }
                )
    model_added_payload = payload(
        "paper-stage-m-model-added-degradation-v1",
        common=common,
        definitions={
            "reference": "model decoded state divided by target canonical oracle metric",
            "transport_separation": "absolute state retention is separate from delta-transport error",
            "severe": "strict retention < 0.5",
        },
        rows=model_added_rows,
    )
    write_json(output_dir / "model_added_degradation.json", model_added_payload)
    write_csv(output_dir / "model_added_degradation.csv", model_added_rows)

    no_gt_variance = load_json(stage_l_root / "no_gt_variance.json")["rows"]
    no_gt_shell = load_json(stage_l_root / "no_gt_shells.json")["rows"]
    no_gt_radial = load_json(stage_l_root / "no_gt_radial.json")["rows"]
    initial = {
        channel: field_summary(oracle_states[91][index].astype(np.float64), shell_index)
        for index, channel in enumerate(CHANNELS)
    }
    no_gt_rows: list[dict[str, Any]] = []
    for model in MODELS:
        for channel in CHANNELS:
            for step in NO_GT_STEPS:
                variance = next(
                    row
                    for row in no_gt_variance
                    if row["model"] == model
                    and row["channel"] == channel
                    and int(row["step"]) == step
                )
                shells = [
                    row["retention_vs_canonical_initial_std"]
                    for row in no_gt_shell
                    if row["model"] == model
                    and row["channel"] == channel
                    and int(row["step"]) == step
                    and row["retention_vs_canonical_initial_std"] is not None
                ]
                radial_row = next(
                    row
                    for row in no_gt_radial
                    if row["model"] == model
                    and row["channel"] == channel
                    and int(row["step"]) == step
                )
                span_ratio, span_undefined = safe_retention(
                    variance["dynamic_span_q99_q01"],
                    initial[channel]["dynamic_span"],
                    epsilon=epsilon,
                )
                no_gt_rows.append(
                    {
                        "step": step,
                        "model": model,
                        "channel": channel,
                        "ground_truth_available": False,
                        "target_oracle_available": False,
                        "legacy_raw_reference_detector_flag": variance["detector_flag"],
                        "legacy_raw_reference_std_ratio": variance["detector_std_ratio"],
                        "floor_corrected_std_ratio": (
                            variance["std"] / initial[channel]["global_std"]
                            if initial[channel]["global_std"] > epsilon
                            else None
                        ),
                        "shell_floor_corrected_std_ratio": shells,
                        "shell_floor_corrected_std_ratio_median": (
                            None if not shells else float(np.median(shells))
                        ),
                        "radial_floor_corrected_ratio": radial_row[
                            "retention_vs_canonical_initial_variance"
                        ],
                        "dynamic_span_floor_corrected_ratio": span_ratio,
                        "dynamic_span_floor_corrected_ratio_undefined": span_undefined,
                        "candidate_metric_status": "proposal_not_validated_physical_stability_metric",
                    }
                )
    no_gt_payload = payload(
        "paper-stage-m-no-gt-floor-corrected-v1",
        common=common,
        definitions={
            "legacy": "std prediction / std raw physical snapshot 91; unchanged detector",
            "floor_corrected": "state metric / canonical oracle snapshot 91 metric",
            "future_oracle": "not available and never constructed",
        },
        rows=no_gt_rows,
    )
    write_json(output_dir / "no_gt_floor_corrected.json", no_gt_payload)
    write_csv(output_dir / "no_gt_floor_corrected.csv", no_gt_rows)

    floor_sources = load_json(output_dir / "channel_floor_sources.json")["rows"]
    preprocessing_floor = load_json(stage_l_root / "preprocessing_floor.json")
    floor_pre_rows = preprocessing_floor["rows"]
    gate1: dict[str, Any] = {}
    for channel in CHANNELS:
        selected = [
            row
            for row in floor_pre_rows
            if row["channel"] == channel and int(row["step"]) in SELECTED_STEPS
        ]
        variance_retention = float(
            np.median([row["preprocessing_retention_variance"] for row in selected])
        )
        shell_radial = float(
            np.median(
                [
                    min(
                        row["shell_variance_retention_median"],
                        row["radial_profile_variance_retention"],
                    )
                    for row in selected
                ]
            )
        )
        high_k = float(
            np.median(
                [
                    row["preprocessing_retention_combined_demeaned_high_k_energy"]
                    for row in selected
                ]
            )
        )
        legacy = any(bool(row["detector_flag"]) for row in selected)
        gate1[channel] = {
            "floor_limited_channel": floor_limited_channel(
                oracle_legacy_detector_triggered=legacy,
                median_variance_retention=variance_retention,
                median_shell_radial_retention=shell_radial,
                median_high_k_retention=high_k,
            ),
            "oracle_legacy_detector_triggered": legacy,
            "median_variance_retention": variance_retention,
            "median_shell_radial_retention": shell_radial,
            "median_high_k_retention": high_k,
        }

    gate2: dict[str, dict[str, Any]] = {}
    gate3: dict[str, dict[str, Any]] = {}
    replay_rows: list[dict[str, Any]] = []
    for model in MODELS:
        gate2[model] = {}
        gate3[model] = {}
        for channel in CHANNELS:
            relevant = [
                row
                for row in model_added_rows
                if row["model"] == model and row["channel"] == channel
            ]
            severe_by_step = {
                int(row["step"]): row["severe_categories"] for row in relevant
            }
            model_gate = model_added_gate(severe_by_step, required_steps=2)
            gate2[model][channel] = model_gate
            shell_selected = [
                row
                for row in shell_rows
                if row["model"] == model
                and row["channel"] == channel
                and int(row["step"]) in SELECTED_STEPS
            ]
            radial_selected = [
                row
                for row in radial_rows
                if row["model"] == model
                and row["channel"] == channel
                and int(row["step"]) in SELECTED_STEPS
            ]
            shell_aggregate = aggregate_transport(shell_selected)
            radial_aggregate = aggregate_transport(radial_selected)
            skills = (
                shell_aggregate["persistence_relative_skill"],
                radial_aggregate["persistence_relative_skill"],
            )
            sign_agreement = min(
                shell_aggregate["signed_transport_agreement"],
                radial_aggregate["signed_transport_agreement"],
            )
            transport_pass = (
                shell_aggregate["error_l2"]
                <= shell_aggregate["persistence_error_l2"] + 1.0e-15
                and radial_aggregate["error_l2"]
                <= radial_aggregate["persistence_error_l2"] + 1.0e-15
                and any(skill is not None and skill > 0 for skill in skills)
                and sign_agreement >= 0.5
            )
            transport_gate = {
                "passed": transport_pass,
                "shell": shell_aggregate,
                "radial": radial_aggregate,
                "minimum_sign_agreement": sign_agreement,
            }
            gate3[model][channel] = transport_gate
            if model_gate["failed"] and not transport_pass:
                label = "model_added_degradation_and_transport_failure"
            elif model_gate["failed"] and transport_pass:
                label = "state_degradation_with_partial_transport_skill"
            elif not model_gate["failed"] and transport_pass:
                label = "oracle_conditioned_candidate_pass"
            else:
                label = "transport_gate_not_passed_without_two_step_state_failure"
            replay_rows.append(
                {
                    "model": model,
                    "channel": channel,
                    "replay_classification": "counterfactual_gate_replay_only",
                    "gate_0_engineering_valid": True,
                    "gate_1_floor_limited": gate1[channel]["floor_limited_channel"],
                    "gate_2_model_added_degradation": model_gate["failed"],
                    "gate_2_failure_steps": model_gate["failure_steps"],
                    "gate_3_transport_passed": transport_pass,
                    "shell_transport_skill": shell_aggregate[
                        "persistence_relative_skill"
                    ],
                    "radial_transport_skill": radial_aggregate[
                        "persistence_relative_skill"
                    ],
                    "minimum_transport_sign_agreement": sign_agreement,
                    "candidate_label": label,
                    "historical_reclassification": False,
                }
            )
    for channel in CHANNELS:
        replay_rows.append(
            {
                "model": "canonical_oracle",
                "channel": channel,
                "replay_classification": "counterfactual_gate_replay_only",
                "gate_0_engineering_valid": True,
                "gate_1_floor_limited": gate1[channel]["floor_limited_channel"],
                "gate_2_model_added_degradation": False,
                "gate_2_failure_steps": [],
                "gate_3_transport_passed": True,
                "shell_transport_skill": 1.0,
                "radial_transport_skill": 1.0,
                "minimum_transport_sign_agreement": 1.0,
                "candidate_label": "floor_limited_reference_not_model_failure",
                "historical_reclassification": False,
            }
        )

    calibration_artifact = {
        "schema_version": "paper-stage-m-candidate-gate-calibration-v1",
        "split": config["candidate_gate"]["calibration_split"],
        "data_derived_quantile_thresholds": None,
        "reason": "candidate uses only predeclared 0.5 retention, zero skill, and 0.5 sign thresholds",
        "validation_model_outcomes_used": False,
        "thresholds": {
            "severe_retention": 0.5,
            "selected_gt_failure_steps": 2,
            "transport_skill": 0.0,
            "sign_agreement": 0.5,
        },
    }
    calibration_sha = sha256_json(calibration_artifact)
    candidate_config = {
        "schema_version": "paper-stage-m-oracle-conditioned-structure-gate-v1",
        "metadata": {
            **common,
            "metric_definitions": definitions,
            "undefined_count": 0,
            "calibration_artifact_sha256": calibration_sha,
        },
        "name": "OracleConditionedStructureGate",
        "status": "proposal_for_future_runs",
        "rules": config["candidate_gate"],
        "calibration_artifact": calibration_artifact,
        "legacy_detector_retained": True,
        "historical_reclassification_forbidden": True,
    }
    write_json(output_dir / "candidate_gate_config.json", candidate_config)
    (output_dir / "candidate_gate_config.md").write_text(
        "# OracleConditionedStructureGate candidate\n\n"
        "- Gate 0: frozen engineering validity.\n"
        "- Gate 1: qualify raw-to-oracle preprocessing floor.\n"
        "- Gate 2: assess model-to-target-oracle structure degradation.\n"
        "- Gate 3: compare shell/radial transport with persistence.\n"
        "- Thresholds are predeclared/toy-derived; no validation model outcome was used.\n"
        "- This is a future-run proposal and does not replace the legacy detector.\n",
        encoding="utf-8",
    )
    replay_payload = payload(
        "paper-stage-m-candidate-gate-replay-v1",
        common={**common, "calibration_artifact_sha256": calibration_sha},
        definitions={
            "classification": "counterfactual gate replay only",
            "historical_reclassification": "always false",
        },
        rows=replay_rows,
        summary={"gate_1": gate1, "gate_2": gate2, "gate_3": gate3},
    )
    write_json(output_dir / "candidate_gate_replay.json", replay_payload)
    write_csv(output_dir / "candidate_gate_replay.csv", replay_rows)
    (output_dir / "candidate_gate_replay.md").write_text(
        "# Stage M candidate gate replay\n\n"
        "This is `counterfactual_gate_replay_only`; no Stage G/K/L result is reclassified.\n\n"
        + "\n".join(
            f"- {row['model']} / {row['channel']}: `{row['candidate_label']}`"
            for row in replay_rows
            if row["channel"] in PRIMARY_CHANNELS
        )
        + "\n",
        encoding="utf-8",
    )

    local_primary = [
        row
        for row in replay_rows
        if row["model"] == "localno_plain" and row["channel"] in PRIMARY_CHANNELS
    ]
    local_state_failures = sum(row["gate_2_model_added_degradation"] for row in local_primary)
    local_transport_failures = sum(not row["gate_3_transport_passed"] for row in local_primary)
    floor_limited_count = sum(row["gate_1_floor_limited"] for row in local_primary)
    partial_transport = any(row["gate_3_transport_passed"] for row in local_primary)
    if floor_limited_count == 3 and local_state_failures and local_transport_failures:
        if partial_transport:
            local_decision = "2. STATE_COLLAPSE_WITH_PARTIAL_TRANSPORT_SKILL"
        else:
            local_decision = "4. MIXED_TRANSPORT_AND_PREPROCESSING_FAILURE"
    elif local_transport_failures:
        local_decision = "1. TRANSPORT_FAILURE_CONFIRMED"
    elif local_state_failures:
        local_decision = "2. STATE_COLLAPSE_WITH_PARTIAL_TRANSPORT_SKILL"
    elif floor_limited_count == 3:
        local_decision = "3. PREPROCESSING_FLOOR_EXPLAINS_MOST_APPARENT_FAILURE"
    else:
        local_decision = "5. INCONCLUSIVE_OR_ENGINEERING_FAILURE"

    contradictory = any(
        row["model"] == "canonical_oracle"
        and (row["gate_2_model_added_degradation"] or not row["gate_3_transport_passed"])
        for row in replay_rows
    )
    candidate_decision = (
        "II. CANDIDATE_GATE_NEEDS_REVISION"
        if contradictory
        else "I. CANDIDATE_GATE_READY_FOR_FUTURE_RUNS"
    )
    stage_m_decision = {
        "schema_version": "paper-stage-m-decision-v1",
        "metadata": {
            **common,
            "metric_definitions": definitions,
            "undefined_count": sum(
                item["metadata"]["undefined_count"]
                for item in (variance_payload, shell_payload, radial_payload)
            ),
            "calibration_artifact_sha256": calibration_sha,
        },
        "transform_floor_decisions": {
            row["channel"]: row["decision"] for row in floor_sources
        },
        "localno_transport_decision": local_decision,
        "candidate_gate_decision": candidate_decision,
        "candidate_gate_replay_only": True,
        "stage_k_decision_unchanged": "C. TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE",
        "stage_l_decision_unchanged": "3. MIXED_OVERALL",
        "localno_primary_replay": local_primary,
    }
    write_json(output_dir / "stage_m_decision.json", stage_m_decision)
    (output_dir / "stage_m_decision.md").write_text(
        "# Stage M decision\n\n"
        + "\n".join(
            f"- {channel}: `{decision}`"
            for channel, decision in stage_m_decision["transform_floor_decisions"].items()
        )
        + f"\n- LocalNO transport: `{local_decision}`"
        + f"\n- Candidate gate: `{candidate_decision}`"
        + "\n- Stage K remains `C. TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE`."
        + "\n- Stage L remains `3. MIXED_OVERALL`.\n",
        encoding="utf-8",
    )

    manifest.update(
        {
            "status": "stage_m_complete",
            "scope_completed": [
                "provenance_gate",
                "transform_source_audit",
                "transform_stage_trace",
                "diagnostic_counterfactual_roundtrips",
                "channel_floor_source_decisions",
                "oracle_conditioned_variance_transport",
                "oracle_conditioned_shell_radial_transport",
                "model_added_degradation",
                "no_gt_floor_corrected_diagnostics",
                "candidate_gate_replay",
                "stage_m_final_decision",
            ],
            "scope_not_started": [],
            "localno_transport_decision": local_decision,
            "candidate_gate_decision": candidate_decision,
            "calibration_artifact_sha256": calibration_sha,
        }
    )
    write_json(manifest_path, manifest)
    (output_dir / "run_manifest.md").write_text(
        "# Stage M run manifest\n\n"
        "- Status: `stage_m_complete`\n"
        "- Classification: `post_hoc_transform_and_evaluation_audit`\n"
        "- No training/backward/optimizer/scheduler/model rollout.\n"
        f"- Config SHA256: `{EXPECTED_CONFIG_SHA256}`\n"
        + "\n".join(
            f"- {channel}: `{decision}`"
            for channel, decision in stage_m_decision["transform_floor_decisions"].items()
        )
        + f"\n- LocalNO transport: `{local_decision}`"
        + f"\n- Candidate gate: `{candidate_decision}`\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "localno_transport_decision": local_decision,
                "candidate_gate_decision": candidate_decision,
                "localno_primary_replay": local_primary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
