#!/usr/bin/env python
"""Run Stage M provenance, source, trace, and transform-floor attribution only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping

import h5py
import numpy as np
import torch
import yaml

from grmhd import CHANNELS
from grmhd.dataset import sha256_file
from grmhd.paper_bounds import PaperPhysicalBounds
from grmhd.paper_config import load_paper_experiment_config
from grmhd.paper_data_processor import PaperDataProcessor
from grmhd.paper_preprocessing import PAPER_TRANSFORMS, PaperPreprocessor
from grmhd.paper_stage_l_attribution import (
    basic_field_metrics,
    radial_profile,
    relative_l2,
    safe_retention,
    shell_metrics,
    spectrum_metrics,
)
from grmhd.paper_stage_m import (
    classify_floor_source,
    diagnostic_counterfactual,
    inverse_nonlinear,
    recovery_rate,
    trace_channel_transform,
)


EXPECTED_CONFIG_SHA256 = (
    "91a360fe9087f413fccaeaa6cf56874e980773aa7f584b3bf5b2917e8a677206"
)
EXPECTED_UPSTREAM = "86a8bc7812a31b42c4f7895693cf4ac11521c066"
EXPECTED_DATA = "cd3966bed1f617f49d775151cff87ae92eb2432a560bb82ae9355605e7aea70a"
EXPECTED_PREPROCESSING = (
    "1645714424d95b02de2283fc16a8ef05f3864d15f11025580af13f3e67964001"
)
EXPECTED_SHELLS = "99580fc2c62431f46e6702972660b060aa1294920e5f9530b45b43ce08c190b7"
EXPECTED_CHECKPOINTS = {
    "fno_full": "a18e8711aae8071218959d34d217bb52b51a1fcc23f4e11bdcc5713dbff111e2",
    "fno_plain": "923b233de1d0abfda97f99fda9142dc437dc91f5dbbc3f19e0e5d28f5e862f1b",
    "localno_plain": "f69008f91da80a4a7a6374adc253bbcdf37f58f0d142ea80f2b3f399e1f02a9f",
}
EXPECTED_SELECTED_STATES = {
    "fno_full": "60eeda64167b0ab33ce544ac011ff13bab0f4f4887a4eae4d664436b351e9077",
    "fno_plain": "2a82cb6015fa9316ceeaa7c433423d8cee4ccbd0910884f244f8ea0217b97c7c",
    "localno_plain": "441a1d8348b240cb5df16063ee71de1ba375d97b64e79010ec88330191fbea66",
}
EXPECTED_STAGE_L_CONFIG = (
    "8f9c170be023da938ce6b6d1414ebc459965467080ea397a85fe98998ff34983"
)
EXPECTED_STAGE_L_DECISION = (
    "867c8e155f843e9f5bbdd52de5280f515138fc5e6b29a743e8d7c3d6cca83595"
)
EXPECTED_STAGE_L_ATTRIBUTION = (
    "cb173caf676df312d3f3eaf6cd46ea63f408144476adeb7a8131fcb733596ddf"
)
EXPECTED_DETECTOR_SOURCE = (
    "64125810309c129ca56c326c6a6be7f60f94c706b8362b62810947587d7101e8"
)
CORE_METRICS = (
    "global_variance_retention",
    "shell_radial_variance_retention",
    "high_k_retention",
    "dynamic_span_retention",
)
COUNTERFACTUALS = (
    "CANONICAL_FULL",
    "NO_FINAL_INVERSE_CLAMP",
    "NO_SOFTCLIP_COUNTERFACTUAL",
    "NORMALIZER_ONLY_ROUNDTRIP",
    "NONLINEAR_ONLY_ROUNDTRIP",
    "FLOAT64_REFERENCE",
)


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
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(json_value(value), sort_keys=True, separators=(",", ":"))
    return json_value(value)


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty Stage M CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            [{key: csv_value(value) for key, value in row.items()} for row in rows]
        )


def sha256_json(value: Any) -> str:
    encoded = json.dumps(json_value(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def function_hash(function: Any) -> str:
    return hashlib.sha256(inspect.getsource(function).encode()).hexdigest()


def current_commit(root: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def provenance_gate(root: Path, config_path: Path) -> dict[str, Any]:
    stage_l = load_json(root / "outputs/paper_reduced100/stage_l/run_manifest.json")
    expected = {
        "data": (root / "data_proc/grmhd_regrid_inner_r200_64.h5", EXPECTED_DATA),
        "preprocessing": (
            root / "outputs/paper_reduced100/stats/normalizer.npz",
            EXPECTED_PREPROCESSING,
        ),
        "shell_metadata": (
            root / "outputs/paper_reduced100/priors/shell_metadata.json",
            EXPECTED_SHELLS,
        ),
        "fno_full_checkpoint": (
            root
            / "outputs/paper_reduced100/stage_g/pilot30_full_fno/best_validation_l2/paper_state_dict.pt",
            EXPECTED_CHECKPOINTS["fno_full"],
        ),
        "fno_plain_checkpoint": (
            root
            / "outputs/paper_reduced100/stage_g/pilot30_plain_l2/best_validation_l2/paper_state_dict.pt",
            EXPECTED_CHECKPOINTS["fno_plain"],
        ),
        "localno_plain_checkpoint": (
            root
            / "outputs/paper_reduced100/stage_k/localno_differential_plain/best_validation_l2/paper_state_dict.pt",
            EXPECTED_CHECKPOINTS["localno_plain"],
        ),
        "localno_selected_states": (
            root
            / "outputs/paper_reduced100/stage_k/localno_differential_plain/selected_states.pt",
            EXPECTED_SELECTED_STATES["localno_plain"],
        ),
        "stage_l_config": (
            root / "configs/paper_reduced100/stage_l_collapse_attribution.yaml",
            EXPECTED_STAGE_L_CONFIG,
        ),
        "stage_l_decision": (
            root / "outputs/paper_reduced100/stage_l/stage_l_decision.json",
            EXPECTED_STAGE_L_DECISION,
        ),
        "stage_l_attribution": (
            root / "outputs/paper_reduced100/stage_l/collapse_attribution.json",
            EXPECTED_STAGE_L_ATTRIBUTION,
        ),
        "detector_source_file": (
            root / "src/grmhd/paper_stage_g_evaluation.py",
            EXPECTED_DETECTOR_SOURCE,
        ),
        "stage_m_config": (config_path, EXPECTED_CONFIG_SHA256),
    }
    checks = {}
    for name, (path, wanted) in expected.items():
        actual = sha256_file(path)
        checks[name] = {
            "path": str(path.relative_to(root)),
            "expected_sha256": wanted,
            "actual_sha256": actual,
            "match": actual == wanted,
        }
        if actual != wanted:
            raise RuntimeError(f"Stage M frozen provenance mismatch: {name}")
    if stage_l["status"] != "attribution_complete":
        raise RuntimeError("Stage L is not complete")
    if stage_l["stage_l_phase_2"]["overall_decision"] != "3. MIXED_OVERALL":
        raise RuntimeError("Frozen Stage L decision changed")
    if stage_l["stage_k_decision_unchanged"] != (
        "C. TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE"
    ):
        raise RuntimeError("Frozen Stage K decision changed")
    for model, wanted in EXPECTED_SELECTED_STATES.items():
        actual = stage_l["stage_l_phase_2"]["selected_state_hashes"][model]
        if actual != wanted:
            raise RuntimeError(f"Stage M selected-state provenance changed: {model}")
    upstream = subprocess.check_output(
        ["git", "-C", "external/neuraloperator", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
    ).strip()
    if upstream != EXPECTED_UPSTREAM:
        raise RuntimeError("Stage M upstream commit changed")
    return {
        "checks": checks,
        "upstream_commit": upstream,
        "stage_k_decision": "C. TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE",
        "stage_l_decision": "3. MIXED_OVERALL",
    }


def field_metrics(values: np.ndarray, shell_index: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    basic = basic_field_metrics(values)
    quantiles = np.quantile(values, [0.001, 0.01, 0.5, 0.99, 0.999])
    shells = shell_metrics(values, shell_index)
    radial = radial_profile(values)
    spectrum = spectrum_metrics(values, axis="combined", demean=True)
    return {
        "finite_count": int(np.count_nonzero(np.isfinite(values))),
        "element_count": int(values.size),
        "mean": basic["mean"],
        "std": basic["std"],
        "variance": basic["variance"],
        "rms": basic["rms"],
        "minimum": basic["minimum"],
        "maximum": basic["maximum"],
        "q001": float(quantiles[0]),
        "q01": float(quantiles[1]),
        "q50": float(quantiles[2]),
        "q99": float(quantiles[3]),
        "q999": float(quantiles[4]),
        "dynamic_span_q99_q01": basic["dynamic_span_q99_q01"],
        "sign_balance": basic["sign_balance"],
        "near_zero_occupancy": basic["near_zero_occupancy"],
        "exact_zero_occupancy": basic["zero_fraction"],
        "shell_variance": [row["variance"] for row in shells],
        "radial_profile_variance": radial["variance"],
        "total_variation": basic["total_variation"],
        "gradient_energy": basic["gradient_energy"],
        "low_k_energy": spectrum["low_k_energy"],
        "mid_k_energy": spectrum["mid_k_energy"],
        "high_k_energy": spectrum["high_k_energy"],
        "parseval_relative_error": spectrum["parseval_relative_error"],
    }


def retention_metrics(
    candidate: np.ndarray, raw: np.ndarray, shell_index: np.ndarray, epsilon: float
) -> dict[str, Any]:
    raw_metrics = field_metrics(raw, shell_index)
    candidate_metrics = field_metrics(candidate, shell_index)
    output: dict[str, Any] = {
        "relative_l2": relative_l2(candidate, raw, epsilon=epsilon),
        "sign_agreement": float(
            np.mean(np.signbit(candidate) == np.signbit(raw))
        ),
    }
    for name, key in (
        ("global_variance", "variance"),
        ("std", "std"),
        ("dynamic_span", "dynamic_span_q99_q01"),
        ("radial_variance", "radial_profile_variance"),
        ("high_k", "high_k_energy"),
        ("total_variation", "total_variation"),
    ):
        ratio, undefined = safe_retention(
            candidate_metrics[key], raw_metrics[key], epsilon=epsilon
        )
        output[f"{name}_retention"] = ratio
        output[f"{name}_retention_undefined"] = undefined
        output[f"raw_{key}"] = raw_metrics[key]
        output[f"diagnostic_{key}"] = candidate_metrics[key]
    shell_ratios = [
        safe_retention(candidate_value, raw_value, epsilon=epsilon)[0]
        for candidate_value, raw_value in zip(
            candidate_metrics["shell_variance"], raw_metrics["shell_variance"]
        )
    ]
    defined_shells = [value for value in shell_ratios if value is not None]
    shell_median = None if not defined_shells else float(np.median(defined_shells))
    radial_ratio = output["radial_variance_retention"]
    defined_shell_radial = [
        value for value in (shell_median, radial_ratio) if value is not None
    ]
    output["shell_variance_retention"] = shell_ratios
    output["shell_variance_retention_median"] = shell_median
    output["shell_radial_variance_retention"] = (
        None if not defined_shell_radial else min(defined_shell_radial)
    )
    output["shell_radial_variance_retention_undefined"] = not defined_shell_radial
    output["global_variance_retention"] = output["global_variance_retention"]
    return output


def no_final_clamp_from_encoded(
    encoded: np.ndarray,
    *,
    kind: str,
    epsilon: float,
    median: float,
    scale: float,
    gamma: float,
) -> tuple[np.ndarray | None, int]:
    encoded = np.asarray(encoded, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = float(gamma) * np.arctanh(encoded / float(gamma))
    representation = z * float(scale) + float(median)
    count = int(representation.size - np.count_nonzero(np.isfinite(representation)))
    if count:
        return None, count
    return (
        inverse_nonlinear(
            representation,
            kind=kind,
            epsilon=epsilon,
            physical_limit=np.finfo(np.float32).max,
        ),
        0,
    )


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
    output_dir.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    provenance = provenance_gate(root, config_path)
    epsilon = float(config["retention_contract"]["denominator_absolute_floor"])

    experiment_config = load_paper_experiment_config(
        root / config["provenance"]["experiment_config"], project_root=root
    )
    processor = PaperDataProcessor.from_config(experiment_config).to("cpu")
    processor.eval()
    preprocessor = processor.preprocessor
    h5_path = root / "data_proc/grmhd_regrid_inner_r200_64.h5"
    with h5py.File(h5_path, "r") as handle:
        snapshots = {
            index: np.asarray(handle["snapshots"][index], dtype=np.float32)
            for index in range(91, 111)
        }
        radial = np.asarray(handle["coords/r"][...], dtype=np.float64)
    edges = np.asarray(processor.shell_metadata["edges"], dtype=np.float64)
    shell_index = np.clip(np.searchsorted(edges, radial, side="right") - 1, 0, 7)

    channel_parameters = {
        name: {
            "transform": PAPER_TRANSFORMS[index],
            "epsilon": float(preprocessor.epsilon[index]),
            "median": float(preprocessor.median[index]),
            "scale": float(preprocessor.scale[index]),
            "gamma": preprocessor.gamma,
            "inverse_clamp_fraction": preprocessor.inverse_clamp_fraction,
        }
        for index, name in enumerate(CHANNELS)
    }
    source_hashes = {
        "forward_nonlinear_numpy": function_hash(PaperPreprocessor._transform_values_inplace),
        "canonical_encode_numpy": function_hash(PaperPreprocessor.encode_numpy),
        "canonical_decode_numpy": function_hash(PaperPreprocessor.decode_numpy),
        "canonical_encode_tensor": function_hash(PaperPreprocessor.encode_tensor),
        "canonical_decode_tensor": function_hash(PaperPreprocessor.decode_tensor),
        "evaluation_bounds_clamp": function_hash(PaperPhysicalBounds.clamp_normalized),
        "prediction_decode": function_hash(PaperDataProcessor.decode_prediction),
        "source_file_paper_preprocessing": sha256_file(root / "src/grmhd/paper_preprocessing.py"),
        "source_file_paper_bounds": sha256_file(root / "src/grmhd/paper_bounds.py"),
        "source_file_paper_data_processor": sha256_file(root / "src/grmhd/paper_data_processor.py"),
    }
    common = {
        "project_commit": current_commit(root),
        "classification": "post_hoc_transform_and_evaluation_audit",
        "no_training": True,
        "no_backward": True,
        "no_optimizer_or_scheduler": True,
        "stage_k_decision": "C. TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE",
        "stage_l_decision": "3. MIXED_OVERALL",
        "data_checksum": EXPECTED_DATA,
        "preprocessing_checksum": EXPECTED_PREPROCESSING,
        "normalizer_checksum": EXPECTED_PREPROCESSING,
        "shell_metadata_checksum": EXPECTED_SHELLS,
        "checkpoint_hashes": EXPECTED_CHECKPOINTS,
        "selected_state_hashes": EXPECTED_SELECTED_STATES,
        "stage_l_config_sha256": EXPECTED_STAGE_L_CONFIG,
        "stage_l_decision_sha256": EXPECTED_STAGE_L_DECISION,
        "stage_l_attribution_sha256": EXPECTED_STAGE_L_ATTRIBUTION,
        "stage_m_config_sha256": EXPECTED_CONFIG_SHA256,
        "transform_source_hashes": source_hashes,
        "channel_parameter_sha256": sha256_json(channel_parameters),
        "calibration_split": config["candidate_gate"]["calibration_split"],
        "counterfactual_classification": "diagnostic_counterfactual",
        "metric_definitions": {
            "variance": "population variance over one channel and phi/theta/r",
            "shells": "frozen eight physical-r shells",
            "radial_profile": "mean over phi/theta at every stored r center",
            "spectrum": "orthonormal combined demeaned stored-index-space diagnostic",
            "retention": "diagnostic metric divided by raw metric; null at denominator floor",
        },
        "undefined_count": 0,
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
    }

    source_rows = [
        {
            "stage": "forward nonlinear",
            "function": "PaperPreprocessor._transform_values_inplace / encode_tensor",
            "input_domain": "raw physical",
            "output_domain": "signed-log, positive-log, or linear representation",
            "channel_parameters": "transform, epsilon",
            "invertible": True,
            "lossy": False,
            "frozen": True,
        },
        {
            "stage": "robust normalization",
            "function": "(transformed-median)/scale",
            "input_domain": "nonlinear representation",
            "output_domain": "unbounded robust z",
            "channel_parameters": "train-only median and MAD scale",
            "invertible": True,
            "lossy": False,
            "frozen": True,
        },
        {
            "stage": "forward softclip",
            "function": "gamma*tanh(z/gamma)",
            "input_domain": "unbounded robust z",
            "output_domain": "bounded canonical normalized",
            "channel_parameters": "gamma=6",
            "invertible": "mathematically for abs(output)<gamma",
            "lossy": "numerical saturation possible",
            "frozen": True,
        },
        {
            "stage": "inverse input clamp",
            "function": "clamp(encoded,-0.99*gamma,0.99*gamma)",
            "input_domain": "canonical normalized",
            "output_domain": "clamped normalized",
            "channel_parameters": "inverse_clamp_fraction=0.99",
            "invertible": False,
            "lossy": True,
            "frozen": True,
        },
        {
            "stage": "inverse softclip",
            "function": "gamma*atanh(encoded/gamma)",
            "input_domain": "clamped normalized",
            "output_domain": "robust z",
            "channel_parameters": "gamma=6",
            "invertible": True,
            "lossy": False,
            "frozen": True,
        },
        {
            "stage": "inverse robust normalization",
            "function": "z*scale+median",
            "input_domain": "robust z",
            "output_domain": "nonlinear representation",
            "channel_parameters": "median and scale",
            "invertible": True,
            "lossy": False,
            "frozen": True,
        },
        {
            "stage": "inverse nonlinearity and dtype guard",
            "function": "PaperPreprocessor.decode_tensor",
            "input_domain": "nonlinear representation",
            "output_domain": "physical",
            "channel_parameters": "transform, epsilon, output dtype max",
            "invertible": "before dtype guard",
            "lossy": "only if dtype guard hits",
            "frozen": True,
        },
        {
            "stage": "evaluation-only bounds clamp",
            "function": "PaperPhysicalBounds.clamp_normalized",
            "input_domain": "model normalized prediction",
            "output_domain": "rho/press bounded normalized prediction",
            "channel_parameters": "train-only rho/press bounds",
            "invertible": False,
            "lossy": True,
            "frozen": True,
        },
    ]
    source_payload = {
        "schema_version": "paper-stage-m-transform-source-audit-v1",
        "metadata": common,
        "resolved_transform_config_sha256": sha256_file(
            root / "configs/data/paper_reduced100.yaml"
        ),
        "channel_parameters": channel_parameters,
        "rows": source_rows,
    }
    write_json(output_dir / "transform_source_audit.json", source_payload)

    trace_rows: list[dict[str, Any]] = []
    counter_rows: list[dict[str, Any]] = []
    canonical_states: dict[int, np.ndarray] = {}
    encoded_states: dict[int, np.ndarray] = {}
    evaluation_states: dict[int, np.ndarray] = {}
    evaluation_masks: dict[int, np.ndarray] = {}
    for snapshot, raw_state in snapshots.items():
        tensor = torch.from_numpy(raw_state).unsqueeze(0)
        encoded_tensor = preprocessor.encode(tensor, channel_axis=1)
        canonical_tensor = preprocessor.decode(encoded_tensor, channel_axis=1)
        bounds = processor.bounds.clamp_normalized(encoded_tensor, channel_axis=1)
        evaluation_tensor = preprocessor.decode(bounds.clamped, channel_axis=1)
        encoded_states[snapshot] = encoded_tensor.squeeze(0).numpy()
        canonical_states[snapshot] = canonical_tensor.squeeze(0).numpy()
        evaluation_states[snapshot] = evaluation_tensor.squeeze(0).numpy()
        evaluation_masks[snapshot] = bounds.mask.squeeze(0).numpy()

        for channel, name in enumerate(CHANNELS):
            parameters = channel_parameters[name]
            raw = np.asarray(raw_state[channel], dtype=np.float64)
            transform_arguments = {
                "kind": parameters["transform"],
                "epsilon": parameters["epsilon"],
                "median": parameters["median"],
                "scale": parameters["scale"],
                "gamma": parameters["gamma"],
                "inverse_clamp_fraction": parameters["inverse_clamp_fraction"],
            }
            stages = trace_channel_transform(raw, **transform_arguments)
            stages["T7_inverse_nonlinear_canonical_oracle"] = np.asarray(
                canonical_states[snapshot][channel], dtype=np.float64
            )
            stages["T8_evaluation_bounds_clamp_then_decode"] = np.asarray(
                evaluation_states[snapshot][channel], dtype=np.float64
            )
            encoded = encoded_states[snapshot][channel]
            inverse_hits = np.abs(encoded) > (
                preprocessor.gamma * preprocessor.inverse_clamp_fraction
            )
            for stage, values in stages.items():
                metrics = field_metrics(values, shell_index)
                normalized_domain = stage in {
                    "T3_forward_softclip_canonical_normalized",
                    "T4_decode_input_no_forward_hard_clamp",
                }
                trace_rows.append(
                    {
                        "snapshot": snapshot,
                        "role": "control_input" if snapshot == 91 else "validation_target",
                        "channel": name,
                        "stage": stage,
                        "domain": (
                            "physical"
                            if stage.startswith("T0")
                            or stage.startswith("T7")
                            or stage.startswith("T8")
                            else "nonlinear_representation"
                            if stage.startswith("T1") or stage.startswith("T6")
                            else "robust_z"
                            if stage.startswith("T2") or stage.startswith("T5")
                            else "canonical_normalized"
                        ),
                        **metrics,
                        "saturation_occupancy_abs_ge_0.90_gamma": (
                            float(np.mean(np.abs(values) >= 0.90 * preprocessor.gamma))
                            if normalized_domain
                            else None
                        ),
                        "inverse_clamp_occupancy": (
                            float(np.mean(inverse_hits))
                            if stage.startswith(("T3", "T4", "T5", "T6", "T7"))
                            else None
                        ),
                        "evaluation_bounds_clamp_occupancy": (
                            float(np.mean(evaluation_masks[snapshot][channel]))
                            if stage.startswith("T8")
                            else None
                        ),
                    }
                )

            for label in COUNTERFACTUALS:
                if label == "CANONICAL_FULL":
                    values = canonical_states[snapshot][channel].astype(np.float64)
                    finite = True
                    nonfinite_count = 0
                    note = "frozen canonical tensor encode/decode"
                elif label == "NO_FINAL_INVERSE_CLAMP":
                    values, nonfinite_count = no_final_clamp_from_encoded(
                        encoded,
                        kind=parameters["transform"],
                        epsilon=parameters["epsilon"],
                        median=parameters["median"],
                        scale=parameters["scale"],
                        gamma=parameters["gamma"],
                    )
                    finite = values is not None
                    note = "diagnostic_counterfactual; actual float32 encoded state; inverse clamp bypassed"
                else:
                    result = diagnostic_counterfactual(
                        raw,
                        label=label,
                        kind=parameters["transform"],
                        epsilon=parameters["epsilon"],
                        median=parameters["median"],
                        scale=parameters["scale"],
                        gamma=parameters["gamma"],
                        inverse_clamp_fraction=parameters["inverse_clamp_fraction"],
                    )
                    values = result.values
                    finite = result.finite
                    nonfinite_count = result.nonfinite_count
                    note = result.note
                row: dict[str, Any] = {
                    "snapshot": snapshot,
                    "role": "control_input" if snapshot == 91 else "validation_target",
                    "channel": name,
                    "counterfactual": label,
                    "classification": "diagnostic_counterfactual",
                    "finite": finite,
                    "nonfinite_count": nonfinite_count,
                    "note": note,
                    "canonical_inverse_clamp_occupancy": float(np.mean(inverse_hits)),
                    "saturation_occupancy_abs_ge_0.90_gamma": float(
                        np.mean(np.abs(encoded) >= 0.90 * preprocessor.gamma)
                    ),
                }
                if finite and values is not None:
                    row.update(retention_metrics(values, raw, shell_index, epsilon))
                else:
                    for metric in CORE_METRICS:
                        row[metric] = None
                    row.update(
                        {
                            "relative_l2": None,
                            "sign_agreement": None,
                            "std_retention": None,
                            "radial_variance_retention": None,
                            "shell_variance_retention": None,
                            "total_variation_retention": None,
                        }
                    )
                counter_rows.append(row)

    counter_lookup = {
        (row["snapshot"], row["channel"], row["counterfactual"]): row
        for row in counter_rows
    }
    for row in counter_rows:
        canonical = counter_lookup[
            (row["snapshot"], row["channel"], "CANONICAL_FULL")
        ]
        for metric in CORE_METRICS:
            row[f"recovery_{metric}"] = recovery_rate(
                row.get(metric), canonical.get(metric), epsilon=epsilon
            )

    selected_snapshots = {92, 94, 96, 101, 110}
    floor_rows: list[dict[str, Any]] = []
    for channel in ("Bcc2", "Bcc3", "vel3"):
        medians: dict[str, dict[str, float | None]] = {}
        for label in COUNTERFACTUALS:
            medians[label] = {}
            selected = [
                row
                for row in counter_rows
                if row["channel"] == channel
                and row["counterfactual"] == label
                and row["snapshot"] in selected_snapshots
            ]
            for metric in CORE_METRICS:
                values = [row.get(metric) for row in selected if row.get(metric) is not None]
                medians[label][metric] = (
                    None if not values else float(np.median(values))
                )
                recoveries = [
                    row.get(f"recovery_{metric}")
                    for row in selected
                    if row.get(f"recovery_{metric}") is not None
                ]
                medians[label][f"recovery_{metric}"] = (
                    None if not recoveries else float(np.median(recoveries))
                )
        decision = classify_floor_source(
            recovery_by_component={
                "forward_softclip": [
                    medians["NO_SOFTCLIP_COUNTERFACTUAL"][f"recovery_{metric}"]
                    for metric in CORE_METRICS
                ],
                "final_inverse_clamp": [
                    medians["NO_FINAL_INVERSE_CLAMP"][f"recovery_{metric}"]
                    for metric in CORE_METRICS
                ],
            },
            nonlinear_isolated_retentions=[
                medians["NONLINEAR_ONLY_ROUNDTRIP"][metric]
                for metric in CORE_METRICS
            ],
            normalizer_isolated_retentions=[
                medians["NORMALIZER_ONLY_ROUNDTRIP"][metric]
                for metric in CORE_METRICS
            ],
            float64_recoveries=[
                medians["FLOAT64_REFERENCE"][f"recovery_{metric}"]
                for metric in CORE_METRICS
            ],
        )
        floor_rows.append(
            {
                "channel": channel,
                "decision": decision.choice,
                "reason": decision.reason,
                "recovered_metric_counts": dict(decision.recovered_metric_counts),
                "selected_snapshots": sorted(selected_snapshots),
                "counterfactual_medians": medians,
            }
        )

    undefined_counterfactuals = sum(
        value is None
        for row in counter_rows
        for key, value in row.items()
        if key.endswith("_retention")
    )
    trace_payload = {
        "schema_version": "paper-stage-m-transform-stage-trace-v1",
        "metadata": {**common, "undefined_count": 0},
        "metric_definitions": {
            "trace_order": config["trace"]["real_pipeline_order"],
            "variance": "population variance over phi/theta/r",
            "shells": "frozen eight physical-r shells",
            "radial_profile": "mean over phi/theta for each stored r center",
            "spectrum": config["spectral_contract"],
        },
        "rows": trace_rows,
    }
    counter_payload = {
        "schema_version": "paper-stage-m-transform-counterfactuals-v1",
        "metadata": {**common, "undefined_count": undefined_counterfactuals},
        "metric_definitions": {
            "counterfactuals": config["counterfactuals"],
            "retention": "diagnostic metric divided by raw metric",
            "recovery": "(counterfactual retention - canonical retention) / max(1-canonical retention,epsilon)",
            "shell_radial": config["retention_contract"]["shell_radial_variance_aggregation"],
        },
        "rows": counter_rows,
    }
    floor_payload = {
        "schema_version": "paper-stage-m-channel-floor-sources-v1",
        "metadata": {**common, "undefined_count": undefined_counterfactuals},
        "rules": config["floor_source_rules"],
        "rows": floor_rows,
    }
    write_json(output_dir / "transform_stage_trace.json", trace_payload)
    write_csv(output_dir / "transform_stage_trace.csv", trace_rows)
    write_json(output_dir / "transform_counterfactuals.json", counter_payload)
    write_csv(output_dir / "transform_counterfactuals.csv", counter_rows)
    write_json(output_dir / "channel_floor_sources.json", floor_payload)
    write_csv(output_dir / "channel_floor_sources.csv", floor_rows)

    source_md = [
        "# Stage M transform source audit",
        "",
        f"- Config SHA256: `{EXPECTED_CONFIG_SHA256}`",
        f"- Normalizer SHA256: `{EXPECTED_PREPROCESSING}`",
        f"- Channel-parameter SHA256: `{common['channel_parameter_sha256']}`",
        "- The actual source order normalizes before applying tanh softclip.",
        "- Evaluation bounds clamp applies only to rho/press model predictions.",
        "",
        "| stage | function | input | output | invertible | lossy |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in source_rows:
        source_md.append(
            f"| {row['stage']} | {row['function']} | {row['input_domain']} | "
            f"{row['output_domain']} | {row['invertible']} | {row['lossy']} |"
        )
    (output_dir / "transform_source_audit.md").write_text(
        "\n".join(source_md) + "\n", encoding="utf-8"
    )
    counter_md = [
        "# Stage M diagnostic transform counterfactuals",
        "",
        "All variants are `diagnostic_counterfactual`; none replaces canonical preprocessing.",
        "",
    ]
    for row in floor_rows:
        counter_md.append(f"- {row['channel']}: `{row['decision']}`")
    (output_dir / "transform_counterfactuals.md").write_text(
        "\n".join(counter_md) + "\n", encoding="utf-8"
    )
    floor_md = [
        "# Stage M channel transform-floor sources",
        "",
        "| channel | decision | recovered metric counts |",
        "| --- | --- | --- |",
    ]
    for row in floor_rows:
        floor_md.append(
            f"| {row['channel']} | `{row['decision']}` | "
            f"`{json.dumps(row['recovered_metric_counts'], sort_keys=True)}` |"
        )
    floor_md.extend(
        [
            "",
            "The no-final-clamp diagnostic reports nonfinite values rather than replacing them.",
            "Float64, isolated robust normalization, and isolated channel nonlinearities are reported separately.",
        ]
    )
    (output_dir / "channel_floor_sources.md").write_text(
        "\n".join(floor_md) + "\n", encoding="utf-8"
    )
    run_manifest = {
        "schema_version": "paper-stage-m-run-manifest-v1",
        "status": "transform_floor_complete_gate_replay_not_started",
        "classification": "post_hoc_transform_and_evaluation_audit",
        "metadata": common,
        "provenance_gate": provenance,
        "scope_completed": [
            "provenance_gate",
            "transform_source_audit",
            "transform_stage_trace",
            "diagnostic_counterfactual_roundtrips",
            "channel_floor_source_decisions",
        ],
        "scope_not_started": [
            "oracle_conditioned_transport",
            "candidate_gate_replay",
            "stage_m_final_decision",
        ],
        "channel_floor_decisions": {
            row["channel"]: row["decision"] for row in floor_rows
        },
    }
    write_json(output_dir / "run_manifest.json", run_manifest)
    (output_dir / "run_manifest.md").write_text(
        "# Stage M run manifest\n\n"
        "- Status: `transform_floor_complete_gate_replay_not_started`\n"
        "- No training/backward/optimizer/scheduler/model rollout.\n"
        f"- Config SHA256: `{EXPECTED_CONFIG_SHA256}`\n"
        + "\n".join(
            f"- {row['channel']}: `{row['decision']}`" for row in floor_rows
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"channel_floor_decisions": run_manifest["channel_floor_decisions"], "undefined_count": undefined_counterfactuals}, indent=2))


if __name__ == "__main__":
    main()
