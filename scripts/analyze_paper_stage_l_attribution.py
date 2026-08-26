#!/usr/bin/env python
"""Run the frozen, read-only Stage L preprocessing/model collapse attribution."""

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
from neuralop.layers.spectral_convolution import SpectralConv

from grmhd import CHANNELS
from grmhd.dataset import sha256_file
from grmhd.paper_checkpoint import validate_paper_checkpoint_metadata
from grmhd.paper_config import build_paper_model, load_paper_experiment_config
from grmhd.paper_data_processor import PaperDataProcessor
from grmhd.paper_protocol import PaperReduced100Protocol
from grmhd.paper_stage_g import tensor_state_sha256
from grmhd.paper_stage_g_evaluation import artifact_diagnostics
from grmhd.paper_stage_l_attribution import (
    AXES,
    all_spectrum_metrics,
    basic_field_metrics,
    classify_channel,
    decompose_metric,
    make_region_masks,
    overall_attribution,
    pair_metrics,
    radial_profile,
    relative_l2,
    safe_retention,
    shell_metrics,
)


EXPECTED_CONFIG_SHA256 = (
    "8f9c170be023da938ce6b6d1414ebc459965467080ea397a85fe98998ff34983"
)
EXPECTED_DETECTOR_CONTRACT_SHA256 = (
    "2e0d4df70e9f074f410ea29ad806d03fa44e9b24aa95650e3305d291d996511b"
)
GT_STEPS = (1, 3, 5, 10, 19)
NO_GT_STEPS = (25, 50, 75, 100)
MODELS = ("persistence", "fno_plain", "fno_full", "localno_plain")
MODEL_CONFIGS = {
    "fno_plain": "configs/paper_reduced100/plain_l2_fno.yaml",
    "fno_full": "configs/paper_reduced100/full_fno_proxy.yaml",
    "localno_plain": "configs/paper_reduced100/stage_k_localno_differential_plain.yaml",
}
RETENTION_METRICS = (
    "std",
    "variance",
    "dynamic_span_q99_q01",
    "total_variation",
    "gradient_energy",
    "center_region_variance",
    "polar_region_variance",
    "outer_shell_region_variance",
)
SHELL_METRICS = (
    "std",
    "variance",
    "dynamic_span_q99_q01",
    "total_variation",
    "high_k_energy",
)
SPECTRAL_ENERGIES = (
    "low_k_energy",
    "mid_k_energy",
    "high_k_energy",
    "total_nonzero_mode_energy",
)


def json_value(value: Any) -> Any:
    if torch.is_tensor(value):
        value = value.detach().cpu()
        return float(value) if value.numel() == 1 else value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(json_value(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(json_value(value), sort_keys=True, separators=(",", ":"))
    return json_value(value)


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty Stage L CSV: {path}")
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            [{key: csv_value(value) for key, value in row.items()} for row in rows]
        )


def write_result(
    output_dir: Path,
    stem: str,
    *,
    rows: list[Mapping[str, Any]],
    common_metadata: Mapping[str, Any],
    metric_definitions: Mapping[str, Any],
    summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    undefined = sum(
        1
        for row in rows
        for key, value in row.items()
        if key.endswith("_undefined") and value is True
    )
    payload = {
        "schema_version": f"paper-stage-l-{stem.replace('_', '-')}-v1",
        "metadata": {
            **common_metadata,
            "metric_definitions": metric_definitions,
            "undefined_ratio_count": undefined,
        },
        "summary": dict(summary or {}),
        "rows": rows,
    }
    write_json(output_dir / f"{stem}.json", payload)
    write_csv(output_dir / f"{stem}.csv", rows)
    return payload


def strict_model_only_reload(
    *,
    root: Path,
    checkpoint_dir: Path,
    config_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, int]:
    config = load_paper_experiment_config(config_path, project_root=root)
    metadata = load_json(checkpoint_dir / "paper_grmhd_metadata.json")
    config_hash = sha256_file(config_path)
    validate_paper_checkpoint_metadata(
        metadata, config, expected_config_checksum=config_hash
    )
    model = build_paper_model(config)
    with torch.no_grad(), torch.serialization.safe_globals(
        [torch._C._nn.gelu, SpectralConv]
    ):
        state = torch.load(
            checkpoint_dir / "paper_state_dict.pt",
            map_location="cpu",
            weights_only=True,
        )
        incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Stage L strict reload returned incompatible keys")
    model.to(device).eval()
    processor = PaperDataProcessor.from_config(config).to(device)
    processor.eval()
    processor.set_epoch(int(metadata["epoch"]))
    return model, processor, int(metadata["epoch"])


def deterministic_step_three(
    *,
    model: torch.nn.Module,
    processor: PaperDataProcessor,
    validation_dataset: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    current = validation_dataset.load_snapshot(91).unsqueeze(0)
    selected: dict[int, torch.Tensor] = {}
    times = np.asarray(validation_dataset.times, dtype=np.float64)
    with torch.no_grad():
        for step in range(1, 4):
            source = 90 + step
            target = source + 1
            raw_target = validation_dataset.load_snapshot(target).unsqueeze(0)
            sample = processor.preprocess(
                {
                    "physical_input": current,
                    "physical_target": raw_target,
                    "source_index": torch.tensor([source]),
                    "target_index": torch.tensor([target]),
                    "time": torch.tensor([times[source]], dtype=torch.float64),
                    "target_time": torch.tensor([times[target]], dtype=torch.float64),
                }
            )
            prediction = model(x=sample["x"])
            prediction, _ = processor.postprocess(prediction, sample)
            current = processor.decode_prediction(
                prediction, apply_evaluation_clamp=True
            ).physical_prediction.detach().cpu()
            selected[step] = current
    return selected[1], selected[3]


def state_numpy(state: torch.Tensor, channel: int) -> np.ndarray:
    return state[0, channel].detach().cpu().numpy().astype(np.float64, copy=False)


def analyze_field(
    values: np.ndarray,
    *,
    region_masks: Mapping[str, np.ndarray],
    shell_index: np.ndarray,
) -> dict[str, Any]:
    return {
        "basic": basic_field_metrics(values, region_masks=region_masks),
        "shells": shell_metrics(values, shell_index),
        "radial": radial_profile(values),
        "spectra": all_spectrum_metrics(values),
    }


def spectral_lookup(analysis: Mapping[str, Any], axis: str, demean: bool) -> dict[str, Any]:
    return next(
        row
        for row in analysis["spectra"]
        if row["axis"] == axis and row["demeaned"] is demean
    )


def high_k_axis_anisotropy(analysis: Mapping[str, Any], *, demean: bool) -> float | None:
    energies = [
        spectral_lookup(analysis, axis, demean)["high_k_energy"] for axis in AXES
    ]
    denominator = min(energies)
    return None if denominator <= 1.0e-300 else float(max(energies) / denominator)


def prefixed_decomposition(
    name: str, raw: float, oracle: float, model: float, epsilon: float
) -> dict[str, Any]:
    return {
        f"raw_{name}": result["raw"],
        f"oracle_{name}": result["oracle"],
        f"model_{name}": result["model"],
        f"preprocessing_retention_{name}": result["preprocessing_retention"],
        f"model_retention_{name}": result["model_retention"],
        f"total_retention_{name}": result["total_retention"],
        f"preprocessing_{name}_undefined": result["preprocessing_undefined"],
        f"model_{name}_undefined": result["model_undefined"],
        f"total_{name}_undefined": result["total_undefined"],
    } if (result := decompose_metric(raw, oracle, model, epsilon=epsilon)) else {}


def saturation_record(
    *,
    normalized_model: torch.Tensor,
    normalized_target: torch.Tensor,
    channel: int,
    limit: float,
) -> dict[str, float]:
    model_mask = torch.abs(normalized_model[:, channel]) > limit
    target_mask = torch.abs(normalized_target[:, channel]) > limit
    return {
        "target_clamp_fraction": float(target_mask.float().mean()),
        "model_clamp_fraction": float(model_mask.float().mean()),
        "target_only_saturation": float((target_mask & ~model_mask).float().mean()),
        "model_only_saturation": float((model_mask & ~target_mask).float().mean()),
    }


def metric_envelope(raw_analysis: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    metrics = ("mean", "std", "variance", "dynamic_span_q99_q01", "total_variation")
    output: dict[str, Any] = {}
    for channel in CHANNELS:
        records = [raw_analysis[step][channel]["basic"] for step in range(1, 20)]
        output[channel] = {
            metric: {
                "minimum": min(record[metric] for record in records),
                "maximum": max(record[metric] for record in records),
            }
            for metric in metrics
        }
        output[channel]["raw_value_minimum"] = min(
            record["minimum"] for record in records
        )
        output[channel]["raw_value_maximum"] = max(
            record["maximum"] for record in records
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/paper_reduced100/stage_l_collapse_attribution.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper_reduced100/stage_l"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if sha256_file(config_path) != EXPECTED_CONFIG_SHA256:
        raise RuntimeError("Frozen Stage L attribution config checksum changed")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    epsilon = float(config["retention_contract"]["ratio_denominator_absolute_floor"])
    severe_threshold = float(config["retention_contract"]["severe_loss_threshold"])
    if severe_threshold != 0.5:
        raise RuntimeError("Frozen Stage L severe-loss threshold changed")
    detector_contract = load_json(output_dir / "detector_contract.json")
    if detector_contract["contract_sha256"] != EXPECTED_DETECTOR_CONTRACT_SHA256:
        raise RuntimeError("Frozen Stage L detector contract changed")
    detector_reproduction = load_json(output_dir / "detector_reproduction.json")
    if detector_reproduction["status"] != "passed" or not all(
        record["exact_all_flag_match"]
        for record in detector_reproduction["steps"].values()
    ):
        raise RuntimeError("Frozen detector reproduction no longer passes")
    manifest_path = output_dir / "run_manifest.json"
    manifest = load_json(manifest_path)
    if manifest["status"] not in {"detector_gate_passed", "attribution_complete"}:
        raise RuntimeError("Stage L detector gate is not passed")
    if manifest["stage_k_decision_unchanged"] != (
        "C. TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE"
    ):
        raise RuntimeError("Stage K decision changed before attribution")

    checkpoint_dirs = {
        name: root / manifest["checkpoints"][name]["best_validation_l2"]["path"]
        for name in ("fno_plain", "fno_full", "localno_plain")
    }
    experiment_dirs = {name: path.parent for name, path in checkpoint_dirs.items()}
    selected_paths = {
        name: experiment_dirs[name] / "selected_states.pt"
        for name in experiment_dirs
    }
    selected_payloads = {
        name: torch.load(path, map_location="cpu", weights_only=True)
        for name, path in selected_paths.items()
    }
    selected = {
        name: payload["selected_steps"] for name, payload in selected_payloads.items()
    }
    checkpoint_hashes = {
        name: manifest["checkpoints"][name]["best_validation_l2"][
            "model_state_file_sha256"
        ]
        for name in checkpoint_dirs
    }
    selected_hashes = {name: sha256_file(path) for name, path in selected_paths.items()}

    stage_k_config_path = root / MODEL_CONFIGS["localno_plain"]
    stage_k_config = load_paper_experiment_config(stage_k_config_path, project_root=root)
    processor = PaperDataProcessor.from_config(stage_k_config).to("cpu")
    processor.eval()
    protocol = PaperReduced100Protocol.from_yaml(
        root / "configs/data/paper_reduced100.yaml", project_root=root
    )
    validation_dataset = protocol.make_datasets()["validation"]
    hdf5_path = stage_k_config.resolve_path(stage_k_config.values["protocol"]["dataset"])
    with h5py.File(hdf5_path, "r") as handle:
        phi = np.asarray(handle["coords/phi"][...], dtype=np.float64)
        theta = np.asarray(handle["coords/theta"][...], dtype=np.float64)
        r = np.asarray(handle["coords/r"][...], dtype=np.float64)
    shell_edges = np.asarray(
        load_json(
            stage_k_config.resolve_path(
                stage_k_config.values["provenance"]["artifacts"]["shells"]["path"]
            )
        )["shells"]["edges"],
        dtype=np.float64,
    )
    shell_index = np.clip(
        np.searchsorted(shell_edges, r, side="right") - 1, 0, 7
    )
    spatial_shape = (len(phi), len(theta), len(r))
    region_masks = make_region_masks(
        theta=theta, shell_index=shell_index, spatial_shape=spatial_shape
    )
    initial_raw = validation_dataset.load_snapshot(91).unsqueeze(0)
    initial_normalized = processor.preprocessor.encode(initial_raw, channel_axis=1)
    initial_oracle = processor.preprocessor.decode(initial_normalized, channel_axis=1)
    initial_raw_hash = tensor_state_sha256({"snapshot_91_raw": initial_raw})

    fno_step_three: dict[str, torch.Tensor] = {}
    supplemental_reload: dict[str, Any] = {}
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Stage L FNO step-3 supplement requires CUDA parity with frozen trajectories"
        )
    supplement_device = torch.device("cuda:0")
    for name in ("fno_plain", "fno_full"):
        model, model_processor, epoch = strict_model_only_reload(
            root=root,
            checkpoint_dir=checkpoint_dirs[name],
            config_path=root / MODEL_CONFIGS[name],
            device=supplement_device,
        )
        recomputed_step1, recomputed_step3 = deterministic_step_three(
            model=model,
            processor=model_processor,
            validation_dataset=validation_dataset,
        )
        saved_step1 = selected[name][1]
        step1_difference = float(
            torch.linalg.vector_norm(recomputed_step1 - saved_step1)
            / torch.linalg.vector_norm(saved_step1).clamp_min(1.0e-30)
        )
        if step1_difference > 1.0e-6:
            raise RuntimeError(
                f"{name} CUDA deterministic supplement changed step 1: "
                f"relative_l2={step1_difference}"
            )
        fno_step_three[name] = recomputed_step3
        supplemental_reload[name] = {
            "epoch": epoch,
            "strict_model_reload": True,
            "eval_mode": True,
            "torch_no_grad": True,
            "optimizer_created": False,
            "scheduler_created": False,
            "cuda_vs_saved_cuda_step1_relative_l2": step1_difference,
            "supplemented_step": 3,
        }
        del model, model_processor

    model_states: dict[str, dict[int, torch.Tensor]] = {
        name: {step: state for step, state in selected[name].items()}
        for name in selected
    }
    model_states["fno_plain"][3] = fno_step_three["fno_plain"]
    model_states["fno_full"][3] = fno_step_three["fno_full"]
    model_states["persistence"] = {
        step: initial_oracle for step in (*GT_STEPS, *NO_GT_STEPS)
    }

    raw_states = {
        step: validation_dataset.load_snapshot(91 + step).unsqueeze(0)
        for step in range(1, 20)
    }
    oracle_states = {
        step: processor.preprocessor.decode(
            processor.preprocessor.encode(raw, channel_axis=1), channel_axis=1
        )
        for step, raw in raw_states.items()
    }
    normalized_targets = {
        step: processor.preprocessor.encode(raw, channel_axis=1)
        for step, raw in raw_states.items()
    }

    analysis_cache: dict[tuple[str, int, str], dict[str, Any]] = {}

    def cached(label: str, step: int, state: torch.Tensor, channel: int) -> dict[str, Any]:
        key = (label, step, CHANNELS[channel])
        if key not in analysis_cache:
            analysis_cache[key] = analyze_field(
                state_numpy(state, channel),
                region_masks=region_masks,
                shell_index=shell_index,
            )
        return analysis_cache[key]

    for step in range(1, 20):
        for channel in range(len(CHANNELS)):
            cached("raw", step, raw_states[step], channel)
            cached("oracle", step, oracle_states[step], channel)
    for step in GT_STEPS:
        for model_name in MODELS:
            for channel in range(len(CHANNELS)):
                cached(model_name, step, model_states[model_name][step], channel)
    for step in NO_GT_STEPS:
        for model_name in MODELS:
            for channel in range(len(CHANNELS)):
                cached(model_name, step, model_states[model_name][step], channel)

    truth_envelope = metric_envelope(
        {
            step: {
                channel: cached("raw", step, raw_states[step], index)
                for index, channel in enumerate(CHANNELS)
            }
            for step in range(1, 20)
        }
    )
    data_checksum = stage_k_config.values["provenance"]["dataset"]["sha256"]
    preprocessing_checksum = stage_k_config.values["provenance"]["preprocessing"][
        "sha256"
    ]
    common_metadata = {
        "classification": "post_hoc_attribution",
        "no_training": True,
        "no_backward": True,
        "no_optimizer_or_scheduler": True,
        "stage_k_decision": "C. TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE",
        "config_sha256": EXPECTED_CONFIG_SHA256,
        "detector_contract_sha256": EXPECTED_DETECTOR_CONTRACT_SHA256,
        "data_checksum": data_checksum,
        "preprocessing_checksum": preprocessing_checksum,
        "checkpoint_hashes": checkpoint_hashes,
        "selected_state_hashes": selected_hashes,
        "snapshot_91_raw_state_sha256": initial_raw_hash,
        "selected_gt_steps": list(GT_STEPS),
        "selected_no_gt_steps": list(NO_GT_STEPS),
        "ground_truth_ends_after_step": 19,
        "no_oracle_after_step_19": True,
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }
    metric_definitions = {
        "variance": "population variance (ddof=0) over one channel and phi/theta/r",
        "detector_std": "torch.std correction=1; unchanged frozen detector",
        "retention": "numerator metric / denominator metric; null at configured floor",
        "total_variation": "mean of mean absolute first differences over phi/theta/r",
        "gradient_energy": "sum of mean squared first differences over phi/theta/r",
        "near_zero": "abs(value) <= max(1e-30, 1e-6 * RMS)",
        "center_region": "frozen radial shells 1-2",
        "polar_region": "theta <= pi/6 or theta >= 5pi/6",
        "outer_region": "frozen radial shells 7-8",
        "shells": "eight frozen physical-r shells; no batch/channel aggregation",
        "radial_profile": "mean over phi/theta at every r center",
        "spectrum": {
            "semantics": "stored-index-space diagnostic; not Kerr-Schild invariant",
            "axes": [*AXES, "combined"],
            "combined_frequency": "max absolute cycles/index across phi/theta/r",
            "window": "none",
            "normalization": "orthonormal FFT",
            "variants": ["original", "mean-subtracted"],
            "bands": {
                "low_k": "k <= 0.125",
                "mid_k": "0.125 < k <= 0.25",
                "high_k": "k > 0.25",
            },
        },
    }

    preprocessing_rows: list[dict[str, Any]] = []
    preprocessing_detector_flags: dict[str, list[int]] = {name: [] for name in CHANNELS}
    limit = processor.preprocessor.gamma * processor.preprocessor.inverse_clamp_fraction
    for step in range(1, 20):
        detector = artifact_diagnostics(
            oracle_states[step],
            raw_states[step],
            raw_states[step],
            reference_kind="same_snapshot_raw_truth",
        )
        for channel, channel_name in enumerate(CHANNELS):
            raw = cached("raw", step, raw_states[step], channel)
            oracle = cached("oracle", step, oracle_states[step], channel)
            raw_field = state_numpy(raw_states[step], channel)
            oracle_field = state_numpy(oracle_states[step], channel)
            normalized = normalized_targets[step][:, channel]
            shell_ratios = [
                safe_retention(o["variance"], r_["variance"], epsilon=epsilon)[0]
                for r_, o in zip(raw["shells"], oracle["shells"])
            ]
            radial_ratio, radial_undefined = safe_retention(
                oracle["radial"]["variance"], raw["radial"]["variance"], epsilon=epsilon
            )
            raw_spectrum = spectral_lookup(raw, "combined", True)
            oracle_spectrum = spectral_lookup(oracle, "combined", True)
            row: dict[str, Any] = {
                "step": step,
                "target_snapshot": 91 + step,
                "channel": channel_name,
                "raw_to_oracle_relative_l2": relative_l2(
                    oracle_field, raw_field, epsilon=epsilon
                ),
                "raw_oracle_sign_agreement": float(
                    np.mean(np.signbit(raw_field) == np.signbit(oracle_field))
                ),
                "raw_oracle_sign_change_fraction": float(
                    np.mean(np.signbit(raw_field) != np.signbit(oracle_field))
                ),
                "transform_clamp_occupancy": float(
                    (torch.abs(normalized) > limit).float().mean()
                ),
                "inverse_clamp_occupancy": float(
                    (torch.abs(normalized) > limit).float().mean()
                ),
                "raw_near_zero_occupancy": raw["basic"]["near_zero_occupancy"],
                "oracle_near_zero_occupancy": oracle["basic"]["near_zero_occupancy"],
                "detector_std_ratio_oracle_over_same_raw": detector["channels"][
                    channel_name
                ]["std_ratio_prediction_over_reference"],
                "detector_flag": (
                    f"{channel_name}:possible_field_collapse" in detector["flags"]
                ),
                "shell_variance_retention": shell_ratios,
                "shell_variance_retention_median": float(
                    np.median([value for value in shell_ratios if value is not None])
                ),
                "radial_profile_variance_retention": radial_ratio,
                "radial_profile_variance_undefined": radial_undefined,
            }
            for metric in (
                "std",
                "variance",
                "dynamic_span_q99_q01",
                "total_variation",
                "gradient_energy",
            ):
                ratio, undefined = safe_retention(
                    oracle["basic"][metric], raw["basic"][metric], epsilon=epsilon
                )
                row[f"raw_{metric}"] = raw["basic"][metric]
                row[f"oracle_{metric}"] = oracle["basic"][metric]
                row[f"preprocessing_retention_{metric}"] = ratio
                row[f"preprocessing_{metric}_undefined"] = undefined
            for energy in SPECTRAL_ENERGIES:
                ratio, undefined = safe_retention(
                    oracle_spectrum[energy], raw_spectrum[energy], epsilon=epsilon
                )
                row[f"raw_combined_demeaned_{energy}"] = raw_spectrum[energy]
                row[f"oracle_combined_demeaned_{energy}"] = oracle_spectrum[energy]
                row[f"preprocessing_retention_combined_demeaned_{energy}"] = ratio
                row[f"preprocessing_combined_demeaned_{energy}_undefined"] = undefined
            preprocessing_rows.append(row)
            if row["detector_flag"]:
                preprocessing_detector_flags[channel_name].append(step)

    initial_persistence_detector = artifact_diagnostics(
        initial_oracle,
        initial_raw,
        initial_raw,
        reference_kind="raw_initial_snapshot_91",
    )
    preprocessing_summary = {
        "detector_trigger_steps_by_channel": preprocessing_detector_flags,
        "canonical_persistence_initial_detector_flags": initial_persistence_detector[
            "flags"
        ],
        "canonical_persistence_initial_std_ratios": {
            channel: initial_persistence_detector["channels"][channel][
                "std_ratio_prediction_over_reference"
            ]
            for channel in CHANNELS
        },
    }
    preprocessing_payload = write_result(
        output_dir,
        "preprocessing_floor",
        rows=preprocessing_rows,
        common_metadata=common_metadata,
        metric_definitions=metric_definitions,
        summary=preprocessing_summary,
    )

    gt_variance_rows: list[dict[str, Any]] = []
    gt_shell_rows: list[dict[str, Any]] = []
    gt_radial_rows: list[dict[str, Any]] = []
    gt_spectral_rows: list[dict[str, Any]] = []
    rollout_json = {
        name: load_json(experiment_dirs[name] / "gt_rollout.json")
        for name in ("fno_plain", "fno_full", "localno_plain")
    }
    rollout_records = {
        name: {int(record["step"]): record for record in payload["records"]}
        for name, payload in rollout_json.items()
    }
    for step in GT_STEPS:
        for channel, channel_name in enumerate(CHANNELS):
            raw = cached("raw", step, raw_states[step], channel)
            oracle = cached("oracle", step, oracle_states[step], channel)
            raw_field = state_numpy(raw_states[step], channel)
            oracle_field = state_numpy(oracle_states[step], channel)
            oracle_pair = pair_metrics(oracle_field, raw_field, epsilon=epsilon)
            for model_name in MODELS:
                model = cached(
                    model_name, step, model_states[model_name][step], channel
                )
                model_field = state_numpy(model_states[model_name][step], channel)
                row: dict[str, Any] = {
                    "step": step,
                    "target_snapshot": 91 + step,
                    "model": model_name,
                    "channel": channel_name,
                    "oracle_vs_raw_pearson": oracle_pair["pearson_correlation"],
                    "oracle_vs_raw_cosine": oracle_pair["cosine_similarity"],
                    "oracle_vs_raw_sign_agreement": oracle_pair["sign_agreement"],
                }
                for metric in RETENTION_METRICS:
                    row.update(
                        prefixed_decomposition(
                            metric,
                            raw["basic"][metric],
                            oracle["basic"][metric],
                            model["basic"][metric],
                            epsilon,
                        )
                    )
                for prefix, reference_field in (
                    ("model_vs_oracle", oracle_field),
                    ("model_vs_raw", raw_field),
                ):
                    paired = pair_metrics(model_field, reference_field, epsilon=epsilon)
                    for metric, value in paired.items():
                        row[f"{prefix}_{metric}"] = value
                if model_name == "persistence":
                    saturation = saturation_record(
                        normalized_model=initial_normalized,
                        normalized_target=normalized_targets[step],
                        channel=channel,
                        limit=limit,
                    )
                else:
                    saved = rollout_records[model_name][step]["saturation"]["channels"][
                        channel_name
                    ]
                    saturation = {
                        "target_clamp_fraction": saved["target_clamp_fraction"],
                        "model_clamp_fraction": saved["model_clamp_fraction"],
                        "target_only_saturation": saved["target_only_fraction"],
                        "model_only_saturation": saved["model_only_fraction"],
                    }
                row.update(saturation)
                gt_variance_rows.append(row)

                for raw_shell, oracle_shell, model_shell in zip(
                    raw["shells"], oracle["shells"], model["shells"]
                ):
                    shell_row: dict[str, Any] = {
                        "step": step,
                        "model": model_name,
                        "channel": channel_name,
                        "shell": raw_shell["shell"],
                        "shell_group": raw_shell["shell_group"],
                        "voxel_count": raw_shell["voxel_count"],
                    }
                    for metric in SHELL_METRICS:
                        shell_row.update(
                            prefixed_decomposition(
                                metric,
                                raw_shell[metric],
                                oracle_shell[metric],
                                model_shell[metric],
                                epsilon,
                            )
                        )
                    gt_shell_rows.append(shell_row)

                radial_row = {
                    "step": step,
                    "model": model_name,
                    "channel": channel_name,
                    "profile_length": raw["radial"]["length"],
                    "raw_profile": raw["radial"]["profile"],
                    "oracle_profile": oracle["radial"]["profile"],
                    "model_profile": model["radial"]["profile"],
                }
                for metric in ("variance", "std", "dynamic_span"):
                    radial_row.update(
                        prefixed_decomposition(
                            f"radial_profile_{metric}",
                            raw["radial"][metric],
                            oracle["radial"][metric],
                            model["radial"][metric],
                            epsilon,
                        )
                    )
                gt_radial_rows.append(radial_row)

                for raw_spectrum, oracle_spectrum, model_spectrum in zip(
                    raw["spectra"], oracle["spectra"], model["spectra"]
                ):
                    spectral_row: dict[str, Any] = {
                        "step": step,
                        "model": model_name,
                        "channel": channel_name,
                        "axis": raw_spectrum["axis"],
                        "demeaned": raw_spectrum["demeaned"],
                        "window": "none",
                        "normalization": "ortho",
                        "raw_parseval_relative_error": raw_spectrum[
                            "parseval_relative_error"
                        ],
                        "oracle_parseval_relative_error": oracle_spectrum[
                            "parseval_relative_error"
                        ],
                        "model_parseval_relative_error": model_spectrum[
                            "parseval_relative_error"
                        ],
                        "raw_spectral_centroid": raw_spectrum["spectral_centroid"],
                        "oracle_spectral_centroid": oracle_spectrum[
                            "spectral_centroid"
                        ],
                        "model_spectral_centroid": model_spectrum[
                            "spectral_centroid"
                        ],
                        "raw_high_k_axis_anisotropy": high_k_axis_anisotropy(
                            raw, demean=raw_spectrum["demeaned"]
                        ),
                        "oracle_high_k_axis_anisotropy": high_k_axis_anisotropy(
                            oracle, demean=oracle_spectrum["demeaned"]
                        ),
                        "model_high_k_axis_anisotropy": high_k_axis_anisotropy(
                            model, demean=model_spectrum["demeaned"]
                        ),
                    }
                    for energy in SPECTRAL_ENERGIES:
                        spectral_row.update(
                            prefixed_decomposition(
                                energy,
                                raw_spectrum[energy],
                                oracle_spectrum[energy],
                                model_spectrum[energy],
                                epsilon,
                            )
                        )
                    gt_spectral_rows.append(spectral_row)

    gt_variance_payload = write_result(
        output_dir,
        "gt_variance_retention",
        rows=gt_variance_rows,
        common_metadata=common_metadata,
        metric_definitions=metric_definitions,
    )
    gt_shell_payload = write_result(
        output_dir,
        "gt_shell_retention",
        rows=gt_shell_rows,
        common_metadata=common_metadata,
        metric_definitions=metric_definitions,
    )
    gt_radial_payload = write_result(
        output_dir,
        "gt_radial_retention",
        rows=gt_radial_rows,
        common_metadata=common_metadata,
        metric_definitions=metric_definitions,
    )
    gt_spectral_payload = write_result(
        output_dir,
        "gt_spectral_retention",
        rows=gt_spectral_rows,
        common_metadata=common_metadata,
        metric_definitions=metric_definitions,
        summary={
            "maximum_parseval_relative_error": max(
                row[key]
                for row in gt_spectral_rows
                for key in (
                    "raw_parseval_relative_error",
                    "oracle_parseval_relative_error",
                    "model_parseval_relative_error",
                )
            )
        },
    )
    write_result(
        output_dir,
        "shell_retention_heatmap_data",
        rows=[
            row
            for row in gt_shell_rows
            if row["channel"] in {"Bcc3", "vel3"}
        ],
        common_metadata=common_metadata,
        metric_definitions=metric_definitions,
    )

    no_gt_variance_rows: list[dict[str, Any]] = []
    no_gt_shell_rows: list[dict[str, Any]] = []
    no_gt_radial_rows: list[dict[str, Any]] = []
    no_gt_spectral_rows: list[dict[str, Any]] = []
    previous_steps = {25: 19, 50: 25, 75: 50, 100: 75}
    initial_analysis = {
        "raw": {
            channel: analyze_field(
                state_numpy(initial_raw, index),
                region_masks=region_masks,
                shell_index=shell_index,
            )
            for index, channel in enumerate(CHANNELS)
        },
        "oracle": {
            channel: analyze_field(
                state_numpy(initial_oracle, index),
                region_masks=region_masks,
                shell_index=shell_index,
            )
            for index, channel in enumerate(CHANNELS)
        },
    }
    no_gt_detectors: dict[str, dict[int, Any]] = {name: {} for name in MODELS}
    for model_name in MODELS:
        for step in NO_GT_STEPS:
            state = model_states[model_name][step]
            detector = artifact_diagnostics(
                state,
                initial_raw,
                initial_raw,
                reference_kind="initial_snapshot_91",
            )
            no_gt_detectors[model_name][step] = detector
            previous_state = model_states[model_name][previous_steps[step]]
            for channel, channel_name in enumerate(CHANNELS):
                model = cached(model_name, step, state, channel)
                raw_initial = initial_analysis["raw"][channel_name]
                oracle_initial = initial_analysis["oracle"][channel_name]
                model_field = state_numpy(state, channel)
                raw_initial_field = state_numpy(initial_raw, channel)
                oracle_initial_field = state_numpy(initial_oracle, channel)
                previous_field = state_numpy(previous_state, channel)
                envelope = truth_envelope[channel_name]
                violation_count = sum(
                    not (
                        envelope[metric]["minimum"]
                        <= model["basic"][metric]
                        <= envelope[metric]["maximum"]
                    )
                    for metric in (
                        "mean",
                        "std",
                        "variance",
                        "dynamic_span_q99_q01",
                        "total_variation",
                    )
                )
                variance_row: dict[str, Any] = {
                    "step": step,
                    "model": model_name,
                    "channel": channel_name,
                    "ground_truth_available": False,
                    "oracle_available": False,
                    **model["basic"],
                    "detector_std_ratio": detector["channels"][channel_name][
                        "std_ratio_prediction_over_reference"
                    ],
                    "detector_flag": (
                        f"{channel_name}:possible_field_collapse" in detector["flags"]
                    ),
                    "selected_step_interval": step - previous_steps[step],
                    "selected_step_change_relative_l2": relative_l2(
                        model_field, previous_field, epsilon=epsilon
                    ),
                    "distance_from_raw_initial_relative_l2": relative_l2(
                        model_field, raw_initial_field, epsilon=epsilon
                    ),
                    "distance_from_canonical_initial_relative_l2": relative_l2(
                        model_field, oracle_initial_field, epsilon=epsilon
                    ),
                    "distance_from_persistence_relative_l2": relative_l2(
                        model_field, oracle_initial_field, epsilon=epsilon
                    ),
                    "truth_envelope_metric_violation_count": int(violation_count),
                    "truth_envelope_metric_violation_fraction": violation_count / 5.0,
                    "truth_raw_value_range_violation_fraction": float(
                        np.mean(
                            (model_field < envelope["raw_value_minimum"])
                            | (model_field > envelope["raw_value_maximum"])
                        )
                    ),
                }
                no_gt_variance_rows.append(variance_row)

                for raw_shell, oracle_shell, model_shell in zip(
                    raw_initial["shells"],
                    oracle_initial["shells"],
                    model["shells"],
                ):
                    row = {
                        "step": step,
                        "model": model_name,
                        "channel": channel_name,
                        "shell": model_shell["shell"],
                        "shell_group": model_shell["shell_group"],
                        "voxel_count": model_shell["voxel_count"],
                        "ground_truth_available": False,
                        "oracle_available": False,
                    }
                    for metric in SHELL_METRICS:
                        raw_ratio, raw_undefined = safe_retention(
                            model_shell[metric], raw_shell[metric], epsilon=epsilon
                        )
                        oracle_ratio, oracle_undefined = safe_retention(
                            model_shell[metric], oracle_shell[metric], epsilon=epsilon
                        )
                        row[f"model_{metric}"] = model_shell[metric]
                        row[f"retention_vs_raw_initial_{metric}"] = raw_ratio
                        row[f"retention_vs_raw_initial_{metric}_undefined"] = raw_undefined
                        row[f"retention_vs_canonical_initial_{metric}"] = oracle_ratio
                        row[
                            f"retention_vs_canonical_initial_{metric}_undefined"
                        ] = oracle_undefined
                    no_gt_shell_rows.append(row)

                radial_row = {
                    "step": step,
                    "model": model_name,
                    "channel": channel_name,
                    "ground_truth_available": False,
                    "oracle_available": False,
                    "profile": model["radial"]["profile"],
                    "profile_length": model["radial"]["length"],
                    "variance": model["radial"]["variance"],
                    "std": model["radial"]["std"],
                    "dynamic_span": model["radial"]["dynamic_span"],
                }
                for metric in ("variance", "std", "dynamic_span"):
                    ratio, undefined = safe_retention(
                        model["radial"][metric],
                        oracle_initial["radial"][metric],
                        epsilon=epsilon,
                    )
                    radial_row[f"retention_vs_canonical_initial_{metric}"] = ratio
                    radial_row[
                        f"retention_vs_canonical_initial_{metric}_undefined"
                    ] = undefined
                no_gt_radial_rows.append(radial_row)

                for spectrum in model["spectra"]:
                    initial_spectrum = spectral_lookup(
                        oracle_initial, spectrum["axis"], spectrum["demeaned"]
                    )
                    row = {
                        "step": step,
                        "model": model_name,
                        "channel": channel_name,
                        "axis": spectrum["axis"],
                        "demeaned": spectrum["demeaned"],
                        "ground_truth_available": False,
                        "oracle_available": False,
                        "spectral_centroid": spectrum["spectral_centroid"],
                        "parseval_relative_error": spectrum["parseval_relative_error"],
                        "high_k_axis_anisotropy": high_k_axis_anisotropy(
                            model, demean=spectrum["demeaned"]
                        ),
                    }
                    for energy in SPECTRAL_ENERGIES:
                        ratio, undefined = safe_retention(
                            spectrum[energy], initial_spectrum[energy], epsilon=epsilon
                        )
                        row[energy] = spectrum[energy]
                        row[f"retention_vs_canonical_initial_{energy}"] = ratio
                        row[
                            f"retention_vs_canonical_initial_{energy}_undefined"
                        ] = undefined
                    no_gt_spectral_rows.append(row)

    no_gt_variance_payload = write_result(
        output_dir,
        "no_gt_variance",
        rows=no_gt_variance_rows,
        common_metadata=common_metadata,
        metric_definitions=metric_definitions,
        summary={"explicit_statement": "no oracle exists after step 19"},
    )
    no_gt_shell_payload = write_result(
        output_dir,
        "no_gt_shells",
        rows=no_gt_shell_rows,
        common_metadata=common_metadata,
        metric_definitions=metric_definitions,
        summary={"explicit_statement": "no oracle exists after step 19"},
    )
    no_gt_radial_payload = write_result(
        output_dir,
        "no_gt_radial",
        rows=no_gt_radial_rows,
        common_metadata=common_metadata,
        metric_definitions=metric_definitions,
        summary={"explicit_statement": "no oracle exists after step 19"},
    )
    no_gt_spectrum_payload = write_result(
        output_dir,
        "no_gt_spectrum",
        rows=no_gt_spectral_rows,
        common_metadata=common_metadata,
        metric_definitions=metric_definitions,
        summary={
            "explicit_statement": "no oracle exists after step 19",
            "maximum_parseval_relative_error": max(
                row["parseval_relative_error"] for row in no_gt_spectral_rows
            ),
        },
    )

    comparison_rows: list[dict[str, Any]] = []
    for model_name in MODELS:
        for channel_name in CHANNELS:
            gt_rows = [
                row
                for row in gt_variance_rows
                if row["model"] == model_name and row["channel"] == channel_name
            ]
            spectral_rows = [
                row
                for row in gt_spectral_rows
                if row["model"] == model_name
                and row["channel"] == channel_name
                and row["axis"] == "combined"
                and row["demeaned"] is True
            ]
            no_gt_rows = [
                row
                for row in no_gt_variance_rows
                if row["model"] == model_name and row["channel"] == channel_name
            ]
            collapse_steps = [row["step"] for row in no_gt_rows if row["detector_flag"]]
            comparison_rows.append(
                {
                    "model": model_name,
                    "channel": channel_name,
                    "gt_median_preprocessing_variance_retention": float(
                        np.median(
                            [row["preprocessing_retention_variance"] for row in gt_rows]
                        )
                    ),
                    "gt_median_model_variance_retention": float(
                        np.median([row["model_retention_variance"] for row in gt_rows])
                    ),
                    "gt_median_model_dynamic_span_retention": float(
                        np.median(
                            [
                                row["model_retention_dynamic_span_q99_q01"]
                                for row in gt_rows
                            ]
                        )
                    ),
                    "gt_median_model_total_variation_retention": float(
                        np.median(
                            [row["model_retention_total_variation"] for row in gt_rows]
                        )
                    ),
                    "gt_median_model_high_k_retention": float(
                        np.median(
                            [row["model_retention_high_k_energy"] for row in spectral_rows]
                        )
                    ),
                    "no_gt_detector_collapse_steps": collapse_steps,
                    "earliest_selected_no_gt_collapse_step": (
                        min(collapse_steps) if collapse_steps else None
                    ),
                    "no_gt_std_ratios": {
                        str(row["step"]): row["detector_std_ratio"] for row in no_gt_rows
                    },
                }
            )
    comparison_payload = write_result(
        output_dir,
        "fno_localno_comparison",
        rows=comparison_rows,
        common_metadata=common_metadata,
        metric_definitions=metric_definitions,
    )

    attribution_rows: list[dict[str, Any]] = []
    channel_decisions: dict[str, str] = {}
    for channel_name in ("Bcc3", "vel3"):
        local_variance = [
            row
            for row in gt_variance_rows
            if row["model"] == "localno_plain" and row["channel"] == channel_name
        ]
        local_shell = [
            row
            for row in gt_shell_rows
            if row["model"] == "localno_plain" and row["channel"] == channel_name
        ]
        local_radial = [
            row
            for row in gt_radial_rows
            if row["model"] == "localno_plain" and row["channel"] == channel_name
        ]
        local_spectral = [
            row
            for row in gt_spectral_rows
            if row["model"] == "localno_plain"
            and row["channel"] == channel_name
            and row["axis"] == "combined"
            and row["demeaned"] is True
        ]
        pre_global_steps = [
            row["step"]
            for row in local_variance
            if row["preprocessing_retention_variance"] < severe_threshold
        ]
        model_global_steps = [
            row["step"]
            for row in local_variance
            if row["model_retention_variance"] < severe_threshold
        ]
        pre_shell_radial_steps: list[int] = []
        model_shell_radial_steps: list[int] = []
        shell_step_summary: dict[str, Any] = {}
        for step in GT_STEPS:
            shell_rows = [row for row in local_shell if row["step"] == step]
            radial_row = next(row for row in local_radial if row["step"] == step)
            pre_shell_values = [
                row["preprocessing_retention_variance"]
                for row in shell_rows
                if row["preprocessing_retention_variance"] is not None
            ]
            model_shell_values = [
                row["model_retention_variance"]
                for row in shell_rows
                if row["model_retention_variance"] is not None
            ]
            pre_shell_median = (
                None if not pre_shell_values else float(np.median(pre_shell_values))
            )
            model_shell_median = (
                None if not model_shell_values else float(np.median(model_shell_values))
            )
            pre_radial = radial_row["preprocessing_retention_radial_profile_variance"]
            model_radial = radial_row["model_retention_radial_profile_variance"]
            if (
                pre_shell_median is not None and pre_shell_median < severe_threshold
            ) or (pre_radial is not None and pre_radial < severe_threshold):
                pre_shell_radial_steps.append(step)
            if (
                model_shell_median is not None and model_shell_median < severe_threshold
            ) or (model_radial is not None and model_radial < severe_threshold):
                model_shell_radial_steps.append(step)
            shell_step_summary[str(step)] = {
                "preprocessing_shell_median": pre_shell_median,
                "preprocessing_radial": pre_radial,
                "model_shell_median": model_shell_median,
                "model_radial": model_radial,
            }
        pre_high_steps = [
            row["step"]
            for row in local_spectral
            if row["preprocessing_retention_high_k_energy"] < severe_threshold
        ]
        model_high_steps = [
            row["step"]
            for row in local_spectral
            if row["model_retention_high_k_energy"] < severe_threshold
        ]
        required_steps = int(
            config["retention_contract"]["selected_gt_step_aggregation"][
                "required_steps"
            ]
        )
        pre_evidence = {
            "global_variance": len(pre_global_steps) >= required_steps,
            "shell_radial_variance": len(pre_shell_radial_steps) >= required_steps,
            "high_k_spectral_energy": len(pre_high_steps) >= required_steps,
        }
        model_evidence = {
            "global_variance": len(model_global_steps) >= required_steps,
            "shell_radial_variance": len(model_shell_radial_steps) >= required_steps,
            "high_k_spectral_energy": len(model_high_steps) >= required_steps,
        }
        oracle_trigger_steps = preprocessing_detector_flags[channel_name]
        local_collapse_steps = [
            step
            for step in NO_GT_STEPS
            if f"{channel_name}:possible_field_collapse"
            in no_gt_detectors["localno_plain"][step]["flags"]
        ]
        decision = classify_channel(
            preprocessing_severe=pre_evidence,
            model_severe=model_evidence,
            oracle_detector_triggered=bool(oracle_trigger_steps),
            localno_detector_triggered=bool(local_collapse_steps),
            oracle_degraded=sum(pre_evidence.values()) >= 1,
        )
        channel_decisions[channel_name] = decision.choice
        attribution_rows.append(
            {
                "channel": channel_name,
                "decision": decision.choice,
                "reason": decision.reason,
                "preprocessing_core_evidence": pre_evidence,
                "model_core_evidence": model_evidence,
                "preprocessing_severe_core_count": decision.preprocessing_severe_count,
                "model_severe_core_count": decision.model_severe_count,
                "preprocessing_global_variance_severe_steps": pre_global_steps,
                "model_global_variance_severe_steps": model_global_steps,
                "preprocessing_shell_radial_severe_steps": pre_shell_radial_steps,
                "model_shell_radial_severe_steps": model_shell_radial_steps,
                "preprocessing_high_k_severe_steps": pre_high_steps,
                "model_high_k_severe_steps": model_high_steps,
                "shell_radial_step_values": shell_step_summary,
                "oracle_same_snapshot_detector_trigger_steps": oracle_trigger_steps,
                "localno_frozen_no_gt_detector_trigger_steps": local_collapse_steps,
            }
        )
    overall = overall_attribution(channel_decisions)
    attribution_payload = write_result(
        output_dir,
        "collapse_attribution",
        rows=attribution_rows,
        common_metadata=common_metadata,
        metric_definitions=metric_definitions,
        summary={"channel_decisions": channel_decisions, "overall_decision": overall},
    )
    all_payloads = (
        preprocessing_payload,
        gt_variance_payload,
        gt_shell_payload,
        gt_radial_payload,
        gt_spectral_payload,
        no_gt_variance_payload,
        no_gt_shell_payload,
        no_gt_radial_payload,
        no_gt_spectrum_payload,
        comparison_payload,
        attribution_payload,
    )
    undefined_total = sum(
        payload["metadata"]["undefined_ratio_count"] for payload in all_payloads
    )
    decision_payload = {
        "schema_version": "paper-stage-l-decision-v1",
        "classification": "post_hoc_attribution",
        "channel_decisions": channel_decisions,
        "overall_decision": overall,
        "stage_k_decision_unchanged": "C. TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE",
        "no_training": True,
        "no_oracle_after_step_19": True,
        "metric_definitions": metric_definitions,
        "undefined_ratio_count": undefined_total,
        "core_evidence": {
            row["channel"]: {
                "preprocessing": row["preprocessing_core_evidence"],
                "model_added": row["model_core_evidence"],
                "preprocessing_count": row["preprocessing_severe_core_count"],
                "model_added_count": row["model_severe_core_count"],
            }
            for row in attribution_rows
        },
        "provenance": common_metadata,
    }
    write_json(output_dir / "stage_l_decision.json", decision_payload)
    manifest.update(
        {
            "status": "attribution_complete",
            "scope_completed": [
                "provenance_gate",
                "detector_source_audit",
                "stage_k_flag_reproduction",
                "frozen_attribution_config",
                "preprocessing_floor",
                "gt_variance_shell_radial_spectrum",
                "no_gt_variance_shell_radial_spectrum",
                "fno_localno_comparison",
                "channel_and_overall_attribution",
            ],
            "scope_not_started": [],
            "stage_l_phase_2": {
                "config_sha256": EXPECTED_CONFIG_SHA256,
                "snapshot_91_raw_state_sha256": initial_raw_hash,
                "selected_state_hashes": selected_hashes,
                "checkpoint_hashes": checkpoint_hashes,
                "rollout_hashes": {
                    name: {
                        "gt": sha256_file(experiment_dirs[name] / "gt_rollout.json"),
                        "no_gt": sha256_file(
                            experiment_dirs[name] / "no_gt_rollout.json"
                        ),
                    }
                    for name in experiment_dirs
                },
                "supplemental_deterministic_step3": supplemental_reload,
                "undefined_ratio_count": undefined_total,
                "maximum_gt_parseval_relative_error": gt_spectral_payload["summary"][
                    "maximum_parseval_relative_error"
                ],
                "maximum_no_gt_parseval_relative_error": no_gt_spectrum_payload[
                    "summary"
                ]["maximum_parseval_relative_error"],
                "metric_definitions": metric_definitions,
                "channel_decisions": channel_decisions,
                "overall_decision": overall,
            },
        }
    )
    write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "channel_decisions": channel_decisions,
                "overall_decision": overall,
                "undefined_ratio_count": undefined_total,
                "gt_parseval_max": gt_spectral_payload["summary"][
                    "maximum_parseval_relative_error"
                ],
                "no_gt_parseval_max": no_gt_spectrum_payload["summary"][
                    "maximum_parseval_relative_error"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
