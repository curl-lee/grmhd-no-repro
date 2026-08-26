#!/usr/bin/env python3
"""Evaluate frozen Stage S best checkpoints on one shared validation trajectory."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch
import yaml

from grmhd import CHANNELS
from grmhd.paper_stage_g_evaluation import radial_shell_indices
from grmhd.paper_stage_m import radial_profile_vector, transport_metrics, variance_vector

from train_stage_s import StageSBatchPath, build_frozen_model


TARGET_CHANNELS = ("Bcc2", "Bcc3", "vel3")
SELECTED_STEPS = (1, 2, 3, 5, 10, 19, 25, 42, 50, 100)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def encoded_channel(raw: np.ndarray, channel: int, processor: Any) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64).copy()
    processor.base._transform_values_inplace(values, channel, processor.epsilon[channel])
    z = (values - processor.median[channel]) / processor.scale[channel]
    if processor.spec.channel_policies[channel] != "no_softclip":
        z = processor.gamma * np.tanh(z / processor.gamma)
    return z


def build_train_reference(data: StageSBatchPath, output: Path) -> dict[str, Any]:
    norm_squared = np.zeros(169, dtype=np.float64)
    channels: dict[str, Any] = {}
    for channel, name in enumerate(CHANNELS):
        raw = np.asarray(data.snapshots[0:169, channel], dtype=np.float32)
        normalized = encoded_channel(raw, channel, data.preprocessor)
        norm_squared += np.sum(np.square(normalized), axis=(1, 2, 3), dtype=np.float64)
        raw_q = np.quantile(raw, [0.001, 0.999])
        norm_q = np.quantile(normalized, [0.001, 0.999])
        channels[name] = {
            "physical": {
                "minimum": float(raw.min()), "maximum": float(raw.max()),
                "q001": float(raw_q[0]), "q999": float(raw_q[1]),
            },
            "normalized": {
                "minimum": float(normalized.min()), "maximum": float(normalized.max()),
                "q001": float(norm_q[0]), "q999": float(norm_q[1]),
            },
        }
        del raw, normalized
    norms = np.sqrt(norm_squared)
    rmax = float(norms.max())
    payload = {
        "schema_version": "stage-s-train-only-diagnostic-reference-v1",
        "training_indices": list(range(169)),
        "validation_indices_used_for_fit": [],
        "train_only": True,
        "quantiles": [0.001, 0.999],
        "normalized_global_norm": {
            "per_snapshot": norms.tolist(),
            "Rmax": rmax,
            "Rin": 1.05 * rmax,
            "Rout": 1.5 * 1.05 * rmax,
            "formula": "Rmax=max(train snapshot norms); Rin=1.05*Rmax; Rout=1.5*Rin",
        },
        "channels": channels,
    }
    write_json(output / "diagnostic_reference_train_only.json", payload)
    return payload


def accumulator() -> dict[str, np.ndarray]:
    return {
        key: np.zeros(len(CHANNELS), dtype=np.float64)
        for key in ("num", "den", "physical_num", "physical_den", "res_num", "res_den", "dot", "left", "right")
    }


def finalized(values: Mapping[str, np.ndarray]) -> dict[str, Any]:
    rel = np.sqrt(values["num"] / np.maximum(values["den"], 1e-300))
    physical = np.sqrt(values["physical_num"] / np.maximum(values["physical_den"], 1e-300))
    residual = np.sqrt(values["res_num"] / np.maximum(values["res_den"], 1e-300))
    cosine = values["dot"] / np.maximum(np.sqrt(values["left"] * values["right"]), 1e-300)
    return {
        "normalized_relative_l2": {
            "per_channel": dict(zip(CHANNELS, rel.tolist(), strict=True)),
            "arithmetic_average": float(rel.mean()),
            "global": float(math.sqrt(values["num"].sum() / max(values["den"].sum(), 1e-300))),
        },
        "physical_relative_l2": {
            "per_channel": dict(zip(CHANNELS, physical.tolist(), strict=True)),
            "arithmetic_average": float(physical.mean()),
            "global": float(math.sqrt(values["physical_num"].sum() / max(values["physical_den"].sum(), 1e-300))),
        },
        "residual": {
            "per_channel_relative_l2": dict(zip(CHANNELS, residual.tolist(), strict=True)),
            "arithmetic_average_relative_l2": float(residual.mean()),
            "per_channel_cosine": dict(zip(CHANNELS, cosine.tolist(), strict=True)),
            "arithmetic_average_cosine": float(cosine.mean()),
            "global_cosine": float(values["dot"].sum() / max(math.sqrt(values["left"].sum() * values["right"].sum()), 1e-300)),
        },
    }


def update_accumulator(
    values: dict[str, np.ndarray], *, z_prediction: np.ndarray, z_target: np.ndarray,
    physical_prediction: np.ndarray, physical_target: np.ndarray,
    predicted_residual: np.ndarray, residual_target: np.ndarray,
) -> None:
    axes = (1, 2, 3)
    values["num"] += np.sum((z_prediction - z_target) ** 2, axis=axes)
    values["den"] += np.sum(z_target ** 2, axis=axes)
    values["physical_num"] += np.sum((physical_prediction - physical_target) ** 2, axis=axes)
    values["physical_den"] += np.sum(physical_target ** 2, axis=axes)
    values["res_num"] += np.sum((predicted_residual - residual_target) ** 2, axis=axes)
    values["res_den"] += np.sum(residual_target ** 2, axis=axes)
    values["dot"] += np.sum(predicted_residual * residual_target, axis=axes)
    values["left"] += np.sum(predicted_residual ** 2, axis=axes)
    values["right"] += np.sum(residual_target ** 2, axis=axes)


def transport_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"per_channel": {}}
    all_shell: list[float] = []
    all_radial: list[float] = []
    for name in TARGET_CHANNELS:
        channel_rows = [row for row in rows if row["channel"] == name]
        shell = [row["shell"]["persistence_relative_skill"] for row in channel_rows]
        radial = [row["radial"]["persistence_relative_skill"] for row in channel_rows]
        shell = [float(value) for value in shell if value is not None]
        radial = [float(value) for value in radial if value is not None]
        output["per_channel"][name] = {
            "shell_skill_median": None if not shell else float(np.median(shell)),
            "radial_skill_median": None if not radial else float(np.median(radial)),
            "shell_skill_mean": None if not shell else float(np.mean(shell)),
            "radial_skill_mean": None if not radial else float(np.mean(radial)),
        }
        all_shell.extend(shell)
        all_radial.extend(radial)
    output["shell_skill_median"] = None if not all_shell else float(np.median(all_shell))
    output["radial_skill_median"] = None if not all_radial else float(np.median(all_radial))
    return output


def one_step(
    model: torch.nn.Module, data: StageSBatchPath, validation_pairs: list[int],
    shell_index: np.ndarray, train_reference: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    model.eval()
    learned = accumulator()
    persistence = accumulator()
    oracle_floor_num = np.zeros(8, dtype=np.float64)
    oracle_floor_den = np.zeros(8, dtype=np.float64)
    quantiles: dict[str, dict[str, list[float]]] = {
        kind: {name: [] for name in CHANNELS}
        for kind in ("raw_target_q001", "raw_target_q999", "model_q001", "model_q999", "persistence_q001", "persistence_q999")
    }
    transport_rows: list[dict[str, Any]] = []
    saturation = {kind: np.zeros(8, dtype=np.float64) for kind in ("target", "model", "model_only")}
    count = 0
    above_rout_count = 0
    normalized_norms: list[float] = []
    finite = positive = True
    with torch.no_grad():
        for source in validation_pairs:
            result = data.predict(model, source)
            model_physical = data.preprocessor.decode_tensor(result["z_prediction"], channel_axis=1)
            persistence_physical = data.preprocessor.decode_tensor(result["z_input"], channel_axis=1)
            oracle_physical = data.preprocessor.decode_tensor(result["z_target"], channel_axis=1)
            arrays = {
                key: value[0].detach().cpu().numpy().astype(np.float64)
                for key, value in {
                    "z_prediction": result["z_prediction"], "z_target": result["z_target"],
                    "z_input": result["z_input"], "predicted_residual": result["predicted_residual"],
                    "residual_target": result["residual_target"], "model_physical": model_physical,
                    "persistence_physical": persistence_physical, "oracle_physical": oracle_physical,
                    "raw_target": result["raw_target"],
                }.items()
            }
            update_accumulator(
                learned, z_prediction=arrays["z_prediction"], z_target=arrays["z_target"],
                physical_prediction=arrays["model_physical"], physical_target=arrays["raw_target"],
                predicted_residual=arrays["predicted_residual"], residual_target=arrays["residual_target"],
            )
            update_accumulator(
                persistence, z_prediction=arrays["z_input"], z_target=arrays["z_target"],
                physical_prediction=arrays["persistence_physical"], physical_target=arrays["raw_target"],
                predicted_residual=np.zeros_like(arrays["residual_target"]), residual_target=arrays["residual_target"],
            )
            oracle_floor_num += np.sum((arrays["oracle_physical"] - arrays["raw_target"]) ** 2, axis=(1, 2, 3))
            oracle_floor_den += np.sum(arrays["raw_target"] ** 2, axis=(1, 2, 3))
            target_mask = np.abs(arrays["z_target"]) > 5.94
            model_mask = np.abs(arrays["z_prediction"]) > 5.94
            saturation["target"] += target_mask.mean(axis=(1, 2, 3))
            saturation["model"] += model_mask.mean(axis=(1, 2, 3))
            saturation["model_only"] += (model_mask & ~target_mask).mean(axis=(1, 2, 3))
            count += 1
            normalized_norm = float(np.linalg.norm(arrays["z_prediction"].ravel()))
            normalized_norms.append(normalized_norm)
            above_rout_count += int(
                normalized_norm > float(train_reference["normalized_global_norm"]["Rout"])
            )
            for channel, name in enumerate(CHANNELS):
                raw_q = np.quantile(arrays["raw_target"][channel], [0.001, 0.999])
                model_q = np.quantile(arrays["model_physical"][channel], [0.001, 0.999])
                pers_q = np.quantile(arrays["persistence_physical"][channel], [0.001, 0.999])
                for kind, value in (
                    ("raw_target_q001", raw_q[0]), ("raw_target_q999", raw_q[1]),
                    ("model_q001", model_q[0]), ("model_q999", model_q[1]),
                    ("persistence_q001", pers_q[0]), ("persistence_q999", pers_q[1]),
                ):
                    quantiles[kind][name].append(float(value))
            for name in TARGET_CHANNELS:
                channel = CHANNELS.index(name)
                input_field = arrays["persistence_physical"][channel]
                target_field = arrays["oracle_physical"][channel]
                model_field = arrays["model_physical"][channel]
                transport_rows.append({
                    "source_snapshot": source, "channel": name,
                    "shell": transport_metrics(
                        variance_vector(input_field, shell_index),
                        variance_vector(target_field, shell_index),
                        variance_vector(model_field, shell_index),
                        epsilon=1e-30, sign_zero_tolerance=1e-12,
                    ),
                    "radial": transport_metrics(
                        radial_profile_vector(input_field), radial_profile_vector(target_field),
                        radial_profile_vector(model_field), epsilon=1e-30, sign_zero_tolerance=1e-12,
                    ),
                })
            finite &= all(np.isfinite(value).all() for value in arrays.values())
            positive &= bool(np.all(arrays["model_physical"][3:5] > 0))
    learned_metrics = finalized(learned)
    persistence_metrics = finalized(persistence)
    floor = np.sqrt(oracle_floor_num / np.maximum(oracle_floor_den, 1e-300))
    learned_average = learned_metrics["normalized_relative_l2"]["arithmetic_average"]
    persistence_average = persistence_metrics["normalized_relative_l2"]["arithmetic_average"]
    payload = {
        "schema_version": "stage-s-one-step-metrics-v1",
        "validation_pairs": validation_pairs,
        "validation_pair_count": count,
        "finite": finite, "rho_press_positive": positive,
        "model": learned_metrics,
        "persistence": persistence_metrics,
        "model_error_over_persistence_error": learned_average / persistence_average,
        "oracle_floor_physical_relative_l2": {
            "per_channel": dict(zip(CHANNELS, floor.tolist(), strict=True)),
            "arithmetic_average": float(floor.mean()),
        },
        "q001_q999_pair_medians": {
            kind: {name: float(np.median(values)) for name, values in per_channel.items()}
            for kind, per_channel in quantiles.items()
        },
        "saturation_fraction_mean": {
            kind: dict(zip(CHANNELS, (values / count).tolist(), strict=True))
            for kind, values in saturation.items()
        },
        "transport": transport_summary(transport_rows),
        "transport_rows": transport_rows,
        "normalized_global_norm": {
            "mean": float(np.mean(normalized_norms)),
            "maximum": float(np.max(normalized_norms)),
            "above_Rout_fraction": above_rout_count / count,
        },
        "train_only_reference": {
            "Rout": train_reference["normalized_global_norm"]["Rout"],
            "validation_used_for_fit": False,
        },
    }
    return payload, persistence_metrics


def safe_artifact_metrics(prediction: np.ndarray, reference: np.ndarray, prior: np.ndarray) -> dict[str, Any]:
    channels: dict[str, Any] = {}
    flags: list[str] = []
    for channel, name in enumerate(CHANNELS):
        pred, truth, before = prediction[channel], reference[channel], prior[channel]
        std_ratio = float(np.std(pred) / max(float(np.std(truth)), 1e-30))
        pred_diff = [float(np.mean(np.diff(pred, axis=axis) ** 2)) for axis in range(3)]
        truth_diff = [float(np.mean(np.diff(truth, axis=axis) ** 2)) for axis in range(3)]
        prior_diff = [float(np.mean(np.diff(before, axis=axis) ** 2)) for axis in range(3)]
        creation = sum(pred_diff) / max(sum(truth_diff), sum(prior_diff), 1e-300)
        anisotropy = max(pred_diff) / max(min(pred_diff), 1e-300)
        prior_anisotropy = max(prior_diff) / max(min(prior_diff), 1e-300)
        excess = anisotropy / max(prior_anisotropy, 1e-300)
        channels[name] = {
            "std_ratio_prediction_over_reference": std_ratio,
            "created_roughness_ratio": creation,
            "axis_anisotropy_excess_over_input": excess,
        }
        if std_ratio < 0.05: flags.append(f"{name}:possible_field_collapse")
        if creation > 5: flags.append(f"{name}:possible_high_frequency_ripple")
        if excess > 10: flags.append(f"{name}:possible_stripe_anisotropy")
    return {"channels": channels, "flags": flags}


def spectral_summary(normalized: np.ndarray) -> dict[str, Any]:
    output = {}
    frequency = np.maximum.reduce(np.meshgrid(
        np.abs(np.fft.fftfreq(normalized.shape[1])),
        np.abs(np.fft.fftfreq(normalized.shape[2])),
        np.abs(np.fft.rfftfreq(normalized.shape[3])), indexing="ij",
    ))
    low = frequency <= 0.125
    high = frequency > 0.25
    for channel, name in enumerate(CHANNELS):
        spectrum = np.fft.rfftn(normalized[channel], norm="ortho")
        energy = np.abs(spectrum) ** 2
        output[name] = {
            "low_k_energy": float(energy[low].sum()),
            "high_k_energy": float(energy[high].sum()),
            "high_over_low": float(energy[high].sum() / max(float(energy[low].sum()), 1e-300)),
        }
    return output


def rollout(
    model: torch.nn.Module, data: StageSBatchPath, shell_index: np.ndarray,
    train_reference: Mapping[str, Any], *, steps: int = 100,
) -> dict[str, Any]:
    model.eval()
    current = data.raw(169).to(data.device)
    initial = current.detach().cpu().numpy()[0].astype(np.float64)
    records: list[dict[str, Any]] = []
    first_explosion: int | None = None
    rout = float(train_reference["normalized_global_norm"]["Rout"])
    state_encode_count = prediction_decode_count = 0
    with torch.no_grad():
        for step in range(1, steps + 1):
            z_input = data.preprocessor.encode_tensor(current, channel_axis=1)
            state_encode_count += 1
            model_input = torch.cat((z_input, data.shells), dim=1)
            predicted_residual = model(x=model_input)
            z_prediction = z_input + predicted_residual
            physical = data.preprocessor.decode_tensor(z_prediction, channel_axis=1)
            prediction_decode_count += 1
            pred_np = physical[0].detach().cpu().numpy().astype(np.float64)
            z_np = z_prediction[0].detach().cpu().numpy().astype(np.float64)
            prior_np = current[0].detach().cpu().numpy().astype(np.float64)
            finite = bool(np.isfinite(pred_np).all() and np.isfinite(z_np).all())
            positive = bool(np.all(pred_np[3:5] > 0))
            normalized_norm = float(np.linalg.norm(z_np.ravel()))
            physical_exceeded_train = any(
                np.max(np.abs(pred_np[channel]))
                > 10.0 * max(
                    abs(float(train_reference["channels"][name]["physical"]["minimum"])),
                    abs(float(train_reference["channels"][name]["physical"]["maximum"])),
                    1e-300,
                )
                for channel, name in enumerate(CHANNELS)
            )
            record: dict[str, Any] = {
                "step": step,
                "ground_truth_available": step <= 42,
                "finite": finite, "rho_press_positive": positive,
                "normalized_global_norm": normalized_norm,
                "above_Rout": normalized_norm > rout,
                "physical_range_explosion_vs_train": physical_exceeded_train,
                "normalized_range": {
                    name: {"minimum": float(z_np[channel].min()), "maximum": float(z_np[channel].max())}
                    for channel, name in enumerate(CHANNELS)
                },
                "physical_range": {
                    name: {"minimum": float(pred_np[channel].min()), "maximum": float(pred_np[channel].max())}
                    for channel, name in enumerate(CHANNELS)
                },
                "model_saturation_fraction": {
                    name: float(np.mean(np.abs(z_np[channel]) > 5.94))
                    for channel, name in enumerate(CHANNELS)
                },
                "spectral_normalized": spectral_summary(z_np),
            }
            if step <= 42:
                raw_target = data.raw(169 + step).to(data.device)
                z_target = data.preprocessor.encode_tensor(raw_target, channel_axis=1)
                target_oracle = data.preprocessor.decode_tensor(z_target, channel_axis=1)
                target_np = target_oracle[0].detach().cpu().numpy().astype(np.float64)
                raw_target_np = raw_target[0].detach().cpu().numpy().astype(np.float64)
                z_target_np = z_target[0].detach().cpu().numpy().astype(np.float64)
                residual_true = z_target_np - z_input[0].detach().cpu().numpy().astype(np.float64)
                residual_pred = predicted_residual[0].detach().cpu().numpy().astype(np.float64)
                norm_rel = [
                    np.linalg.norm((z_np[ch] - z_target_np[ch]).ravel())
                    / max(np.linalg.norm(z_target_np[ch].ravel()), 1e-300)
                    for ch in range(8)
                ]
                phys_rel = [
                    np.linalg.norm((pred_np[ch] - raw_target_np[ch]).ravel())
                    / max(np.linalg.norm(raw_target_np[ch].ravel()), 1e-300)
                    for ch in range(8)
                ]
                residual_rel = np.linalg.norm((residual_pred - residual_true).ravel()) / max(np.linalg.norm(residual_true.ravel()), 1e-300)
                residual_cosine = float(
                    np.dot(residual_pred.ravel(), residual_true.ravel())
                    / max(np.linalg.norm(residual_pred.ravel()) * np.linalg.norm(residual_true.ravel()), 1e-300)
                )
                physical_exceeded = any(
                    np.max(np.abs(pred_np[channel]))
                    > 10.0 * max(
                        abs(float(train_reference["channels"][name]["physical"]["minimum"])),
                        abs(float(train_reference["channels"][name]["physical"]["maximum"])),
                        float(np.max(np.abs(raw_target_np[channel]))),
                        1e-300,
                    )
                    for channel, name in enumerate(CHANNELS)
                )
                transport_rows = []
                for name in TARGET_CHANNELS:
                    channel = CHANNELS.index(name)
                    transport_rows.append({
                        "channel": name,
                        "shell": transport_metrics(
                            variance_vector(prior_np[channel], shell_index),
                            variance_vector(target_np[channel], shell_index),
                            variance_vector(pred_np[channel], shell_index), epsilon=1e-30,
                            sign_zero_tolerance=1e-12,
                        ),
                        "radial": transport_metrics(
                            radial_profile_vector(prior_np[channel]), radial_profile_vector(target_np[channel]),
                            radial_profile_vector(pred_np[channel]), epsilon=1e-30, sign_zero_tolerance=1e-12,
                        ),
                    })
                record.update({
                    "physical_range_explosion": physical_exceeded,
                    "normalized_relative_l2_per_channel": dict(zip(CHANNELS, map(float, norm_rel), strict=True)),
                    "normalized_relative_l2_average": float(np.mean(norm_rel)),
                    "physical_relative_l2_per_channel": dict(zip(CHANNELS, map(float, phys_rel), strict=True)),
                    "physical_relative_l2_average": float(np.mean(phys_rel)),
                    "residual_relative_l2": float(residual_rel),
                    "residual_cosine": residual_cosine,
                    "transport": transport_summary([
                        {"channel": row["channel"], "shell": row["shell"], "radial": row["radial"]}
                        for row in transport_rows
                    ]),
                    "artifacts": safe_artifact_metrics(pred_np, target_np, prior_np),
                })
            else:
                physical_exceeded = physical_exceeded_train
                record.update({
                    "physical_range_explosion": physical_exceeded,
                    "normalized_relative_l2_average": None,
                    "physical_relative_l2_average": None,
                    "residual_relative_l2": None,
                    "residual_cosine": None,
                    "transport": None,
                    "artifacts": safe_artifact_metrics(pred_np, initial, prior_np),
                    "no_gt_metrics_omitted": True,
                })
            if physical_exceeded and first_explosion is None:
                first_explosion = step
            records.append(record)
            current = physical.detach()
            if not finite or not positive:
                break
    return {
        "schema_version": "stage-s-rollout-metrics-v1",
        "initial_snapshot": 169,
        "requested_steps": steps,
        "completed_steps": len(records),
        "ground_truth_steps": 42,
        "teacher_forcing": False,
        "physical_state_rollout": True,
        "decode_once_per_step": prediction_decode_count == len(records),
        "encode_once_per_step": state_encode_count == len(records),
        "state_encode_count": state_encode_count,
        "prediction_decode_count": prediction_decode_count,
        "extra_target_encodes_for_gt_diagnostics": min(len(records), 42),
        "finite": all(row["finite"] for row in records),
        "rho_press_positive": all(row["rho_press_positive"] for row in records),
        "Rout": rout,
        "first_physical_range_explosion_step": first_explosion,
        "physical_range_explosion_definition": "with GT: model max-abs > 10x max(expanded-train max-abs, current raw-GT max-abs); without GT: >10x expanded-train max-abs",
        "selected_steps": [step for step in SELECTED_STEPS if step <= len(records)],
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("artifacts/stage_s"))
    parser.add_argument("--config", type=Path, default=Path("configs/stage_s/expanded_localno_p3_residual.yaml"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("Stage S evaluation requires CUDA; refusing CPU fallback")
    device = torch.device("cuda:0")
    data = StageSBatchPath(Path(config["data"]["dataset"]), Path(config["preprocessing"]["artifact"]), device)
    split = json.loads(Path(config["data"]["split"]).read_text())
    validation_pairs = [int(value) for value in split["validation_pair_source_indices"]]
    r = np.asarray(data.handle["coords/r"], dtype=np.float64)
    _, shell_index = radial_shell_indices(r, 8)
    reference = build_train_reference(data, args.root)
    shared_persistence = None
    for experiment in ("s_small_matched", "s_full_matched", "s_full_30epoch"):
        output = args.root / experiment
        checkpoint = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
        model = build_frozen_model(config)
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        model.to(device).eval()
        metrics, persistence = one_step(model, data, validation_pairs, shell_index, reference)
        if shared_persistence is None:
            shared_persistence = persistence
        elif json.dumps(shared_persistence, sort_keys=True) != json.dumps(persistence, sort_keys=True):
            raise ValueError("Persistence baseline changed between Stage S models")
        write_json(output / "one_step_metrics.json", metrics)
        rollout_metrics = rollout(model, data, shell_index, reference, steps=100)
        write_json(output / "rollout_metrics.json", rollout_metrics)
        print(json.dumps({
            "experiment": experiment,
            "one_step_average": metrics["model"]["normalized_relative_l2"]["arithmetic_average"],
            "persistence_ratio": metrics["model_error_over_persistence_error"],
            "residual_cosine": metrics["model"]["residual"]["global_cosine"],
            "shell_skill": metrics["transport"]["shell_skill_median"],
            "radial_skill": metrics["transport"]["radial_skill_median"],
            "first_explosion": rollout_metrics["first_physical_range_explosion_step"],
            "rollout_finite": rollout_metrics["finite"],
        }, sort_keys=True), flush=True)
        del model
        torch.cuda.empty_cache()
    write_json(args.root / "persistence_one_step_metrics.json", shared_persistence)
    data.close()


if __name__ == "__main__":
    main()
