#!/usr/bin/env python3
"""Evaluate every frozen Stage T checkpoint without changing model selection."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.torch_version import TorchVersion
import yaml

from grmhd import CHANNELS
from grmhd.paper_stage_g_evaluation import radial_shell_indices
from grmhd.paper_stage_m import radial_profile_vector, transport_metrics, variance_vector
from grmhd.paper_stage_p import decoder_derivative_channel
from grmhd.stage_t_training import validation_thirds

from evaluate_stage_s import (
    accumulator,
    build_train_reference,
    finalized,
    safe_artifact_metrics,
    spectral_summary,
    update_accumulator,
)
from train_stage_s import StageSBatchPath, build_frozen_model, tensor_sha256
from train_stage_t import verify_frozen_contract


ROOT = Path(__file__).resolve().parents[1]
TAIL_CHANNELS = ("rho", "press", "Bcc2", "Bcc3", "vel3")
PRIMARY_TRANSPORT_CHANNELS = ("Bcc2", "Bcc3", "vel3")
ROLLOUT_EPOCHS = (30, 75, 150, 300, 600, 1200)
SELECTED_STEPS = (1, 2, 3, 5, 10, 19, 25, 42, 50, 75, 100)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty Stage T metric table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def finite_median(values: Sequence[float | None]) -> float | None:
    kept = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return None if not kept else float(np.median(kept))


def transport_aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    per_channel: dict[str, Any] = {}
    shell_all: list[float | None] = []
    radial_all: list[float | None] = []
    for name in CHANNELS:
        selected = [row for row in rows if row["channel"] == name]
        shell = [row["shell"]["persistence_relative_skill"] for row in selected]
        radial = [row["radial"]["persistence_relative_skill"] for row in selected]
        per_channel[name] = {
            "shell_skill_median": finite_median(shell),
            "radial_skill_median": finite_median(radial),
            "shell_skill_mean": None if finite_median(shell) is None else float(np.mean([x for x in shell if x is not None])),
            "radial_skill_mean": None if finite_median(radial) is None else float(np.mean([x for x in radial if x is not None])),
        }
        if name in PRIMARY_TRANSPORT_CHANNELS:
            shell_all.extend(shell)
            radial_all.extend(radial)
    return {
        "per_channel": per_channel,
        "shell_skill_median": finite_median(shell_all),
        "radial_skill_median": finite_median(radial_all),
        "aggregate_channels": list(PRIMARY_TRANSPORT_CHANNELS),
    }


def evaluate_one_step(
    model: torch.nn.Module,
    data: StageSBatchPath,
    pairs: Sequence[int],
    shell_index: np.ndarray,
    train_reference: Mapping[str, Any],
    *,
    include_transport_rows: bool = True,
) -> dict[str, Any]:
    model.eval()
    learned = accumulator()
    persistence = accumulator()
    transport_rows: list[dict[str, Any]] = []
    z_min = np.full(8, np.inf)
    z_max = np.full(8, -np.inf)
    physical_min = np.full(8, np.inf)
    physical_max = np.full(8, -np.inf)
    physical_q001: list[list[float]] = [[] for _ in CHANNELS]
    physical_q999: list[list[float]] = [[] for _ in CHANNELS]
    derivative_q50: list[list[float]] = [[] for _ in CHANNELS]
    derivative_q999: list[list[float]] = [[] for _ in CHANNELS]
    derivative_max = np.zeros(8, dtype=np.float64)
    extreme_tail = np.zeros(8, dtype=np.float64)
    target_tail = np.zeros(8, dtype=np.float64)
    normalized_norms: list[float] = []
    above_rout = 0
    finite = True
    positive = True
    count = 0
    with torch.no_grad():
        for source in pairs:
            result = data.predict(model, int(source))
            physical_prediction = data.preprocessor.decode_tensor(result["z_prediction"], channel_axis=1)
            persistence_physical = data.preprocessor.decode_tensor(result["z_input"], channel_axis=1)
            oracle_physical = data.preprocessor.decode_tensor(result["z_target"], channel_axis=1)
            arrays = {
                key: value[0].detach().cpu().numpy().astype(np.float64)
                for key, value in {
                    "z_prediction": result["z_prediction"], "z_target": result["z_target"],
                    "z_input": result["z_input"], "predicted_residual": result["predicted_residual"],
                    "residual_target": result["residual_target"], "physical_prediction": physical_prediction,
                    "persistence_physical": persistence_physical, "oracle_physical": oracle_physical,
                    "raw_target": result["raw_target"],
                }.items()
            }
            update_accumulator(
                learned, z_prediction=arrays["z_prediction"], z_target=arrays["z_target"],
                physical_prediction=arrays["physical_prediction"], physical_target=arrays["raw_target"],
                predicted_residual=arrays["predicted_residual"], residual_target=arrays["residual_target"],
            )
            update_accumulator(
                persistence, z_prediction=arrays["z_input"], z_target=arrays["z_target"],
                physical_prediction=arrays["persistence_physical"], physical_target=arrays["raw_target"],
                predicted_residual=np.zeros_like(arrays["residual_target"]),
                residual_target=arrays["residual_target"],
            )
            norm = float(np.linalg.norm(arrays["z_prediction"].ravel()))
            normalized_norms.append(norm)
            above_rout += int(norm > float(train_reference["normalized_global_norm"]["Rout"]))
            for channel, name in enumerate(CHANNELS):
                predicted = arrays["physical_prediction"][channel]
                z_predicted = arrays["z_prediction"][channel]
                derivative = decoder_derivative_channel(
                    data.preprocessor, z_predicted, channel=channel, source_dtype=np.dtype(np.float32)
                )
                z_min[channel] = min(z_min[channel], float(z_predicted.min()))
                z_max[channel] = max(z_max[channel], float(z_predicted.max()))
                physical_min[channel] = min(physical_min[channel], float(predicted.min()))
                physical_max[channel] = max(physical_max[channel], float(predicted.max()))
                quantile = np.quantile(predicted, [0.001, 0.999])
                physical_q001[channel].append(float(quantile[0]))
                physical_q999[channel].append(float(quantile[1]))
                dq = np.quantile(derivative, [0.5, 0.999])
                derivative_q50[channel].append(float(dq[0]))
                derivative_q999[channel].append(float(dq[1]))
                derivative_max[channel] = max(derivative_max[channel], float(derivative.max()))
                extreme_tail[channel] += float(np.mean(np.abs(z_predicted) > 5.94))
                target_tail[channel] += float(np.mean(np.abs(arrays["z_target"][channel]) > 5.94))
                transport_rows.append({
                    "source_snapshot": int(source), "channel": name,
                    "shell": transport_metrics(
                        variance_vector(arrays["persistence_physical"][channel], shell_index),
                        variance_vector(arrays["oracle_physical"][channel], shell_index),
                        variance_vector(predicted, shell_index), epsilon=1e-30,
                        sign_zero_tolerance=1e-12,
                    ),
                    "radial": transport_metrics(
                        radial_profile_vector(arrays["persistence_physical"][channel]),
                        radial_profile_vector(arrays["oracle_physical"][channel]),
                        radial_profile_vector(predicted), epsilon=1e-30,
                        sign_zero_tolerance=1e-12,
                    ),
                })
            finite &= all(np.isfinite(value).all() for value in arrays.values())
            positive &= bool(np.all(arrays["physical_prediction"][3:5] > 0))
            count += 1
    model_metrics = finalized(learned)
    persistence_metrics = finalized(persistence)
    norm_l2 = float(model_metrics["normalized_relative_l2"]["arithmetic_average"])
    persistence_l2 = float(persistence_metrics["normalized_relative_l2"]["arithmetic_average"])
    physical_l2 = float(model_metrics["physical_relative_l2"]["arithmetic_average"])
    persistence_physical_l2 = float(persistence_metrics["physical_relative_l2"]["arithmetic_average"])
    transport = transport_aggregate(transport_rows)
    physical_contribution = learned["physical_num"] / max(float(learned["physical_num"].sum()), 1e-300)
    tail = {
        name: {
            "normalized_prediction_minimum": float(z_min[channel]),
            "normalized_prediction_maximum": float(z_max[channel]),
            "decoder_derivative_pair_median_q50": float(np.median(derivative_q50[channel])),
            "decoder_derivative_pair_median_q999": float(np.median(derivative_q999[channel])),
            "decoder_derivative_maximum": float(derivative_max[channel]),
            "extreme_decoder_tail_fraction": float(extreme_tail[channel] / count),
            "target_extreme_tail_fraction": float(target_tail[channel] / count),
            "physical_squared_error_contribution_fraction": float(physical_contribution[channel]),
        }
        for channel, name in enumerate(CHANNELS) if name in TAIL_CHANNELS
    }
    payload = {
        "schema_version": "stage-t-one-step-v1",
        "validation_pairs": [int(value) for value in pairs],
        "validation_pair_count": count,
        "finite": finite,
        "rho_press_positive": positive,
        "model": model_metrics,
        "persistence": persistence_metrics,
        "model_error_over_persistence_error": norm_l2 / persistence_l2,
        "transport": transport,
        "normalized_prediction_range": {
            name: {"minimum": float(z_min[channel]), "maximum": float(z_max[channel])}
            for channel, name in enumerate(CHANNELS)
        },
        "physical_prediction_distribution": {
            name: {
                "minimum": float(physical_min[channel]), "maximum": float(physical_max[channel]),
                "pair_median_q001": float(np.median(physical_q001[channel])),
                "pair_median_q999": float(np.median(physical_q999[channel])),
            }
            for channel, name in enumerate(CHANNELS)
        },
        "preprocessing_tail": tail,
        "normalized_global_norm": {
            "mean": float(np.mean(normalized_norms)), "maximum": float(np.max(normalized_norms)),
            "above_Rout_fraction": float(above_rout / count),
            "Rout": float(train_reference["normalized_global_norm"]["Rout"]),
        },
        "gates": {
            "state_prediction_below_persistence": norm_l2 < persistence_l2,
            "residual_relative_l2_below_one": float(model_metrics["residual"]["arithmetic_average_relative_l2"]) < 1.0,
            "residual_direction_positive": float(model_metrics["residual"]["global_cosine"]) > 0.0,
            "residual_cosine_above_0_5": float(model_metrics["residual"]["global_cosine"]) > 0.5,
            "residual_cosine_above_0_7": float(model_metrics["residual"]["global_cosine"]) > 0.7,
            "residual_cosine_above_0_9": float(model_metrics["residual"]["global_cosine"]) > 0.9,
            "shell_transport_positive": bool(transport["shell_skill_median"] is not None and transport["shell_skill_median"] > 0),
            "radial_transport_positive": bool(transport["radial_skill_median"] is not None and transport["radial_skill_median"] > 0),
            "physical_finite_without_catastrophic_inverse_tail": finite and physical_l2 <= 10.0 * persistence_physical_l2,
        },
        "catastrophic_inverse_tail_definition": "physical average relative L2 > 10x persistence physical average relative L2",
    }
    if include_transport_rows:
        payload["transport_rows"] = transport_rows
    return payload


def model_from_checkpoint(
    checkpoint_path: Path, stage_s: Mapping[str, Any], device: torch.device,
    *, expected_updates: int | None = None,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    # Older Stage-T checkpoints created before torch.__version__ was coerced to
    # text contain the benign TorchVersion metadata class.  Keep safe weights
    # loading and allowlist only that exact locally-generated type.
    with torch.serialization.safe_globals([TorchVersion]):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = build_frozen_model(stage_s)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(stage_s["optimizer"]["learning_rate"]),
        weight_decay=float(stage_s["optimizer"]["weight_decay"]),
    )
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    if expected_updates is not None:
        observed = int(payload["scheduler_state"]["completed_updates"])
        if observed != expected_updates or observed != int(payload["training_state"]["optimizer_updates"]):
            raise ValueError(f"Checkpoint scheduler/update mismatch: {checkpoint_path}")
    model.to(device).eval()
    return model, payload, {"strict_model_reload": True, "optimizer_reload": True, "scheduler_reload": True}


def checkpoint_row(epoch: int, metrics: Mapping[str, Any], reload: Mapping[str, Any], updates: int) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "optimizer_updates": updates,
        "normalized_relative_l2": metrics["model"]["normalized_relative_l2"]["arithmetic_average"],
        "persistence_ratio": metrics["model_error_over_persistence_error"],
        "residual_relative_l2": metrics["model"]["residual"]["arithmetic_average_relative_l2"],
        "residual_global_cosine": metrics["model"]["residual"]["global_cosine"],
        "shell_skill": metrics["transport"]["shell_skill_median"],
        "radial_skill": metrics["transport"]["radial_skill_median"],
        "physical_relative_l2": metrics["model"]["physical_relative_l2"]["arithmetic_average"],
        "Rout_fraction": metrics["normalized_global_norm"]["above_Rout_fraction"],
        "finite": metrics["finite"],
        "rho_press_positive": metrics["rho_press_positive"],
        **reload,
    }


def rollout(
    model: torch.nn.Module, data: StageSBatchPath, shell_index: np.ndarray,
    train_reference: Mapping[str, Any], *, epoch: int, steps: int = 100,
) -> dict[str, Any]:
    model.eval()
    current = data.raw(169).to(data.device)
    initial = current[0].detach().cpu().numpy().astype(np.float64)
    records: list[dict[str, Any]] = []
    rout = float(train_reference["normalized_global_norm"]["Rout"])
    encode_count = decode_count = 0
    with torch.no_grad():
        for step in range(1, steps + 1):
            z_input = data.preprocessor.encode_tensor(current, channel_axis=1)
            encode_count += 1
            predicted_residual = model(x=torch.cat((z_input, data.shells), dim=1))
            z_prediction = z_input + predicted_residual
            physical = data.preprocessor.decode_tensor(z_prediction, channel_axis=1)
            decode_count += 1
            z_np = z_prediction[0].detach().cpu().numpy().astype(np.float64)
            physical_np = physical[0].detach().cpu().numpy().astype(np.float64)
            prior_np = current[0].detach().cpu().numpy().astype(np.float64)
            finite = bool(np.isfinite(z_np).all() and np.isfinite(physical_np).all())
            positive = bool(np.all(physical_np[3:5] > 0))
            norm = float(np.linalg.norm(z_np.ravel()))
            train_explosion = any(
                np.max(np.abs(physical_np[channel])) > 10.0 * max(
                    abs(float(train_reference["channels"][name]["physical"]["minimum"])),
                    abs(float(train_reference["channels"][name]["physical"]["maximum"])), 1e-300,
                ) for channel, name in enumerate(CHANNELS)
            )
            record: dict[str, Any] = {
                "step": step, "ground_truth_available": step <= 42,
                "finite": finite, "rho_press_positive": positive,
                "normalized_global_norm": norm, "above_Rout": norm > rout,
                "normalized_range": {
                    name: {"minimum": float(z_np[channel].min()), "maximum": float(z_np[channel].max()),
                           "q001": float(np.quantile(z_np[channel], 0.001)),
                           "q999": float(np.quantile(z_np[channel], 0.999))}
                    for channel, name in enumerate(CHANNELS)
                },
                "physical_range": {
                    name: {"minimum": float(physical_np[channel].min()), "maximum": float(physical_np[channel].max()),
                           "q001": float(np.quantile(physical_np[channel], 0.001)),
                           "q999": float(np.quantile(physical_np[channel], 0.999))}
                    for channel, name in enumerate(CHANNELS)
                },
                "spectral_normalized": spectral_summary(z_np),
            }
            if step <= 42:
                raw_target = data.raw(169 + step).to(data.device)
                z_target = data.preprocessor.encode_tensor(raw_target, channel_axis=1)
                oracle = data.preprocessor.decode_tensor(z_target, channel_axis=1)
                raw_np = raw_target[0].detach().cpu().numpy().astype(np.float64)
                target_np = oracle[0].detach().cpu().numpy().astype(np.float64)
                z_target_np = z_target[0].detach().cpu().numpy().astype(np.float64)
                z_input_np = z_input[0].detach().cpu().numpy().astype(np.float64)
                residual_np = predicted_residual[0].detach().cpu().numpy().astype(np.float64)
                true_residual = z_target_np - z_input_np
                norm_rel = [
                    np.linalg.norm((z_np[ch] - z_target_np[ch]).ravel()) /
                    max(np.linalg.norm(z_target_np[ch].ravel()), 1e-300) for ch in range(8)
                ]
                physical_rel = [
                    np.linalg.norm((physical_np[ch] - raw_np[ch]).ravel()) /
                    max(np.linalg.norm(raw_np[ch].ravel()), 1e-300) for ch in range(8)
                ]
                per_cosine = []
                for channel in range(8):
                    left, right = residual_np[channel].ravel(), true_residual[channel].ravel()
                    per_cosine.append(float(np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), 1e-300)))
                global_cosine = float(
                    np.dot(residual_np.ravel(), true_residual.ravel()) /
                    max(np.linalg.norm(residual_np.ravel()) * np.linalg.norm(true_residual.ravel()), 1e-300)
                )
                transport_rows = []
                for channel, name in enumerate(CHANNELS):
                    transport_rows.append({
                        "channel": name,
                        "shell": transport_metrics(
                            variance_vector(prior_np[channel], shell_index),
                            variance_vector(target_np[channel], shell_index),
                            variance_vector(physical_np[channel], shell_index), epsilon=1e-30,
                            sign_zero_tolerance=1e-12,
                        ),
                        "radial": transport_metrics(
                            radial_profile_vector(prior_np[channel]), radial_profile_vector(target_np[channel]),
                            radial_profile_vector(physical_np[channel]), epsilon=1e-30,
                            sign_zero_tolerance=1e-12,
                        ),
                    })
                transport = transport_aggregate(transport_rows)
                gt_explosion = any(
                    np.max(np.abs(physical_np[channel])) > 10.0 * max(
                        abs(float(train_reference["channels"][name]["physical"]["minimum"])),
                        abs(float(train_reference["channels"][name]["physical"]["maximum"])),
                        float(np.max(np.abs(raw_np[channel]))), 1e-300,
                    ) for channel, name in enumerate(CHANNELS)
                )
                record.update({
                    "physical_range_explosion": gt_explosion,
                    "normalized_relative_l2_per_channel": dict(zip(CHANNELS, map(float, norm_rel), strict=True)),
                    "normalized_relative_l2_average": float(np.mean(norm_rel)),
                    "physical_relative_l2_per_channel": dict(zip(CHANNELS, map(float, physical_rel), strict=True)),
                    "physical_relative_l2_average": float(np.mean(physical_rel)),
                    "residual_relative_l2": float(np.linalg.norm((residual_np - true_residual).ravel()) / max(np.linalg.norm(true_residual.ravel()), 1e-300)),
                    "residual_cosine": global_cosine,
                    "per_channel_residual_cosine": dict(zip(CHANNELS, per_cosine, strict=True)),
                    "transport": transport,
                    "artifacts": safe_artifact_metrics(physical_np, target_np, prior_np),
                })
            else:
                record.update({
                    "physical_range_explosion": train_explosion,
                    "normalized_relative_l2_average": None,
                    "physical_relative_l2_average": None,
                    "residual_relative_l2": None, "residual_cosine": None,
                    "per_channel_residual_cosine": None, "transport": None,
                    "artifacts": safe_artifact_metrics(physical_np, initial, prior_np),
                    "no_gt_metrics_omitted": True,
                })
            records.append(record)
            current = physical.detach()
            if not finite or not positive:
                break
    def first(predicate: Any) -> int | None:
        return next((int(row["step"]) for row in records if predicate(row)), None)
    return {
        "schema_version": "stage-t-rollout-v1", "checkpoint_epoch": epoch,
        "initial_snapshot": 169, "requested_steps": steps, "completed_steps": len(records),
        "ground_truth_steps": 42, "teacher_forcing": False, "physical_state_rollout": True,
        "encode_once_per_step": encode_count == len(records),
        "decode_once_per_step": decode_count == len(records),
        "finite": all(row["finite"] for row in records),
        "rho_press_positive": all(row["rho_press_positive"] for row in records),
        "FIRST_NORMALIZED_OOD_STEP": first(lambda row: row["above_Rout"]),
        "FIRST_10X_PHYSICAL_RANGE_STEP": first(lambda row: row["physical_range_explosion"]),
        "FIRST_NEGATIVE_RESIDUAL_COSINE_STEP": first(lambda row: row.get("residual_cosine") is not None and row["residual_cosine"] < 0),
        "FIRST_NEGATIVE_SHELL_SKILL_STEP": first(lambda row: row.get("transport") is not None and row["transport"]["shell_skill_median"] is not None and row["transport"]["shell_skill_median"] < 0),
        "FIRST_NEGATIVE_RADIAL_SKILL_STEP": first(lambda row: row.get("transport") is not None and row["transport"]["radial_skill_median"] is not None and row["transport"]["radial_skill_median"] < 0),
        "closed_loop_gate_first_10x_after_step_3": (lambda value: value is None or value > 3)(first(lambda row: row["physical_range_explosion"])),
        "closed_loop_strong_gate_first_10x_after_step_19": (lambda value: value is None or value > 19)(first(lambda row: row["physical_range_explosion"])),
        "selected_steps": [value for value in SELECTED_STEPS if value <= len(records)],
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("small", "checkpoints", "rollouts", "all"), default="all")
    parser.add_argument("--config", type=Path, default=Path("configs/stage_t/optimization_convergence.yaml"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/stage_t"))
    args = parser.parse_args()
    stage_t, stage_s, freeze, _, _, validation_pairs = verify_frozen_contract(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("Stage T evaluation requires CUDA; refusing silent CPU fallback")
    device = torch.device("cuda:0")
    data = StageSBatchPath(
        ROOT / stage_s["data"]["dataset"], ROOT / stage_s["preprocessing"]["artifact"], device
    )
    r = np.asarray(data.handle["coords/r"], dtype=np.float64)
    _, shell_index = radial_shell_indices(r, int(stage_s["representation"]["shell_count"]))
    reference_path = ROOT / "artifacts/stage_s/diagnostic_reference_train_only.json"
    reference = json.loads(reference_path.read_text()) if reference_path.exists() else build_train_reference(data, ROOT / "artifacts/stage_s")

    if args.phase in ("small", "all"):
        checkpoint = args.root / "causal_control/small_1260.pt"
        model, payload, reload = model_from_checkpoint(checkpoint, stage_s, device, expected_updates=1260)
        metrics = evaluate_one_step(model, data, validation_pairs, shell_index, reference)
        metrics["checkpoint_reload"] = reload
        metrics["optimizer_updates"] = int(payload["training_state"]["optimizer_updates"])
        write_json(args.root / "causal_control/small_1260_metrics.json", metrics)
        print(json.dumps({"small_1260_norm_l2": metrics["model"]["normalized_relative_l2"]["arithmetic_average"]}, sort_keys=True), flush=True)
        del model
        torch.cuda.empty_cache()

    if args.phase in ("checkpoints", "all"):
        rows: list[dict[str, Any]] = []
        stratum_rows: list[dict[str, Any]] = []
        epochs = [int(value) for value in stage_t["full_long"]["mandatory_checkpoint_epochs"]]
        thirds = validation_thirds(validation_pairs)
        for epoch in epochs:
            checkpoint = args.root / "full_long/checkpoints" / f"epoch_{epoch:04d}.pt"
            expected_updates = epoch * 42
            model, payload, reload = model_from_checkpoint(checkpoint, stage_s, device, expected_updates=expected_updates)
            with torch.no_grad():
                probe_a = tensor_sha256(data.predict(model, 169)["z_prediction"])
                probe_b = tensor_sha256(data.predict(model, 169)["z_prediction"])
            if probe_a != probe_b:
                raise ValueError(f"Checkpoint epoch {epoch} prediction is nondeterministic")
            reload["deterministic_probe_prediction"] = True
            reload["probe_prediction_sha256"] = probe_a
            metrics = evaluate_one_step(model, data, validation_pairs, shell_index, reference)
            metrics["checkpoint_epoch"] = epoch
            metrics["optimizer_updates"] = expected_updates
            metrics["checkpoint_reload"] = reload
            write_json(args.root / "full_long" / f"checkpoint_metrics_epoch_{epoch:04d}.json", metrics)
            rows.append(checkpoint_row(epoch, metrics, reload, expected_updates))
            for name, pairs in thirds.items():
                stratum = evaluate_one_step(
                    model, data, pairs, shell_index, reference, include_transport_rows=False
                )
                for channel in CHANNELS:
                    stratum_rows.append({
                        "epoch": epoch, "optimizer_updates": expected_updates, "stratum": name,
                        "channel": channel,
                        "model_normalized_relative_l2": stratum["model"]["normalized_relative_l2"]["per_channel"][channel],
                        "persistence_normalized_relative_l2": stratum["persistence"]["normalized_relative_l2"]["per_channel"][channel],
                        "residual_relative_l2": stratum["model"]["residual"]["per_channel_relative_l2"][channel],
                        "residual_cosine": stratum["model"]["residual"]["per_channel_cosine"][channel],
                    })
            write_csv(args.root / "full_long/checkpoint_metrics.csv", rows)
            write_csv(args.root / "metrics/validation_time_stratification.csv", stratum_rows)
            print(json.dumps({"epoch": epoch, "norm_l2": rows[-1]["normalized_relative_l2"], "residual_l2": rows[-1]["residual_relative_l2"]}, sort_keys=True), flush=True)
            del model
            torch.cuda.empty_cache()

    if args.phase in ("rollouts", "all"):
        rollout_dir = args.root / "full_long/rollout"
        for epoch in ROLLOUT_EPOCHS:
            model, _, _ = model_from_checkpoint(
                args.root / "full_long/checkpoints" / f"epoch_{epoch:04d}.pt",
                stage_s, device, expected_updates=epoch * 42,
            )
            result = rollout(model, data, shell_index, reference, epoch=epoch, steps=100)
            write_json(rollout_dir / f"epoch_{epoch:04d}_rollout.json", result)
            print(json.dumps({
                "epoch": epoch, "first_10x": result["FIRST_10X_PHYSICAL_RANGE_STEP"],
                "finite": result["finite"], "completed_steps": result["completed_steps"],
            }, sort_keys=True), flush=True)
            del model
            torch.cuda.empty_cache()
    data.close()


if __name__ == "__main__":
    main()
