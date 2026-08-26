#!/usr/bin/env python3
"""Post-hoc Stage AH analysis of the completed Stage AG controlled run."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from grmhd import CHANNELS
from grmhd.dataset import sha256_file
from grmhd.paper_stage_g_evaluation import radial_shell_indices
from grmhd.paper_stage_m import radial_profile_vector, transport_metrics, variance_vector
from grmhd.stage_ah import (
    RADIAL_REGION_BOUNDS,
    THETA_REGION_BOUNDS,
    field_metrics,
    paired_bootstrap,
    primary_gates,
    scientific_decision,
    validate_region_bounds,
)

from evaluate_stage_ad import compact_metrics, model_from_checkpoint as stage_ad_model_from_checkpoint
from evaluate_stage_s import build_train_reference, safe_artifact_metrics, spectral_summary
from evaluate_stage_t import evaluate_one_step, transport_aggregate
from preflight_stage_ad import load_contract as load_stage_ad_contract
from train_stage_ag import (
    build_initial_model,
    dataset_record,
    load_contract as load_stage_ag_contract,
    pair_orders,
    tensor_hash,
    trainable_hash,
)
from train_stage_s import StageSBatchPath, validation_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/stage_ah"
CONFIG = ROOT / "configs/stage_ah/posthoc_analysis.yaml"
PRIMARY_TRANSPORT_CHANNELS = ("Bcc2", "Bcc3", "vel3")
SELECTED_STEPS = (1, 2, 3, 5, 10, 11, 19, 25, 42, 50, 75, 100)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True))


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty Stage AH table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def completion_audit(
    config: Mapping[str, Any], stage_s: Mapping[str, Any], stage_t: Mapping[str, Any]
) -> dict[str, Any]:
    stage_ag = ROOT / "artifacts/stage_ag"
    rows = csv_rows(stage_ag / "training/train_log.csv")
    epochs = [int(row["epoch"]) for row in rows]
    microbatches = [int(row["microbatches_completed"]) for row in rows]
    updates = [int(row["optimizer_updates"]) for row in rows]
    nonfinite = [int(row["nonfinite_count"]) for row in rows]
    orders, pair_record, train_pairs, validation_pairs = pair_orders(config, stage_s, stage_t)
    required = [
        stage_ag / "training/checkpoints" / name
        for name in (
            "epoch_0002.pt", "epoch_0010.pt", "epoch_0030.pt", "epoch_0075.pt",
            "epoch_0150.pt", "epoch_0300.pt", "best.pt", "last.pt",
        )
    ]
    reload_records = [
        json.loads((stage_ag / "integrity" / name).read_text(encoding="utf-8"))
        for name in ("best_reload_test.json", "last_reload_test.json")
    ]
    pretrain_records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((stage_ag / "pretrain").glob("*.json"))
    ]
    identity = dataset_record(config)
    gates = {
        "training_started": bool(rows),
        "epoch_sequence_exact": epochs == list(range(1, 301)),
        "microbatch_sequence_exact": microbatches == [168 * value for value in range(1, 301)],
        "optimizer_update_sequence_exact": updates == [42 * value for value in range(1, 301)],
        "per_epoch_microbatches_exact": all(int(row["microbatches_this_epoch"]) == 168 for row in rows),
        "per_epoch_optimizer_updates_exact": all(int(row["optimizer_updates_this_epoch"]) == 42 for row in rows),
        "nonfinite_count_zero": sum(nonfinite) == 0,
        "required_checkpoints_exist": all(path.is_file() for path in required),
        "pretrain_identity_pass": all(record.get("pass") is True for record in pretrain_records),
        "recorded_reload_pass": all(record.get("pass") is True for record in reload_records),
        "model_identity_match": config["model"]["name"] == "ANISOTROPIC_SPHERICAL_DISCO_LOCALNO",
        "parameter_count_match": int(config["model"]["parameter_count"]) == 363480,
        "initial_state_match": config["initialization"]["expected_trainable_state_sha256"]
        == "77252855f054a199500fa779f7340b33b6a2ce546b2082f17f93666275f4c588",
        "dataset_identity_match": bool(identity["pass"]),
        "split_match": train_pairs == list(range(168)) and validation_pairs == list(range(169, 211)),
        "pair_order_match": bool(pair_record["pass"]) and all(
            sorted(order) == list(range(168)) and len(set(order)) == 168 for order in orders
        ),
        "loss_match": config["loss"] == {
            "name": "PlainL2Loss", "space": "normalized_residual", "enabled_priors": []
        },
        "budget_match": int(config["training"]["epochs"]) == 300,
    }
    complete = all(gates.values())
    return {
        "schema_version": "stage-ah-stage-ag-completion-audit-v1",
        "STAGE_AG_COMPLETE": complete,
        "STAGE_AG_CONTROLLED_COMPARISON_VALID": complete,
        "gates": gates,
        "epochs_completed": epochs[-1] if epochs else 0,
        "microbatches_completed": microbatches[-1] if microbatches else 0,
        "optimizer_updates_completed": updates[-1] if updates else 0,
        "nonfinite_count": sum(nonfinite),
        "total_training_seconds": float(rows[-1]["cumulative_runtime_seconds"]),
        "peak_allocated_vram_mib": max(float(row["gpu_peak_allocated_mib"]) for row in rows),
        "peak_reserved_vram_mib": max(float(row["gpu_peak_reserved_mib"]) for row in rows),
        "model": config["model"],
        "initial_trainable_state_sha256": config["initialization"]["expected_trainable_state_sha256"],
        "dataset": identity,
        "pair_order_sha256": pair_record["observed_flat_sha256"],
        "required_checkpoint_sha256": {
            str(path.relative_to(ROOT)): sha256_file(path) for path in required
        },
    }


def strict_load_stage_ag(
    checkpoint: Path,
    config: Mapping[str, Any],
    stage_s: Mapping[str, Any],
    data: StageSBatchPath,
    *,
    expected_selector: float,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = build_initial_model(config, stage_s)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    updates = int(payload["training_state"]["optimizer_updates"])
    scheduler_updates = int(payload["scheduler_state"]["completed_updates"])
    model.to(data.device).eval()
    with torch.no_grad():
        probe_a = data.predict(model, 169)["predicted_residual"]
        probe_b = data.predict(model, 169)["predicted_residual"]
    observed_hash = trainable_hash(model)
    record = {
        "checkpoint": str(checkpoint.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "epoch": int(payload["training_state"]["completed_epoch"]),
        "optimizer_updates": updates,
        "scheduler_completed_updates": scheduler_updates,
        "strict_model_reload": True,
        "optimizer_state_reload": True,
        "scheduler_state_reload": scheduler_updates == updates == 12600,
        "stored_trainable_state_sha256": payload["trainable_state_sha256"],
        "reloaded_trainable_state_sha256": observed_hash,
        "trainable_hash_match": observed_hash == payload["trainable_state_sha256"],
        "deterministic_probe_sha256": tensor_hash(probe_a),
        "deterministic_probe_pass": tensor_hash(probe_a) == tensor_hash(probe_b),
        "expected_selector_metric": expected_selector,
    }
    record["pass_before_metric_recompute"] = all((
        record["scheduler_state_reload"], record["trainable_hash_match"],
        record["deterministic_probe_pass"], record["epoch"] == 300,
    ))
    return model, payload, record


def transport_for_channel(
    physical_input: np.ndarray,
    physical_target: np.ndarray,
    physical_prediction: np.ndarray,
    shell_index: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    shell = transport_metrics(
        variance_vector(physical_input, shell_index),
        variance_vector(physical_target, shell_index),
        variance_vector(physical_prediction, shell_index),
        epsilon=1.0e-30,
        sign_zero_tolerance=1.0e-12,
    )
    radial = transport_metrics(
        radial_profile_vector(physical_input),
        radial_profile_vector(physical_target),
        radial_profile_vector(physical_prediction),
        epsilon=1.0e-30,
        sign_zero_tolerance=1.0e-12,
    )
    return shell, radial


def scalar_transport(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    selected = [row for row in rows if row["channel"] in PRIMARY_TRANSPORT_CHANNELS]
    return {
        "shell_skill": float(np.median([row["shell_skill"] for row in selected])),
        "radial_skill": float(np.median([row["radial_skill"] for row in selected])),
        "shell_absolute_error": float(np.median([row["shell_absolute_error"] for row in selected])),
        "radial_absolute_error": float(np.median([row["radial_absolute_error"] for row in selected])),
    }


def aggregate_region_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    keys = sorted({(str(row["model"]), str(row["region"]), str(row["channel"])) for row in rows})
    metrics = ("state_l2", "residual_l2", "residual_cosine", "absolute_residual_error")
    for model, region, channel in keys:
        selected = [row for row in rows if (row["model"], row["region"], row["channel"]) == (model, region, channel)]
        record: dict[str, Any] = {
            "model": model, "region": region, "channel": channel, "pair_count": len(selected)
        }
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in selected], dtype=np.float64)
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_median"] = float(np.median(values))
        output.append(record)
    indexed = {(row["model"], row["region"], row["channel"]): row for row in output}
    for row in output:
        if row["model"] != "Stage AG":
            continue
        baseline = indexed[("Stage AD", row["region"], row["channel"])]
        for metric in metrics:
            row[f"stage_ag_minus_stage_ad_{metric}_mean"] = (
                float(row[f"{metric}_mean"]) - float(baseline[f"{metric}_mean"])
            )
    return output


def analyze_validation_pairs(
    models: Mapping[str, torch.nn.Module],
    data: StageSBatchPath,
    pairs: Sequence[int],
    shell_index: np.ndarray,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
    list[dict[str, Any]], list[dict[str, Any]],
]:
    pair_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    transport_rows: list[dict[str, Any]] = []
    theta_rows: list[dict[str, Any]] = []
    radial_rows: list[dict[str, Any]] = []
    validate_region_bounds(THETA_REGION_BOUNDS, size=64)
    validate_region_bounds(RADIAL_REGION_BOUNDS, size=64)
    for source in pairs:
        for model_name, model in models.items():
            with torch.no_grad():
                result = data.predict(model, int(source))
                physical_prediction = data.preprocessor.decode_tensor(
                    result["z_prediction"], channel_axis=1
                )
                physical_target = data.preprocessor.decode_tensor(
                    result["z_target"], channel_axis=1
                )
            arrays = {
                key: value[0].detach().cpu().numpy().astype(np.float64)
                for key, value in {
                    "state_prediction": result["z_prediction"],
                    "state_target": result["z_target"],
                    "residual_prediction": result["predicted_residual"],
                    "residual_target": result["residual_target"],
                    "physical_input": result["raw_input"],
                    "physical_target": physical_target,
                    "physical_prediction": physical_prediction,
                }.items()
            }
            aggregate = field_metrics(
                arrays["state_prediction"], arrays["state_target"],
                arrays["residual_prediction"], arrays["residual_target"],
            )
            local_transport: list[dict[str, Any]] = []
            local_channel_metrics: list[dict[str, float]] = []
            for channel, name in enumerate(CHANNELS):
                metrics = field_metrics(
                    arrays["state_prediction"][channel], arrays["state_target"][channel],
                    arrays["residual_prediction"][channel], arrays["residual_target"][channel],
                )
                local_channel_metrics.append(metrics)
                shell, radial = transport_for_channel(
                    arrays["physical_input"][channel], arrays["physical_target"][channel],
                    arrays["physical_prediction"][channel], shell_index,
                )
                transport = {
                    "model": model_name,
                    "source_snapshot": int(source),
                    "target_snapshot": int(source) + 1,
                    "channel": name,
                    "shell_skill": float(shell["persistence_relative_skill"]),
                    "shell_absolute_error": float(shell["error_l2"]),
                    "shell_persistence_absolute_error": float(shell["persistence_error_l2"]),
                    "radial_skill": float(radial["persistence_relative_skill"]),
                    "radial_absolute_error": float(radial["error_l2"]),
                    "radial_persistence_absolute_error": float(radial["persistence_error_l2"]),
                }
                local_transport.append(transport)
                transport_rows.append(transport)
                channel_rows.append({
                    "model": model_name, "source_snapshot": int(source),
                    "target_snapshot": int(source) + 1, "channel": name,
                    **metrics, **{key: transport[key] for key in (
                        "shell_skill", "shell_absolute_error", "radial_skill", "radial_absolute_error"
                    )},
                })
            aggregate["state_l2"] = float(np.mean([row["state_l2"] for row in local_channel_metrics]))
            aggregate["residual_l2"] = float(np.mean([row["residual_l2"] for row in local_channel_metrics]))
            pair_rows.append({
                "model": model_name, "source_snapshot": int(source),
                "target_snapshot": int(source) + 1, **aggregate, **scalar_transport(local_transport),
            })
            for region, (start, stop) in THETA_REGION_BOUNDS.items():
                for channel, name in [*enumerate(CHANNELS), (-1, "ALL")]:
                    selected = {
                        key: value[:, :, start:stop, :] if channel == -1
                        else value[channel, :, start:stop, :]
                        for key, value in arrays.items() if key.startswith(("state_", "residual_"))
                    }
                    theta_rows.append({
                        "model": model_name, "source_snapshot": int(source),
                        "region": region, "channel": name,
                        **field_metrics(
                            selected["state_prediction"], selected["state_target"],
                            selected["residual_prediction"], selected["residual_target"],
                        ),
                    })
            for region, (start, stop) in RADIAL_REGION_BOUNDS.items():
                for channel, name in [*enumerate(CHANNELS), (-1, "ALL")]:
                    selected = {
                        key: value[..., start:stop] if channel == -1 else value[channel, ..., start:stop]
                        for key, value in arrays.items() if key.startswith(("state_", "residual_"))
                    }
                    radial_rows.append({
                        "model": model_name, "source_snapshot": int(source),
                        "region": region, "channel": name,
                        **field_metrics(
                            selected["state_prediction"], selected["state_target"],
                            selected["residual_prediction"], selected["residual_target"],
                        ),
                    })
    return pair_rows, channel_rows, transport_rows, theta_rows, radial_rows


def geometry_benefit_rows(
    models: Mapping[str, torch.nn.Module],
    data: StageSBatchPath,
    pairs: Sequence[int],
) -> list[dict[str, Any]]:
    geometry = {row["target"]: row for row in csv_rows(
        ROOT / "artifacts/stage_af/comparison/stage_ad_vs_stage_af_geometry.csv"
    )}
    targets = {
        f"{radial}_{latitude}": (radial, latitude)
        for radial in ("inner", "middle", "outer")
        for latitude in ("equator", "near_pole")
    }
    raw: list[dict[str, Any]] = []
    for source in pairs:
        for model_name, model in models.items():
            with torch.no_grad():
                result = data.predict(model, int(source))
            arrays = {
                key: value[0].detach().cpu().numpy().astype(np.float64)
                for key, value in {
                    "state_prediction": result["z_prediction"],
                    "state_target": result["z_target"],
                    "residual_prediction": result["predicted_residual"],
                    "residual_target": result["residual_target"],
                }.items()
            }
            for target, (radial_name, latitude) in targets.items():
                r_start, r_stop = RADIAL_REGION_BOUNDS[radial_name]
                if latitude == "equator":
                    t_start, t_stop = THETA_REGION_BOUNDS["equatorial"]
                    selected = {
                        key: value[:, :, t_start:t_stop, r_start:r_stop]
                        for key, value in arrays.items()
                    }
                else:
                    north = THETA_REGION_BOUNDS["near_north_pole"]
                    south = THETA_REGION_BOUNDS["near_south_pole"]
                    selected = {
                        key: np.concatenate((
                            value[:, :, north[0]:north[1], r_start:r_stop],
                            value[:, :, south[0]:south[1], r_start:r_stop],
                        ), axis=2)
                        for key, value in arrays.items()
                    }
                raw.append({
                    "model": model_name, "source_snapshot": int(source), "target": target,
                    **field_metrics(
                        selected["state_prediction"], selected["state_target"],
                        selected["residual_prediction"], selected["residual_target"],
                    ),
                })
    output: list[dict[str, Any]] = []
    for target in targets:
        ad = [row for row in raw if row["model"] == "Stage AD" and row["target"] == target]
        ag = [row for row in raw if row["model"] == "Stage AG" and row["target"] == target]
        record: dict[str, Any] = {
            "target": target,
            "stage_af_vs_stage_ad_kernel_relative_l2": float(geometry[target]["af_vs_ad_relative_l2"]),
        }
        for metric in ("state_l2", "residual_l2", "residual_cosine", "absolute_residual_error"):
            ad_mean = float(np.mean([row[metric] for row in ad]))
            ag_mean = float(np.mean([row[metric] for row in ag]))
            record[f"stage_ad_{metric}_mean"] = ad_mean
            record[f"stage_ag_{metric}_mean"] = ag_mean
            record[f"prediction_improvement_{metric}"] = (
                ag_mean - ad_mean if metric == "residual_cosine" else ad_mean - ag_mean
            )
        output.append(record)
    geometry_delta = np.asarray([row["stage_af_vs_stage_ad_kernel_relative_l2"] for row in output])
    for metric in ("state_l2", "residual_l2", "residual_cosine", "absolute_residual_error"):
        improvement = np.asarray([row[f"prediction_improvement_{metric}"] for row in output])
        correlation = float(np.corrcoef(geometry_delta, improvement)[0, 1])
        for row in output:
            row[f"exploratory_pearson_geometry_vs_{metric}_improvement"] = correlation
    return output


def rollout_stage_ah(
    model: torch.nn.Module,
    data: StageSBatchPath,
    shell_index: np.ndarray,
    train_reference: Mapping[str, Any],
    *,
    steps: int = 100,
) -> dict[str, Any]:
    model.eval()
    current = data.raw(169).to(data.device)
    initial = current[0].detach().cpu().numpy().astype(np.float64)
    records: list[dict[str, Any]] = []
    rout = float(train_reference["normalized_global_norm"]["Rout"])
    encode_count = 0
    decode_count = 0
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
            positive = bool(np.all(physical_np[3:5] > 0.0))
            norm = float(np.linalg.norm(z_np.ravel()))
            train_explosion = any(
                np.max(np.abs(physical_np[channel])) > 10.0 * max(
                    abs(float(train_reference["channels"][name]["physical"]["minimum"])),
                    abs(float(train_reference["channels"][name]["physical"]["maximum"])),
                    1.0e-300,
                )
                for channel, name in enumerate(CHANNELS)
            )
            record: dict[str, Any] = {
                "step": step,
                "ground_truth_available": step <= 42,
                "finite": finite,
                "rho_press_positive": positive,
                "normalized_global_norm": norm,
                "above_Rout": norm > rout,
                "normalized_range": {
                    name: {
                        "minimum": float(z_np[channel].min()),
                        "maximum": float(z_np[channel].max()),
                        "q001": float(np.quantile(z_np[channel], 0.001)),
                        "q999": float(np.quantile(z_np[channel], 0.999)),
                    }
                    for channel, name in enumerate(CHANNELS)
                },
                "physical_range": {
                    name: {
                        "minimum": float(physical_np[channel].min()),
                        "maximum": float(physical_np[channel].max()),
                        "q001": float(np.quantile(physical_np[channel], 0.001)),
                        "q999": float(np.quantile(physical_np[channel], 0.999)),
                    }
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
                residual_np = predicted_residual[0].detach().cpu().numpy().astype(np.float64)
                true_residual = z_target_np - z_input[0].detach().cpu().numpy().astype(np.float64)
                metrics = field_metrics(z_np, z_target_np, residual_np, true_residual)
                physical_relative = [
                    np.linalg.norm((physical_np[channel] - raw_np[channel]).ravel())
                    / max(np.linalg.norm(raw_np[channel].ravel()), 1.0e-300)
                    for channel in range(8)
                ]
                detail: list[dict[str, Any]] = []
                for channel, name in enumerate(CHANNELS):
                    shell, radial = transport_for_channel(
                        prior_np[channel], target_np[channel], physical_np[channel], shell_index
                    )
                    detail.append({
                        "channel": name,
                        "shell_skill": float(shell["persistence_relative_skill"]),
                        "shell_absolute_error": float(shell["error_l2"]),
                        "radial_skill": float(radial["persistence_relative_skill"]),
                        "radial_absolute_error": float(radial["error_l2"]),
                    })
                transport = scalar_transport(detail)
                gt_explosion = any(
                    np.max(np.abs(physical_np[channel])) > 10.0 * max(
                        abs(float(train_reference["channels"][name]["physical"]["minimum"])),
                        abs(float(train_reference["channels"][name]["physical"]["maximum"])),
                        float(np.max(np.abs(raw_np[channel]))),
                        1.0e-300,
                    )
                    for channel, name in enumerate(CHANNELS)
                )
                record.update({
                    "physical_range_explosion": gt_explosion,
                    "normalized_relative_l2_average": metrics["state_l2"],
                    "residual_relative_l2": metrics["residual_l2"],
                    "residual_cosine": metrics["residual_cosine"],
                    "absolute_residual_error": metrics["absolute_residual_error"],
                    "physical_relative_l2_average": float(np.mean(physical_relative)),
                    "transport": transport,
                    "transport_per_channel": detail,
                    "artifacts": safe_artifact_metrics(physical_np, target_np, prior_np),
                })
            else:
                record.update({
                    "physical_range_explosion": train_explosion,
                    "normalized_relative_l2_average": None,
                    "residual_relative_l2": None,
                    "residual_cosine": None,
                    "absolute_residual_error": None,
                    "physical_relative_l2_average": None,
                    "transport": None,
                    "transport_per_channel": None,
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
        "schema_version": "stage-ah-rollout-v1",
        "checkpoint_epoch": 300,
        "initial_snapshot": 169,
        "requested_steps": steps,
        "completed_steps": len(records),
        "ground_truth_steps": 42,
        "teacher_forcing": False,
        "physical_state_rollout": True,
        "encode_once_per_step": encode_count == len(records),
        "decode_once_per_step": decode_count == len(records),
        "finite": all(row["finite"] for row in records),
        "rho_press_positive": all(row["rho_press_positive"] for row in records),
        "FIRST_NORMALIZED_OOD_STEP": first(lambda row: row["above_Rout"]),
        "FIRST_10X_PHYSICAL_RANGE_STEP": first(lambda row: row["physical_range_explosion"]),
        "FIRST_NEGATIVE_RESIDUAL_COSINE_STEP": first(
            lambda row: row.get("residual_cosine") is not None and row["residual_cosine"] < 0.0
        ),
        "FIRST_NEGATIVE_SHELL_SKILL_STEP": first(
            lambda row: row.get("transport") is not None and row["transport"]["shell_skill"] < 0.0
        ),
        "FIRST_NEGATIVE_RADIAL_SKILL_STEP": first(
            lambda row: row.get("transport") is not None and row["transport"]["radial_skill"] < 0.0
        ),
        "selected_steps": [step for step in SELECTED_STEPS if step <= len(records)],
        "records": records,
    }


def rollout_csv_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in result["records"]:
        transport = record.get("transport") or {}
        rows.append({
            "step": record["step"],
            "selected_step": int(record["step"]) in SELECTED_STEPS,
            "ground_truth_available": record["ground_truth_available"],
            "state_l2": record.get("normalized_relative_l2_average"),
            "residual_l2": record.get("residual_relative_l2"),
            "residual_cosine": record.get("residual_cosine"),
            "absolute_residual_error": record.get("absolute_residual_error"),
            "shell_skill": transport.get("shell_skill"),
            "shell_absolute_error": transport.get("shell_absolute_error"),
            "radial_skill": transport.get("radial_skill"),
            "radial_absolute_error": transport.get("radial_absolute_error"),
            "physical_l2": record.get("physical_relative_l2_average"),
            "normalized_global_norm": record["normalized_global_norm"],
            "finite": record["finite"],
            "rho_press_positive": record["rho_press_positive"],
            "physical_range_explosion": record["physical_range_explosion"],
            "rho_q001": record["physical_range"]["rho"]["q001"],
            "rho_q999": record["physical_range"]["rho"]["q999"],
            "press_q001": record["physical_range"]["press"]["q001"],
            "press_q999": record["physical_range"]["press"]["q999"],
        })
    return rows


def branch_norms(model: torch.nn.Module, data: StageSBatchPath) -> list[dict[str, Any]]:
    records = {index: {} for index in range(4)}
    blocks = model.local_no_blocks
    handles = []

    def capture(layer: int, branch: str):
        def hook(_module: torch.nn.Module, _inputs: Any, output: torch.Tensor) -> None:
            records[layer][f"{branch}_output_norm"] = float(
                torch.linalg.vector_norm(output.detach()).cpu()
            )
        return hook

    for layer in range(4):
        handles.extend((
            blocks.convs[layer].register_forward_hook(capture(layer, "spectral")),
            blocks.differential[layer].register_forward_hook(capture(layer, "differential")),
            blocks.local_convs[layer].register_forward_hook(capture(layer, "spherical_disco")),
        ))
    with torch.no_grad():
        data.predict(model, 169)
    for handle in handles:
        handle.remove()
    return [{"model": "Stage AG", "layer": layer, **records[layer]} for layer in range(4)]


def plot_bar(path: Path, labels: Sequence[str], values: Sequence[float], ylabel: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar(labels, values, color=("#777777", "#3b82f6", "#f97316", "#16a34a")[:len(labels)])
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=18)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def make_figures(
    table: Sequence[Mapping[str, Any]],
    theta_rows: Sequence[Mapping[str, Any]],
    radial_rows: Sequence[Mapping[str, Any]],
    stage_ag_rollout: Mapping[str, Any],
    stage_ad_rollout: Mapping[str, Any],
) -> None:
    figures = OUT / "figures"
    labels = [str(row["model"]) for row in table]
    for metric, name, ylabel in (
        ("state_l2", "state_l2_comparison.png", "normalized state relative L2"),
        ("residual_l2", "residual_l2_comparison.png", "normalized residual relative L2"),
        ("cosine", "cosine_comparison.png", "residual cosine"),
        ("shell_skill", "shell_transport.png", "shell persistence-relative skill"),
        ("radial_skill", "radial_transport.png", "radial persistence-relative skill"),
    ):
        values = [float(row[metric]) for row in table]
        plot_bar(figures / name, labels, values, ylabel)

    for rows, regions, name, title in (
        (theta_rows, list(THETA_REGION_BOUNDS), "theta_region_comparison.png", "theta regions"),
        (radial_rows, list(RADIAL_REGION_BOUNDS), "radial_region_comparison.png", "radial regions"),
    ):
        figure, axis = plt.subplots(figsize=(8.0, 4.5))
        for model, color in (("Stage AD", "#f97316"), ("Stage AG", "#16a34a")):
            values = [next(
                float(row["residual_l2_mean"]) for row in rows
                if row["model"] == model and row["channel"] == "ALL" and row["region"] == region
            ) for region in regions]
            axis.plot(regions, values, marker="o", label=model, color=color)
        axis.set_ylabel("mean regional residual relative L2")
        axis.set_title(title)
        axis.tick_params(axis="x", rotation=18)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(figures / name, dpi=150)
        plt.close(figure)

    rollout_pairs = (("Stage AD", stage_ad_rollout, "#f97316"), ("Stage AG", stage_ag_rollout, "#16a34a"))
    for metric, name, ylabel in (
        ("normalized_relative_l2_average", "rollout_state_error.png", "normalized state relative L2"),
        ("residual_cosine", "rollout_cosine.png", "residual cosine"),
    ):
        figure, axis = plt.subplots(figsize=(7.4, 4.3))
        for label, result, color in rollout_pairs:
            rows = [row for row in result["records"] if row.get(metric) is not None]
            axis.plot([row["step"] for row in rows], [row[metric] for row in rows], label=label, color=color)
        axis.set_xlabel("rollout step")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(figures / name, dpi=150)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.4, 4.3))
    for label, result, color in rollout_pairs:
        values = [
            max(max(abs(value["minimum"]), abs(value["maximum"])) for value in row["physical_range"].values())
            for row in result["records"]
        ]
        axis.semilogy([row["step"] for row in result["records"]], values, label=label, color=color)
    axis.set_xlabel("rollout step")
    axis.set_ylabel("maximum absolute decoded value")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(figures / "rollout_physical_range.png", dpi=150)
    plt.close(figure)


def fmt(value: Any) -> str:
    if value is None:
        return "not reached"
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage AH requires CUDA for frozen checkpoint inference; refusing CPU fallback")
    analysis_config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config, stage_s, stage_t = load_stage_ag_contract()
    audit = completion_audit(config, stage_s, stage_t)
    if not audit["STAGE_AG_COMPLETE"] or not audit["STAGE_AG_CONTROLLED_COMPARISON_VALID"]:
        raise RuntimeError("Stage AG completion/identity gate failed; refusing Stage AH analysis")
    atomic_json(OUT / "integrity/stage_ag_completion_audit.json", audit)
    atomic_json(OUT / "stage_ag_integrity_audit.json", audit)

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    data = StageSBatchPath(
        ROOT / stage_s["data"]["dataset"], ROOT / stage_s["preprocessing"]["artifact"], device
    )
    split = json.loads((ROOT / stage_s["data"]["split"]).read_text(encoding="utf-8"))
    validation_pairs = [int(value) for value in split["validation_pair_source_indices"]]
    if validation_pairs != list(range(169, 211)):
        raise ValueError("Stage AH validation population changed")
    r = np.asarray(data.handle["coords/r"], dtype=np.float64)
    _, shell_index = radial_shell_indices(r, int(stage_s["representation"]["shell_count"]))
    reference_path = ROOT / "artifacts/stage_s/diagnostic_reference_train_only.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8")) \
        if reference_path.exists() else build_train_reference(data, ROOT / "artifacts/stage_s")

    best_selector = json.loads(
        (ROOT / "artifacts/stage_ag/validation/best_selector_metric.json").read_text(encoding="utf-8")
    )
    last_selector = json.loads(
        (ROOT / "artifacts/stage_ag/validation/last_selector_metric.json").read_text(encoding="utf-8")
    )
    best_model, best_payload, best_reload = strict_load_stage_ag(
        ROOT / "artifacts/stage_ag/training/checkpoints/best.pt", config, stage_s, data,
        expected_selector=float(best_selector["normalized_per_channel_relative_l2_arithmetic_average"]),
    )
    last_model, last_payload, last_reload = strict_load_stage_ag(
        ROOT / "artifacts/stage_ag/training/checkpoints/last.pt", config, stage_s, data,
        expected_selector=float(last_selector["normalized_per_channel_relative_l2_arithmetic_average"]),
    )
    del last_model, last_payload
    torch.cuda.empty_cache()

    stage_ad_config, _, stage_ad_stage_s = load_stage_ad_contract()
    if stage_ad_stage_s != stage_s:
        raise RuntimeError("Stage AD and Stage AG Stage-S contracts differ")
    stage_ad_model, stage_ad_payload, stage_ad_reload = stage_ad_model_from_checkpoint(
        ROOT / "artifacts/stage_ad/training/disco3d_localno/checkpoints/epoch_0300.pt",
        stage_ad_config, stage_s, device, expected_updates=12600,
    )
    started = time.perf_counter()
    stage_ag_metrics = evaluate_one_step(best_model, data, validation_pairs, shell_index, reference)
    stage_ag_evaluation_seconds = time.perf_counter() - started
    started = time.perf_counter()
    stage_ad_metrics = evaluate_one_step(stage_ad_model, data, validation_pairs, shell_index, reference)
    stage_ad_evaluation_seconds = time.perf_counter() - started
    selector_recompute = validation_score(best_model, data, validation_pairs)
    observed_selector = float(selector_recompute["normalized_relative_l2"]["arithmetic_average"])
    best_reload["recomputed_selector_metric"] = observed_selector
    best_reload["selector_absolute_difference"] = abs(observed_selector - best_reload["expected_selector_metric"])
    best_reload["selector_recompute_pass"] = best_reload["selector_absolute_difference"] <= 1.0e-12
    best_reload["pass"] = best_reload["pass_before_metric_recompute"] and best_reload["selector_recompute_pass"]
    last_reload["recorded_completion_reload_pass"] = json.loads(
        (ROOT / "artifacts/stage_ag/integrity/last_reload_test.json").read_text(encoding="utf-8")
    )["pass"]
    last_reload["pass"] = last_reload["pass_before_metric_recompute"] and last_reload["recorded_completion_reload_pass"]
    reload_audit = {
        "schema_version": "stage-ah-checkpoint-reload-audit-v1",
        "best": best_reload,
        "last": last_reload,
        "pass": best_reload["pass"] and last_reload["pass"],
    }
    atomic_json(OUT / "integrity/checkpoint_reload_audit.json", reload_audit)
    if not reload_audit["pass"]:
        raise RuntimeError("Stage AH checkpoint reload audit failed")

    stored_stage_ad = json.loads(
        (ROOT / "artifacts/stage_ad/training/disco3d_localno/one_step_metrics.json").read_text(encoding="utf-8")
    )["metrics"]
    recompute_delta = max(
        abs(compact_metrics(stage_ad_metrics)[key] - compact_metrics(stored_stage_ad)[key])
        for key in compact_metrics(stage_ad_metrics)
    )
    if recompute_delta > 1.0e-12:
        raise RuntimeError(f"Stage AD formal metrics changed: maximum delta {recompute_delta}")
    identity = {
        "schema_version": "stage-ah-controlled-comparison-identity-v1",
        "pass": True,
        "stage_ad_model": "INDEX_SPACE_DISCO3D_LOCALNO",
        "stage_ag_model": config["model"]["name"],
        "parameter_count_both": 363480,
        "initial_trainable_state_sha256_both": config["initialization"]["expected_trainable_state_sha256"],
        "dataset_sha256_both": config["data"]["dataset_sha256"],
        "normalizer_sha256_both": config["preprocessing"]["normalizer_sha256"],
        "pair_order_sha256_both": audit["pair_order_sha256"],
        "train_pairs_both": 168,
        "validation_pairs_both": 42,
        "optimizer_updates_both": 12600,
        "objective_both": "PlainL2Loss_normalized_residual",
        "formal_selector_both": config["validation"]["selector"],
        "only_intended_intervention": "index-space DISCO3D to anisotropic spherical tangent-proxy DISCO3D",
        "stage_ad_metric_maximum_recompute_delta": recompute_delta,
    }
    atomic_json(OUT / "integrity/controlled_comparison_identity.json", identity)

    models = {"Stage AD": stage_ad_model, "Stage AG": best_model}
    pair_rows, channel_pair_rows, transport_rows, theta_raw, radial_raw = analyze_validation_pairs(
        models, data, validation_pairs, shell_index
    )
    indexed_pairs = {(row["model"], int(row["source_snapshot"])): row for row in pair_rows}
    paired: list[dict[str, Any]] = []
    pair_metrics = (
        "state_l2", "residual_l2", "residual_cosine", "absolute_residual_error",
        "shell_skill", "shell_absolute_error", "radial_skill", "radial_absolute_error",
    )
    for source in validation_pairs:
        row = {"source_snapshot": source, "target_snapshot": source + 1}
        for model_name, prefix in (("Stage AD", "stage_ad"), ("Stage AG", "stage_ag")):
            source_row = indexed_pairs[(model_name, source)]
            row.update({f"{prefix}_{metric}": source_row[metric] for metric in pair_metrics})
        paired.append(row)
    atomic_csv(OUT / "one_step/per_pair_metrics.csv", paired)
    bootstrap = paired_bootstrap(
        paired,
        ("state_l2", "residual_l2", "residual_cosine", "shell_absolute_error", "radial_absolute_error"),
        seed=int(analysis_config["validation"]["bootstrap_seed"]),
        samples=int(analysis_config["validation"]["bootstrap_samples"]),
    )
    atomic_csv(OUT / "statistics/paired_bootstrap.csv", bootstrap)

    channel_summary: list[dict[str, Any]] = []
    model_metrics = {"Stage AD": stage_ad_metrics, "Stage AG": stage_ag_metrics}
    for model_name in ("Stage AD", "Stage AG"):
        for channel in CHANNELS:
            selected = [
                row for row in channel_pair_rows
                if row["model"] == model_name and row["channel"] == channel
            ]
            metric = model_metrics[model_name]
            channel_summary.append({
                "model": model_name,
                "channel": channel,
                "state_l2": metric["model"]["normalized_relative_l2"]["per_channel"][channel],
                "residual_l2": metric["model"]["residual"]["per_channel_relative_l2"][channel],
                "residual_cosine": metric["model"]["residual"]["per_channel_cosine"][channel],
                "shell_skill": metric["transport"]["per_channel"][channel]["shell_skill_median"],
                "shell_absolute_error": float(np.median([row["shell_absolute_error"] for row in selected])),
                "radial_skill": metric["transport"]["per_channel"][channel]["radial_skill_median"],
                "radial_absolute_error": float(np.median([row["radial_absolute_error"] for row in selected])),
            })
    channel_index = {(row["model"], row["channel"]): row for row in channel_summary}
    for row in channel_summary:
        if row["model"] == "Stage AG":
            baseline = channel_index[("Stage AD", row["channel"])]
            for metric in (
                "state_l2", "residual_l2", "residual_cosine", "shell_skill",
                "shell_absolute_error", "radial_skill", "radial_absolute_error",
            ):
                row[f"stage_ag_minus_stage_ad_{metric}"] = float(row[metric]) - float(baseline[metric])
    atomic_csv(OUT / "one_step/per_channel_metrics.csv", channel_summary)

    shell_rows = [{key: row[key] for key in (
        "model", "source_snapshot", "target_snapshot", "channel", "shell_skill",
        "shell_absolute_error", "shell_persistence_absolute_error"
    )} for row in transport_rows]
    radial_transport_rows = [{key: row[key] for key in (
        "model", "source_snapshot", "target_snapshot", "channel", "radial_skill",
        "radial_absolute_error", "radial_persistence_absolute_error"
    )} for row in transport_rows]
    atomic_csv(OUT / "transport/shell_metrics.csv", shell_rows)
    atomic_csv(OUT / "transport/radial_metrics.csv", radial_transport_rows)
    atomic_csv(OUT / "transport/absolute_transport_errors.csv", [{
        "model": row["model"], "source_snapshot": row["source_snapshot"], "channel": row["channel"],
        "shell_absolute_error": row["shell_absolute_error"],
        "radial_absolute_error": row["radial_absolute_error"],
    } for row in transport_rows])

    theta_summary = aggregate_region_rows(theta_raw)
    radial_summary = aggregate_region_rows(radial_raw)
    atomic_csv(OUT / "regional/theta_region_metrics.csv", theta_summary)
    atomic_csv(OUT / "regional/radial_region_metrics.csv", radial_summary)
    geometry_rows = geometry_benefit_rows(models, data, validation_pairs)
    atomic_csv(OUT / "regional/geometry_benefit_comparison.csv", geometry_rows)

    branch_rows = branch_norms(best_model, data)
    for row in csv_rows(ROOT / "artifacts/stage_ad/attribution/branch_activity.csv"):
        branch_rows.append({
            "model": "Stage AD",
            "layer": int(row["layer"]),
            "spectral_output_norm": float(row["spectral_output_norm"]),
            "differential_output_norm": float(row["differential_output_norm"]),
            "spherical_disco_output_norm": float(row["disco3d_output_norm"]),
        })
    atomic_csv(OUT / "attribution/branch_norms.csv", branch_rows)
    with torch.no_grad():
        full_probe = data.predict(best_model, 169)["predicted_residual"]
    original_indices = list(best_model.local_no_blocks.disco_idx_list)
    best_model.local_no_blocks.disco_idx_list = [-1] * 4
    without_disco_metrics = evaluate_one_step(
        best_model, data, validation_pairs, shell_index, reference, include_transport_rows=False
    )
    with torch.no_grad():
        without_probe = data.predict(best_model, 169)["predicted_residual"]
    best_model.local_no_blocks.disco_idx_list = original_indices
    contribution_norm = float(torch.linalg.vector_norm(full_probe - without_probe).detach().cpu())
    full_compact = compact_metrics(stage_ag_metrics)
    without_compact = compact_metrics(without_disco_metrics)
    ablation_rows: list[dict[str, Any]] = []
    for metric in full_compact:
        ablation_rows.extend((
            {"variant": "FULL", "metric": metric, "value": full_compact[metric]},
            {"variant": "FULL_MINUS_DISCO", "metric": metric, "value": without_compact[metric]},
            {"variant": "DISCO_CONTRIBUTION", "metric": metric,
             "value": full_compact[metric] - without_compact[metric]},
        ))
    ablation_rows.append({
        "variant": "DISCO_CONTRIBUTION", "metric": "probe_residual_output_l2_norm",
        "value": contribution_norm,
    })
    stage_ad_contribution = next(
        float(row["full"])
        for row in csv_rows(ROOT / "artifacts/stage_ad/attribution/branch_ablation.csv")
        if row["metric"] == "probe_residual_disco_contribution_norm"
    )
    ablation_rows.append({
        "variant": "STAGE_AD_DISCO_CONTRIBUTION", "metric": "probe_residual_output_l2_norm",
        "value": stage_ad_contribution,
    })
    atomic_csv(OUT / "attribution/branch_ablation.csv", ablation_rows)
    train_rows = csv_rows(ROOT / "artifacts/stage_ag/training/train_log.csv")
    spherical_branch_active = bool(
        contribution_norm > 0.0
        and all(float(row["spherical_disco_output_norm"]) > 0.0 for row in branch_rows if row["model"] == "Stage AG")
        and all(float(row["disco_gradient_norm_min"]) > 0.0 for row in train_rows)
    )

    rollout_result = rollout_stage_ah(best_model, data, shell_index, reference, steps=100)
    atomic_csv(OUT / "rollout/rollout_metrics.csv", rollout_csv_rows(rollout_result))
    landmarks = {
        key: rollout_result[key] for key in (
            "FIRST_NEGATIVE_RESIDUAL_COSINE_STEP", "FIRST_NEGATIVE_SHELL_SKILL_STEP",
            "FIRST_NEGATIVE_RADIAL_SKILL_STEP", "FIRST_10X_PHYSICAL_RANGE_STEP",
            "FIRST_NORMALIZED_OOD_STEP", "finite", "rho_press_positive", "completed_steps",
        )
    }
    landmarks.update({
        "formal_best_epoch": int(best_selector["epoch"]),
        "initial_snapshot": 169,
        "teacher_forcing": False,
    })
    atomic_json(OUT / "rollout/stability_landmarks.json", landmarks)

    stage_ad_summary = json.loads(
        (ROOT / "artifacts/stage_ad/training/disco3d_localno/training_summary.json").read_text(encoding="utf-8")
    )
    af_cost = json.loads(
        (ROOT / "artifacts/stage_af/preflight/cuda_feasibility.json").read_text(encoding="utf-8")
    )
    ad_cost = next(
        row for row in csv_rows(ROOT / "artifacts/stage_af/comparison/computational_cost.csv")
        if row["stage"] == "AD"
    )
    efficiency = [
        {
            "model": "Stage AD", "geometry": "INDEX_SPACE_DISCO3D", "parameter_count": 363480,
            "training_runtime_seconds": float(stage_ad_summary["runtime_seconds"]),
            "evaluation_runtime_seconds": stage_ad_evaluation_seconds,
            "forward_runtime_seconds": float(ad_cost["forward_seconds"]),
            "backward_runtime_seconds": float(ad_cost["backward_seconds"]),
            "peak_allocated_vram_mib": float(stage_ad_summary["peak_allocated_mib_max"]),
            "peak_reserved_vram_mib": float(stage_ad_summary["peak_reserved_mib_max"]),
        },
        {
            "model": "Stage AG", "geometry": "ANISOTROPIC_LOCAL_SPHERICAL_TANGENT_PROXY",
            "parameter_count": 363480,
            "training_runtime_seconds": audit["total_training_seconds"],
            "evaluation_runtime_seconds": stage_ag_evaluation_seconds,
            "forward_runtime_seconds": float(af_cost["forward_seconds"]),
            "backward_runtime_seconds": float(af_cost["backward_seconds"]),
            "peak_allocated_vram_mib": audit["peak_allocated_vram_mib"],
            "peak_reserved_vram_mib": audit["peak_reserved_vram_mib"],
        },
    ]
    atomic_csv(OUT / "efficiency/stage_ad_vs_stage_ag_cost.csv", efficiency)

    ad_compact = compact_metrics(stage_ad_metrics)
    ag_compact = compact_metrics(stage_ag_metrics)
    stage_t_metrics = json.loads(
        (ROOT / "artifacts/stage_t/full_long/checkpoint_metrics_epoch_0150.json").read_text(encoding="utf-8")
    )
    stage_t_compact = compact_metrics(stage_t_metrics)
    stage_t_rollout = json.loads(
        (ROOT / "artifacts/stage_t/full_long/rollout/epoch_0150_rollout.json").read_text(encoding="utf-8")
    )
    stage_ad_rollout = json.loads(
        (ROOT / "artifacts/stage_ad/training/disco3d_localno/rollout_metrics.json").read_text(encoding="utf-8")
    )
    persistence_state = float(stage_ag_metrics["persistence"]["normalized_relative_l2"]["arithmetic_average"])
    table = [
        {"model": "Persistence", "geometry": "none", "params": 0,
         "state_l2": persistence_state, "residual_l2": 1.0, "cosine": 0.0,
         "shell_skill": 0.0, "radial_skill": 0.0, "first10x": "stable"},
        {"model": "Stage-T LocalNO", "geometry": "no DISCO", "params": 358296,
         **stage_t_compact, "first10x": stage_t_rollout["FIRST_10X_PHYSICAL_RANGE_STEP"]},
        {"model": "Stage-AD DISCO3D", "geometry": "index-space", "params": 363480,
         **ad_compact, "first10x": stage_ad_rollout["FIRST_10X_PHYSICAL_RANGE_STEP"]},
        {"model": "Stage-AG spherical DISCO3D", "geometry": "anisotropic spherical tangent",
         "params": 363480, **ag_compact, "first10x": rollout_result["FIRST_10X_PHYSICAL_RANGE_STEP"]},
    ]

    model_transport_absolute: dict[str, dict[str, float]] = {}
    for model_name in ("Stage AD", "Stage AG"):
        selected = [row for row in transport_rows if row["model"] == model_name]
        model_transport_absolute[model_name] = scalar_transport(selected)
    gates = primary_gates(
        ad_compact, ag_compact,
        stage_ad_shell_absolute_error=model_transport_absolute["Stage AD"]["shell_absolute_error"],
        stage_ag_shell_absolute_error=model_transport_absolute["Stage AG"]["shell_absolute_error"],
        stage_ad_radial_absolute_error=model_transport_absolute["Stage AD"]["radial_absolute_error"],
        stage_ag_radial_absolute_error=model_transport_absolute["Stage AG"]["radial_absolute_error"],
        first_10x_step=rollout_result["FIRST_10X_PHYSICAL_RANGE_STEP"],
    )
    decision_code = scientific_decision(gates, ad_compact, ag_compact, integrity_valid=True)
    decision_names = {
        "A": "SPHERICAL_GEOMETRY_RESCUES_DYNAMICS",
        "B": "SPHERICAL_GEOMETRY_PARTIALLY_IMPROVES_DYNAMICS",
        "C": "SPHERICAL_GEOMETRY_IMPROVES_ONE_STEP_ONLY",
        "D": "SPHERICAL_GEOMETRY_NOT_SUPPORTED",
        "E": "SPHERICAL_GEOMETRY_MODEL_WORSE",
        "F": "TRAINING_OR_ARTIFACT_INVALID",
    }
    decision = {
        "STAGE_AG_COMPLETE": True,
        "STAGE_AG_CONTROLLED_COMPARISON_VALID": True,
        "PRIMARY_DECISION": decision_code,
        "PRIMARY_DECISION_NAME": decision_names[decision_code],
        "REPRODUCTION_SCOPE": "ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION",
        "MODEL": "ANISOTROPIC_SPHERICAL_DISCO_LOCALNO",
        "DISCO_GEOMETRY": "ANISOTROPIC_LOCAL_SPHERICAL_TANGENT_PROXY",
        **gates,
        "SPHERICAL_DISCO_BRANCH_ACTIVE": spherical_branch_active,
        "EXACT_3D_DISCO_IMPLEMENTATION_FOUND": False,
        "EXACT_REPRODUCTION_BLOCKED": True,
        "AUTHORIZE_NEXT_STAGE": "adapted_final_baseline_benchmark"
        if decision_code != "F" else "training_integrity_repair",
    }
    atomic_json(OUT / "STAGE_AH_DECISION.json", decision)

    aggregate = {
        "schema_version": "stage-ah-one-step-aggregate-v1",
        "formal_best_epoch": int(best_selector["epoch"]),
        "formal_selector": config["validation"]["selector"],
        "models": {
            "persistence": table[0], "stage_t": table[1], "stage_ad": table[2], "stage_ag": table[3],
        },
        "stage_ag_minus_stage_ad": {
            key: ag_compact[key] - ad_compact[key]
            for key in ("state_l2", "residual_l2", "cosine", "shell_skill", "radial_skill", "physical_l2")
        },
        "absolute_transport": model_transport_absolute,
        "stage_ag_normalized_space": {
            "state": stage_ag_metrics["model"]["normalized_relative_l2"],
            "residual": stage_ag_metrics["model"]["residual"],
        },
        "stage_ag_inverse_p3_physical_space": {
            "physical_relative_l2": stage_ag_metrics["model"]["physical_relative_l2"],
            "physical_prediction_distribution": stage_ag_metrics["physical_prediction_distribution"],
            "decoder_tail": stage_ag_metrics["preprocessing_tail"],
            "rho_press_positive": stage_ag_metrics["rho_press_positive"],
            "finite": stage_ag_metrics["finite"],
        },
        "paired_bootstrap": bootstrap,
        "gates": gates,
        "decision": decision,
    }
    atomic_json(OUT / "one_step/aggregate_metrics.json", aggregate)
    make_figures(table, theta_summary, radial_summary, rollout_result, stage_ad_rollout)

    table_lines = [
        "| model | geometry | params | state L2 | residual L2 | cosine | shell skill | radial skill | first10x |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    table_lines.extend(
        f"| {row['model']} | {row['geometry']} | {row['params']} | {fmt(row['state_l2'])} | "
        f"{fmt(row['residual_l2'])} | {fmt(row['cosine'])} | {fmt(row['shell_skill'])} | "
        f"{fmt(row['radial_skill'])} | {fmt(row['first10x'])} |"
        for row in table
    )
    report = f"""# Stage AH Scientific Result Analysis

{chr(10).join(table_lines)}

## Controlled result

- formal selector: `{config['validation']['selector']}`
- formal best epoch: `{best_selector['epoch']}`
- Stage AG minus Stage AD state L2: `{ag_compact['state_l2'] - ad_compact['state_l2']}`
- Stage AG minus Stage AD residual L2: `{ag_compact['residual_l2'] - ad_compact['residual_l2']}`
- Stage AG minus Stage AD residual cosine: `{ag_compact['cosine'] - ad_compact['cosine']}`
- Stage AG minus Stage AD shell skill: `{ag_compact['shell_skill'] - ad_compact['shell_skill']}`
- Stage AG minus Stage AD radial skill: `{ag_compact['radial_skill'] - ad_compact['radial_skill']}`
- Stage AG first 10x physical-range step: `{fmt(rollout_result['FIRST_10X_PHYSICAL_RANGE_STEP'])}`
- spherical DISCO branch active: `{str(spherical_branch_active).lower()}`

The paired, channel, theta-region, radial-region, geometry-benefit, branch-ablation,
rollout, and efficiency tables are stored in their respective Stage AH directories.
Kernel-geometry correlation is exploratory and is not treated as causal proof.

## Interpretation boundary

This is an adapted spherical Kerr--Schild workflow using a Euclidean spherical
tangent proxy. It is not a Kerr--Schild covariant operator and is not the exact
volumetric 3-D DISCO implementation from the paper.

`REPRODUCTION_SCOPE = ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION`

`EXACT_REPRODUCTION_BLOCKED = true`
"""
    atomic_text(OUT / "STAGE_AH_REPORT.md", report)
    gate_lines = "\n".join(
        f"{key} = {'PASS' if value else 'FAIL'}"
        for key, value in gates.items() if key.endswith(("GATE", "GAIN"))
    )
    decision_text = f"""# Stage AH Decision

```text
STAGE_AG_COMPLETE = true
STAGE_AG_CONTROLLED_COMPARISON_VALID = true

PRIMARY_DECISION = {decision_code}
PRIMARY_DECISION_NAME = {decision_names[decision_code]}

REPRODUCTION_SCOPE = ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION
MODEL = ANISOTROPIC_SPHERICAL_DISCO_LOCALNO
DISCO_GEOMETRY = ANISOTROPIC_LOCAL_SPHERICAL_TANGENT_PROXY

{gate_lines}

SPHERICAL_DISCO_BRANCH_ACTIVE = {str(spherical_branch_active).lower()}
EXACT_3D_DISCO_IMPLEMENTATION_FOUND = false
EXACT_REPRODUCTION_BLOCKED = true
AUTHORIZE_NEXT_STAGE = {decision['AUTHORIZE_NEXT_STAGE']}
```

The conclusion applies only to the tested anisotropic spherical tangent-proxy
DISCO in this adapted workflow; it is not a general claim about DISCO or the
paper's unavailable exact implementation.
"""
    atomic_text(OUT / "STAGE_AH_DECISION.md", decision_text)
    print(json.dumps({
        "STAGE_AG_COMPLETE": True,
        "formal_best_epoch": int(best_selector["epoch"]),
        "stage_ag": ag_compact,
        "stage_ad": ad_compact,
        "rollout": landmarks,
        "decision": decision,
    }, indent=2, sort_keys=True), flush=True)
    del best_model, best_payload, stage_ad_model, stage_ad_payload
    data.close()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
