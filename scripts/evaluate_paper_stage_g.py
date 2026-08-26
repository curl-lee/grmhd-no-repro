#!/usr/bin/env python
"""Strict validation and 19/100-step physical rollout for a paired paper pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

from grmhd import CHANNELS
from grmhd.dataset import sha256_file
from grmhd.paper_checkpoint import load_paper_checkpoint
from grmhd.paper_config import build_paper_model, load_paper_experiment_config
from grmhd.paper_data_processor import PaperDataProcessor
from grmhd.paper_dissipation import PaperDissipativeReference, global_state_norm
from grmhd.paper_metrics import ClampMaskAccumulator, PaperMetricAccumulator
from grmhd.paper_priors import PaperResidualEnvelope
from grmhd.paper_protocol import PaperReduced100Protocol
from grmhd.paper_references import PaperReferenceStates, paper_reference_metadata
from grmhd.paper_stage_g_evaluation import (
    artifact_diagnostics,
    physical_state_statistics,
    temporal_series_statistics,
    total_variation_and_high_k,
)
from grmhd.paper_stage_r import reconstruct_normalized_state
from grmhd.paper_trainer import build_paper_optimizer, build_warmup_cosine_scheduler
from grmhd.upstream_adapters import GRMHDNextStepDataset


SELECTED_STEPS = (1, 3, 5, 10, 19, 25, 50, 75, 100)


def json_value(value: Any) -> Any:
    if torch.is_tensor(value):
        detached = value.detach().cpu()
        return float(detached) if detached.numel() == 1 else detached.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def manifest_pairing_hashes(manifest: Mapping[str, Any]) -> tuple[str, str]:
    shared_hash = manifest.get("shared_initial_state", {}).get(
        "tensor_state_sha256"
    )
    if shared_hash is None:
        shared_hash = manifest.get("checksums", {}).get(
            "shared_initial_state_tensor"
        )
    if shared_hash is None:
        shared_hash = manifest.get("initial_state", {}).get(
            "tensor_state_sha256"
        )
    if shared_hash is None:
        shared_hash = manifest.get("pairing", {}).get(
            "initial_state_tensor_sha256"
        )
    order_hash = manifest.get("checksums", {}).get("pair_order_file")
    if order_hash is None:
        order_hash = manifest.get("pairing", {}).get("pair_order_sha256")
    for name, value in (
        ("shared initial state", shared_hash),
        ("pair order", order_hash),
    ):
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Run manifest {name} hash is invalid")
    return shared_hash, order_hash


def interpret_rollout_output(
    raw_output: torch.Tensor,
    normalized_input: torch.Tensor,
    *,
    residual_contract: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Interpret the network output without scaling, clipping, or repair."""
    if residual_contract:
        return (
            reconstruct_normalized_state(normalized_input, raw_output),
            raw_output,
        )
    return raw_output, None


def expected_artifact_arguments(config) -> dict[str, Any]:
    values = config.values
    return {
        "source_hdf5_checksum": values["provenance"]["dataset"]["sha256"],
        "preprocessing_stats_checksum": config.prior_preprocessing_checksum,
        "training_indices": tuple(range(*values["protocol"]["train_snapshots"])),
        "protocol_name": values["protocol"]["name"],
        "thermal_channel": values["thermal"]["channel"],
    }


def reference_states(
    *,
    fields: Mapping[str, Any],
    normalized_prediction: torch.Tensor,
    physical_prediction: torch.Tensor,
    processor: PaperDataProcessor,
) -> PaperReferenceStates:
    return PaperReferenceStates(
        raw_physical_target=fields["raw_physical_target"],
        oracle_physical_target=fields["oracle_physical_target"],
        normalized_target=fields["normalized_target"],
        normalized_prediction=normalized_prediction,
        model_physical_prediction=physical_prediction,
        metadata=paper_reference_metadata(processor.preprocessor),
    )


def load_strict_checkpoint(
    *,
    checkpoint_dir: Path,
    config,
    config_checksum: str,
    device: torch.device,
    shared_hash: str,
    order_hash: str,
) -> tuple[Any, PaperDataProcessor, dict[str, Any]]:
    model = build_paper_model(config)
    optimizer = build_paper_optimizer(
        model,
        learning_rate=config.values["optimizer"]["learning_rate"],
        weight_decay=config.values["optimizer"]["weight_decay"],
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_epochs=30,
        warmup_epochs=2,
        min_learning_rate=config.values["scheduler"]["min_learning_rate"],
    )
    loaded = load_paper_checkpoint(
        checkpoint_dir,
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_config_checksum=config_checksum,
    )
    stage_g = loaded.metadata.get("stage_g")
    stage_i = loaded.metadata.get("stage_i")
    stage_k = loaded.metadata.get("stage_k")
    stage_o = loaded.metadata.get("stage_o")
    stage_r = loaded.metadata.get("stage_r")
    pairing = stage_k if config.stage_k else stage_g
    if not isinstance(pairing, Mapping):
        raise ValueError("Checkpoint lacks controlled-pair metadata")
    initial_key = (
        "initial_state_sha256" if config.stage_k else "shared_initial_state_sha256"
    )
    if pairing[initial_key] != shared_hash:
        raise ValueError("Checkpoint initial-state hash changed")
    if pairing["pair_order_sha256"] != order_hash:
        raise ValueError("Checkpoint pair-order hash changed")
    loaded.model.to(device).eval()
    processor = PaperDataProcessor.from_config(config).to(device)
    processor.eval()
    processor.set_epoch(loaded.epoch)
    return loaded, processor, {
        "strict_model_reload": True,
        "optimizer_reload": loaded.optimizer is not None,
        "scheduler_reload": loaded.scheduler is not None,
        "epoch": loaded.epoch,
        "optimizer_step": pairing["optimizer_step"],
        "roi_ramp": loaded.metadata["roi_ramp"],
        "provenance_validated": True,
        "initial_state_sha256": pairing[initial_key],
        "pair_order_sha256": pairing["pair_order_sha256"],
        "stage_i": stage_i,
        "stage_k": stage_k,
        "stage_o": stage_o,
        "stage_r": stage_r,
    }


def shell_clamp_summary(
    counts: dict[str, np.ndarray], shells: torch.Tensor
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, values in counts.items():
        output[name] = {
            channel: {
                str(shell): float(values[channel_index, shell, 0])
                / max(float(values[channel_index, shell, 1]), 1.0)
                for shell in range(shells.shape[1])
            }
            for channel_index, channel in enumerate(CHANNELS)
        }
    return output


def evaluate_validation(
    *,
    model: torch.nn.Module,
    processor: PaperDataProcessor,
    loader: DataLoader,
    envelope: PaperResidualEnvelope,
    dissipation: PaperDissipativeReference,
) -> dict[str, Any]:
    model_metric = PaperMetricAccumulator()
    persistence_metric = PaperMetricAccumulator()
    clamp = ClampMaskAccumulator(
        gamma=processor.preprocessor.gamma,
        inverse_clamp_fraction=processor.preprocessor.inverse_clamp_fraction,
    )
    shell_count = int(processor.shells.shape[1])
    shell_counts = {
        name: np.zeros((len(CHANNELS), shell_count, 2), dtype=np.int64)
        for name in ("target", "model", "intersection", "model_only", "target_only")
    }
    finite = True
    positive = True
    bound_clamp = []
    envelope_rho = []
    envelope_press = []
    input_norms = []
    prediction_norms = []
    gates = []
    above_rin = []
    above_rout = []
    roi_clamp_intersection = 0
    roi_voxels = 0
    probe_hash = None
    with torch.no_grad():
        for batch in loader:
            sample = processor.preprocess(batch)
            prediction = model(x=sample["x"])
            prediction, fields = processor.postprocess(prediction, sample)
            decoded = processor.decode_prediction(
                prediction,
                apply_evaluation_clamp=bool(
                    processor.config.values["evaluation"]["rho_press_eval_clamp"]
                ),
            )
            model_metric.update(
                reference_states(
                    fields=fields,
                    normalized_prediction=prediction,
                    physical_prediction=decoded.physical_prediction,
                    processor=processor,
                ),
                channel_axis=1,
            )
            persistence_physical = processor.preprocessor.decode(
                fields["normalized_input"], channel_axis=1
            )
            persistence_metric.update(
                reference_states(
                    fields=fields,
                    normalized_prediction=fields["normalized_input"],
                    physical_prediction=persistence_physical,
                    processor=processor,
                ),
                channel_axis=1,
            )
            clamp.update(
                normalized_target=fields["normalized_target"],
                normalized_prediction=prediction,
                channel_axis=1,
            )
            limit = (
                processor.preprocessor.gamma
                * processor.preprocessor.inverse_clamp_fraction
            )
            target_mask = torch.abs(fields["normalized_target"]) > limit
            model_mask = torch.abs(prediction) > limit
            velocity_target_clamp = torch.any(target_mask[:, 5:8], dim=1)
            canonical_roi = fields["canonical_roi_mask"]
            roi_clamp_intersection += int(
                (canonical_roi & velocity_target_clamp).sum().cpu()
            )
            roi_voxels += int(canonical_roi.sum().cpu())
            masks = {
                "target": target_mask,
                "model": model_mask,
                "intersection": target_mask & model_mask,
                "model_only": model_mask & ~target_mask,
                "target_only": target_mask & ~model_mask,
            }
            shell_masks = processor.shells[0].bool()
            for mask_name, mask in masks.items():
                for channel in range(len(CHANNELS)):
                    for shell in range(shell_count):
                        selected = mask[:, channel, shell_masks[shell]]
                        shell_counts[mask_name][channel, shell, 0] += int(
                            selected.sum().cpu()
                        )
                        shell_counts[mask_name][channel, shell, 1] += selected.numel()
            violations = envelope.violation_fractions(
                prediction, fields["radial_baseline_normalized"], channel_axis=1
            )
            envelope_rho.append(violations["rho"])
            envelope_press.append(violations["press"])
            dissipative = dissipation.apply(fields["normalized_input"], prediction)
            input_norm = global_state_norm(fields["normalized_input"])
            prediction_norm = global_state_norm(prediction)
            input_norms.extend(input_norm.detach().cpu().tolist())
            prediction_norms.extend(prediction_norm.detach().cpu().tolist())
            gates.extend(dissipative.gate.detach().cpu().tolist())
            above_rin.extend((prediction_norm > dissipation.rin).detach().cpu().tolist())
            above_rout.extend((prediction_norm > dissipation.rout).detach().cpu().tolist())
            bound_clamp.append(decoded.clamp_fraction["combined"])
            finite &= bool(
                torch.isfinite(prediction).all()
                and torch.isfinite(decoded.physical_prediction).all()
            )
            positive &= bool(torch.all(decoded.physical_prediction[:, 3:5] > 0))
            if probe_hash is None:
                probe_hash = tensor_sha256(prediction)
    return {
        "model": model_metric.finalize(),
        "persistence": persistence_metric.finalize(),
        "saturation": clamp.finalize(),
        "shell_wise_saturation": shell_clamp_summary(shell_counts, processor.shells),
        "constraints": {
            "finite": finite,
            "rho_press_positive": positive,
            "evaluation_bound_clamp_fraction_mean": float(np.mean(bound_clamp)),
            "envelope_violation_fraction_mean": {
                "rho": float(np.mean(envelope_rho)),
                "press": float(np.mean(envelope_press)),
            },
            "input_global_norm_mean": float(np.mean(input_norms)),
            "prediction_global_norm_mean": float(np.mean(prediction_norms)),
            "dissipation_gate_mean": float(np.mean(gates)),
            "prediction_above_rin_fraction": float(np.mean(above_rin)),
            "prediction_above_rout_fraction": float(np.mean(above_rout)),
            "roi_target_clamp_overlap": (
                None
                if roi_voxels == 0
                else roi_clamp_intersection / roi_voxels
            ),
        },
        "probe_prediction_sha256": probe_hash,
    }


def compact_rollout_row(record: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        "step": record["step"],
        "finite": record["finite"],
        "rho_press_positive": record["rho_press_positive"],
        "evaluation_bound_clamp_fraction": record[
            "evaluation_bound_clamp_fraction"
        ],
        "prediction_global_norm": record["prediction_global_norm"],
        "above_rin": record["above_rin"],
        "above_rout": record["above_rout"],
        "artifact_flag_count": len(record["artifacts"]["flags"]),
    }
    if "oracle_aware" in record:
        metrics = record["oracle_aware"]["metrics"]
        for metric_name in ("E_norm", "E_model_oracle", "E_model_raw", "E_oracle_raw"):
            row[metric_name] = metrics[metric_name]["arithmetic_average"]
        for channel, value in metrics["E_norm"]["per_channel"].items():
            row[f"E_norm_{channel}"] = value
        row["model_only_saturation_mean"] = float(
            np.mean(
                [
                    channel["model_only_fraction"]
                    for channel in record["saturation"]["channels"].values()
                ]
            )
        )
    return row


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_rollout(
    *,
    model: torch.nn.Module,
    processor: PaperDataProcessor,
    validation_dataset,
    r: np.ndarray,
    dissipation: PaperDissipativeReference,
) -> tuple[dict[str, Any], dict[str, Any], dict[int, torch.Tensor]]:
    initial = validation_dataset.load_snapshot(91).unsqueeze(0)
    current = initial
    initial_device = initial.to(processor.device)
    initial_normalized = processor.preprocessor.encode(initial_device, channel_axis=1)
    initial_physical_oracle = processor.preprocessor.decode(
        initial_normalized, channel_axis=1
    )
    initial_stats = physical_state_statistics(initial_device, r)
    current_normalized = initial_normalized
    times = np.asarray(validation_dataset.times, dtype=np.float64)
    dt = float(np.median(np.diff(times)))
    selected_states: dict[int, torch.Tensor] = {}
    gt_records: list[dict[str, Any]] = []
    no_gt_records: list[dict[str, Any]] = []
    temporal_means: list[list[float]] = []
    temporal_stds: list[list[float]] = []
    transform_before = dict(processor.transform_counts)
    with torch.no_grad():
        for step in range(1, 101):
            source_index = 90 + step
            target_index = source_index + 1
            has_gt = step <= 19
            if has_gt:
                raw_target = validation_dataset.load_snapshot(target_index).unsqueeze(0).to(
                    processor.device
                )
                normalized_target = processor.preprocessor.encode(
                    raw_target, channel_axis=1
                )
                processor.transform_counts["target_encode"] += 1
                oracle_target = processor.preprocessor.decode(
                    normalized_target, channel_axis=1
                )
                processor.transform_counts["oracle_decode"] += 1
                fields = {
                    "raw_physical_target": raw_target,
                    "oracle_physical_target": oracle_target,
                    "normalized_target": normalized_target,
                    "normalized_input": current_normalized,
                }
            shells = processor.shells.expand(current.shape[0], -1, -1, -1, -1).to(
                dtype=current_normalized.dtype
            )
            model_input = torch.cat((current_normalized, shells), dim=1)
            raw_output = model(x=model_input)
            if raw_output.shape != current_normalized.shape:
                raise ValueError("Rollout model output shape changed")
            prediction, predicted_residual = interpret_rollout_output(
                raw_output,
                current_normalized,
                residual_contract=processor.config.stage_r,
            )
            decoded = processor.decode_prediction(
                prediction,
                apply_evaluation_clamp=bool(
                    processor.config.values["evaluation"]["rho_press_eval_clamp"]
                ),
            )
            physical = decoded.physical_prediction
            step_finite = bool(
                torch.isfinite(prediction).all() and torch.isfinite(physical).all()
            )
            step_positive = bool(torch.all(physical[:, 3:5] > 0))
            if not step_finite or not step_positive:
                raise FloatingPointError(
                    f"Physical rollout failed finite/positivity at step {step}"
                )
            physical_flat = physical.flatten(start_dim=2).to(dtype=torch.float64)
            temporal_means.append(
                physical_flat.mean(dim=2)[0].detach().cpu().tolist()
            )
            temporal_stds.append(
                physical_flat.std(dim=2)[0].detach().cpu().tolist()
            )
            prediction_norm = float(global_state_norm(prediction)[0].cpu())
            dissipative = dissipation.apply(current_normalized, prediction)
            model_clamp = {
                name: float(
                    (
                        torch.abs(prediction[:, channel])
                        > processor.preprocessor.gamma
                        * processor.preprocessor.inverse_clamp_fraction
                    )
                    .float()
                    .mean()
                    .cpu()
                )
                for channel, name in enumerate(CHANNELS)
            }
            reference = fields["raw_physical_target"] if has_gt else initial_device
            record: dict[str, Any] = {
                "step": step,
                "source_snapshot": source_index if has_gt else None,
                "target_snapshot": target_index if has_gt else None,
                "ground_truth_available": has_gt,
                "finite": step_finite,
                "rho_press_positive": step_positive,
                "evaluation_bound_clamp_fraction": decoded.clamp_fraction["combined"],
                "model_clamp_fraction_by_channel": model_clamp,
                "prediction_global_norm": prediction_norm,
                "predicted_residual_norm": (
                    None
                    if predicted_residual is None
                    else float(global_state_norm(predicted_residual)[0].cpu())
                ),
                "normalized_state_change_norm": float(
                    global_state_norm(prediction - current_normalized)[0].cpu()
                ),
                "dissipation_gate": float(dissipative.gate[0].cpu()),
                "above_rin": prediction_norm > dissipation.rin,
                "above_rout": prediction_norm > dissipation.rout,
                "physical_range": {
                    name: {
                        "minimum": float(physical[:, channel].min().cpu()),
                        "maximum": float(physical[:, channel].max().cpu()),
                    }
                    for channel, name in enumerate(CHANNELS)
                },
                "normalized_range": {
                    name: {
                        "minimum": float(prediction[:, channel].min().cpu()),
                        "maximum": float(prediction[:, channel].max().cpu()),
                    }
                    for channel, name in enumerate(CHANNELS)
                },
                "artifacts": artifact_diagnostics(
                    physical,
                    reference,
                    fields["raw_physical_target"] if has_gt else initial_device,
                    reference_kind="ground_truth" if has_gt else "initial_snapshot_91",
                ),
            }
            if has_gt:
                if predicted_residual is not None:
                    true_residual = fields["normalized_target"] - current_normalized
                    pred64 = predicted_residual.to(torch.float64)
                    true64 = true_residual.to(torch.float64)
                    pred_norm = torch.linalg.vector_norm(pred64)
                    true_norm = torch.linalg.vector_norm(true64)
                    record["residual_skill"] = {
                        "predicted_residual_norm": float(pred_norm.cpu()),
                        "true_residual_norm": float(true_norm.cpu()),
                        "norm_ratio": float(
                            (pred_norm / true_norm.clamp_min(1.0e-30)).cpu()
                        ),
                        "cosine": float(
                            (
                                torch.sum(pred64 * true64)
                                / (pred_norm * true_norm).clamp_min(1.0e-30)
                            ).cpu()
                        ),
                        "sign_agreement": float(
                            (torch.signbit(pred64) == torch.signbit(true64))
                            .to(torch.float64)
                            .mean()
                            .cpu()
                        ),
                    }
                metric = PaperMetricAccumulator()
                metric.update(
                    reference_states(
                        fields=fields,
                        normalized_prediction=prediction,
                        physical_prediction=physical,
                        processor=processor,
                    ),
                    channel_axis=1,
                )
                persistence = PaperMetricAccumulator()
                persistence.update(
                    reference_states(
                        fields=fields,
                        normalized_prediction=initial_normalized,
                        physical_prediction=initial_physical_oracle,
                        processor=processor,
                    ),
                    channel_axis=1,
                )
                clamp = ClampMaskAccumulator(
                    gamma=processor.preprocessor.gamma,
                    inverse_clamp_fraction=processor.preprocessor.inverse_clamp_fraction,
                )
                clamp.update(
                    normalized_target=fields["normalized_target"],
                    normalized_prediction=prediction,
                    channel_axis=1,
                )
                record["oracle_aware"] = metric.finalize()
                record["persistence"] = persistence.finalize()
                record["saturation"] = clamp.finalize()
            else:
                record["model_only_saturation"] = (
                    "omitted_without_ground_truth_by_protocol"
                )
            if step in SELECTED_STEPS:
                statistics = physical_state_statistics(physical, r)
                spectral = total_variation_and_high_k(physical)
                radial_drift = {}
                for name in CHANNELS:
                    current_profile = np.asarray(
                        statistics["channels"][name]["radial_mean"]
                    )
                    initial_profile = np.asarray(
                        initial_stats["channels"][name]["radial_mean"]
                    )
                    radial_drift[name] = float(
                        np.linalg.norm(current_profile - initial_profile)
                        / max(np.linalg.norm(initial_profile), 1e-30)
                    )
                record["statistics"] = statistics
                record["spectral"] = spectral
                record["radial_profile_drift_from_snapshot_91"] = radial_drift
                selected_states[step] = physical.detach().cpu()
            (gt_records if has_gt else no_gt_records).append(record)
            current = physical.detach()
            next_input = processor.encode_rollout_input(current)
            current_normalized = next_input["normalized_input"]
    transform_delta = {
        key: processor.transform_counts[key] - transform_before[key]
        for key in transform_before
    }
    gt = {
        "schema_version": (
            "paper-stage-r-gt-rollout-v1"
            if processor.config.stage_r
            else "paper-stage-o-gt-rollout-v1"
            if processor.config.stage_o
            else "paper-stage-k-gt-rollout-v1"
            if processor.config.stage_k
            else "paper-stage-g-gt-rollout-v1"
        ),
        "initial_snapshot": 91,
        "transitions": 19,
        "teacher_forcing": False,
        "physical_state_rollout": True,
        "selected_steps": [1, 3, 5, 10, 19],
        "finite": all(record["finite"] for record in gt_records),
        "rho_press_positive": all(
            record["rho_press_positive"] for record in gt_records
        ),
        "records": gt_records,
    }
    no_gt = {
        "schema_version": (
            "paper-stage-r-no-gt-rollout-v1"
            if processor.config.stage_r
            else "paper-stage-o-no-gt-rollout-v1"
            if processor.config.stage_o
            else "paper-stage-k-no-gt-rollout-v1"
            if processor.config.stage_k
            else "paper-stage-g-no-gt-rollout-v1"
        ),
        "initial_snapshot": 91,
        "steps": 100,
        "ground_truth_ends_after_step": 19,
        "selected_steps": list(SELECTED_STEPS),
        "finite": all(record["finite"] for record in gt_records + no_gt_records),
        "rho_press_positive": all(
            record["rho_press_positive"] for record in gt_records + no_gt_records
        ),
        "no_gt_error_reported_after_step_19": True,
        "transform_count_delta": transform_delta,
        "encode_once_per_step": transform_delta["input_encode"] == 100,
        "decode_once_per_step": transform_delta["prediction_decode"] == 100,
        "target_oracle_only_with_ground_truth": (
            transform_delta["target_encode"] == 19
            and transform_delta["oracle_decode"] == 19
        ),
        "temporal_channel_means": temporal_series_statistics(temporal_means),
        "temporal_channel_stds": temporal_series_statistics(temporal_stds),
        "records": no_gt_records,
    }
    return gt, no_gt, selected_states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument(
        "--run-manifest",
        type=Path,
        default=Path("outputs/paper_reduced100/stage_g/run_manifest.json"),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.project_root.resolve()
    if not torch.cuda.is_available():
        raise RuntimeError("Controlled evaluation requires CUDA; refusing CPU fallback")
    device = torch.device("cuda:0")
    config_path = args.config if args.config.is_absolute() else root / args.config
    experiment_dir = (
        args.experiment_dir
        if args.experiment_dir.is_absolute()
        else root / args.experiment_dir
    )
    manifest_path = (
        args.run_manifest
        if args.run_manifest.is_absolute()
        else root / args.run_manifest
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    shared_hash, order_hash = manifest_pairing_hashes(manifest)
    config = load_paper_experiment_config(config_path, project_root=root)
    config_checksum = sha256_file(config_path)

    protocol = PaperReduced100Protocol.from_yaml(
        root / "configs/data/paper_reduced100.yaml", project_root=root
    )
    datasets = protocol.make_datasets()
    validation_dataset = datasets["validation"]
    validation_loader = DataLoader(
        GRMHDNextStepDataset(validation_dataset),
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    expected = expected_artifact_arguments(config)
    artifacts = config.values["provenance"]["artifacts"]
    envelope = PaperResidualEnvelope.load(
        config.resolve_path(artifacts["envelope"]["path"]), **expected
    )
    dissipation = PaperDissipativeReference.load(
        config.resolve_path(artifacts["dissipation"]["path"]), **expected
    )
    with h5py.File(config.resolve_path(config.values["protocol"]["dataset"]), "r") as handle:
        r = np.asarray(handle["coords/r"][...], dtype=np.float64)

    summaries = {}
    loaded_models = {}
    for name, checkpoint_name in (
        ("best_validation_l2", "best_validation_l2"),
        ("last", "last"),
    ):
        loaded, processor, reload_summary = load_strict_checkpoint(
            checkpoint_dir=experiment_dir / checkpoint_name,
            config=config,
            config_checksum=config_checksum,
            device=device,
            shared_hash=shared_hash,
            order_hash=order_hash,
        )
        validation = evaluate_validation(
            model=loaded.model,
            processor=processor,
            loader=validation_loader,
            envelope=envelope,
            dissipation=dissipation,
        )
        if validation["probe_prediction_sha256"] != loaded.metadata[
            "validation_probe_sha256"
        ]:
            raise ValueError(f"{name} deterministic validation probe changed")
        saved_epoch = json.loads(
            (
                experiment_dir
                / f"validation_epoch_{loaded.epoch:03d}.json"
            ).read_text(encoding="utf-8")
        )
        recomputed = validation["model"]["metrics"]["E_norm"]
        saved = saved_epoch["validation"]["oracle_aware"]["metrics"]["E_norm"]
        parity = {
            "arithmetic_average": bool(
                np.isclose(
                    recomputed["arithmetic_average"],
                    saved["arithmetic_average"],
                    rtol=1e-7,
                    atol=1e-9,
                )
            ),
            "global_relative_l2": bool(
                np.isclose(
                    recomputed["global_relative_l2"],
                    saved["global_relative_l2"],
                    rtol=1e-7,
                    atol=1e-9,
                )
            ),
        }
        if not all(parity.values()):
            raise ValueError(f"{name} recomputed validation metrics changed")
        summary = {
            "schema_version": (
                "paper-stage-r-checkpoint-evaluation-v1"
                if config.stage_r
                else "paper-stage-o-checkpoint-evaluation-v1"
                if config.stage_o
                else "paper-stage-k-checkpoint-evaluation-v1"
                if config.stage_k
                else "paper-stage-i-checkpoint-evaluation-v1"
                if config.diagnostic_h1_mode is not None
                else "paper-stage-g-checkpoint-evaluation-v1"
            ),
            "checkpoint": name,
            "reload": reload_summary,
            "validation": validation,
            "training_time_metric_parity": parity,
        }
        summaries[name] = summary
        loaded_models[name] = (loaded.model, processor)
        (experiment_dir / f"evaluation_{name}.json").write_text(
            json.dumps(json_value(summary), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    best_model, best_processor = loaded_models["best_validation_l2"]
    gt, no_gt, selected_states = evaluate_rollout(
        model=best_model,
        processor=best_processor,
        validation_dataset=validation_dataset,
        r=r,
        dissipation=dissipation,
    )
    (experiment_dir / "gt_rollout.json").write_text(
        json.dumps(json_value(gt), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (experiment_dir / "no_gt_rollout.json").write_text(
        json.dumps(json_value(no_gt), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(
        experiment_dir / "gt_rollout.csv",
        [compact_rollout_row(record) for record in gt["records"]],
    )
    write_csv(
        experiment_dir / "no_gt_rollout.csv",
        [compact_rollout_row(record) for record in no_gt["records"]],
    )
    torch.save(
        {
            "schema_version": (
                "paper-stage-r-selected-states-v1"
                if config.stage_r
                else "paper-stage-o-selected-states-v1"
                if config.stage_o
                else "paper-stage-k-selected-states-v1"
                if config.stage_k
                else "paper-stage-i-selected-states-v1"
                if config.diagnostic_h1_mode is not None
                else "paper-stage-g-selected-states-v1"
            ),
            "mode": config.mode,
            "diagnostic_h1_mode": config.diagnostic_h1_mode,
            "checkpoint_epoch": summaries["best_validation_l2"]["reload"]["epoch"],
            "selected_steps": selected_states,
        },
        experiment_dir / "selected_states.pt",
    )
    result = {
        "schema_version": (
            "paper-stage-r-evaluation-summary-v1"
            if config.stage_r
            else "paper-stage-o-evaluation-summary-v1"
            if config.stage_o
            else "paper-stage-k-evaluation-summary-v1"
            if config.stage_k
            else "paper-stage-i-evaluation-summary-v1"
            if config.diagnostic_h1_mode is not None
            else "paper-stage-g-evaluation-summary-v1"
        ),
        "mode": config.mode,
        "diagnostic_h1_mode": config.diagnostic_h1_mode,
        "best": summaries["best_validation_l2"],
        "last": summaries["last"],
        "gt_rollout": {
            "finite": gt["finite"],
            "rho_press_positive": gt["rho_press_positive"],
            "selected": {
                str(step): compact_rollout_row(gt["records"][step - 1])
                for step in (1, 3, 5, 10, 19)
            },
        },
        "no_gt_rollout": {
            "finite": no_gt["finite"],
            "rho_press_positive": no_gt["rho_press_positive"],
            "selected": {
                str(step): compact_rollout_row(no_gt["records"][step - 20])
                for step in (50, 100)
            },
            "temporal_channel_means": no_gt["temporal_channel_means"],
            "temporal_channel_stds": no_gt["temporal_channel_stds"],
        },
    }
    (experiment_dir / "evaluation_summary.json").write_text(
        json.dumps(json_value(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(json_value(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
