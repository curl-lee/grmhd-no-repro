#!/usr/bin/env python
"""Train a small FNO/LocalNO standalone GRMHD one-step surrogate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from grmhd import CHANNELS
from grmhd.checkpoint_metrics import (
    CHECKPOINT_METRIC_FILENAMES,
    COMPOSITE_STABILITY_FORMULA,
    composite_stability_metric,
    relative_l2,
)
from grmhd.dataset import GRMHDPairedDataset, make_temporal_datasets
from grmhd.losses import WeightedGRMHDLoss
from grmhd.models import (
    PersistenceBaseline,
    apply_prediction_mode,
    build_model,
    parameters_without_grad,
    trainable_parameter_count,
    zero_initialize_residual_head,
)
from grmhd.normalizer import GRMHDNormalizer, validate_stats_bundle
from grmhd.priors import QuantileBounds, ResidualEnvelope
from grmhd.residual import ResidualScaleStats, validate_residual_wrapper_metadata
from grmhd.shells import radial_shells_tensor


def nested_set(config: dict[str, Any], keys: tuple[str, ...], value: Any) -> None:
    target = config
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: Path) -> dict[str, Any]:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    base_specification = values.pop("base_config", None)
    if base_specification is None:
        return values
    base_path = Path(base_specification)
    if not base_path.is_absolute() and not base_path.exists():
        base_path = path.parent / base_path
    return deep_merge(load_config(base_path.resolve()), values)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def learning_rate_for_epoch(
    epoch: int, total_epochs: int, warmup_epochs: int, base_lr: float, min_lr: float
) -> float:
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return base_lr * (epoch + 1) / warmup_epochs
    decay_epochs = max(total_epochs - warmup_epochs, 1)
    progress = min(max((epoch - warmup_epochs) / decay_epochs, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


def prepare_batch(
    batch: dict[str, Any],
    normalizer: GRMHDNormalizer,
    shells: torch.Tensor | None,
    downsample: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw_x = batch["x"].to(device, non_blocking=True)
    raw_y = batch["y"].to(device, non_blocking=True)
    x = normalizer.encode_tensor(raw_x, channel_axis=1)
    y = normalizer.encode_tensor(raw_y, channel_axis=1)
    if downsample > 1:
        x = x[..., ::downsample, ::downsample, ::downsample]
        y = y[..., ::downsample, ::downsample, ::downsample]
    if shells is not None:
        x = torch.cat([x, shells.unsqueeze(0).expand(x.shape[0], -1, -1, -1, -1)], dim=1)
    return x, y


def run_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    loss_function: WeightedGRMHDLoss,
    normalizer: GRMHDNormalizer,
    shells: torch.Tensor | None,
    downsample: int,
    device: torch.device,
    training: bool,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    mixed_precision: bool,
    accumulation: int,
    gradient_clip: float,
    max_batches: int | None,
    predict_residual: bool,
    bounded_residual: bool,
    residual_scale: tuple[float, ...] | None,
) -> tuple[dict[str, float], int]:
    model.train(training)
    totals: dict[str, float] = {}
    batches = 0
    gradient_norm_total = 0.0
    gradient_steps = 0
    if training:
        assert optimizer is not None
        optimizer.zero_grad(set_to_none=True)
    context = torch.enable_grad if training else torch.no_grad
    with context():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            x, y = prepare_batch(batch, normalizer, shells, downsample, device)
            with torch.amp.autocast(
                device_type=device.type,
                enabled=mixed_precision,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
            ):
                prediction = apply_prediction_mode(
                    model(x),
                    x,
                    predict_residual=predict_residual,
                    bounded_residual=bounded_residual,
                    residual_scale=residual_scale,
                )
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError("Model prediction contains NaN/Inf")
                components = loss_function.components(prediction, y, input_state=x[:, :8])
                scaled_loss = components["loss"] / accumulation
            if training:
                scaler.scale(scaled_loss).backward()
                missing_grad = parameters_without_grad(model)
                if missing_grad:
                    raise RuntimeError(f"Trainable parameters without gradients: {missing_grad}")
                should_step = (batch_index + 1) % accumulation == 0
                is_last = batch_index + 1 == len(loader) or (
                    max_batches is not None and batch_index + 1 == max_batches
                )
                if should_step or is_last:
                    scaler.unscale_(optimizer)
                    gradient_norm = clip_grad_norm_(model.parameters(), gradient_clip)
                    gradient_norm_total += float(gradient_norm.detach().cpu())
                    gradient_steps += 1
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            for name, value in components.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().cpu())
            with torch.no_grad():
                decoded = normalizer.decode_tensor(prediction.detach(), channel_axis=1)
            totals["prediction_nonfinite"] = totals.get("prediction_nonfinite", 0.0) + float(
                (~torch.isfinite(decoded)).sum().item()
            )
            totals["rho_positivity_violations"] = totals.get(
                "rho_positivity_violations", 0.0
            ) + float((decoded[:, 3] <= 0).sum().item())
            totals["press_positivity_violations"] = totals.get(
                "press_positivity_violations", 0.0
            ) + float((decoded[:, 4] <= 0).sum().item())
            batches += 1
    if batches == 0:
        raise RuntimeError("DataLoader produced no batches")
    metrics = {name: value / batches for name, value in totals.items()}
    metrics["gradient_norm"] = gradient_norm_total / gradient_steps if gradient_steps else 0.0
    return metrics, batches


def prediction_probe_sha256(
    model: torch.nn.Module,
    dataset: GRMHDPairedDataset,
    normalizer: GRMHDNormalizer,
    shells: torch.Tensor | None,
    downsample: int,
    device: torch.device,
    predict_residual: bool,
    bounded_residual: bool,
    residual_scale: tuple[float, ...] | None,
) -> str:
    sample = dataset[0]
    batch = {"x": sample["x"].unsqueeze(0), "y": sample["y"].unsqueeze(0)}
    was_training = model.training
    model.eval()
    with torch.no_grad():
        x, _ = prepare_batch(batch, normalizer, shells, downsample, device)
        prediction = apply_prediction_mode(
            model(x),
            x,
            predict_residual=predict_residual,
            bounded_residual=bounded_residual,
            residual_scale=residual_scale,
        )
    model.train(was_training)
    values = prediction.detach().cpu().contiguous().numpy()
    return hashlib.sha256(values.tobytes()).hexdigest()


def _accumulate_relative_l2(
    prediction: torch.Tensor,
    truth: torch.Tensor,
    error_sums: torch.Tensor,
    truth_sums: torch.Tensor,
) -> None:
    """Accumulate per-channel sums in float64 without retaining model tensors."""
    difference = (prediction.detach() - truth.detach()).double()
    reference = truth.detach().double()
    reduction_axes = (0,) + tuple(range(2, difference.ndim))
    error_sums += difference.square().sum(dim=reduction_axes).cpu()
    truth_sums += reference.square().sum(dim=reduction_axes).cpu()


def _relative_l2_metrics(
    error_sums: torch.Tensor, truth_sums: torch.Tensor
) -> tuple[float, list[float]]:
    per_channel = [
        relative_l2(float(error_sums[index]), float(truth_sums[index]))
        for index in range(len(CHANNELS))
    ]
    return relative_l2(float(error_sums.sum()), float(truth_sums.sum())), per_channel


def validation_stability_metrics(
    model: torch.nn.Module,
    dataset: GRMHDPairedDataset,
    normalizer: GRMHDNormalizer,
    shells: torch.Tensor | None,
    downsample: int,
    device: torch.device,
    predict_residual: bool,
    bounded_residual: bool,
    residual_scale: tuple[float, ...] | None,
    *,
    rollout_steps: int = 3,
) -> dict[str, Any]:
    """Evaluate aggregate teacher-forced and rolling-origin validation errors.

    The one-step metrics include every validation pair. The rollout metric uses
    every validation origin with ``rollout_steps`` in-split targets and scores
    the endpoint. No validation statistics are fit here.
    """
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    if dataset.stride != 1:
        raise ValueError("short-rollout checkpoint metrics currently require stride=1")

    was_training = model.training
    model.eval()
    one_norm_error = torch.zeros(len(CHANNELS), dtype=torch.float64)
    one_norm_truth = torch.zeros(len(CHANNELS), dtype=torch.float64)
    one_decoded_error = torch.zeros(len(CHANNELS), dtype=torch.float64)
    one_decoded_truth = torch.zeros(len(CHANNELS), dtype=torch.float64)
    rollout_error = torch.zeros(len(CHANNELS), dtype=torch.float64)
    rollout_truth = torch.zeros(len(CHANNELS), dtype=torch.float64)
    magnetic_range_excess: list[float] = []
    prediction_nonfinite = 0
    rho_positivity_violations = 0
    press_positivity_violations = 0

    def encode_snapshot(snapshot: torch.Tensor) -> torch.Tensor:
        encoded = normalizer.encode_tensor(
            snapshot.unsqueeze(0).to(device, non_blocking=True), channel_axis=1
        )
        return encoded[..., ::downsample, ::downsample, ::downsample]

    def operator_input(state: torch.Tensor) -> torch.Tensor:
        if shells is None:
            return state
        return torch.cat(
            [state, shells.unsqueeze(0).expand(state.shape[0], -1, -1, -1, -1)],
            dim=1,
        )

    with torch.no_grad():
        for item in range(len(dataset)):
            sample = dataset[item]
            input_state = encode_snapshot(sample["x"])
            truth = encode_snapshot(sample["y"])
            model_input = operator_input(input_state)
            prediction = apply_prediction_mode(
                model(model_input),
                model_input,
                predict_residual=predict_residual,
                bounded_residual=bounded_residual,
                residual_scale=residual_scale,
            )
            _accumulate_relative_l2(prediction, truth, one_norm_error, one_norm_truth)
            decoded_prediction = normalizer.decode_tensor(prediction, channel_axis=1)
            decoded_truth = sample["y"].unsqueeze(0).to(device, non_blocking=True)[
                ..., ::downsample, ::downsample, ::downsample
            ]
            _accumulate_relative_l2(
                decoded_prediction, decoded_truth, one_decoded_error, one_decoded_truth
            )
            prediction_nonfinite += int((~torch.isfinite(decoded_prediction)).sum().item())
            rho_positivity_violations += int((decoded_prediction[:, 3] <= 0).sum().item())
            press_positivity_violations += int((decoded_prediction[:, 4] <= 0).sum().item())

        last_origin = dataset.split.stop - rollout_steps
        rollout_origins = tuple(range(dataset.split.start, last_origin))
        for origin in rollout_origins:
            state = encode_snapshot(dataset.load_snapshot(origin))
            for _ in range(rollout_steps):
                model_input = operator_input(state)
                state = apply_prediction_mode(
                    model(model_input),
                    model_input,
                    predict_residual=predict_residual,
                    bounded_residual=bounded_residual,
                    residual_scale=residual_scale,
                )
            decoded_prediction = normalizer.decode_tensor(state, channel_axis=1)
            decoded_truth = dataset.load_snapshot(origin + rollout_steps).unsqueeze(0).to(
                device, non_blocking=True
            )[..., ::downsample, ::downsample, ::downsample]
            _accumulate_relative_l2(
                decoded_prediction, decoded_truth, rollout_error, rollout_truth
            )
            prediction_nonfinite += int((~torch.isfinite(decoded_prediction)).sum().item())
            rho_positivity_violations += int((decoded_prediction[:, 3] <= 0).sum().item())
            press_positivity_violations += int((decoded_prediction[:, 4] <= 0).sum().item())
            for channel in range(3):
                predicted_range = decoded_prediction[:, channel].max() - decoded_prediction[
                    :, channel
                ].min()
                truth_range = decoded_truth[:, channel].max() - decoded_truth[:, channel].min()
                denominator = max(float(truth_range), torch.finfo(torch.float64).tiny)
                relative_excess = max(float(predicted_range - truth_range) / denominator, 0.0)
                magnetic_range_excess.append(relative_excess)

    model.train(was_training)
    normalized_global, normalized_per_channel = _relative_l2_metrics(
        one_norm_error, one_norm_truth
    )
    decoded_one_global, decoded_one_per_channel = _relative_l2_metrics(
        one_decoded_error, one_decoded_truth
    )
    decoded_three_global, decoded_three_per_channel = _relative_l2_metrics(
        rollout_error, rollout_truth
    )
    magnetic_range_violation = float(np.mean(magnetic_range_excess))
    return {
        "normalized_one_step_global": normalized_global,
        "normalized_one_step_per_channel": normalized_per_channel,
        "decoded_one_step_global": decoded_one_global,
        "decoded_one_step_per_channel": decoded_one_per_channel,
        "decoded_three_step_global": decoded_three_global,
        "decoded_three_step_per_channel": decoded_three_per_channel,
        "magnetic_range_violation": magnetic_range_violation,
        "composite_stability": composite_stability_metric(
            decoded_one_global, decoded_three_global, magnetic_range_violation
        ),
        "rollout_steps": rollout_steps,
        "one_step_transition_count": len(dataset),
        "rolling_origin_count": len(rollout_origins),
        "prediction_nonfinite": prediction_nonfinite,
        "rho_positivity_violations": rho_positivity_violations,
        "press_positivity_violations": press_positivity_violations,
    }


def checkpoint_payload(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    metrics: dict[str, Any],
    normalizer: GRMHDNormalizer,
    shell_metadata: dict[str, Any] | None,
    residual_wrapper_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "metrics": metrics,
        "normalizer": normalizer.as_dict(),
        "shell_metadata": shell_metadata,
        "residual_wrapper": residual_wrapper_metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/grmhd_localno.yaml"))
    parser.add_argument("--experiment-name")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--device")
    parser.add_argument("--model-type", choices=["fno", "localno_diff"])
    parser.add_argument("--downsample", type=int)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run two epochs with at most two training and one validation batch per epoch.",
    )
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    if args.experiment_name:
        config["experiment_name"] = args.experiment_name
    if args.epochs is not None:
        nested_set(config, ("training", "epochs"), args.epochs)
    if args.max_train_batches is not None:
        nested_set(config, ("training", "max_train_batches"), args.max_train_batches)
    if args.max_val_batches is not None:
        nested_set(config, ("training", "max_val_batches"), args.max_val_batches)
    if args.device:
        nested_set(config, ("training", "device"), args.device)
    if args.model_type:
        nested_set(config, ("model", "type"), args.model_type)
    if args.downsample is not None:
        nested_set(config, ("data", "downsample"), args.downsample)
    if args.smoke:
        nested_set(config, ("training", "epochs"), 2)
        nested_set(config, ("training", "max_train_batches"), 2)
        nested_set(config, ("training", "max_val_batches"), 1)
        nested_set(config, ("training", "gradient_accumulation"), 1)

    seed = int(config["seed"])
    seed_everything(seed)
    requested_device = str(config["training"]["device"])
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(requested_device)
    if device.type == "cuda":
        torch.cuda.set_device(0 if device.index is None else device.index)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    data_config = config["data"]
    time_range = (data_config.get("train_time_min"), data_config.get("train_time_max"))
    datasets = make_temporal_datasets(
        data_config["path"],
        stride=int(data_config["stride"]),
        train_fraction=float(data_config["train_fraction"]),
        val_fraction=float(data_config["val_fraction"]),
        snapshot_start=int(data_config.get("snapshot_start", 0)),
        snapshot_end=(
            None if data_config.get("snapshot_end") is None else int(data_config["snapshot_end"])
        ),
        train_snapshot_count=(
            None
            if data_config.get("train_snapshot_count") is None
            else int(data_config["train_snapshot_count"])
        ),
        val_snapshot_count=(
            None
            if data_config.get("val_snapshot_count") is None
            else int(data_config["val_snapshot_count"])
        ),
        test_snapshot_count=(
            None
            if data_config.get("test_snapshot_count") is None
            else int(data_config["test_snapshot_count"])
        ),
        train_time_range=time_range,
    )
    if len(datasets["train"]) == 0 or len(datasets["val"]) == 0:
        raise RuntimeError("Training and validation splits must each contain at least one pair")
    normalizer_stats_path = (
        Path(data_config["normalizer_stats_path"])
        if data_config.get("normalizer_stats_path")
        else None
    )
    if normalizer_stats_path is not None and normalizer_stats_path.exists():
        stats_bundle_validation = validate_stats_bundle(
            normalizer_stats_path.parent,
            data_config["path"],
            expected_training_indices=datasets["train"].owned_snapshot_indices,
            expected_window_name=str(data_config["window_name"]),
        )
        normalizer = GRMHDNormalizer.load(
            normalizer_stats_path,
            h5_path=data_config["path"],
            expected_training_indices=datasets["train"].owned_snapshot_indices,
        )
    else:
        stats_bundle_validation = None
        normalizer = GRMHDNormalizer.fit(
            datasets["train"],
            max_samples_per_channel=int(data_config["normalizer_samples_per_channel"]),
            seed=seed,
            radial_baseline=bool(config.get("priors", {}).get("radial_baseline", {}).get("enabled", False)),
        )
        if normalizer_stats_path is not None:
            normalizer.save(normalizer_stats_path)
    priors_config = config.get("priors", {})
    prior_stats: dict[str, Any] = {}
    quantile_bounds = None
    quantile_config = priors_config.get("quantile_bounds", {})
    if quantile_config.get("enabled", False):
        quantile_bounds = QuantileBounds.fit(
            datasets["train"],
            normalizer,
            quantiles=tuple(float(value) for value in quantile_config["quantiles"]),
            max_samples_per_channel=int(quantile_config["samples_per_channel"]),
            seed=seed,
        )
        prior_stats["quantile_bounds"] = quantile_bounds.as_dict()
    residual_envelope = None
    residual_config = priors_config.get("residual_envelope", {})
    if residual_config.get("enabled", False):
        residual_envelope = ResidualEnvelope.fit(
            datasets["train"],
            normalizer,
            quantiles=tuple(float(value) for value in residual_config["quantiles"]),
            max_samples_per_channel=int(residual_config["samples_per_channel"]),
            seed=seed,
        )
        prior_stats["residual_envelope"] = residual_envelope.as_dict()

    downsample = int(data_config.get("downsample", 1))
    source_shape = datasets["train"].snapshot_shape[1:]
    if any(size % downsample for size in source_shape):
        raise ValueError(f"downsample={downsample} does not divide source shape {source_shape}")
    actual_shape = tuple(size // downsample for size in source_shape)
    shells: torch.Tensor | None = None
    shell_metadata = None
    if data_config.get("with_shells", True):
        r = datasets["train"].coords["r"][::downsample]
        shells, metadata = radial_shells_tensor(
            r,
            actual_shape[0],
            actual_shape[1],
            n_shells=int(data_config.get("n_shells", 8)),
            device=device,
        )
        shell_metadata = metadata.as_dict()
    in_channels = 8 + (shells.shape[0] if shells is not None else 0)

    generator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=int(config["training"]["batch_size"]),
            shuffle=True,
            generator=generator,
            num_workers=int(data_config.get("num_workers", 0)),
            pin_memory=device.type == "cuda",
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=int(config["training"]["batch_size"]),
            shuffle=False,
            num_workers=int(data_config.get("num_workers", 0)),
            pin_memory=device.type == "cuda",
        ),
    }

    model_config = config["model"]
    predict_residual = bool(model_config.get("predict_residual", False))
    bounded_residual = bool(model_config.setdefault("bounded_residual", False))
    zero_init_residual_head = bool(
        model_config.setdefault("zero_init_residual_head", False)
    )
    if (bounded_residual or zero_init_residual_head) and not predict_residual:
        raise ValueError("bounded/zero-init residual options require predict_residual=true")
    residual_scale_stats: ResidualScaleStats | None = None
    residual_scale: tuple[float, ...] | None = None
    residual_wrapper_config: dict[str, Any] | None = None
    if bounded_residual:
        residual_scale_source = str(
            model_config.setdefault("residual_scale_source", "train_abs_quantile")
        )
        if residual_scale_source != "train_abs_quantile":
            raise ValueError("Only residual_scale_source=train_abs_quantile is supported")
        residual_scale_quantile = float(
            model_config.setdefault("residual_scale_quantile", 0.999)
        )
        residual_scale_multiplier = float(
            model_config.setdefault("residual_scale_multiplier", 1.0)
        )
        residual_scale_stats_path = model_config.get("residual_scale_stats_path")
        if residual_scale_stats_path and Path(residual_scale_stats_path).exists():
            residual_scale_stats = ResidualScaleStats.load(
                residual_scale_stats_path,
                h5_path=data_config["path"],
                expected_training_indices=datasets["train"].owned_snapshot_indices,
                expected_quantile=residual_scale_quantile,
                expected_multiplier=residual_scale_multiplier,
            )
        else:
            residual_scale_stats = ResidualScaleStats.fit(
                datasets["train"],
                normalizer,
                quantile=residual_scale_quantile,
                multiplier=residual_scale_multiplier,
                min_alpha=float(model_config.setdefault("residual_scale_min", 1e-6)),
                max_samples_per_channel=int(
                    model_config.setdefault("residual_scale_samples_per_channel", 200_000)
                ),
                seed=seed,
            )
        residual_scale = residual_scale_stats.alpha
        residual_wrapper_config = {
            "predict_residual": True,
            "zero_init_residual_head": zero_init_residual_head,
            "bounded_residual": True,
            "residual_scale_source": residual_scale_source,
            "residual_scale_quantile": residual_scale_quantile,
            "residual_scale_multiplier": residual_scale_multiplier,
        }
    model_kwargs: dict[str, Any] = {}
    if str(model_config["type"]).startswith("localno"):
        model_kwargs["conv_padding_mode"] = model_config.get("conv_padding_mode", "zeros")
    model = build_model(
        str(model_config["type"]),
        in_channels=in_channels,
        out_channels=8,
        default_in_shape=actual_shape,
        n_modes=tuple(int(value) for value in model_config["n_modes"]),
        hidden_channels=int(model_config["hidden_channels"]),
        n_layers=int(model_config["n_layers"]),
        positional_embedding=model_config.get("positional_embedding"),
        **model_kwargs,
    ).to(device)
    zero_initialized_head = (
        zero_initialize_residual_head(model) if zero_init_residual_head else None
    )
    parameter_count = trainable_parameter_count(model)
    if parameter_count == 0:
        raise ValueError("Selected model has no trainable parameters")
    initial_persistence_max_abs_difference: float | None = None
    if zero_init_residual_head:
        sample = datasets["val"][0]
        batch = {"x": sample["x"].unsqueeze(0), "y": sample["y"].unsqueeze(0)}
        model.eval()
        with torch.no_grad():
            initial_input, _ = prepare_batch(
                batch, normalizer, shells, downsample, device
            )
            initial_prediction = apply_prediction_mode(
                model(initial_input),
                initial_input,
                predict_residual=True,
                bounded_residual=bounded_residual,
                residual_scale=residual_scale,
            )
            initial_persistence_max_abs_difference = float(
                torch.max(torch.abs(initial_prediction - initial_input[:, :8])).cpu()
            )
    if initial_persistence_max_abs_difference != 0.0:
            raise RuntimeError(
                "Zero-initialized residual head is not exactly persistence: "
                f"max difference={initial_persistence_max_abs_difference}"
            )

    persistence_metrics = validation_stability_metrics(
        PersistenceBaseline().to(device),
        datasets["val"],
        normalizer,
        shells,
        downsample,
        device,
        False,
        False,
        None,
        rollout_steps=3,
    )

    training_config = config["training"]
    checkpoint_selection_metric = str(
        training_config.setdefault("checkpoint_selection_metric", "normalized_total")
    )
    if checkpoint_selection_metric not in CHECKPOINT_METRIC_FILENAMES:
        raise ValueError(
            "training.checkpoint_selection_metric must be one of "
            f"{sorted(CHECKPOINT_METRIC_FILENAMES)}, found {checkpoint_selection_metric!r}"
        )
    save_epoch_checkpoints = bool(training_config.setdefault("save_epoch_checkpoints", False))
    stability_rollout_steps = int(training_config.setdefault("stability_rollout_steps", 3))
    if stability_rollout_steps != 3:
        raise ValueError("Round-two checkpoint alignment requires stability_rollout_steps=3")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    loss_function = WeightedGRMHDLoss(
        **config["loss"],
        quantile_bounds=quantile_bounds,
        lambda_quantile_bounds=float(quantile_config.get("lambda", 0.0)),
        residual_envelope=residual_envelope,
        lambda_residual_envelope=float(residual_config.get("lambda", 0.0)),
        lambda_velocity_roi=float(priors_config.get("velocity_roi", {}).get("lambda", 0.0))
        if priors_config.get("velocity_roi", {}).get("enabled", False)
        else 0.0,
        velocity_roi_quantile=float(
            priors_config.get("velocity_roi", {}).get("quantile", 0.8)
        ),
        lambda_dissipative=float(priors_config.get("dissipative", {}).get("lambda", 0.0))
        if priors_config.get("dissipative", {}).get("enabled", False)
        else 0.0,
    )
    mixed_precision = bool(training_config["mixed_precision"])
    scaler = torch.amp.GradScaler(device.type, enabled=mixed_precision)

    output_dir = Path(config["logging"]["root"]) / str(config["experiment_name"])
    output_dir.mkdir(parents=True, exist_ok=True)
    normalizer.save(output_dir / "normalizer_stats.npz")
    if residual_scale_stats is not None:
        residual_scale_stats.save(output_dir / "residual_scale.json")
    residual_wrapper_metadata = (
        {
            "config": residual_wrapper_config,
            "alpha": list(residual_scale_stats.alpha),
            "stats": residual_scale_stats.as_dict(),
            "zero_initialized_head": zero_initialized_head,
            "initial_persistence_max_abs_difference": (
                initial_persistence_max_abs_difference
            ),
        }
        if residual_scale_stats is not None
        else None
    )
    config["resolved"] = {
        "actual_spatial_shape": list(actual_shape),
        "in_channels": in_channels,
        "trainable_parameters": parameter_count,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "split_snapshot_ranges": {
            name: [dataset.split.start, dataset.split.stop] for name, dataset in datasets.items()
        },
        "split_pair_counts": {name: len(dataset) for name, dataset in datasets.items()},
        "shell_metadata": shell_metadata,
        "h1_definition": "unweighted value+first differences on phi/theta/log-r index grid; not KS metric",
        "wandb_enabled": False,
        "prior_stats": prior_stats,
        "prediction_mode": "residual" if predict_residual else "direct",
        "predict_residual": predict_residual,
        "bounded_residual": bounded_residual,
        "zero_init_residual_head": zero_init_residual_head,
        "zero_initialized_head": zero_initialized_head,
        "initial_persistence_max_abs_difference": (
            initial_persistence_max_abs_difference
        ),
        "residual_scale_stats": (
            residual_scale_stats.as_dict() if residual_scale_stats is not None else None
        ),
        "validation_persistence_reference": persistence_metrics,
        "source_hdf5_checksum": normalizer.source_hdf5_checksum,
        "normalizer_stats_checksum": (
            stats_bundle_validation["normalizer_stats_checksum"]
            if stats_bundle_validation is not None
            else None
        ),
        "stats_bundle_validation": stats_bundle_validation,
        "data_window": data_config.get("window_name"),
        "channel_order": list(CHANNELS),
        "checkpoint_selection_metric": checkpoint_selection_metric,
        "checkpoint_metric_files": CHECKPOINT_METRIC_FILENAMES,
        "checkpoint_metric_definitions": {
            "normalized_total": "validation WeightedGRMHDLoss total (val_loss)",
            "decoded_one_step": (
                "aggregate decoded global relative L2 over every validation transition"
            ),
            "decoded_three_step": (
                "aggregate decoded global relative L2 at the 3-step endpoint over every "
                "available validation rolling origin"
            ),
            "magnetic_range_violation": (
                "mean over rolling origins and Bcc1/Bcc2/Bcc3 of "
                "max((predicted_range - truth_range) / truth_range, 0) at step 3"
            ),
            "composite_stability": COMPOSITE_STABILITY_FORMULA,
        },
        "split_manifests": {name: dataset.manifest() for name, dataset in datasets.items()},
    }
    (output_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    capture_script = Path(__file__).with_name("capture_environment.py")
    subprocess.run(
        [sys.executable, str(capture_script), "--out", str(output_dir / "environment.json")],
        check=True,
        stdout=subprocess.DEVNULL,
    )

    writer = (
        SummaryWriter(output_dir / "tensorboard")
        if config["logging"].get("tensorboard", True)
        else None
    )
    csv_path = output_dir / "train_log.csv"
    fieldnames = [
        "epoch",
        "learning_rate",
        "train_loss",
        "train_weighted_l2",
        "train_h1_index_grid",
        "train_quantile_bounds_penalty",
        "train_residual_envelope_penalty",
        "train_velocity_roi_mse",
        "train_dissipative_excess_index_grid",
        "train_gradient_norm",
        "train_prediction_nonfinite",
        "train_rho_positivity_violations",
        "train_press_positivity_violations",
        "val_loss",
        "val_weighted_l2",
        "val_h1_index_grid",
        "val_quantile_bounds_penalty",
        "val_residual_envelope_penalty",
        "val_velocity_roi_mse",
        "val_dissipative_excess_index_grid",
        "val_prediction_nonfinite",
        "val_rho_positivity_violations",
        "val_press_positivity_violations",
        "val_normalized_one_step_global",
        "val_normalized_one_step_per_channel_json",
        "val_decoded_one_step_global",
        "val_decoded_one_step_per_channel_json",
        "val_decoded_three_step_global",
        "val_decoded_three_step_per_channel_json",
        "val_magnetic_range_violation",
        "val_composite_stability",
        "val_persistence_decoded_one_step_global",
        "val_persistence_decoded_three_step_global",
        "val_decoded_one_step_improvement_over_persistence_fraction",
        "val_decoded_three_step_improvement_over_persistence_fraction",
        "val_stability_prediction_nonfinite",
        "val_stability_rho_positivity_violations",
        "val_stability_press_positivity_violations",
        "val_one_step_transition_count",
        "val_rolling_origin_count",
        "epoch_seconds",
        "train_batches",
        "val_batches",
        "samples_per_second",
        "peak_cuda_memory_mib",
    ]
    stream = csv_path.open("w", newline="", encoding="utf-8")
    csv_writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
    csv_writer.writeheader()

    epochs = int(training_config["epochs"])
    best_values = {name: math.inf for name in CHECKPOINT_METRIC_FILENAMES}
    best_epochs: dict[str, int | None] = {
        name: None for name in CHECKPOINT_METRIC_FILENAMES
    }
    epochs_without_improvement = 0
    run_start = time.perf_counter()
    history: list[dict[str, Any]] = []
    epoch_checkpoint_dir = output_dir / "epoch_checkpoints"
    if save_epoch_checkpoints:
        epoch_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        for epoch in range(epochs):
            epoch_start = time.perf_counter()
            learning_rate = learning_rate_for_epoch(
                epoch,
                epochs,
                int(training_config["warmup_epochs"]),
                float(training_config["learning_rate"]),
                float(training_config["min_learning_rate"]),
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            train_metrics, train_batches = run_epoch(
                model=model,
                loader=loaders["train"],
                loss_function=loss_function,
                normalizer=normalizer,
                shells=shells,
                downsample=downsample,
                device=device,
                training=True,
                optimizer=optimizer,
                scaler=scaler,
                mixed_precision=mixed_precision,
                accumulation=int(training_config["gradient_accumulation"]),
                gradient_clip=float(training_config["gradient_clip"]),
                max_batches=training_config.get("max_train_batches"),
                predict_residual=predict_residual,
                bounded_residual=bounded_residual,
                residual_scale=residual_scale,
            )
            val_metrics, val_batches = run_epoch(
                model=model,
                loader=loaders["val"],
                loss_function=loss_function,
                normalizer=normalizer,
                shells=shells,
                downsample=downsample,
                device=device,
                training=False,
                optimizer=None,
                scaler=scaler,
                mixed_precision=mixed_precision,
                accumulation=1,
                gradient_clip=float(training_config["gradient_clip"]),
                max_batches=training_config.get("max_val_batches"),
                predict_residual=predict_residual,
                bounded_residual=bounded_residual,
                residual_scale=residual_scale,
            )
            stability_metrics = validation_stability_metrics(
                model,
                datasets["val"],
                normalizer,
                shells,
                downsample,
                device,
                predict_residual,
                bounded_residual,
                residual_scale,
                rollout_steps=stability_rollout_steps,
            )
            epoch_seconds = time.perf_counter() - epoch_start
            peak_memory = (
                torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
            )
            row = {
                "epoch": epoch + 1,
                "learning_rate": learning_rate,
                "train_loss": train_metrics["loss"],
                "train_weighted_l2": train_metrics["weighted_l2"],
                "train_h1_index_grid": train_metrics["h1_index_grid"],
                "train_quantile_bounds_penalty": train_metrics["quantile_bounds_penalty"],
                "train_residual_envelope_penalty": train_metrics["residual_envelope_penalty"],
                "train_velocity_roi_mse": train_metrics["velocity_roi_mse"],
                "train_dissipative_excess_index_grid": train_metrics[
                    "dissipative_excess_index_grid"
                ],
                "train_gradient_norm": train_metrics["gradient_norm"],
                "train_prediction_nonfinite": train_metrics["prediction_nonfinite"],
                "train_rho_positivity_violations": train_metrics[
                    "rho_positivity_violations"
                ],
                "train_press_positivity_violations": train_metrics[
                    "press_positivity_violations"
                ],
                "val_loss": val_metrics["loss"],
                "val_weighted_l2": val_metrics["weighted_l2"],
                "val_h1_index_grid": val_metrics["h1_index_grid"],
                "val_quantile_bounds_penalty": val_metrics["quantile_bounds_penalty"],
                "val_residual_envelope_penalty": val_metrics["residual_envelope_penalty"],
                "val_velocity_roi_mse": val_metrics["velocity_roi_mse"],
                "val_dissipative_excess_index_grid": val_metrics[
                    "dissipative_excess_index_grid"
                ],
                "val_prediction_nonfinite": val_metrics["prediction_nonfinite"],
                "val_rho_positivity_violations": val_metrics[
                    "rho_positivity_violations"
                ],
                "val_press_positivity_violations": val_metrics[
                    "press_positivity_violations"
                ],
                "val_normalized_one_step_global": stability_metrics[
                    "normalized_one_step_global"
                ],
                "val_normalized_one_step_per_channel_json": json.dumps(
                    dict(zip(CHANNELS, stability_metrics["normalized_one_step_per_channel"])),
                    sort_keys=True,
                ),
                "val_decoded_one_step_global": stability_metrics[
                    "decoded_one_step_global"
                ],
                "val_decoded_one_step_per_channel_json": json.dumps(
                    dict(zip(CHANNELS, stability_metrics["decoded_one_step_per_channel"])),
                    sort_keys=True,
                ),
                "val_decoded_three_step_global": stability_metrics[
                    "decoded_three_step_global"
                ],
                "val_decoded_three_step_per_channel_json": json.dumps(
                    dict(zip(CHANNELS, stability_metrics["decoded_three_step_per_channel"])),
                    sort_keys=True,
                ),
                "val_magnetic_range_violation": stability_metrics[
                    "magnetic_range_violation"
                ],
                "val_composite_stability": stability_metrics["composite_stability"],
                "val_persistence_decoded_one_step_global": persistence_metrics[
                    "decoded_one_step_global"
                ],
                "val_persistence_decoded_three_step_global": persistence_metrics[
                    "decoded_three_step_global"
                ],
                "val_decoded_one_step_improvement_over_persistence_fraction": (
                    1.0
                    - stability_metrics["decoded_one_step_global"]
                    / persistence_metrics["decoded_one_step_global"]
                ),
                "val_decoded_three_step_improvement_over_persistence_fraction": (
                    1.0
                    - stability_metrics["decoded_three_step_global"]
                    / persistence_metrics["decoded_three_step_global"]
                ),
                "val_stability_prediction_nonfinite": stability_metrics[
                    "prediction_nonfinite"
                ],
                "val_stability_rho_positivity_violations": stability_metrics[
                    "rho_positivity_violations"
                ],
                "val_stability_press_positivity_violations": stability_metrics[
                    "press_positivity_violations"
                ],
                "val_one_step_transition_count": stability_metrics[
                    "one_step_transition_count"
                ],
                "val_rolling_origin_count": stability_metrics["rolling_origin_count"],
                "epoch_seconds": epoch_seconds,
                "train_batches": train_batches,
                "val_batches": val_batches,
                "samples_per_second": train_batches * int(training_config["batch_size"]) / epoch_seconds,
                "peak_cuda_memory_mib": peak_memory,
            }
            csv_writer.writerow(row)
            stream.flush()
            history.append(row)
            if writer is not None:
                for name, value in row.items():
                    if isinstance(value, (int, float)) and value is not None:
                        writer.add_scalar(name, value, epoch + 1)
            checkpoint_metrics = {
                **val_metrics,
                **stability_metrics,
                "normalized_total": val_metrics["loss"],
            }
            checkpoint_values = {
                "normalized_total": float(val_metrics["loss"]),
                "decoded_one_step": float(stability_metrics["decoded_one_step_global"]),
                "decoded_three_step": float(stability_metrics["decoded_three_step_global"]),
                "composite_stability": float(stability_metrics["composite_stability"]),
            }
            payload = checkpoint_payload(
                model,
                optimizer,
                epoch + 1,
                config,
                checkpoint_metrics,
                normalizer,
                shell_metadata,
                residual_wrapper_metadata,
            )
            payload["checkpoint_metric_values"] = checkpoint_values
            improved_metrics = [
                name
                for name, value in checkpoint_values.items()
                if best_epochs[name] is None or value < best_values[name]
            ]
            selected_improved = checkpoint_selection_metric in improved_metrics
            if selected_improved:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if improved_metrics:
                probe_sha256 = prediction_probe_sha256(
                    model,
                    datasets["val"],
                    normalizer,
                    shells,
                    downsample,
                    device,
                    predict_residual,
                    bounded_residual,
                    residual_scale,
                )
                for metric_name in improved_metrics:
                    best_values[metric_name] = checkpoint_values[metric_name]
                    best_epochs[metric_name] = epoch + 1
                    selected_payload = dict(payload)
                    selected_payload["checkpoint_selection_metric"] = metric_name
                    selected_payload["checkpoint_selection_value"] = checkpoint_values[
                        metric_name
                    ]
                    selected_payload["prediction_probe_sha256"] = probe_sha256
                    torch.save(
                        selected_payload,
                        output_dir / CHECKPOINT_METRIC_FILENAMES[metric_name],
                    )
                    if metric_name == checkpoint_selection_metric:
                        torch.save(selected_payload, output_dir / "best.pt")
            payload["checkpoint_selection_metric"] = checkpoint_selection_metric
            payload["checkpoint_selection_value"] = checkpoint_values[
                checkpoint_selection_metric
            ]
            torch.save(payload, output_dir / "last.pt")
            if save_epoch_checkpoints:
                lightweight_payload = {
                    key: value
                    for key, value in payload.items()
                    if key != "optimizer_state_dict"
                }
                lightweight_payload["lightweight_checkpoint"] = True
                torch.save(
                    lightweight_payload,
                    epoch_checkpoint_dir / f"epoch_{epoch + 1:04d}.pt",
                )
            print(json.dumps(row), flush=True)
            if epochs_without_improvement >= int(training_config["early_stopping_patience"]):
                print(f"early stopping at epoch {epoch + 1}", flush=True)
                break
    finally:
        stream.close()
        if writer is not None:
            writer.close()

    # Prove that the produced checkpoint is readable and architecture-compatible.
    best_checkpoint = torch.load(output_dir / "best.pt", map_location=device, weights_only=False)
    if residual_wrapper_metadata is not None:
        validate_residual_wrapper_metadata(
            best_checkpoint["residual_wrapper"],
            h5_path=data_config["path"],
            expected_training_indices=datasets["train"].owned_snapshot_indices,
            expected_config=residual_wrapper_config,
        )
    model.load_state_dict(best_checkpoint["model_state_dict"], strict=True)
    reloaded_probe_sha256 = prediction_probe_sha256(
        model,
        datasets["val"],
        normalizer,
        shells,
        downsample,
        device,
        predict_residual,
        bounded_residual,
        residual_scale,
    )
    checkpoint_reload_verified = (
        reloaded_probe_sha256 == best_checkpoint["prediction_probe_sha256"]
    )
    if not checkpoint_reload_verified:
        raise RuntimeError("Checkpoint reload prediction probe mismatch")
    metrics = {
        "status": "completed",
        "experiment_name": config["experiment_name"],
        "epochs_completed": len(history),
        "best_epoch": best_epochs[checkpoint_selection_metric],
        "best_validation_loss": best_values["normalized_total"],
        "best_checkpoint_epochs": best_epochs,
        "best_checkpoint_values": best_values,
        "wall_seconds": time.perf_counter() - run_start,
        "peak_cuda_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        ),
        "checkpoint_reload_verified": checkpoint_reload_verified,
        "prediction_probe_sha256": reloaded_probe_sha256,
        "checkpoint_selection_metric": checkpoint_selection_metric,
        "checkpoint_metric_files": CHECKPOINT_METRIC_FILENAMES,
        "composite_stability_formula": COMPOSITE_STABILITY_FORMULA,
        "epoch_checkpoints_saved": save_epoch_checkpoints,
        "residual_wrapper": residual_wrapper_metadata,
        "validation_persistence_reference": persistence_metrics,
        "history": history,
        "smoke_limited_batches": bool(
            training_config.get("max_train_batches") is not None
            or training_config.get("max_val_batches") is not None
        ),
        "snapshot_count": len(datasets["train"].times),
        "time_range": [
            float(datasets["train"].times[0]),
            float(datasets["train"].times[-1]),
        ],
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    # WandB is deliberately not imported; CSV/JSON/TensorBoard are the default loggers.
    os.environ.setdefault("WANDB_MODE", "disabled")
    main()
