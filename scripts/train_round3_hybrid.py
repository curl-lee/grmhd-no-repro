#!/usr/bin/env python
"""Run one gated Round-3 hybrid-target ablation (R1--R4)."""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
import yaml

from grmhd import CHANNELS
from grmhd.data_processor import GRMHDDataProcessor
from grmhd.dataset import make_temporal_datasets, sha256_file
from grmhd.fold_b import recency_weights
from grmhd.hybrid import HybridTargetStats
from grmhd.models import zero_initialize_residual_head
from grmhd.normalizer import GRMHDNormalizer
from grmhd.shells import radial_shells_tensor
from grmhd.trainers import GRMHDRolloutTrainer
from grmhd.upstream_adapters import (
    PINNED_NEURALOP_COMMIT,
    GRMHDNextStepDataset,
    UpstreamFNOConfig,
    build_upstream_fno,
    build_upstream_optimizer,
    save_upstream_training_bundle,
)


VARIANT_TERMS = {
    "R1": ("target",),
    "R2": ("target", "decoded"),
    "R3": ("target", "decoded", "rollout"),
    "R4": ("target", "decoded", "rollout", "range"),
}


def _lr_lambda(epoch: int, total: int, warmup: int, minimum_ratio: float) -> float:
    if epoch < warmup:
        return (epoch + 1) / warmup
    progress = min(max((epoch - warmup) / max(total - warmup - 1, 1), 0.0), 1.0)
    return minimum_ratio + 0.5 * (1.0 - minimum_ratio) * (1.0 + math.cos(math.pi * progress))


def _model_input(model_state, normalizer, shells):
    encoded = normalizer.encode_tensor(model_state, channel_axis=1)
    if shells is None:
        return encoded
    return torch.cat(
        (encoded, shells.unsqueeze(0).expand(encoded.shape[0], -1, -1, -1, -1)), dim=1
    )


def _accumulate(prediction, truth, error, reference):
    axes = (0,) + tuple(range(2, prediction.ndim))
    error += (prediction.detach().double() - truth.detach().double()).square().sum(dim=axes).cpu()
    reference += truth.detach().double().square().sum(dim=axes).cpu()


def _relative(error, reference):
    per_channel = torch.sqrt(error / reference.clamp_min(1.0e-30))
    return float(torch.sqrt(error.sum() / reference.sum().clamp_min(1.0e-30))), [
        float(value) for value in per_channel
    ]


def evaluate(model, dataset, normalizer, hybrid, shells, device):
    model.eval()
    one_error = torch.zeros(8, dtype=torch.float64)
    one_truth = torch.zeros(8, dtype=torch.float64)
    persistence_error = torch.zeros(8, dtype=torch.float64)
    persistence_truth = torch.zeros(8, dtype=torch.float64)
    target_total = 0.0
    target_count = 0
    late_error = torch.zeros(8, dtype=torch.float64)
    late_truth = torch.zeros(8, dtype=torch.float64)
    midpoint = dataset.split.start + dataset.split.snapshot_count // 2
    nonfinite = positivity = 0
    with torch.no_grad():
        for item in range(len(dataset)):
            sample = dataset[item]
            physical_input = sample["x"].unsqueeze(0).to(device)
            physical_target = sample["y"].unsqueeze(0).to(device)
            raw = model(x=_model_input(physical_input, normalizer, shells))
            target_prediction = hybrid.bounded_target_prediction(raw)
            target = hybrid.encode_target(physical_input, physical_target)
            target_total += float(torch.nn.functional.smooth_l1_loss(target_prediction, target))
            target_count += 1
            prediction = hybrid.reconstruct(physical_input, raw)
            _accumulate(prediction, physical_target, one_error, one_truth)
            _accumulate(physical_input, physical_target, persistence_error, persistence_truth)
            if int(sample["index"]) >= midpoint:
                _accumulate(prediction, physical_target, late_error, late_truth)
            nonfinite += int((~torch.isfinite(prediction)).sum())
            positivity += int((prediction[:, 3:5] <= 0).sum())

        trajectory = dataset.trajectory_indices()
        initial = dataset.load_snapshot(trajectory[0]).unsqueeze(0).to(device)
        current = initial
        steps = []
        range_ratios = []
        predicted_ranges = []
        collapse = ripple = stripe = False
        for step, truth_index in enumerate(trajectory[1:], start=1):
            truth = dataset.load_snapshot(truth_index).unsqueeze(0).to(device)
            raw = model(x=_model_input(current, normalizer, shells))
            prediction = hybrid.reconstruct(current, raw)
            model_global = float(
                torch.linalg.vector_norm(prediction - truth)
                / torch.linalg.vector_norm(truth).clamp_min(1e-12)
            )
            persistence_global = float(
                torch.linalg.vector_norm(initial - truth)
                / torch.linalg.vector_norm(truth).clamp_min(1e-12)
            )
            channel_error = []
            persistence_channel = []
            for channel in range(8):
                channel_error.append(
                    float(
                        torch.linalg.vector_norm(prediction[:, channel] - truth[:, channel])
                        / torch.linalg.vector_norm(truth[:, channel]).clamp_min(1e-12)
                    )
                )
                persistence_channel.append(
                    float(
                        torch.linalg.vector_norm(initial[:, channel] - truth[:, channel])
                        / torch.linalg.vector_norm(truth[:, channel]).clamp_min(1e-12)
                    )
                )
            predicted_range = prediction[:, :3].amax(dim=(-3, -2, -1)) - prediction[
                :, :3
            ].amin(dim=(-3, -2, -1))
            truth_range = truth[:, :3].amax(dim=(-3, -2, -1)) - truth[:, :3].amin(
                dim=(-3, -2, -1)
            )
            ratio = predicted_range / truth_range.clamp_min(1e-12)
            range_ratios.append(float(torch.max(ratio)))
            predicted_ranges.append(float(torch.max(predicted_range)))
            pred_std = prediction.std(dim=(-3, -2, -1))
            truth_std = truth.std(dim=(-3, -2, -1)).clamp_min(1e-12)
            collapse |= bool(torch.any(pred_std / truth_std < 0.1))
            pred_tv = torch.stack(
                [torch.diff(prediction, dim=axis).abs().mean() for axis in (-3, -2, -1)]
            )
            truth_tv = torch.stack(
                [torch.diff(truth, dim=axis).abs().mean() for axis in (-3, -2, -1)]
            ).clamp_min(1e-12)
            ripple |= bool(torch.mean(pred_tv / truth_tv) > 2.0)
            stripe |= bool(
                (pred_tv.max() / pred_tv.min().clamp_min(1e-12))
                > 2.0 * (truth_tv.max() / truth_tv.min().clamp_min(1e-12))
            )
            nonfinite += int((~torch.isfinite(prediction)).sum())
            positivity += int((prediction[:, 3:5] <= 0).sum())
            steps.append(
                {
                    "step": step,
                    "truth_index": truth_index,
                    "model_global": model_global,
                    "persistence_global": persistence_global,
                    "model_per_channel": dict(zip(CHANNELS, channel_error)),
                    "persistence_per_channel": dict(zip(CHANNELS, persistence_channel)),
                    "magnetic_max_range_ratio": range_ratios[-1],
                }
            )
            current = prediction
    one_global, one_channels = _relative(one_error, one_truth)
    persistence_global, persistence_channels = _relative(
        persistence_error, persistence_truth
    )
    late_global, _ = _relative(late_error, late_truth)
    horizons = {
        str(horizon): next((record for record in steps if record["step"] == horizon), None)
        for horizon in (1, 3, 5, 9)
    }
    rollout3 = horizons["3"]["model_global"] if horizons["3"] else float("inf")
    range_penalty = float(np.mean([max(value - 1.0, 0.0) for value in range_ratios[:3]]))
    return {
        "target_smooth_l1": target_total / target_count,
        "decoded_one_step_global": one_global,
        "decoded_one_step_per_channel": dict(zip(CHANNELS, one_channels)),
        "persistence_one_step_global": persistence_global,
        "persistence_one_step_per_channel": dict(zip(CHANNELS, persistence_channels)),
        "one_step_improvement_fraction": 1.0 - one_global / persistence_global,
        "late_half_decoded_one_step_global": late_global,
        "rollout3_global": rollout3,
        "rollout3_persistence_global": (
            horizons["3"]["persistence_global"] if horizons["3"] else float("inf")
        ),
        "magnetic_range_penalty3": range_penalty,
        "composite": one_global + 0.5 * rollout3 + 0.1 * range_penalty,
        "maximum_magnetic_range_ratio": max(range_ratios),
        "maximum_consecutive_magnetic_range_growth": max(
            [predicted_ranges[index] / max(predicted_ranges[index - 1], 1e-30) for index in range(1, len(predicted_ranges))]
            or [1.0]
        ),
        "prediction_nonfinite": nonfinite,
        "rho_press_positivity_violations": positivity,
        "artifact_flags": {"collapse": collapse, "ripple": ripple, "stripe": stripe},
        "horizons": horizons,
        "steps": steps,
    }


def _gradient_norm(model):
    return math.sqrt(
        sum(
            float(parameter.grad.detach().abs().double().square().sum())
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    )


def calibrate(config, datasets, normalizer, hybrid, shells, device, batch):
    model = build_upstream_fno(UpstreamFNOConfig(in_channels=16)).to(device)
    zero_initialize_residual_head(model)
    processor = GRMHDDataProcessor(
        normalizer=normalizer, shells=shells, target_mode="hybrid", hybrid_stats=hybrid, device=device
    )
    trainer = GRMHDRolloutTrainer(
        model=model,
        n_epochs=1,
        device=device,
        data_processor=processor,
        rollout_dataset=datasets["train"],
        loss_weights=config["loss"],
        gradient_accumulation=4,
    )
    trainer.rollout_k = 1
    measurements = {}
    for name in ("target", "decoded", "rollout", "range"):
        model.zero_grad(set_to_none=True)
        components = trainer.loss_components(batch)
        components[name].backward()
        measurements[name] = {
            "value": float(components[name].detach().cpu()),
            "gradient_norm": _gradient_norm(model),
            "default_weight": float(config["loss"][name]),
        }
    target_contribution = measurements["target"]["gradient_norm"]
    calibrated = {name: float(config["loss"][name]) for name in measurements}
    maximum_ratio = float(config["loss"]["calibration_max_ratio"])
    for name in ("decoded", "rollout", "range"):
        contribution = measurements[name]["gradient_norm"] * calibrated[name]
        if contribution == 0 or target_contribution == 0:
            continue
        if contribution > maximum_ratio * target_contribution:
            calibrated[name] *= maximum_ratio * target_contribution / contribution
        elif contribution < target_contribution / maximum_ratio:
            calibrated[name] *= (target_contribution / maximum_ratio) / contribution
    for name in measurements:
        measurements[name]["calibrated_weight"] = calibrated[name]
        measurements[name]["weighted_gradient_contribution"] = (
            calibrated[name] * measurements[name]["gradient_norm"]
        )
    return {"measurements": measurements, "weights": calibrated}


def _loader(strategy, base_dataset, config):
    wrapped = GRMHDNextStepDataset(base_dataset)
    generator = torch.Generator().manual_seed(int(config["seed"]))
    if strategy == "uniform_history":
        return DataLoader(wrapped, batch_size=1, shuffle=True, generator=generator)
    if strategy == "recency_weighted":
        weights = recency_weights(
            [int(value) for value in base_dataset.pair_starts],
            tau=float(config["training"]["recency_tau"]),
        )
        sampler = WeightedRandomSampler(
            weights, num_samples=len(wrapped), replacement=True, generator=generator
        )
        return DataLoader(wrapped, batch_size=1, sampler=sampler)
    if strategy == "recent_window":
        start, stop = config["training"]["recent_window"]
        selected = [
            index
            for index, source in enumerate(base_dataset.pair_starts)
            if start <= int(source) < stop
        ]
        return DataLoader(
            Subset(wrapped, selected), batch_size=1, shuffle=True, generator=generator
        )
    raise ValueError(f"Unknown strategy {strategy}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/round3/hybrid_fold_b.yaml"))
    parser.add_argument("--variant", choices=sorted(VARIANT_TERMS), required=True)
    parser.add_argument(
        "--strategy",
        choices=("uniform_history", "recency_weighted", "recent_window"),
        default="uniform_history",
    )
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Round 3 hybrid experiment requires CUDA")
    device = torch.device(config["training"]["device"])
    torch.cuda.reset_peak_memory_stats(device)
    data = config["data"]
    datasets = make_temporal_datasets(
        data["path"], snapshot_start=0, snapshot_end=100,
        train_snapshot_count=80, val_snapshot_count=10, test_snapshot_count=10,
    )
    normalizer = GRMHDNormalizer.load(
        data["normalizer_stats_path"], h5_path=data["path"],
        expected_training_indices=datasets["train"].owned_snapshot_indices,
    )
    hybrid = HybridTargetStats.load(
        data["hybrid_target_stats_path"], h5_path=data["path"],
        expected_training_indices=datasets["train"].owned_snapshot_indices,
    )
    shape = datasets["train"].snapshot_shape[1:]
    shells, shell_metadata = radial_shells_tensor(
        datasets["train"].coords["r"], shape[0], shape[1], n_shells=8, device=device
    )
    train_loader = _loader(args.strategy, datasets["train"], config)
    first_batch = next(iter(_loader("uniform_history", datasets["train"], config)))
    calibration_path = Path("outputs/experiment_round3/loss_calibration.json")
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    else:
        calibration = calibrate(
            config, datasets, normalizer, hybrid, shells, device, first_batch
        )
        calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")
    weights = {
        name: calibration["weights"][name] if name in VARIANT_TERMS[args.variant] else 0.0
        for name in ("target", "decoded", "rollout", "range")
    }
    model_config = UpstreamFNOConfig(in_channels=16)
    model = build_upstream_fno(model_config).to(device)
    zero_head = zero_initialize_residual_head(model)
    with torch.no_grad():
        physical_probe = datasets["val"].load_snapshot(80).unsqueeze(0).to(device)
        zero_prediction = hybrid.reconstruct(
            physical_probe, model(x=_model_input(physical_probe, normalizer, shells))
        )
        zero_persistence_difference = float(torch.max(torch.abs(zero_prediction - physical_probe)))
    if zero_persistence_difference != 0.0:
        raise RuntimeError("Hybrid zero head is not exact persistence")
    optimizer = build_upstream_optimizer(
        model, learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    epochs = int(config["training"]["epochs"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda epoch: _lr_lambda(
            epoch, epochs, int(config["training"]["warmup_epochs"]),
            float(config["training"]["min_learning_rate"]) / float(config["training"]["learning_rate"]),
        ),
    )
    processor = GRMHDDataProcessor(
        normalizer=normalizer, shells=shells, target_mode="hybrid", hybrid_stats=hybrid, device=device
    )
    trainer = GRMHDRolloutTrainer(
        model=model, n_epochs=epochs, device=device, data_processor=processor,
        rollout_dataset=datasets["train"], loss_weights=weights,
        gradient_accumulation=int(config["training"]["gradient_accumulation"]),
        gradient_clip=float(config["training"]["gradient_clip"]), mixed_precision=False,
    )
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler
    trainer.regularizer = None
    experiment_name = f"{args.variant}_{args.strategy}"
    output_dir = Path(config["logging"]["root"]) / experiment_name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing Round 3 experiment: {output_dir}")
    output_dir.mkdir(parents=True)
    resolved = deepcopy(config)
    resolved["variant"] = args.variant
    resolved["strategy"] = args.strategy
    resolved["calibrated_loss_weights"] = weights
    resolved["zero_initialized_head"] = zero_head
    resolved["zero_persistence_max_abs_difference"] = zero_persistence_difference
    (output_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    metadata_base = {
        "neuraloperator_commit": PINNED_NEURALOP_COMMIT,
        "model_config": model_config.as_dict(),
        "optimizer": "neuralop.training.AdamW",
        "scheduler": "torch.optim.lr_scheduler.LambdaLR",
        "prediction_mode": "hybrid_physical_residual",
        "target_mode": "signed additive plus positive log-ratio",
        "normalizer_checksum": sha256_file(data["normalizer_stats_path"]),
        "target_stats_checksum": hybrid.stats_checksum,
        "hdf5_checksum": sha256_file(data["path"]),
        "train_indices": list(datasets["train"].owned_snapshot_indices),
        "fold": "B",
        "shells": shell_metadata.as_dict(),
        "loss_weights": weights,
        "strategy": args.strategy,
        "variant": args.variant,
    }
    checkpoint_names = config["checkpoint"]["names"]
    best = {name: float("inf") for name in ("target", "decoded_one_step", "rollout3", "composite")}
    best_epoch = {name: None for name in best}
    best_states = {}
    history = []
    patience = int(config["training"]["early_stopping_patience"])
    without_composite = 0
    started = time.perf_counter()
    for epoch in range(epochs):
        epoch_started = time.perf_counter()
        trainer.train_one_epoch(epoch, train_loader)
        validation = evaluate(model, datasets["val"], normalizer, hybrid, shells, device)
        values = {
            "target": validation["target_smooth_l1"],
            "decoded_one_step": validation["decoded_one_step_global"],
            "rollout3": validation["rollout3_global"],
            "composite": validation["composite"],
        }
        improved_composite = False
        for name, value in values.items():
            if value < best[name]:
                best[name] = value
                best_epoch[name] = epoch + 1
                best_states[name] = deepcopy(model.state_dict())
                metadata = {
                    **metadata_base,
                    "rollout_k": trainer.rollout_k,
                    "checkpoint_selection_metric": name,
                    "checkpoint_selection_value": value,
                    "validation": validation,
                }
                save_upstream_training_bundle(
                    output_dir / checkpoint_names[name], name,
                    model=model, optimizer=optimizer, scheduler=scheduler,
                    epoch=epoch + 1, metadata=metadata,
                )
                improved_composite |= name == "composite"
        without_composite = 0 if improved_composite else without_composite + 1
        row = {
            "epoch": epoch + 1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            **trainer.epoch_component_metrics,
            **{f"val_{key}": value for key, value in values.items()},
            "val_one_step_improvement_fraction": validation["one_step_improvement_fraction"],
            "val_late_half_decoded_one_step_global": validation["late_half_decoded_one_step_global"],
            "epoch_seconds": time.perf_counter() - epoch_started,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if without_composite >= patience:
            break
    save_upstream_training_bundle(
        output_dir / checkpoint_names["last"], "last",
        model=model, optimizer=optimizer, scheduler=scheduler, epoch=len(history),
        metadata={**metadata_base, "rollout_k": trainer.rollout_k, "checkpoint_selection_metric": "last"},
    )
    model.load_state_dict(deepcopy(best_states["composite"]), strict=True)
    validation_best = evaluate(model, datasets["val"], normalizer, hybrid, shells, device)
    test = evaluate(model, datasets["test"], normalizer, hybrid, shells, device)
    report = {
        "status": "completed",
        "variant": args.variant,
        "strategy": args.strategy,
        "epochs_completed": len(history),
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_memory_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "loss_calibration": calibration,
        "loss_weights": weights,
        "zero_persistence_max_abs_difference": zero_persistence_difference,
        "best_epochs": best_epoch,
        "best_values": best,
        "validation_best_composite": validation_best,
        "test_best_composite": test,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(history)
    print(json.dumps({key: value for key, value in report.items() if key != "history"}, indent=2))


if __name__ == "__main__":
    main()
