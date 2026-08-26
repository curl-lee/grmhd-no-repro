#!/usr/bin/env python
"""Two-epoch GRMHD closure test using the pinned upstream Trainer unchanged."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from neuralop.training import Trainer

from grmhd.data_processor import GRMHDDataProcessor
from grmhd.dataset import make_temporal_datasets, sha256_file
from grmhd.normalizer import GRMHDNormalizer
from grmhd.shells import radial_shells_tensor
from grmhd.upstream_adapters import (
    PINNED_NEURALOP_COMMIT,
    GRMHDAutoregressiveDataset,
    GRMHDNextStepDataset,
    UpstreamFNOConfig,
    build_upstream_fno,
    build_upstream_loss,
    build_upstream_optimizer,
    save_upstream_training_bundle,
)


def _jsonable(values):
    if isinstance(values, torch.Tensor):
        return float(values.detach().cpu()) if values.numel() == 1 else values.detach().cpu().tolist()
    if isinstance(values, dict):
        return {key: _jsonable(value) for key, value in values.items()}
    if isinstance(values, (list, tuple)):
        return [_jsonable(value) for value in values]
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/upstream_baselines/fno_next_step.yaml"),
    )
    parser.add_argument("--device")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.device:
        config["training"]["device"] = args.device
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    requested_device = config["training"]["device"]
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(requested_device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    data = config["data"]
    datasets = make_temporal_datasets(
        data["path"],
        stride=int(data["stride"]),
        snapshot_start=int(data["snapshot_start"]),
        snapshot_end=int(data["snapshot_end"]),
        train_snapshot_count=int(data["train_snapshot_count"]),
        val_snapshot_count=int(data["val_snapshot_count"]),
        test_snapshot_count=int(data["test_snapshot_count"]),
    )
    normalizer = GRMHDNormalizer.load(
        data["normalizer_stats_path"],
        h5_path=data["path"],
        expected_training_indices=datasets["train"].owned_snapshot_indices,
    )
    downsample = int(data["downsample"])
    source_shape = datasets["train"].snapshot_shape[1:]
    actual_shape = tuple(size // downsample for size in source_shape)
    shells = None
    shell_metadata = None
    if data.get("with_shells", True):
        shells, shell_metadata_object = radial_shells_tensor(
            datasets["train"].coords["r"][::downsample],
            actual_shape[0],
            actual_shape[1],
            n_shells=int(data["n_shells"]),
            device=device,
        )
        shell_metadata = shell_metadata_object.as_dict()

    processor = GRMHDDataProcessor(
        normalizer=normalizer,
        shells=shells,
        target_mode="state",
        downsample=downsample,
        device=device,
    )
    train_loader = DataLoader(
        GRMHDNextStepDataset(datasets["train"]),
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
    )
    test_loaders = {
        "next_step": DataLoader(
            GRMHDNextStepDataset(datasets["val"]), batch_size=1, shuffle=False
        ),
        "autoregression": DataLoader(
            GRMHDAutoregressiveDataset(datasets["val"]), batch_size=1, shuffle=False
        ),
    }

    model_config = UpstreamFNOConfig(
        in_channels=int(config["model"]["in_channels"]),
        out_channels=int(config["model"]["out_channels"]),
        n_modes=tuple(config["model"]["n_modes"]),
        hidden_channels=int(config["model"]["hidden_channels"]),
        n_layers=int(config["model"]["n_layers"]),
        positional_embedding=None,
    )
    model = build_upstream_fno(model_config)
    optimizer = build_upstream_optimizer(
        model,
        learning_rate=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(config["training"]["epochs"])
    )
    loss = build_upstream_loss(config["loss"]["name"])
    trainer = Trainer(
        model=model,
        n_epochs=int(config["training"]["epochs"]),
        device=device,
        data_processor=processor,
        mixed_precision=bool(config["training"]["mixed_precision"]),
        eval_interval=int(config["training"]["eval_interval"]),
        verbose=True,
    )

    output_dir = Path(config["logging"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    started = time.perf_counter()
    metrics = trainer.train(
        train_loader=train_loader,
        test_loaders=test_loaders,
        eval_modes={"autoregression": "autoregression"},
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=None,
        training_loss=loss,
        eval_losses={"l2": loss},
        save_best="next_step_l2",
        save_dir=output_dir / "best_upstream",
        max_autoregressive_steps=int(config["training"]["max_autoregressive_steps"]),
    )
    wall_seconds = time.perf_counter() - started
    metadata = {
        "model_config": model_config.as_dict(),
        "optimizer": "neuralop.training.AdamW",
        "trainer": "neuralop.training.Trainer",
        "loss": "neuralop.LpLoss(d=3,p=2,reduction='sum')",
        "prediction_mode": "direct_state",
        "target_mode": "encoded state during training; physical state during evaluation",
        "normalizer_checksum": sha256_file(data["normalizer_stats_path"]),
        "hdf5_checksum": sha256_file(data["path"]),
        "train_indices": list(datasets["train"].owned_snapshot_indices),
        "fold": "all111_80_10_21",
        "shells": shell_metadata,
        "rollout_k": None,
        "checkpoint_selection_metric": "next_step_l2",
    }
    save_upstream_training_bundle(
        output_dir / "last_upstream",
        "last",
        model=trainer.model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=int(config["training"]["epochs"]),
        metadata=metadata,
    )
    report = {
        "status": "completed",
        "neuraloperator_commit": PINNED_NEURALOP_COMMIT,
        "model_class": type(trainer.model).__name__,
        "model_module": type(trainer.model).__module__,
        "trainer_class": type(trainer).__name__,
        "trainer_module": type(trainer).__module__,
        "optimizer_class": type(optimizer).__name__,
        "optimizer_module": type(optimizer).__module__,
        "data_processor_class": type(processor).__name__,
        "epochs": int(config["training"]["epochs"]),
        "wall_seconds": wall_seconds,
        "peak_cuda_memory_mib": (
            torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
        ),
        "metrics": _jsonable(metrics),
        "metadata": metadata,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
