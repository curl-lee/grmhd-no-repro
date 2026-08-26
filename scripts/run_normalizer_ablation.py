#!/usr/bin/env python
"""Statistics and two-epoch smoke for the controlled normalizer ablation."""

from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from neuralop.training import Trainer

from grmhd import CHANNELS
from grmhd.data_processor import GRMHDDataProcessor
from grmhd.dataset import make_temporal_datasets
from grmhd.gaussian_normalizer import TransformedGaussianNormalizer
from grmhd.normalizer import GRMHDNormalizer
from grmhd.shells import radial_shells_tensor
from grmhd.upstream_adapters import (
    GRMHDNextStepDataset,
    UpstreamFNOConfig,
    build_upstream_fno,
    build_upstream_loss,
    build_upstream_optimizer,
)


def _encoding_statistics(normalizer, physical: torch.Tensor) -> dict[str, object]:
    sampled = physical[..., ::4, ::4, ::4]
    encoded = normalizer.encode_tensor(sampled, channel_axis=1)
    encoded_leaf = encoded.detach().clone().requires_grad_(True)
    decoded = normalizer.decode_tensor(encoded_leaf, channel_axis=1)
    derivatives = []
    for channel in range(8):
        gradient = torch.autograd.grad(
            decoded[:, channel].sum(), encoded_leaf, retain_graph=channel < 7
        )[0][:, channel]
        absolute = torch.abs(gradient).detach().cpu().numpy().reshape(-1)
        derivatives.append(
            {
                "median": float(np.median(absolute)),
                "q99": float(np.quantile(absolute, 0.99)),
                "max": float(np.max(absolute)),
            }
        )
    absolute_encoded = torch.abs(encoded.detach()).cpu().numpy()
    roundtrip = normalizer.decode_tensor(encoded.detach(), channel_axis=1)
    return {
        "encoded_abs_q99_by_channel": {
            channel: float(np.quantile(absolute_encoded[:, index], 0.99))
            for index, channel in enumerate(CHANNELS)
        },
        "encoded_abs_max_by_channel": {
            channel: float(np.max(absolute_encoded[:, index]))
            for index, channel in enumerate(CHANNELS)
        },
        "fraction_abs_above_0.95_gamma_by_channel": {
            channel: float(np.mean(absolute_encoded[:, index] > 0.95 * 6.0))
            for index, channel in enumerate(CHANNELS)
        },
        "decoder_derivative_by_channel": dict(zip(CHANNELS, derivatives)),
        "roundtrip_max_abs_difference": float(torch.max(torch.abs(roundtrip - sampled)).cpu()),
    }


def _train_smoke(
    *,
    name: str,
    normalizer,
    initial_state,
    datasets,
    shells,
    device,
) -> dict[str, object]:
    model = build_upstream_fno(UpstreamFNOConfig(in_channels=16)).to(device)
    model.load_state_dict(deepcopy(initial_state), strict=True)
    processor = GRMHDDataProcessor(
        normalizer=normalizer,
        shells=shells,
        target_mode="state",
        device=device,
    )
    optimizer = build_upstream_optimizer(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    loss = build_upstream_loss("l2")
    trainer = Trainer(
        model=model,
        n_epochs=2,
        device=device,
        data_processor=processor,
        mixed_precision=False,
        eval_interval=1,
        verbose=False,
    )
    started = time.perf_counter()
    metrics = trainer.train(
        train_loader=DataLoader(
            GRMHDNextStepDataset(datasets["train"]), batch_size=1, shuffle=False
        ),
        test_loaders={
            "next_step": DataLoader(
                GRMHDNextStepDataset(datasets["val"]), batch_size=1, shuffle=False
            )
        },
        optimizer=optimizer,
        scheduler=scheduler,
        training_loss=loss,
        eval_losses={"l2": loss},
    )
    return {
        "name": name,
        "epochs": 2,
        "wall_seconds": time.perf_counter() - started,
        "train_loss": float(metrics["avg_loss"]),
        "validation_l2": float(metrics["next_step_l2"]),
    }


def main() -> None:
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Normalizer smoke requires CUDA")
    device = torch.device("cuda")
    data_path = Path("data_proc/grmhd_regrid_inner_r200_64.h5")
    datasets = make_temporal_datasets(
        data_path,
        snapshot_start=0,
        snapshot_end=111,
        train_snapshot_count=80,
        val_snapshot_count=10,
        test_snapshot_count=21,
    )
    paper = GRMHDNormalizer.load(
        "outputs/stats/all111/normalizer_stats.npz",
        h5_path=data_path,
        expected_training_indices=datasets["train"].owned_snapshot_indices,
    )
    gaussian = TransformedGaussianNormalizer.fit(
        datasets["train"], max_samples_per_channel=200_000, seed=seed
    )
    output_dir = Path("outputs/experiment_round3")
    output_dir.mkdir(parents=True, exist_ok=True)
    gaussian.save(output_dir / "upstream_gaussian_transformed_stats.json")
    physical_probe = datasets["val"].load_snapshot(datasets["val"].split.start).unsqueeze(0)
    statistics = {
        "paper_robust": _encoding_statistics(paper, physical_probe),
        "upstream_gaussian_transformed": _encoding_statistics(gaussian, physical_probe),
    }
    shape = datasets["train"].snapshot_shape[1:]
    shells, _ = radial_shells_tensor(
        datasets["train"].coords["r"], shape[0], shape[1], n_shells=8, device=device
    )
    initial_model = build_upstream_fno(UpstreamFNOConfig(in_channels=16))
    initial_state = deepcopy(initial_model.state_dict())
    smokes = [
        _train_smoke(
            name="paper_robust",
            normalizer=paper,
            initial_state=initial_state,
            datasets=datasets,
            shells=shells,
            device=device,
        ),
        _train_smoke(
            name="upstream_gaussian_transformed",
            normalizer=gaussian,
            initial_state=initial_state,
            datasets=datasets,
            shells=shells,
            device=device,
        ),
    ]
    rows = []
    for smoke in smokes:
        stats = statistics[smoke["name"]]
        rows.append(
            {
                **smoke,
                "Bcc1_encoded_abs_q99": stats["encoded_abs_q99_by_channel"]["Bcc1"],
                "Bcc1_decoder_derivative_q99": stats["decoder_derivative_by_channel"]["Bcc1"]["q99"],
                "Bcc1_decoder_derivative_max": stats["decoder_derivative_by_channel"]["Bcc1"]["max"],
                "roundtrip_max_abs_difference": stats["roundtrip_max_abs_difference"],
            }
        )
    with (output_dir / "normalizer_ablation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "status": "completed_two_epoch_smoke_only",
        "statistics": statistics,
        "smokes": smokes,
        "conclusion": (
            "This ablation changes only transformed-space centering/scaling and soft clipping; "
            "the signed/positive physical transforms remain identical."
        ),
    }
    (output_dir / "normalizer_ablation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    lines = [
        "# Normalizer ablation",
        "",
        "Both variants retain the same signed/positive physical transform.  The Gaussian variant uses the pinned upstream `UnitGaussianNormalizer` after that transform and has no soft clip.",
        "",
        "| Variant | train loss | validation L2 | Bcc1 encoded q99 | Bcc1 decoder derivative q99/max |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['train_loss']:.6g} | {row['validation_l2']:.6g} | "
            f"{row['Bcc1_encoded_abs_q99']:.6g} | {row['Bcc1_decoder_derivative_q99']:.6g} / "
            f"{row['Bcc1_decoder_derivative_max']:.6g} |"
        )
    (output_dir / "normalizer_ablation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
