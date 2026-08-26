#!/usr/bin/env python
"""Compare one actual GRMHD batch through upstream and literal manual paths."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from grmhd.dataset import make_temporal_datasets
from grmhd.normalizer import GRMHDNormalizer
from grmhd.parity import compare_upstream_and_manual_train_batch
from grmhd.shells import radial_shells_tensor
from grmhd.upstream_adapters import GRMHDNextStepDataset, UpstreamFNOConfig


def main() -> None:
    data_path = Path("data_proc/grmhd_regrid_inner_r200_64.h5")
    datasets = make_temporal_datasets(
        data_path,
        snapshot_start=0,
        snapshot_end=111,
        train_snapshot_count=80,
        val_snapshot_count=10,
        test_snapshot_count=21,
    )
    normalizer = GRMHDNormalizer.load(
        "outputs/stats/all111/normalizer_stats.npz",
        h5_path=data_path,
        expected_training_indices=datasets["train"].owned_snapshot_indices,
    )
    downsample = 4
    shape = tuple(size // downsample for size in datasets["train"].snapshot_shape[1:])
    shells, _ = radial_shells_tensor(
        datasets["train"].coords["r"][::downsample],
        shape[0],
        shape[1],
        n_shells=8,
    )
    batch = next(
        iter(DataLoader(GRMHDNextStepDataset(datasets["train"]), batch_size=1))
    )
    report = compare_upstream_and_manual_train_batch(
        batch=batch,
        normalizer=normalizer,
        shells=shells,
        downsample=downsample,
        model_config=UpstreamFNOConfig(in_channels=16),
    )
    output_dir = Path("outputs/experiment_round3")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "trainer_parity.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    markdown = "# Trainer batch parity\n\n" + "\n".join(
        f"- `{key}`: `{value}`" for key, value in report.items()
    ) + "\n"
    (output_dir / "trainer_parity.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
