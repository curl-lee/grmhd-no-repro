#!/usr/bin/env python
"""Cross-check one checkpoint with upstream and local physical AR loops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from grmhd.dataset import make_temporal_datasets
from grmhd.hybrid import HybridTargetStats
from grmhd.normalizer import GRMHDNormalizer
from grmhd.parity import compare_upstream_and_local_autoregression
from grmhd.shells import radial_shells_tensor
from grmhd.upstream_adapters import GRMHDAutoregressiveDataset, UpstreamFNOConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "outputs/experiment_round3/hybrid_experiments/R3_recency_weighted/"
            "best_composite.pt/composite_state_dict.pt"
        ),
    )
    parser.add_argument("--target-mode", choices=("state", "hybrid"), default="hybrid")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    data_path = Path("data_proc/grmhd_regrid_inner_r200_64.h5")
    snapshot_end = 100 if args.target_mode == "hybrid" else 111
    datasets = make_temporal_datasets(
        data_path, snapshot_start=0, snapshot_end=snapshot_end,
        train_snapshot_count=80, val_snapshot_count=10,
        test_snapshot_count=snapshot_end - 90,
    )
    normalizer_path = (
        "outputs/experiment_round3/fold_b_stats/normalizer_stats.npz"
        if args.target_mode == "hybrid"
        else "outputs/stats/all111/normalizer_stats.npz"
    )
    normalizer = GRMHDNormalizer.load(
        normalizer_path,
        h5_path=data_path,
        expected_training_indices=datasets["train"].owned_snapshot_indices,
    )
    hybrid = None
    if args.target_mode == "hybrid":
        hybrid = HybridTargetStats.load(
            "outputs/experiment_round3/fold_b_stats/hybrid_target_stats.json",
            h5_path=data_path,
            expected_training_indices=datasets["train"].owned_snapshot_indices,
        )
    shape = datasets["train"].snapshot_shape[1:]
    shells, _ = radial_shells_tensor(
        datasets["train"].coords["r"], shape[0], shape[1], n_shells=8
    )
    sample = next(
        iter(DataLoader(GRMHDAutoregressiveDataset(datasets[args.split]), batch_size=1))
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    report = compare_upstream_and_local_autoregression(
        sample=sample,
        normalizer=normalizer,
        shells=shells,
        downsample=1,
        model_config=UpstreamFNOConfig(in_channels=16),
        model_state_dict=state,
        max_steps=100,
        target_mode=args.target_mode,
        hybrid_stats=hybrid,
        device=device,
    )
    report["checkpoint"] = str(args.checkpoint.resolve())
    report["checkpoint_selection"] = "validation best composite"
    report["split"] = args.split
    report["trajectory_indices"] = list(datasets[args.split].trajectory_indices())
    output_dir = Path("outputs/experiment_round3")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "autoregression_parity.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    markdown = "# Autoregression parity\n\n" + "\n".join(
        f"- `{key}`: `{value}`" for key, value in report.items() if key != "steps"
    ) + "\n"
    (output_dir / "autoregression_parity.md").write_text(markdown, encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
