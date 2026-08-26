#!/usr/bin/env python
"""Validate the fixed n111 HDF5 and both train-only stats bundles without refitting."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py

from fit_grmhd_stats import load_config, make_datasets
from grmhd.dataset import sha256_file
from grmhd.normalizer import validate_stats_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/experiment_round1/stats_validation.json"),
    )
    args = parser.parse_args()
    configs = [Path("configs/data/all111.yaml"), Path("configs/data/late100.yaml")]
    first = load_config(configs[0].resolve())
    h5_path = Path(first["data"]["path"])
    checksum = sha256_file(h5_path)
    with h5py.File(h5_path, "r") as handle:
        shape = list(handle["snapshots"].shape)
    if shape != [111, 8, 64, 64, 64]:
        raise ValueError(f"Unexpected HDF5 shape: {shape}")
    expected_checksum = "cd3966bed1f617f49d775151cff87ae92eb2432a560bb82ae9355605e7aea70a"
    if checksum != expected_checksum:
        raise ValueError(f"Unexpected HDF5 checksum: {checksum}")

    reports = {}
    for config_path in configs:
        config = load_config(config_path.resolve())
        data = config["data"]
        datasets = make_datasets(data)
        stats_dir = Path(data["normalizer_stats_path"]).parent
        report = validate_stats_bundle(
            stats_dir,
            h5_path,
            expected_training_indices=datasets["train"].owned_snapshot_indices,
            expected_window_name=str(data["window_name"]),
        )
        rejection_checks = {}
        try:
            validate_stats_bundle(
                stats_dir,
                h5_path,
                expected_training_indices=datasets["train"].owned_snapshot_indices,
                expected_window_name=f"wrong-{data['window_name']}",
            )
        except ValueError as exc:
            rejection_checks["window_mismatch"] = {"rejected": True, "error": str(exc)}
        try:
            validate_stats_bundle(
                stats_dir,
                h5_path,
                expected_training_indices=datasets["train"].owned_snapshot_indices[:-1],
                expected_window_name=str(data["window_name"]),
            )
        except ValueError as exc:
            rejection_checks["training_indices_mismatch"] = {
                "rejected": True,
                "error": str(exc),
            }
        report["rejection_checks"] = rejection_checks
        reports[str(data["window_name"])] = report

    payload = {
        "status": "passed",
        "hdf5": {
            "path": str(h5_path.resolve()),
            "shape": shape,
            "sha256": checksum,
        },
        "windows": reports,
        "note": "Read-only validation; no statistics were refit or modified.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
