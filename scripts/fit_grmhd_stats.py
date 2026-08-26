#!/usr/bin/env python
"""Fit all train-only normalization and optional-prior statistics for a data window."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import yaml

from grmhd import CHANNELS
from grmhd.dataset import GRMHDPairedDataset, make_temporal_datasets
from grmhd.normalizer import DEFAULT_EPSILON, GRMHDNormalizer
from grmhd.priors import QuantileBounds, ResidualEnvelope
from grmhd.shells import radial_shells_tensor


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        result[key] = (
            deep_merge(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else value
        )
    return result


def load_config(path: Path) -> dict[str, Any]:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    base = values.pop("base_config", None)
    if base is None:
        return values
    base_path = Path(base)
    if not base_path.is_absolute() and not base_path.exists():
        base_path = path.parent / base_path
    return deep_merge(load_config(base_path.resolve()), values)


def make_datasets(data: dict[str, Any]) -> dict[str, GRMHDPairedDataset]:
    return make_temporal_datasets(
        data["path"],
        stride=int(data["stride"]),
        train_fraction=float(data["train_fraction"]),
        val_fraction=float(data["val_fraction"]),
        snapshot_start=int(data.get("snapshot_start", 0)),
        snapshot_end=None if data.get("snapshot_end") is None else int(data["snapshot_end"]),
        train_snapshot_count=None if data.get("train_snapshot_count") is None else int(data["train_snapshot_count"]),
        val_snapshot_count=None if data.get("val_snapshot_count") is None else int(data["val_snapshot_count"]),
        test_snapshot_count=None if data.get("test_snapshot_count") is None else int(data["test_snapshot_count"]),
    )


def fit_radial_baseline(dataset: GRMHDPairedDataset) -> dict[str, Any]:
    r = np.asarray(dataset.coords["r"], dtype=np.float64)
    log_r = np.log10(r)
    coefficients: dict[str, dict[str, float]] = {}
    with h5py.File(dataset.h5_path, "r") as handle:
        for channel in (3, 4):
            profiles = []
            for snapshot_index in dataset.owned_snapshot_indices:
                raw = np.asarray(handle["snapshots"][snapshot_index, channel], dtype=np.float64)
                transformed = np.log10(
                    np.maximum(raw + DEFAULT_EPSILON[channel], np.finfo(np.float64).tiny)
                )
                profiles.append(np.median(transformed, axis=(0, 1)))
            profile = np.median(np.stack(profiles), axis=0)
            slope, intercept = np.polyfit(log_r, profile, deg=1)
            coefficients[CHANNELS[channel]] = {
                "slope": float(slope),
                "intercept": float(intercept),
            }
    return {
        "enabled_for_fit": False,
        "radial_coordinate": "log10(r)",
        "fit_space": "raw positive-log before robust normalization",
        "fit_statistic": "median over phi/theta, then median over training snapshots",
        "fit_separately": True,
        "r": r.tolist(),
        "channels": coefficients,
    }


def fit_velocity_roi_threshold(
    dataset: GRMHDPairedDataset,
    normalizer: GRMHDNormalizer,
    *,
    quantile: float = 0.8,
    max_samples: int = 100_000,
    seed: int = 42,
) -> dict[str, Any]:
    voxels = int(np.prod(dataset.snapshot_shape[1:]))
    indices = dataset.owned_snapshot_indices
    population = len(indices) * voxels
    choices = np.sort(
        np.random.default_rng(seed + 8675309).choice(
            population, size=min(max_samples, population), replace=False
        )
    )
    samples = np.empty(len(choices), dtype=np.float32)
    offset = 0
    for slot, snapshot_index in enumerate(indices):
        mask = choices // voxels == slot
        local = choices[mask] % voxels
        count = int(mask.sum())
        if count:
            encoded = normalizer.encode_tensor(dataset.load_snapshot(snapshot_index)).numpy()
            norm = np.sqrt(np.sum(encoded[5:8] ** 2, axis=0)).reshape(-1)
            samples[offset : offset + count] = norm[local]
            offset += count
    return {
        "threshold": float(np.quantile(samples, quantile)),
        "quantile": quantile,
        "sample_count": len(samples),
        "seed": seed,
        "space": "normalized stored velocity coordinate-component Euclidean norm",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path)
    args = parser.parse_args()

    config = load_config(args.config.resolve())
    data = config["data"]
    seed = int(config["seed"])
    datasets = make_datasets(data)
    train = datasets["train"]
    normalizer = GRMHDNormalizer.fit(
        train,
        max_samples_per_channel=int(data["normalizer_samples_per_channel"]),
        seed=seed,
        radial_baseline=False,
    )
    configured_stats_path = Path(data["normalizer_stats_path"])
    out_dir = args.out_dir or configured_stats_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    normalizer_path = out_dir / "normalizer_stats.npz"
    normalizer.save(normalizer_path)

    bounds = QuantileBounds.fit(train, normalizer, seed=seed)
    envelope = ResidualEnvelope.fit(train, normalizer, seed=seed)
    radial_baseline = fit_radial_baseline(train)
    _, shell_metadata = radial_shells_tensor(
        train.coords["r"],
        train.snapshot_shape[1],
        train.snapshot_shape[2],
        n_shells=int(data.get("n_shells", 8)),
    )
    payload = {
        "window_name": data.get("window_name"),
        "source_hdf5": str(train.h5_path),
        "source_hdf5_checksum": normalizer.source_hdf5_checksum,
        "train_snapshot_indices": list(train.owned_snapshot_indices),
        "split_manifests": {name: dataset.manifest() for name, dataset in datasets.items()},
        "normalizer": normalizer.as_dict(),
        "shell_metadata": shell_metadata.as_dict(),
        "radial_baseline": radial_baseline,
        "quantile_bounds": bounds.as_dict(),
        "residual_envelope": envelope.as_dict(),
        "velocity_roi": fit_velocity_roi_threshold(train, normalizer, seed=seed),
        "command": shlex.join([sys.executable, *sys.argv]),
    }
    (out_dir / "priors.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (out_dir / "manifest.json").write_text(
        json.dumps(payload["split_manifests"], indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "window_name": payload["window_name"],
        "source_hdf5_checksum": payload["source_hdf5_checksum"],
        "train_snapshot_indices": payload["train_snapshot_indices"],
        "normalizer_path": str(normalizer_path),
        "radial_baseline": radial_baseline["channels"],
        "velocity_roi": payload["velocity_roi"],
    }, indent=2))


if __name__ == "__main__":
    main()
