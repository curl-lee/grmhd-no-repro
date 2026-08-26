#!/usr/bin/env python3
"""Fit resolution-specific train-only P3 statistics and audit oracle floors."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import yaml

from grmhd import CHANNELS
from grmhd.dataset import sha256_file
from grmhd.paper_preprocessing import (
    PAPER_GAMMA,
    PAPER_INVERSE_CLAMP_FRACTION,
    PaperPreprocessor,
)
from grmhd.paper_stage_n import NO_SOFTCLIP, PrototypePreprocessor, prototype_specs
from grmhd.stage_z import REPRODUCTION_SCOPE


ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty Stage Z oracle table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def fit_processor(dataset: Path, resolution: int, checksum: str) -> PrototypePreprocessor:
    base = PaperPreprocessor.fit_hdf5(
        dataset,
        training_indices=range(169),
        protocol_name=f"stage_z_z{resolution}_train_only_p3",
        expected_source_hdf5_checksum=checksum,
        gamma=PAPER_GAMMA,
        inverse_clamp_fraction=PAPER_INVERSE_CLAMP_FRACTION,
        thermal_channel="press",
        paper_adaptation=True,
        eos_conversion="disabled_unverified_gamma",
    )
    return PrototypePreprocessor(base, prototype_specs(NO_SOFTCLIP)["P3"])


def processor_summary(
    processor: PrototypePreprocessor, *, resolution: int, dataset: Path,
    dataset_sha256: str, artifact_dir: Path, reused: bool,
) -> dict[str, Any]:
    normalizer_path = artifact_dir / "normalizer.npz"
    return {
        "schema_version": "stage-z-resolution-p3-v1",
        "reproduction_scope": REPRODUCTION_SCOPE,
        "resolution": resolution,
        "dataset": str(dataset.relative_to(ROOT)),
        "dataset_sha256": dataset_sha256,
        "artifact_directory": str(artifact_dir.resolve().relative_to(ROOT)),
        "normalizer_sha256": sha256_file(normalizer_path),
        "frozen_stage_s_artifact_reused": reused,
        "family": "P3",
        "channel_policies": dict(zip(CHANNELS, processor.spec.channel_policies, strict=True)),
        "epsilon": dict(zip(CHANNELS, processor.epsilon.tolist(), strict=True)),
        "median": dict(zip(CHANNELS, processor.median.tolist(), strict=True)),
        "mad_scale": dict(zip(CHANNELS, processor.scale.tolist(), strict=True)),
        "gamma": processor.gamma,
        "inverse_clamp_fraction": processor.inverse_clamp_fraction,
        "fit_scope": processor.base.fit_scope,
        "fit_snapshot_indices": list(processor.training_indices),
        "fit_snapshot_count": len(processor.training_indices),
        "sample_count_per_channel": processor.base.sample_count_per_channel,
        "validation_indices_used_for_fit": [],
        "thermal_channel": processor.thermal_channel,
        "paper_adaptation": processor.paper_adaptation,
        "eos_conversion": processor.eos_conversion,
    }


def oracle_rows(
    dataset: Path, processor: PrototypePreprocessor, resolution: int,
) -> list[dict[str, Any]]:
    numerator = np.zeros(8, dtype=np.float64)
    denominator = np.zeros(8, dtype=np.float64)
    maximum = np.zeros(8, dtype=np.float64)
    encoded_saturation = np.zeros(8, dtype=np.float64)
    voxel_count = 0
    finite = True
    positive = True
    with h5py.File(dataset, "r") as handle:
        snapshots = handle["snapshots"]
        for index in range(169, 212):
            raw = np.asarray(snapshots[index], dtype=np.float32)
            encoded = processor.encode_numpy(raw, channel_axis=0)
            decoded = processor.decode_numpy(encoded, channel_axis=0)
            difference = decoded.astype(np.float64) - raw.astype(np.float64)
            axes = (1, 2, 3)
            numerator += np.sum(np.square(difference), axis=axes)
            denominator += np.sum(np.square(raw, dtype=np.float64), axis=axes)
            maximum = np.maximum(maximum, np.max(np.abs(difference), axis=axes))
            encoded_saturation += np.count_nonzero(
                np.abs(encoded) >= processor.gamma * processor.inverse_clamp_fraction,
                axis=axes,
            )
            voxel_count += int(np.prod(raw.shape[1:]))
            finite &= bool(np.isfinite(encoded).all() and np.isfinite(decoded).all())
            positive &= bool(np.all(decoded[3] > 0) and np.all(decoded[4] > 0))
    if not finite or not positive:
        raise FloatingPointError(f"Z{resolution} preprocessing oracle is nonfinite/nonpositive")
    relative = np.sqrt(numerator / np.maximum(denominator, 1.0e-300))
    return [
        {
            "reproduction_scope": REPRODUCTION_SCOPE,
            "resolution": resolution,
            "channel": name,
            "validation_snapshot_start": 169,
            "validation_snapshot_end_inclusive": 211,
            "validation_snapshot_count": 43,
            "physical_relative_l2": float(relative[channel]),
            "maximum_absolute_error": float(maximum[channel]),
            "encoded_inverse_limit_fraction": float(encoded_saturation[channel] / voxel_count),
            "finite": True,
            "rho_press_positive": True,
            "validation_used_for_fit": False,
        }
        for channel, name in enumerate(CHANNELS)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/stage_z/higher_resolution.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/stage_z/preprocessing"))
    parser.add_argument("--resolutions", type=int, nargs="+", default=[64, 96, 128])
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["reproduction_scope"] != REPRODUCTION_SCOPE:
        raise ValueError("Stage Z scope changed")
    available = {int(key): ROOT / value for key, value in config["data"]["resolutions"].items()}
    requested = [int(value) for value in args.resolutions]
    if not set(requested) <= set(available):
        raise ValueError("unknown Stage Z resolution")

    processors: dict[int, PrototypePreprocessor] = {}
    summaries: dict[int, dict[str, Any]] = {}
    for resolution in requested:
        dataset = available[resolution]
        checksum = sha256_file(dataset)
        if resolution == 64:
            artifact_dir = ROOT / "artifacts/stage_s/p3_normalizer_expanded"
            processor = PrototypePreprocessor.load(
                artifact_dir, spec=prototype_specs(NO_SOFTCLIP)["P3"]
            )
            processor.base.validate_compatibility(
                h5_path=dataset,
                expected_source_hdf5_checksum=checksum,
                expected_training_indices=range(169),
            )
            reused = True
        else:
            artifact_dir = args.out_dir / f"p3_{resolution}"
            print(f"fitting Z{resolution} P3 on train snapshots 0..168", flush=True)
            processor = fit_processor(dataset, resolution, checksum)
            processor.save(artifact_dir)
            reused = False
        processors[resolution] = processor
        summary = processor_summary(
            processor, resolution=resolution, dataset=dataset,
            dataset_sha256=checksum, artifact_dir=artifact_dir, reused=reused,
        )
        summaries[resolution] = summary
        atomic_json(args.out_dir / f"p3_{resolution}.json", summary)

    rows: list[dict[str, Any]] = []
    for resolution in requested:
        print(f"auditing Z{resolution} validation preprocessing oracle", flush=True)
        rows.extend(oracle_rows(available[resolution], processors[resolution], resolution))
    atomic_csv(args.out_dir / "oracle_floor_comparison.csv", rows)
    oracle_summary = {}
    for resolution in requested:
        selected = [row for row in rows if int(row["resolution"]) == resolution]
        oracle_summary[str(resolution)] = {
            "PREPROCESSING_ORACLE": "passed",
            "physical_relative_l2_arithmetic_average": float(np.mean([
                float(row["physical_relative_l2"]) for row in selected
            ])),
            "per_channel": {
                row["channel"]: row["physical_relative_l2"] for row in selected
            },
            "finite": True,
            "rho_press_positive": True,
        }
    atomic_json(args.out_dir / "oracle_floor_summary.json", {
        "schema_version": "stage-z-oracle-floor-summary-v1",
        "reproduction_scope": REPRODUCTION_SCOPE,
        "validation_indices_used_for_fit": [],
        "resolutions": oracle_summary,
    })
    print(json.dumps({
        "normalizer_sha256": {str(key): value["normalizer_sha256"] for key, value in summaries.items()},
        "oracle": oracle_summary,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
