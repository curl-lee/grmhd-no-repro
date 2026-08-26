#!/usr/bin/env python3
"""Evaluate the same-resolution Stage Z persistence baselines on CPU."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import yaml

from grmhd import CHANNELS
from grmhd.paper_stage_n import NO_SOFTCLIP, PrototypePreprocessor, prototype_specs
from grmhd.stage_z import REPRODUCTION_SCOPE, spherical_volume_proxy_weights


ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = list(rows[0])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def finalize(numerator: np.ndarray, denominator: np.ndarray) -> dict[str, Any]:
    values = np.sqrt(numerator / np.maximum(denominator, 1.0e-300))
    return {
        "per_channel": dict(zip(CHANNELS, values.tolist(), strict=True)),
        "arithmetic_average": float(values.mean()),
        "global": float(math.sqrt(numerator.sum() / max(float(denominator.sum()), 1.0e-300))),
    }


def evaluate(dataset: Path, normalizer: Path, resolution: int) -> dict[str, Any]:
    processor = PrototypePreprocessor.load(
        normalizer, spec=prototype_specs(NO_SOFTCLIP)["P3"]
    )
    state_num = np.zeros(8, dtype=np.float64)
    state_den = np.zeros(8, dtype=np.float64)
    physical_num = np.zeros(8, dtype=np.float64)
    physical_den = np.zeros(8, dtype=np.float64)
    proxy_num = np.zeros(8, dtype=np.float64)
    proxy_den = np.zeros(8, dtype=np.float64)
    with h5py.File(dataset, "r") as handle:
        r = np.asarray(handle["coords/r"], dtype=np.float64)
        theta = np.asarray(handle["coords/theta"], dtype=np.float64)
        phi = np.asarray(handle["coords/phi"], dtype=np.float64)
        metadata = handle["metadata"].attrs
        dtheta = float(theta[1] - theta[0])
        dphi = float(phi[1] - phi[0])
        weights = spherical_volume_proxy_weights(
            r, theta, phi,
            r_bounds=(float(metadata["requested_r_min"]), float(metadata["requested_r_max"])),
            theta_bounds=(float(theta[0] - 0.5 * dtheta), float(theta[-1] + 0.5 * dtheta)),
            phi_bounds=(float(phi[0] - 0.5 * dphi), float(phi[-1] + 0.5 * dphi)),
        )
        for source in range(169, 211):
            raw_input = np.asarray(handle["snapshots"][source], dtype=np.float32)
            raw_target = np.asarray(handle["snapshots"][source + 1], dtype=np.float32)
            z_input = processor.encode_numpy(raw_input, channel_axis=0)
            z_target = processor.encode_numpy(raw_target, channel_axis=0)
            physical = processor.decode_numpy(z_input, channel_axis=0)
            state_difference = z_input.astype(np.float64) - z_target.astype(np.float64)
            physical_difference = physical.astype(np.float64) - raw_target.astype(np.float64)
            axes = (1, 2, 3)
            state_num += np.sum(np.square(state_difference), axis=axes)
            state_den += np.sum(np.square(z_target, dtype=np.float64), axis=axes)
            physical_num += np.sum(np.square(physical_difference), axis=axes)
            physical_den += np.sum(np.square(raw_target, dtype=np.float64), axis=axes)
            proxy_num += np.sum(weights[None, ...] * np.square(physical_difference), axis=axes)
            proxy_den += np.sum(weights[None, ...] * np.square(raw_target, dtype=np.float64), axis=axes)
    state = finalize(state_num, state_den)
    physical = finalize(physical_num, physical_den)
    proxy = finalize(proxy_num, proxy_den)
    return {
        "reproduction_scope": REPRODUCTION_SCOPE,
        "resolution": resolution,
        "validation_pair_sources": list(range(169, 211)),
        "validation_pair_count": 42,
        "prediction": "z_t (zero residual)",
        "normalized_state_relative_l2": state,
        "physical_relative_l2": physical,
        "spherical_coordinate_volume_proxy_relative_l2": proxy,
        "volume_proxy_semantics": "SPHERICAL_COORDINATE_VOLUME_PROXY_NOT_STRICT_KERR_SCHILD_PROPER_VOLUME",
        "residual_relative_l2": 1.0,
        "residual_cosine": 0.0,
        "shell_skill": 0.0,
        "radial_skill": 0.0,
        "finite": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/stage_z/higher_resolution.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/stage_z/comparison"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    results = {}
    for resolution in (64, 96, 128):
        summary = json.loads(
            (ROOT / f"artifacts/stage_z/preprocessing/p3_{resolution}.json").read_text(encoding="utf-8")
        )
        print(f"evaluating Z{resolution} same-resolution persistence", flush=True)
        results[str(resolution)] = evaluate(
            ROOT / config["data"]["resolutions"][resolution],
            ROOT / summary["artifact_directory"], resolution,
        )
    atomic_json(args.out_dir / "persistence_by_resolution.json", {
        "schema_version": "stage-z-persistence-v1",
        "reproduction_scope": REPRODUCTION_SCOPE,
        "resolutions": results,
    })
    rows = []
    for resolution, result in results.items():
        rows.append({
            "reproduction_scope": REPRODUCTION_SCOPE,
            "resolution": resolution,
            "normalized_state_relative_l2_average": result["normalized_state_relative_l2"]["arithmetic_average"],
            "normalized_state_relative_l2_global": result["normalized_state_relative_l2"]["global"],
            "physical_relative_l2_average": result["physical_relative_l2"]["arithmetic_average"],
            "physical_relative_l2_global": result["physical_relative_l2"]["global"],
            "spherical_volume_proxy_relative_l2_average": result["spherical_coordinate_volume_proxy_relative_l2"]["arithmetic_average"],
            "spherical_volume_proxy_relative_l2_global": result["spherical_coordinate_volume_proxy_relative_l2"]["global"],
            "residual_relative_l2": 1.0,
            "residual_cosine": 0.0,
            "shell_skill": 0.0,
            "radial_skill": 0.0,
        })
    atomic_csv(args.out_dir / "persistence_by_resolution.csv", rows)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
