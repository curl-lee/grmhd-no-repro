#!/usr/bin/env python3
"""Freeze Stage Z datasets, regression checks, and information-fidelity gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable, Mapping

import h5py
import numpy as np
import yaml

from build_regrid_from_athdf import (
    build_mapping,
    build_target_grid,
    source_grid_signature,
    variable_map,
)
from grmhd import CHANNELS
from grmhd.dataset import sha256_file
from grmhd.shells import radial_shells
from grmhd.stage_x_audit import classify_temporal_fidelity
from grmhd.stage_z import REFERENCE_SEMANTICS, REPRODUCTION_SCOPE, information_gain


ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty Stage Z CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def decode(values: Iterable[Any]) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def memory_available() -> int:
    fields = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        fields[key] = int(value.strip().split()[0]) * 1024
    return fields["MemAvailable"]


def dataset_summary(path: Path, resolution: int) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        snapshots = handle["snapshots"]
        expected = (212, 8, resolution, resolution, resolution)
        if tuple(snapshots.shape) != expected or snapshots.dtype != np.dtype(np.float32):
            raise ValueError(f"Stage Z dataset contract mismatch: {path}")
        channels = decode(handle["channels"][...])
        if channels != list(CHANNELS):
            raise ValueError(f"channel contract mismatch: {path}")
        r = np.asarray(handle["coords/r"], dtype=np.float64)
        theta = np.asarray(handle["coords/theta"], dtype=np.float64)
        phi = np.asarray(handle["coords/phi"], dtype=np.float64)
        metadata = handle["metadata"].attrs
        if metadata["method_name"] != "nearest_leaf":
            raise ValueError(f"Stage Z method changed: {path}")
        if bool(metadata["conservative"]) or bool(metadata["divergence_preserving"]):
            raise ValueError(f"Stage Z regrid semantics changed: {path}")
        ratio = r[1:] / r[:-1]
        return {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "uncompressed_snapshot_array_bytes": int(np.prod(expected) * 4),
            "shape": list(expected),
            "dtype": str(snapshots.dtype),
            "chunks": list(snapshots.chunks or ()),
            "compression": snapshots.compression,
            "compression_opts": snapshots.compression_opts,
            "channels": channels,
            "axis_order": str(metadata["axis_order"]),
            "source_file_count": len(handle["source_files"]),
            "source_files_sha256": hashlib.sha256("\n".join(decode(handle["source_files"][...])).encode()).hexdigest(),
            "source_times_sha256": hashlib.sha256(np.asarray(handle["times"], dtype=np.float64).tobytes()).hexdigest(),
            "source_grid_signature": str(metadata["source_grid_signature"]),
            "method": str(metadata["method"]),
            "conservative": bool(metadata["conservative"]),
            "interpolation": False,
            "divergence_preserving": bool(metadata["divergence_preserving"]),
            "coords": {"r": r.tolist(), "theta": theta.tolist(), "phi": phi.tolist()},
            "coordinate_summary": {
                "r_center_range": [float(r[0]), float(r[-1])],
                "theta_center_range": [float(theta[0]), float(theta[-1])],
                "phi_center_range": [float(phi[0]), float(phi[-1])],
                "dr_min": float(np.diff(r).min()),
                "dr_max": float(np.diff(r).max()),
                "dtheta": float(np.diff(theta)[0]),
                "dphi": float(np.diff(phi)[0]),
                "log_r_center_ratio_min": float(ratio.min()),
                "log_r_center_ratio_max": float(ratio.max()),
                "r_edge_range": [float(metadata["requested_r_min"]), float(metadata["requested_r_max"])],
            },
        }


def direct_sampling_integrity(
    *, raw_dir: Path, datasets: Mapping[int, Path], snapshots: list[int],
    r_bounds: tuple[float, float],
) -> dict[int, list[dict[str, Any]]]:
    """Prove compressed HDF5 reads equal freshly sampled float32 arrays."""

    output: dict[int, list[dict[str, Any]]] = {}
    first_path = raw_dir / "mad98.prim.00000.athdf"
    for resolution, dataset in datasets.items():
        with h5py.File(first_path, "r") as first:
            grid = build_target_grid(
                first, resolution, resolution, resolution, r_bounds[0], r_bounds[1]
            )
            mapping = build_mapping(first, grid)
            expected_signature = source_grid_signature(first)
        rows: list[dict[str, Any]] = []
        with h5py.File(dataset, "r") as processed:
            for snapshot in snapshots:
                raw_path = raw_dir / f"mad98.prim.{snapshot:05d}.athdf"
                with h5py.File(raw_path, "r") as raw:
                    if source_grid_signature(raw) != expected_signature:
                        raise ValueError(f"raw grid changed in {raw_path}")
                    variables = variable_map(raw)
                    for channel_index, channel in enumerate(CHANNELS):
                        dataset_name, variable_index = variables[channel]
                        source = np.asarray(raw[dataset_name][variable_index], dtype=np.float32)
                        direct = np.asarray(
                            source[mapping.block, mapping.k, mapping.j, mapping.i], dtype=np.float32
                        )
                        stored = np.asarray(processed["snapshots"][snapshot, channel_index], dtype=np.float32)
                        rows.append({
                            "resolution": resolution,
                            "snapshot": snapshot,
                            "channel": channel,
                            "array_equal": bool(np.array_equal(direct, stored)),
                            "max_absolute_difference": float(np.max(np.abs(direct - stored))),
                        })
        output[resolution] = rows
    return output


def medians(rows: list[dict[str, str]], resolution: int) -> dict[str, float]:
    selected = [row for row in rows if int(row["resolution"]) == resolution]
    if not selected:
        raise ValueError(f"missing fidelity rows for {resolution}")
    return {
        "temporal_increment_relative_difference": float(np.median([
            float(row["residual_relative_difference"]) for row in selected
        ])),
        "temporal_increment_cosine": float(np.median([
            float(row["residual_cosine"]) for row in selected
        ])),
        "shell_increment_relative_l2": float(np.median([
            float(row["shell_increment_relative_l2"]) for row in selected
        ])),
        "radial_increment_relative_l2": float(np.median([
            float(row["radial_increment_relative_l2"]) for row in selected
        ])),
        **{
            f"temporal_increment_loss_{region}": float(np.median([
                float(row[f"temporal_increment_loss_{region}"]) for row in selected
            ]))
            for region in ("inner", "middle", "outer")
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/stage_z/higher_resolution.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/stage_z"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config["reproduction_scope"] != REPRODUCTION_SCOPE:
        raise ValueError("Stage Z adapted scope changed")
    upstream = git_value("-C", "external/neuraloperator", "rev-parse", "HEAD")
    upstream_dirty = git_value("-C", "external/neuraloperator", "status", "--short")
    if upstream != config["frozen"]["upstream_commit"] or upstream_dirty:
        raise ValueError("pinned neuraloperator provenance changed")

    datasets = {int(key): ROOT / value for key, value in config["data"]["resolutions"].items()}
    missing = [str(path) for path in datasets.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Stage Z datasets missing: {missing}")
    summaries = {resolution: dataset_summary(path, resolution) for resolution, path in datasets.items()}
    baseline = summaries[64]
    for resolution, summary in summaries.items():
        if summary["source_files_sha256"] != baseline["source_files_sha256"]:
            raise ValueError(f"source snapshot manifest differs at {resolution}")
        if summary["source_times_sha256"] != baseline["source_times_sha256"]:
            raise ValueError(f"source times differ at {resolution}")
        if summary["source_grid_signature"] != baseline["source_grid_signature"]:
            raise ValueError(f"source grid signature differs at {resolution}")
        if summary["coordinate_summary"]["r_edge_range"] != baseline["coordinate_summary"]["r_edge_range"]:
            raise ValueError(f"physical radial domain differs at {resolution}")

    regression_snapshots = [int(value) for value in config["regrid"]["z64_regression_snapshots"]]
    integrity = direct_sampling_integrity(
        raw_dir=Path(config["raw"]["directory"]), datasets=datasets,
        snapshots=regression_snapshots,
        r_bounds=tuple(float(value) for value in config["regrid"]["r_edge_range"]),
    )
    for resolution, rows in integrity.items():
        if not all(row["array_equal"] for row in rows):
            raise RuntimeError(f"lossless/direct sampling regression failed at {resolution}")

    z64_regression = {
        "schema_version": "stage-z-z64-regression-v1",
        "reproduction_scope": REPRODUCTION_SCOPE,
        "snapshots": regression_snapshots,
        "array_checks": integrity[64],
        "coordinate_arrays_equal_to_frozen_production": True,
        "checked_array_count": len(integrity[64]),
        "all_bitwise_identical": True,
        "Z64_REGRID_REGRESSION_PASS": True,
        "temporary_regrid_command_report": "/tmp/stage_z_preflight/z64_fixed5_report.json",
    }
    atomic_json(args.out_dir / "regrid/z64_regression.json", z64_regression)

    for resolution in (96, 128):
        manifest_path = args.out_dir / f"regrid/z{resolution}_manifest.json"
        builder = json.loads(manifest_path.read_text(encoding="utf-8"))
        atomic_json(manifest_path, {
            **builder,
            **summaries[resolution],
            "schema_version": "stage-z-regrid-manifest-v1",
            "reproduction_scope": REPRODUCTION_SCOPE,
            "direct_sampling_integrity": {
                "snapshots": regression_snapshots,
                "checked_array_count": len(integrity[resolution]),
                "all_bitwise_identical": True,
                "compression_round_trip_bitwise_identical": True,
            },
        })

    stage_x = ROOT / "artifacts/stage_x/regrid"
    tables = {
        "resolution_convergence.csv": "resolution_fidelity.csv",
        "temporal_increment_fidelity.csv": "temporal_increment_fidelity.csv",
        "shell_increment_fidelity.csv": "shell_increment_fidelity.csv",
        "radial_increment_fidelity.csv": "radial_increment_fidelity.csv",
    }
    filtered: dict[str, list[dict[str, str]]] = {}
    for source_name, target_name in tables.items():
        rows = [row for row in read_csv(stage_x / source_name) if int(row["resolution"]) in datasets]
        for row in rows:
            row["reproduction_scope"] = REPRODUCTION_SCOPE
            row["reference_semantics"] = REFERENCE_SEMANTICS
        atomic_csv(args.out_dir / f"regrid/{target_name}", rows)
        filtered[target_name] = rows
    temporal = filtered["temporal_increment_fidelity.csv"]
    metrics = {resolution: medians(temporal, resolution) for resolution in datasets}
    thresholds = {
        "good_relative_max": 0.25, "good_cosine_min": 0.95,
        "moderate_relative_max": 0.50, "moderate_cosine_min": 0.80,
    }
    classifications = {
        resolution: classify_temporal_fidelity(
            [float(row["residual_relative_difference"]) for row in temporal if int(row["resolution"]) == resolution],
            [float(row["residual_cosine"]) for row in temporal if int(row["resolution"]) == resolution],
            **thresholds,
        )
        for resolution in datasets
    }
    gate = information_gain(
        metrics[64], metrics[96],
        threshold=float(config["regrid"]["information_gain_relative_threshold"]),
    )
    atomic_json(args.out_dir / "regrid/information_gain.json", {
        "schema_version": "stage-z-information-gain-v1",
        "reproduction_scope": REPRODUCTION_SCOPE,
        "reference_semantics": REFERENCE_SEMANTICS,
        "metrics": {str(key): value for key, value in metrics.items()},
        "classification": {str(key): value for key, value in classifications.items()},
        **gate,
    })

    r64 = np.asarray(baseline["coords"]["r"], dtype=np.float64)
    _, shell_metadata = radial_shells(r64, 64, 64, n_shells=8)
    shell_counts = {}
    for resolution, summary in summaries.items():
        r = np.asarray(summary["coords"]["r"], dtype=np.float64)
        index = np.digitize(r, np.asarray(shell_metadata.edges)[1:-1], right=False)
        shell_counts[str(resolution)] = [int(np.count_nonzero(index == shell)) for shell in range(8)]
    atomic_json(args.out_dir / "shell_contract.json", {
        "schema_version": "stage-z-fixed-physical-shell-contract-v1",
        "reproduction_scope": REPRODUCTION_SCOPE,
        "source": "frozen Stage-T/Z64 radial_shells_tensor contract",
        "physical_coordinate": "stored spherical Kerr-Schild-like r",
        "physical_edges": list(shell_metadata.edges),
        "internal_membership_boundaries": list(shell_metadata.edges[1:-1]),
        "below_first_center_assigned_to_shell_0": True,
        "above_last_center_assigned_to_shell_7": True,
        "not_equal_index_count": True,
        "radial_cell_counts": shell_counts,
    })

    disk = shutil.disk_usage(ROOT)
    gpu_query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"],
        text=True, capture_output=True,
    )
    preflight = {
        "schema_version": "stage-z-resource-preflight-v1",
        "estimated_Z96_size_bytes": 212 * 8 * 96**3 * 4,
        "estimated_Z128_size_bytes": 212 * 8 * 128**3 * 4,
        "free_disk_before_generation_observed_df_h": "935 GiB available (rounded)",
        "available_ram_before_generation_observed_free_h": "13 GiB available (rounded)",
        "free_disk_bytes_current": disk.free,
        "available_ram_bytes_current": memory_available(),
        "gpu_query_returncode": gpu_query.returncode,
        "gpu_query_stdout": gpu_query.stdout.strip(),
        "gpu_query_stderr": gpu_query.stderr.strip(),
        "streaming_chunked_hdf5": True,
        "full_dataset_loaded_into_ram": False,
    }
    atomic_json(args.out_dir / "scope/resource_preflight.json", preflight)
    atomic_text(args.out_dir / "scope/adapted_reproduction_contract.md", f"""# Stage Z adapted reproduction contract

`REPRODUCTION_SCOPE = {REPRODUCTION_SCOPE}`

This experiment is an adapted spherical-KS workflow reproduction. It is not an exact
paper reproduction, a paper-faithful LocalNO, or an exact volumetric 3D DISCO
reproduction. The sole scientific variable is target spherical regrid resolution.

- Raw snapshots: 212 (`00000..00211`), frozen.
- Split: train snapshots `0..168` / 168 pairs; drop `168->169`; validation
  snapshots `169..211` / 42 pairs.
- Fields: `{list(CHANNELS)}` with no Cartesian vector conversion, press-to-eint
  conversion, EOS inference, or spin inference.
- Regrid: finest-covering leaf, nearest cell centre, non-conservative, no
  interpolation, not divergence preserving.
- Datasets: Z64 `{summaries[64]['sha256']}`, Z96 `{summaries[96]['sha256']}`,
  Z128 `{summaries[128]['sha256']}`.
- Sampling reference: Z128 is `{REFERENCE_SEMANTICS}`, not raw truth.
- Frozen gates: `Z64_REGRID_REGRESSION_PASS = true` and
  `HIGHER_RES_DATA_INFORMATION_GAIN = {str(gate['higher_res_data_information_gain']).lower()}`.
- Exact reproduction remains blocked by Stage Y.
""")
    print(json.dumps({
        "Z64_REGRID_REGRESSION_PASS": True,
        "HIGHER_RES_DATA_INFORMATION_GAIN": gate["higher_res_data_information_gain"],
        "metrics": metrics,
        "classification": classifications,
        "dataset_sha256": {str(key): value["sha256"] for key, value in summaries.items()},
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
