#!/usr/bin/env python
"""Audit Stage S raw/processed expanded data without training or mutation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping

import h5py
import numpy as np

from grmhd import CHANNELS
from grmhd.dataset import sha256_file
from grmhd.stage_s_data import (
    chronological_split,
    classify_distribution_shift,
    stratified_small_pair_indices,
)


SNAPSHOT_RE = re.compile(r"\.(\d+)\.athdf$")
GRID_DATASETS = ("Levels", "LogicalLocations", "x1f", "x1v", "x2f", "x2v", "x3f", "x3v")
EXPECTED_UPSTREAM = "86a8bc7812a31b42c4f7895693cf4ac11521c066"
QUANTILES = (0.001, 0.01, 0.5, 0.99, 0.999)


def decode(values: Iterable[Any]) -> list[str]:
    return [value.decode() if isinstance(value, (bytes, np.bytes_)) else str(value) for value in values]


def snapshot_index(path: Path) -> int:
    match = SNAPSHOT_RE.search(path.name)
    if match is None:
        raise ValueError(f"Cannot parse snapshot index: {path}")
    return int(match.group(1))


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    return value


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def variable_map(handle: h5py.File) -> dict[str, tuple[str, int]]:
    result: dict[str, tuple[str, int]] = {}
    offset = 0
    for dataset, count in zip(
        decode(handle.attrs["DatasetNames"]),
        (int(value) for value in handle.attrs["NumVariables"]),
        strict=True,
    ):
        names = decode(handle.attrs["VariableNames"])[offset : offset + count]
        result.update({name: (dataset, local) for local, name in enumerate(names)})
        offset += count
    return result


def grid_signature(handle: h5py.File) -> str:
    digest = hashlib.sha256()
    for name in GRID_DATASETS:
        values = np.ascontiguousarray(handle[name][...])
        digest.update(name.encode())
        digest.update(str(values.shape).encode())
        digest.update(values.view(np.uint8))
    return digest.hexdigest()


def field_stats(
    handle: h5py.File, mapping: Mapping[str, tuple[str, int]], *, chunk_blocks: int
) -> tuple[dict[str, Any], str]:
    result: dict[str, Any] = {}
    digest = hashlib.sha256()
    for channel in CHANNELS:
        dataset_name, variable_index = mapping[channel]
        dataset = handle[dataset_name]
        minimum, maximum = math.inf, -math.inf
        total = total_square = 0.0
        finite_count = nan_count = inf_count = 0
        digest.update(channel.encode())
        for start in range(0, dataset.shape[1], chunk_blocks):
            stop = min(start + chunk_blocks, dataset.shape[1])
            raw = np.ascontiguousarray(dataset[variable_index, start:stop])
            digest.update(raw.view(np.uint8))
            values = raw.astype(np.float64)
            nan_count += int(np.isnan(values).sum())
            inf_count += int(np.isinf(values).sum())
            finite = values[np.isfinite(values)]
            if finite.size:
                minimum = min(minimum, float(finite.min()))
                maximum = max(maximum, float(finite.max()))
                total += float(finite.sum(dtype=np.float64))
                total_square += float(np.square(finite).sum(dtype=np.float64))
                finite_count += int(finite.size)
        mean = total / max(finite_count, 1)
        variance = max(total_square / max(finite_count, 1) - mean * mean, 0.0)
        result[channel] = {
            "minimum": minimum if finite_count else None,
            "maximum": maximum if finite_count else None,
            "mean": mean if finite_count else None,
            "std": math.sqrt(variance) if finite_count else None,
            "finite_count": finite_count,
            "nan_count": nan_count,
            "inf_count": inf_count,
        }
    return result, digest.hexdigest()


def audit_active_file(path: Path, *, chunk_blocks: int) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        mapping = variable_map(handle)
        missing = sorted(set(CHANNELS) - set(mapping))
        if missing:
            raise KeyError(f"{path.name} missing channels {missing}")
        stats, physical_content_sha256 = field_stats(handle, mapping, chunk_blocks=chunk_blocks)
        levels, counts = np.unique(handle["Levels"][...], return_counts=True)
        attributes = set(handle.attrs)
        code_metadata = {
            key: json_safe(handle.attrs[key])
            for key in sorted(attributes)
            if any(token in key.lower() for token in ("version", "code", "git", "commit"))
        }
        return {
            "file_path": str(path.resolve()),
            "file": path.name,
            "snapshot_index": snapshot_index(path),
            "physical_time": float(handle.attrs["Time"]),
            "num_cycles": int(handle.attrs["NumCycles"]),
            "file_size": path.stat().st_size,
            "physical_content_sha256": physical_content_sha256,
            "dataset_shapes": {
                name: list(handle[name].shape) for name in decode(handle.attrs["DatasetNames"])
            },
            "dataset_storage_order": "variable,meshblock,x3,x2,x1",
            "coordinates": decode([handle.attrs["Coordinates"]])[0],
            "root_grid_size_x1_x2_x3": [int(value) for value in handle.attrs["RootGridSize"]],
            "mesh_block_size_x1_x2_x3": [int(value) for value in handle.attrs["MeshBlockSize"]],
            "num_meshblocks": int(handle.attrs["NumMeshBlocks"]),
            "max_level": int(handle.attrs["MaxLevel"]),
            "level_distribution": {
                str(int(level)): int(count) for level, count in zip(levels, counts, strict=True)
            },
            "variable_names": decode(handle.attrs["VariableNames"]),
            "axis_mapping": {"x1": "r", "x2": "theta", "x3": "phi"},
            "coordinate_bounds": {
                "r": [float(np.min(handle["x1f"][:, 0])), float(np.max(handle["x1f"][:, -1]))],
                "theta": [float(np.min(handle["x2f"][:, 0])), float(np.max(handle["x2f"][:, -1]))],
                "phi": [float(np.min(handle["x3f"][:, 0])), float(np.max(handle["x3f"][:, -1]))],
            },
            "grid_signature": grid_signature(handle),
            "component_basis_metadata": "not_available; project treats stored spherical KS coordinate components",
            "physical_units_metadata": "not_available",
            "code_version_metadata": code_metadata or "not_available",
            "fields": stats,
            "nan_count": sum(row["nan_count"] for row in stats.values()),
            "inf_count": sum(row["inf_count"] for row in stats.values()),
            "rho_strictly_positive": bool(stats["rho"]["minimum"] > 0),
            "press_strictly_positive": bool(stats["press"]["minimum"] > 0),
        }


def flatten_manifest(record: Mapping[str, Any]) -> dict[str, Any]:
    row = {
        key: value
        for key, value in record.items()
        if key not in {"fields", "dataset_shapes", "level_distribution", "variable_names", "coordinate_bounds", "axis_mapping", "code_version_metadata"}
    }
    for key in ("dataset_shapes", "level_distribution", "variable_names", "coordinate_bounds", "axis_mapping", "code_version_metadata"):
        row[key] = json.dumps(json_safe(record[key]), sort_keys=True)
    for channel in CHANNELS:
        for statistic, value in record["fields"][channel].items():
            row[f"{channel}_{statistic}"] = value
    return row


def full_sha256(path: Path) -> str:
    return sha256_file(path)


def recycle_inventory(
    recycle_dir: Path | None, active_by_time: Mapping[float, Path]
) -> dict[str, Any]:
    if recycle_dir is None or not recycle_dir.exists():
        return {"directory": None, "files": 0, "hdf5_files": 0, "status": "not_present"}
    candidates = sorted(recycle_dir.glob("*.athdf"))
    hdf5_files = [path for path in candidates if h5py.is_hdf5(path)]
    active_hashes: dict[Path, str] = {}
    rows = []
    for index, path in enumerate(hdf5_files):
        with h5py.File(path, "r") as handle:
            time = float(handle.attrs["Time"])
        active = active_by_time.get(time)
        if active is None:
            rows.append({"file": path.name, "time": time, "active_file": None, "byte_identical": False})
            continue
        active_hashes.setdefault(active, full_sha256(active))
        candidate_hash = full_sha256(path)
        rows.append(
            {
                "file": path.name,
                "time": time,
                "active_file": active.name,
                "sha256": candidate_hash,
                "active_sha256": active_hashes[active],
                "byte_identical": candidate_hash == active_hashes[active],
            }
        )
        if (index + 1) % 10 == 0:
            print(f"[recycle {index + 1}/{len(hdf5_files)}]", flush=True)
    time_counts: dict[str, int] = {}
    for row in rows:
        key = repr(row["time"])
        time_counts[key] = time_counts.get(key, 0) + 1
    return {
        "directory": str(recycle_dir),
        "files": len(candidates),
        "non_hdf5_metadata_files": len(candidates) - len(hdf5_files),
        "hdf5_files": len(hdf5_files),
        "unique_hdf5_times": len(time_counts),
        "duplicate_copy_time_count": sum(count > 1 for count in time_counts.values()),
        "all_have_active_time_counterpart": all(row["active_file"] is not None for row in rows),
        "all_byte_identical_to_active": all(row["byte_identical"] for row in rows),
        "rows": rows,
    }


def compatibility_summary(records: list[Mapping[str, Any]]) -> dict[str, Any]:
    old = [row for row in records if row["snapshot_index"] <= 110]
    new = [row for row in records if row["snapshot_index"] >= 111]
    keys = (
        "coordinates",
        "root_grid_size_x1_x2_x3",
        "mesh_block_size_x1_x2_x3",
        "num_meshblocks",
        "max_level",
        "level_distribution",
        "variable_names",
        "dataset_shapes",
        "axis_mapping",
        "coordinate_bounds",
        "grid_signature",
    )
    checks = {
        key: len({json.dumps(json_safe(row[key]), sort_keys=True) for row in records}) == 1
        for key in keys
    }
    return {
        "old_snapshot_indices": [old[0]["snapshot_index"], old[-1]["snapshot_index"]],
        "new_snapshot_indices": [new[0]["snapshot_index"], new[-1]["snapshot_index"]],
        "checks": checks,
        "component_basis": "not explicitly stored in either interval; same continuous mad98.prim series and exact schema/grid/channel contract",
        "physical_units": "not available in either interval",
        "preprocessing_input_contract": {
            "shape_after_regrid": "N,8,Nphi,Ntheta,Nr",
            "channels": list(CHANNELS),
            "stored_components_not_cartesian": True,
            "thermal_channel": "press",
        },
        "operationally_compatible": all(checks.values()),
    }


def write_markdown(payload: Mapping[str, Any], output: Path) -> None:
    summary = payload["summary"]
    split = payload["split"]
    processed = payload.get("processed_dataset")
    lines = [
        "# Stage S expanded-data audit",
        "",
        "## Counts",
        "",
        f"- `OLD_SNAPSHOT_COUNT = {summary['old_snapshot_count']}`",
        f"- `NEW_SNAPSHOT_COUNT = {summary['new_snapshot_count']}`",
        f"- `TOTAL_UNIQUE_SNAPSHOT_COUNT = {summary['total_unique_snapshot_count']}`",
        f"- `TIME_SERIES_CONTIGUOUS = {str(summary['time_series_contiguous']).lower()}`",
        f"- `CADENCE_CONSTANT = {str(summary['cadence_constant']).lower()}`",
        f"- `GRID_LAYOUT_STATIC = {str(summary['grid_layout_static']).lower()}`",
        f"- Raw merge gate: `{summary['raw_merge_gate']}`.",
        "",
        "## Time and split",
        "",
        f"- Time range: `{summary['time_range'][0]}..{summary['time_range'][1]}`.",
        f"- Delta-t min/median/max: `{summary['dt']['minimum']}` / `{summary['dt']['median']}` / `{summary['dt']['maximum']}`.",
        f"- Train snapshots/pairs: `{split['train_snapshot_count']}` / `{split['train_pair_count']}` over `[{split['train_start']},{split['train_end']})`.",
        f"- Validation snapshots/pairs: `{split['validation_snapshot_count']}` / `{split['validation_pair_count']}` over `[{split['validation_start']},{split['validation_end']})`.",
        f"- Dropped boundary pair: `{split['dropped_boundary_pair']}`.",
        "",
        "## Compatibility",
        "",
        "- Old/new coordinate system, channel ordering, dataset shapes, root/block grids, AMR levels/layout, domain, and coordinate arrays are identical.",
        "- Component-basis and physical-unit labels are absent from both intervals. Compatibility is established operationally by the continuous file/time/cycle series and identical schema/grid contract; stored components remain spherical Kerr-Schild coordinate components.",
        "- Recycle-bin ATHDF candidates are excluded copies, not additional samples.",
    ]
    if processed is not None:
        lines.extend(
            [
                "",
                "## Processed expanded dataset",
                "",
                f"- Path: `{processed['path']}`.",
                f"- Shape/dtype: `{processed['shape']}` / `{processed['dtype']}`.",
                f"- SHA-256: `{processed['sha256']}`.",
                f"- `DISTRIBUTION_SHIFT = {payload['distribution_shift']['classification']}`.",
                "- The shift is reported, not filtered or used to remove difficult snapshots.",
            ]
        )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def raw_phase(args: argparse.Namespace) -> None:
    root = args.project_root.resolve()
    output = args.output_dir.resolve()
    raw_dir = args.raw_dir.resolve()
    files = sorted(raw_dir.glob("mad98.prim.*.athdf"), key=snapshot_index)
    if not files:
        raise FileNotFoundError(f"No active ATHDF files in {raw_dir}")
    records = []
    for index, path in enumerate(files):
        print(f"[{index + 1}/{len(files)}] {path.name}", flush=True)
        records.append(audit_active_file(path, chunk_blocks=args.chunk_blocks))
    indices = np.asarray([row["snapshot_index"] for row in records], dtype=np.int64)
    times = np.asarray([row["physical_time"] for row in records], dtype=np.float64)
    deltas = np.diff(times)
    if not len(deltas):
        raise RuntimeError("Stage S requires multiple snapshots")
    expected_indices = np.arange(indices[0], indices[-1] + 1)
    active_by_time = {row["physical_time"]: Path(row["file_path"]) for row in records}
    recycle = recycle_inventory(args.recycle_dir, active_by_time)
    compatibility = compatibility_summary(records)
    grid_static = len({row["grid_signature"] for row in records}) == 1
    strict_time = bool(np.all(deltas > 0))
    no_time_duplicates = len(times) == len(np.unique(times))
    median_dt = float(np.median(deltas))
    cadence_relative_deviation = float(np.max(np.abs(deltas - median_dt)) / abs(median_dt))
    cadence_constant = cadence_relative_deviation <= 1.0e-3
    contiguous = bool(np.array_equal(indices, expected_indices) and strict_time and no_time_duplicates and cadence_constant)
    all_finite = all(row["nan_count"] == 0 and row["inf_count"] == 0 for row in records)
    all_positive = all(row["rho_strictly_positive"] and row["press_strictly_positive"] for row in records)
    recycle_safe = recycle.get("status") == "not_present" or (
        recycle.get("all_have_active_time_counterpart") and recycle.get("all_byte_identical_to_active")
    )
    merge_safe = all((compatibility["operationally_compatible"], grid_static, contiguous, all_finite, all_positive, recycle_safe))
    split = chronological_split(len(records)).as_dict()
    split.update(
        {
            "schema_version": "stage-s-chronological-split-v1",
            "validation_fraction_requested": 0.20,
            "validation_held_out": True,
            "validation_used_for_fit": False,
            "train_snapshot_indices": list(range(split["train_start"], split["train_end"])),
            "validation_snapshot_indices": list(range(split["validation_start"], split["validation_end"])),
            "train_pair_source_indices": list(range(split["train_start"], split["train_end"] - 1)),
            "validation_pair_source_indices": list(range(split["validation_start"], split["validation_end"] - 1)),
        }
    )
    small = stratified_small_pair_indices(split["train_pair_source_indices"], count=79, seed=42)
    split["small_control"] = {
        "count": len(small),
        "seed": 42,
        "algorithm": "one seeded uniform-random adjacent pair from each of 79 non-overlapping temporal strata",
        "pair_source_indices": small,
        "uses_validation": False,
    }
    project_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    upstream_commit = subprocess.check_output(["git", "-C", "external/neuraloperator", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if upstream_commit != EXPECTED_UPSTREAM:
        raise RuntimeError("Pinned neuraloperator commit changed")
    payload = {
        "schema_version": "stage-s-expanded-data-audit-v1",
        "project_commit": project_commit,
        "upstream_commit": upstream_commit,
        "source": {
            "configured_path": str(args.raw_dir),
            "resolved_path": str(raw_dir),
            "active_pattern": "mad98.prim.*.athdf",
            "recycle_inventory": recycle,
        },
        "summary": {
            "old_snapshot_count": int(np.count_nonzero(indices <= 110)),
            "new_snapshot_count": int(np.count_nonzero(indices >= 111)),
            "total_unique_snapshot_count": len(records),
            "snapshot_index_range": [int(indices[0]), int(indices[-1])],
            "missing_snapshot_indices": sorted(set(expected_indices.tolist()) - set(indices.tolist())),
            "time_range": [float(times[0]), float(times[-1])],
            "dt": {
                "minimum": float(deltas.min()),
                "maximum": float(deltas.max()),
                "median": median_dt,
                "mean": float(deltas.mean()),
                "standard_deviation": float(deltas.std()),
                "maximum_relative_deviation_from_median": cadence_relative_deviation,
            },
            "strictly_increasing": strict_time,
            "duplicated_times": sorted(float(value) for value, count in zip(*np.unique(times, return_counts=True), strict=True) if count > 1),
            "missing_intervals": [int(i) for i in np.flatnonzero(deltas > 1.5 * median_dt)],
            "irregular_cadence_intervals": [int(i) for i in np.flatnonzero(np.abs(deltas - median_dt) > 1.0e-3 * abs(median_dt))],
            "time_series_contiguous": contiguous,
            "cadence_constant": cadence_constant,
            "grid_layout_static": grid_static,
            "all_fields_finite": all_finite,
            "rho_press_positive": all_positive,
            "raw_merge_gate": "passed" if merge_safe else "failed",
        },
        "compatibility": compatibility,
        "grid_comparison_snapshots": [0, 110, 111, 161, len(records) - 1],
        "split": split,
        "preprocessing_contract": {
            "builder": "src/build_regrid_from_athdf.py",
            "builder_sha256": sha256_file(root / "src/build_regrid_from_athdf.py"),
            "output_shape": [len(records), 8, 64, 64, 64],
            "axis_order": "N,C,Nphi,Ntheta,Nr",
            "channels": list(CHANNELS),
            "target_r": "geometric edges over [1.1,200], geometric cell centres",
            "target_theta": "linear edges over [0,pi], arithmetic cell centres",
            "target_phi": "linear-periodic edges over [0,2pi], unique arithmetic cell centres",
            "method": "nearest_leaf (finest-containing meshblock nearest cell centre)",
            "interpolation": False,
            "conservative": False,
            "divergence_preserving": False,
        },
        "records": records,
    }
    output.mkdir(parents=True, exist_ok=True)
    rows = [flatten_manifest(row) for row in records]
    with (output / "data_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(output / "expanded_data_audit.json", payload)
    write_json(output / "train_val_split.json", split)
    (output / "small_train_pairs.txt").write_text(
        "\n".join(f"{source} {source + 1}" for source in small) + "\n", encoding="utf-8"
    )
    write_markdown(payload, output / "expanded_data_audit.md")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True), flush=True)
    if not merge_safe:
        raise RuntimeError("Stage S raw merge gate failed")


def processed_segment_stats(dataset: h5py.Dataset, indices: slice) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for channel, name in enumerate(CHANNELS):
        print(f"[processed stats] {indices.start}:{indices.stop} {name}", flush=True)
        values = np.asarray(dataset[indices, channel], dtype=np.float32).reshape(-1)
        quantiles = np.quantile(values, QUANTILES)
        result[name] = {
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "mean": float(np.mean(values, dtype=np.float64)),
            "std": float(np.std(values, dtype=np.float64)),
            "q001": float(quantiles[0]),
            "q01": float(quantiles[1]),
            "q50": float(quantiles[2]),
            "q99": float(quantiles[3]),
            "q999": float(quantiles[4]),
        }
        del values
    return result


def processed_phase(args: argparse.Namespace) -> None:
    output = args.output_dir.resolve()
    audit_path = output / "expanded_data_audit.json"
    payload = json.loads(audit_path.read_text(encoding="utf-8"))
    if payload["summary"]["raw_merge_gate"] != "passed":
        raise RuntimeError("Raw merge gate did not pass")
    processed = args.processed.resolve()
    old_processed = args.old_processed.resolve()
    prefix_equal = True
    prefix_maximum_absolute_difference = 0.0
    with h5py.File(processed, "r") as handle, h5py.File(old_processed, "r") as old_handle:
        snapshots = handle["snapshots"]
        shape = list(snapshots.shape)
        channels = decode(handle["channels"][...])
        times = np.asarray(handle["times"][...], dtype=np.float64)
        metadata = {str(key): json_safe(value) for key, value in handle["metadata"].attrs.items()}
        if shape != [212, 8, 64, 64, 64] or channels != list(CHANNELS):
            raise RuntimeError(f"Expanded processed contract mismatch: {shape}, {channels}")
        if not np.all(np.diff(times) > 0):
            raise RuntimeError("Expanded processed times are not strictly increasing")
        if list(old_handle["snapshots"].shape) != [111, 8, 64, 64, 64]:
            raise RuntimeError("Canonical old processed dataset shape changed")
        for name in ("r", "theta", "phi"):
            if not np.array_equal(handle[f"coords/{name}"][...], old_handle[f"coords/{name}"][...]):
                raise RuntimeError(f"Expanded coordinate {name} changed from the old artifact")
        if not np.array_equal(times[:111], old_handle["times"][...]):
            raise RuntimeError("Expanded old-prefix times changed")
        for snapshot_index in range(111):
            current = np.asarray(snapshots[snapshot_index], dtype=np.float32)
            reference = np.asarray(old_handle["snapshots"][snapshot_index], dtype=np.float32)
            if not np.array_equal(current, reference):
                prefix_equal = False
                prefix_maximum_absolute_difference = max(
                    prefix_maximum_absolute_difference,
                    float(np.max(np.abs(current.astype(np.float64) - reference.astype(np.float64)))),
                )
        old = processed_segment_stats(snapshots, slice(0, 111))
        new = processed_segment_stats(snapshots, slice(111, 212))
        coordinates = {
            name: np.asarray(handle[f"coords/{name}"][...], dtype=np.float64).tolist()
            for name in ("r", "theta", "phi")
        }
    classification, diagnostics = classify_distribution_shift(old, new)
    payload["processed_dataset"] = {
        "path": str(processed),
        "shape": shape,
        "dtype": "float32",
        "sha256": sha256_file(processed),
        "axis_order": metadata.get("axis_order"),
        "channels": channels,
        "times_match_raw": times.tolist() == [row["physical_time"] for row in payload["records"]],
        "metadata": metadata,
        "coordinates": coordinates,
        "old_prefix_parity": {
            "reference_path": str(old_processed),
            "reference_sha256": sha256_file(old_processed),
            "snapshot_count": 111,
            "bitwise_equal": prefix_equal,
            "maximum_absolute_difference": prefix_maximum_absolute_difference,
        },
    }
    if not prefix_equal:
        raise RuntimeError("Expanded processed old prefix is not bitwise equal to the frozen 111-snapshot artifact")
    payload["grid_comparison_snapshots"] = [0, 110, 111, 161, 211]
    payload["distribution_shift"] = {
        "classification": classification,
        "rule": "fixed stage_s_data.classify_distribution_shift thresholds",
        "old_snapshot_indices": [0, 110],
        "new_snapshot_indices": [111, 211],
        "old": old,
        "new": new,
        "diagnostics": diagnostics,
        "snapshots_removed": [],
    }
    write_json(audit_path, payload)
    write_markdown(payload, output / "expanded_data_audit.md")
    print(json.dumps({"processed_dataset": payload["processed_dataset"], "distribution_shift": payload["distribution_shift"]}, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("raw", "processed"), required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--raw-dir", type=Path, default=Path("data_raw"))
    parser.add_argument("--processed", type=Path, default=Path("data_proc/grmhd_regrid_inner_r200_64_expanded.h5"))
    parser.add_argument("--old-processed", type=Path, default=Path("data_proc/grmhd_regrid_inner_r200_64.h5"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/stage_s"))
    parser.add_argument("--chunk-blocks", type=int, default=128)
    parser.add_argument(
        "--recycle-dir",
        type=Path,
        default=Path("/mnt/d/$RECYCLE.BIN/S-1-5-21-2594875378-1454575701-991183375-500"),
    )
    args = parser.parse_args()
    if args.phase == "raw":
        raw_phase(args)
    else:
        processed_phase(args)


if __name__ == "__main__":
    main()
