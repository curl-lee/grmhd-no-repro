#!/usr/bin/env python
"""Stream a complete, numerically ordered audit of GRMHD ATHDF snapshots."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from grmhd import CHANNELS


SNAPSHOT_NUMBER = re.compile(r"\.(\d+)\.athdf$")


def decode_strings(values: Iterable[Any]) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


def snapshot_number(path: Path) -> int:
    match = SNAPSHOT_NUMBER.search(path.name)
    if match is None:
        raise ValueError(f"Cannot extract numeric snapshot number from {path.name}")
    return int(match.group(1))


def sorted_snapshot_files(raw_dir: Path, pattern: str) -> list[Path]:
    return sorted(raw_dir.glob(pattern), key=lambda path: (snapshot_number(path), path.name))


def variable_map(handle: h5py.File) -> dict[str, tuple[str, int]]:
    datasets = decode_strings(handle.attrs["DatasetNames"])
    counts = [int(value) for value in handle.attrs["NumVariables"]]
    variables = decode_strings(handle.attrs["VariableNames"])
    mapping: dict[str, tuple[str, int]] = {}
    offset = 0
    for dataset, count in zip(datasets, counts, strict=True):
        for local_index, variable in enumerate(variables[offset : offset + count]):
            mapping[variable] = (dataset, local_index)
        offset += count
    if offset != len(variables):
        raise ValueError("NumVariables does not cover VariableNames")
    return mapping


def grid_signature(handle: h5py.File) -> str:
    digest = hashlib.sha256()
    for name in ("Levels", "LogicalLocations", "x1f", "x1v", "x2f", "x2v", "x3f", "x3v"):
        dataset = handle[name]
        digest.update(name.encode())
        digest.update(str(dataset.shape).encode())
        digest.update(np.ascontiguousarray(dataset[...]).view(np.uint8))
    return digest.hexdigest()


def streaming_stats(
    dataset: h5py.Dataset,
    variable_index: int,
    chunk_blocks: int,
    radial_centres: np.ndarray,
    inner_r_max: float,
) -> dict[str, float | int]:
    nblocks = dataset.shape[1]
    minimum, maximum = math.inf, -math.inf
    total = total_sq = 0.0
    finite_count = nan_count = inf_count = 0
    inner_minimum, inner_maximum = math.inf, -math.inf
    inner_total = inner_total_sq = 0.0
    inner_count = 0
    for start in range(0, nblocks, chunk_blocks):
        stop = min(start + chunk_blocks, nblocks)
        values = np.asarray(dataset[variable_index, start:stop], dtype=np.float64)
        nan_count += int(np.isnan(values).sum())
        inf_count += int(np.isinf(values).sum())
        finite = values[np.isfinite(values)]
        if finite.size:
            minimum = min(minimum, float(finite.min()))
            maximum = max(maximum, float(finite.max()))
            total += float(finite.sum(dtype=np.float64))
            total_sq += float(np.square(finite).sum(dtype=np.float64))
            finite_count += int(finite.size)
        radial_mask = radial_centres[start:stop] <= inner_r_max
        expanded_mask = np.broadcast_to(
            radial_mask[:, None, None, :], values.shape
        )
        inner = values[expanded_mask & np.isfinite(values)]
        if inner.size:
            inner_minimum = min(inner_minimum, float(inner.min()))
            inner_maximum = max(inner_maximum, float(inner.max()))
            inner_total += float(inner.sum(dtype=np.float64))
            inner_total_sq += float(np.square(inner).sum(dtype=np.float64))
            inner_count += int(inner.size)
    mean = total / finite_count if finite_count else math.nan
    variance = max(total_sq / finite_count - mean * mean, 0.0) if finite_count else math.nan
    inner_mean = inner_total / inner_count if inner_count else math.nan
    inner_variance = (
        max(inner_total_sq / inner_count - inner_mean * inner_mean, 0.0)
        if inner_count
        else math.nan
    )
    return {
        "min": minimum if finite_count else math.nan,
        "max": maximum if finite_count else math.nan,
        "mean": mean,
        "std": math.sqrt(variance),
        "max_abs": max(abs(minimum), abs(maximum)) if finite_count else math.nan,
        "finite_count": finite_count,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "inner_r_max": float(inner_r_max),
        "inner_min": inner_minimum if inner_count else math.nan,
        "inner_max": inner_maximum if inner_count else math.nan,
        "inner_mean": inner_mean,
        "inner_std": math.sqrt(inner_variance),
        "inner_max_abs": (
            max(abs(inner_minimum), abs(inner_maximum)) if inner_count else math.nan
        ),
        "inner_count": inner_count,
    }


def longest_constant_dt_run(times: np.ndarray, rtol: float = 1.0e-3) -> dict[str, Any]:
    if len(times) < 2:
        return {"start_index": 0, "stop_index_exclusive": len(times), "length": len(times)}
    deltas = np.diff(times)
    positive = deltas[deltas > 0]
    if not len(positive):
        return {"start_index": 0, "stop_index_exclusive": 1, "length": 1}
    scale = max(float(np.median(positive)) * rtol, 1.0e-12)
    keys = np.rint(deltas / scale).astype(np.int64)
    modal_key = Counter(keys.tolist()).most_common(1)[0][0]
    target = float(np.median(deltas[keys == modal_key]))
    matches = np.isclose(deltas, target, rtol=rtol, atol=max(abs(target) * 1e-6, 1e-12))
    best_start = current_start = 0
    best_edges = current_edges = 0
    for edge, match in enumerate(matches):
        if match:
            if current_edges == 0:
                current_start = edge
            current_edges += 1
            if current_edges > best_edges:
                best_start, best_edges = current_start, current_edges
        else:
            current_edges = 0
    return {
        "target_dt": target,
        "start_index": best_start,
        "stop_index_exclusive": best_start + best_edges + 1,
        "length": best_edges + 1,
    }


def audit_file(
    path: Path, chunk_blocks: int, inner_r_max: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    with h5py.File(path, "r") as handle:
        variables = decode_strings(handle.attrs["VariableNames"])
        mapping = variable_map(handle)
        missing = [channel for channel in CHANNELS if channel not in mapping]
        if missing:
            raise KeyError(f"{path.name}: missing variables {missing}")
        datasets = {
            name: tuple(int(value) for value in handle[name].shape)
            for name in decode_strings(handle.attrs["DatasetNames"])
        }
        levels, counts = np.unique(handle["Levels"][...], return_counts=True)
        radial_centres = np.asarray(handle["x1v"][...], dtype=np.float64)
        stats = {
            channel: streaming_stats(
                handle[dataset], index, chunk_blocks, radial_centres, inner_r_max
            )
            for channel, (dataset, index) in mapping.items()
            if channel in CHANNELS
        }
        coordinates = handle.attrs["Coordinates"]
        coordinates = (
            coordinates.decode() if isinstance(coordinates, (bytes, np.bytes_)) else str(coordinates)
        )
        record: dict[str, Any] = {
            "file": path.name,
            "file_size": path.stat().st_size,
            "snapshot_number": snapshot_number(path),
            "time": float(handle.attrs["Time"]),
            "dt": None,
            "num_meshblocks": int(handle.attrs.get("NumMeshBlocks", handle["Levels"].shape[0])),
            "max_level": int(handle.attrs.get("MaxLevel", int(np.max(levels)))),
            "level_distribution": {
                str(int(k)): int(v) for k, v in zip(levels, counts, strict=True)
            },
            "coordinates": coordinates,
            "variables": variables,
            "dataset_shapes": datasets,
            "grid_signature": grid_signature(handle),
            "nan_count": sum(int(value["nan_count"]) for value in stats.values()),
            "inf_count": sum(int(value["inf_count"]) for value in stats.values()),
            "rho_strictly_positive": bool(stats["rho"]["min"] > 0),
            "press_strictly_positive": bool(stats["press"]["min"] > 0),
            "stats": stats,
        }
        record["inner_activity"] = {
            "rho_rms": math.sqrt(stats["rho"]["inner_std"] ** 2 + stats["rho"]["inner_mean"] ** 2),
            "press_rms": math.sqrt(stats["press"]["inner_std"] ** 2 + stats["press"]["inner_mean"] ** 2),
            "stored_B_component_rms": math.sqrt(
                sum(stats[channel]["inner_std"] ** 2 + stats[channel]["inner_mean"] ** 2 for channel in CHANNELS[:3])
            ),
            "stored_v_component_rms": math.sqrt(
                sum(stats[channel]["inner_std"] ** 2 + stats[channel]["inner_mean"] ** 2 for channel in CHANNELS[5:])
            ),
        }
        reference = {
            "variables": variables,
            "dataset_shapes": datasets,
            "coordinates": coordinates,
            "grid_signature": record["grid_signature"],
            "level_distribution": record["level_distribution"],
        }
        return record, reference


def normalized_residuals(
    previous: Path, current: Path, chunk_blocks: int
) -> tuple[float, dict[str, float]]:
    per_channel: dict[str, float] = {}
    with h5py.File(previous, "r") as before, h5py.File(current, "r") as after:
        before_map, after_map = variable_map(before), variable_map(after)
        for channel in CHANNELS:
            before_dataset, before_index = before_map[channel]
            after_dataset, after_index = after_map[channel]
            numerator = denominator = after_denominator = 0.0
            count = 0
            blocks = before[before_dataset].shape[1]
            for start in range(0, blocks, chunk_blocks):
                stop = min(start + chunk_blocks, blocks)
                left = np.asarray(before[before_dataset][before_index, start:stop], dtype=np.float64)
                right = np.asarray(after[after_dataset][after_index, start:stop], dtype=np.float64)
                finite = np.isfinite(left) & np.isfinite(right)
                delta = right[finite] - left[finite]
                numerator += float(np.square(delta).sum(dtype=np.float64))
                denominator += float(np.square(left[finite]).sum(dtype=np.float64))
                after_denominator += float(np.square(right[finite]).sum(dtype=np.float64))
                count += int(finite.sum())
            before_rms = math.sqrt(denominator / max(count, 1))
            after_rms = math.sqrt(after_denominator / max(count, 1))
            # A channel can be identically zero at the first snapshot (Bcc3 in
            # this dataset). Use a symmetric scale so zero-to-nonzero evolution
            # remains finite instead of becoming a denominator artefact.
            per_channel[channel] = math.sqrt(numerator / max(count, 1)) / max(
                before_rms, after_rms, np.finfo(float).tiny
            )
    return float(np.mean(list(per_channel.values()))), per_channel


def robust_anomaly_flags(records: list[dict[str, Any]], threshold: float = 8.0) -> None:
    for record in records:
        record["anomaly_flags"] = []
    for channel in CHANNELS:
        for statistic in ("mean", "std", "max_abs"):
            values = np.asarray([record["stats"][channel][statistic] for record in records])
            centre = float(np.median(values))
            mad = float(np.median(np.abs(values - centre)))
            if mad <= np.finfo(float).eps:
                continue
            score = np.abs(values - centre) / (1.4826 * mad)
            for index in np.flatnonzero(score > threshold):
                records[int(index)]["anomaly_flags"].append(
                    f"{channel}_{statistic}_robust_z={score[index]:.3g}"
                )


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "file": record["file"],
        "snapshot_number": record["snapshot_number"],
        "time": record["time"],
        "dt": record["dt"],
        "num_meshblocks": record["num_meshblocks"],
        "max_level": record["max_level"],
        "level_distribution": json.dumps(record["level_distribution"], sort_keys=True),
        "variables_consistent": record["variables_consistent"],
        "shapes_consistent": record["shapes_consistent"],
        "coordinates_consistent": record["coordinates_consistent"],
        "levels_consistent": record["levels_consistent"],
        "grid_consistent": record["grid_consistent"],
        "nan_count": record["nan_count"],
        "inf_count": record["inf_count"],
        "rho_strictly_positive": record["rho_strictly_positive"],
        "press_strictly_positive": record["press_strictly_positive"],
        "normalized_residual_from_previous": record.get("normalized_residual_from_previous"),
        "anomaly_flags": ";".join(record["anomaly_flags"]),
        "exclusion_reasons": ";".join(record["exclusion_reasons"]),
    }
    for channel in CHANNELS:
        for statistic in (
            "min", "max", "mean", "std", "max_abs", "nan_count", "inf_count",
            "inner_mean", "inner_std", "inner_max_abs",
        ):
            row[f"{channel}_{statistic}"] = record["stats"][channel][statistic]
        row[f"{channel}_normalized_residual"] = record.get(
            "normalized_residual_by_channel", {}
        ).get(channel)
    row.update({f"inner_{key}": value for key, value in record["inner_activity"].items()})
    return row


def segment_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    segments = np.array_split(np.arange(len(records)), 3)
    names = ("early", "middle", "late")
    result: dict[str, Any] = {}
    for name, indices in zip(names, segments, strict=True):
        result[name] = {
            "indices": [int(indices[0]), int(indices[-1]) + 1],
            "time_range": [records[int(indices[0])]["time"], records[int(indices[-1])]["time"]],
            "channels": {
                channel: {
                    statistic: float(np.mean([records[int(i)]["stats"][channel][statistic] for i in indices]))
                    for statistic in ("mean", "std", "max_abs", "inner_std", "inner_max_abs")
                }
                for channel in CHANNELS
            },
        }
    return result


def write_time_sequence(records: list[dict[str, Any]], dt_breakpoints: set[int], path: Path) -> None:
    fields = [
        "sequence_index", "file", "snapshot_number", "number_delta", "time", "dt",
        "time_strictly_increasing_from_previous", "duplicate_time", "dt_breakpoint",
        "normalized_residual_from_previous", "anomaly_flags", "exclusion_reasons",
    ]
    times = [record["time"] for record in records]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, record in enumerate(records):
            writer.writerow({
                "sequence_index": index,
                "file": record["file"],
                "snapshot_number": record["snapshot_number"],
                "number_delta": None if index == 0 else record["snapshot_number"] - records[index - 1]["snapshot_number"],
                "time": record["time"],
                "dt": record["dt"],
                "time_strictly_increasing_from_previous": index == 0 or record["time"] > records[index - 1]["time"],
                "duplicate_time": times.count(record["time"]) > 1,
                "dt_breakpoint": index in dt_breakpoints,
                "normalized_residual_from_previous": record.get("normalized_residual_from_previous"),
                "anomaly_flags": ";".join(record["anomaly_flags"]),
                "exclusion_reasons": ";".join(record["exclusion_reasons"]),
            })


def plot_temporal_diagnostics(rows: list[dict[str, Any]], path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str((path.parent.parent / ".matplotlib").resolve()))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.asarray([row["time"] for row in rows])
    fig, axes = plt.subplots(3, 2, figsize=(15, 13), constrained_layout=True)
    for channel, style in (("rho", "-"), ("press", "--")):
        axes[0, 0].plot(times, [row[f"{channel}_mean"] for row in rows], style, label=f"{channel} mean")
        axes[0, 1].plot(times, [row[f"{channel}_std"] for row in rows], style, label=f"{channel} std")
        axes[0, 1].plot(times, [row[f"{channel}_max"] for row in rows], style, alpha=0.55, label=f"{channel} max")
    for channel in CHANNELS[:3]:
        axes[1, 0].plot(times, [row[f"{channel}_std"] for row in rows], label=f"{channel} std")
        axes[1, 0].plot(times, [row[f"{channel}_max_abs"] for row in rows], alpha=0.45, label=f"{channel} maxabs")
    for channel in CHANNELS[5:]:
        axes[1, 1].plot(times, [row[f"{channel}_std"] for row in rows], label=f"{channel} std")
        axes[1, 1].plot(times, [row[f"{channel}_max_abs"] for row in rows], alpha=0.45, label=f"{channel} maxabs")
    for key in ("rho_rms", "press_rms", "stored_B_component_rms", "stored_v_component_rms"):
        axes[2, 0].plot(times, [row[f"inner_{key}"] for row in rows], label=key)
    axes[2, 1].plot(times, [row["normalized_residual_from_previous"] for row in rows], label="mean channel-relative residual")
    for axis, title in zip(axes.flat, (
        "rho/press means", "rho/press std and max", "stored B components", "stored velocity components",
        "inner r activity RMS", "adjacent normalized residual",
    ), strict=True):
        axis.set_title(title)
        axis.set_xlabel("Time")
        axis.set_yscale("symlog", linthresh=1e-8)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7, ncol=2)
    fig.suptitle("GRMHD temporal diagnostics (sampled coordinate components; not metric-weighted)")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_summary_markdown(summary: dict[str, Any], path: Path) -> None:
    exclusions = summary["manifest"]["excluded"]
    anomalies = summary["anomaly_snapshot_candidates"]
    lines = [
        "# GRMHD ATHDF audit summary",
        "",
        f"- Snapshots audited: {summary['snapshot_count']}",
        f"- Qualified snapshots: {len(summary['manifest']['included'])}",
        f"- Numeric range: {summary['snapshot_number_start']}..{summary['snapshot_number_end']}",
        f"- Missing numbers: {summary['missing_snapshot_numbers']}",
        f"- Duplicate numbers: {summary['duplicate_snapshot_numbers']}",
        f"- Time range: {summary['time_start']}..{summary['time_end']}",
        f"- Time strictly increasing: {summary['time_strictly_increasing']}",
        f"- Duplicate times: {summary['duplicate_times']}",
        f"- Delta-t min/max/mean/std: {summary['dt_min']} / {summary['dt_max']} / {summary['dt_mean']} / {summary['dt_std']}",
        f"- Delta-t breakpoint target indices: {summary['dt_breakpoint_target_indices']}",
        f"- Variables/shapes/coordinates/levels/full grids consistent: {summary['all_variables_consistent']} / {summary['all_shapes_consistent']} / {summary['all_coordinates_consistent']} / {summary['all_levels_consistent']} / {summary['all_grids_identical']}",
        f"- Total NaN/Inf: {summary['total_nan_count']} / {summary['total_inf_count']}",
        f"- rho/press strictly positive in every file: {summary['all_rho_strictly_positive']} / {summary['all_press_strictly_positive']}",
        f"- Bcc3 first nonzero sequence index: {summary['bcc3']['first_nonzero_index']}",
        f"- Exclusions: {json.dumps(exclusions, ensure_ascii=False)}",
        f"- Statistical anomaly candidates (not automatically excluded): {json.dumps(anomalies, ensure_ascii=False)}",
        "",
        "The early/middle/late and inner-r diagnostics are sampled coordinate-component statistics, not metric-weighted physical integrals. The late100 window is therefore described only as a late-window experiment, not as a proven steady-state dataset.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/audit_111"))
    parser.add_argument("--pattern", default="mad98.prim.*.athdf")
    parser.add_argument("--max_files", type=int)
    parser.add_argument("--chunk_blocks", type=int, default=128)
    parser.add_argument("--inner_r_max", type=float, default=200.0)
    args = parser.parse_args()

    files = sorted_snapshot_files(args.raw_dir, args.pattern)
    if args.max_files is not None:
        files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No files matching {args.pattern} in {args.raw_dir}")

    records: list[dict[str, Any]] = []
    reference: dict[str, Any] | None = None
    read_errors: dict[str, str] = {}
    qualified_files: list[Path] = []
    for index, path in enumerate(files):
        print(f"[{index + 1}/{len(files)}] auditing {path.name}", flush=True)
        try:
            record, current = audit_file(path, args.chunk_blocks, args.inner_r_max)
        except Exception as exc:
            read_errors[path.name] = repr(exc)
            continue
        if reference is None:
            reference = current
        record["variables_consistent"] = current["variables"] == reference["variables"]
        record["shapes_consistent"] = current["dataset_shapes"] == reference["dataset_shapes"]
        record["coordinates_consistent"] = current["coordinates"] == reference["coordinates"]
        record["levels_consistent"] = current["level_distribution"] == reference["level_distribution"]
        record["grid_consistent"] = current["grid_signature"] == reference["grid_signature"]
        records.append(record)
        qualified_files.append(path)
    if not records:
        raise RuntimeError(f"No readable snapshots; errors={read_errors}")

    times = np.asarray([record["time"] for record in records], dtype=np.float64)
    numbers = np.asarray([record["snapshot_number"] for record in records], dtype=np.int64)
    deltas = np.diff(times)
    for index in range(1, len(records)):
        records[index]["dt"] = float(deltas[index - 1])
        residual, by_channel = normalized_residuals(
            qualified_files[index - 1], qualified_files[index], args.chunk_blocks
        )
        records[index]["normalized_residual_from_previous"] = residual
        records[index]["normalized_residual_by_channel"] = by_channel
        print(f"[{index}/{len(records)-1}] residual {records[index]['file']}: {residual:.6g}", flush=True)
    records[0]["normalized_residual_from_previous"] = None
    records[0]["normalized_residual_by_channel"] = {}
    robust_anomaly_flags(records)

    median_dt = float(np.median(deltas)) if len(deltas) else None
    dt_tolerance = max(abs(median_dt or 0.0) * 1.0e-3, 1.0e-12)
    dt_breakpoints = {
        int(index + 1)
        for index, delta in enumerate(deltas)
        if abs(float(delta) - float(median_dt)) > dt_tolerance
    }
    duplicate_number_values = sorted(
        int(value) for value, count in Counter(numbers.tolist()).items() if count > 1
    )
    expected_numbers = set(range(int(numbers.min()), int(numbers.max()) + 1))
    missing_numbers = sorted(expected_numbers - set(int(value) for value in numbers))
    unique_times, time_counts = np.unique(times, return_counts=True)
    duplicate_times = [float(value) for value in unique_times[time_counts > 1]]

    exclusions: dict[str, list[str]] = {name: [reason] for name, reason in read_errors.items()}
    for index, record in enumerate(records):
        reasons: list[str] = []
        for key in ("variables_consistent", "shapes_consistent", "coordinates_consistent", "levels_consistent", "grid_consistent"):
            if not record[key]:
                reasons.append(key.replace("_consistent", " mismatch"))
        if record["nan_count"] or record["inf_count"]:
            reasons.append("NaN/Inf present")
        if not record["rho_strictly_positive"]:
            reasons.append("rho is not strictly positive")
        if not record["press_strictly_positive"]:
            reasons.append("press is not strictly positive")
        if index and record["time"] <= records[index - 1]["time"]:
            reasons.append("time is not strictly increasing")
        if record["snapshot_number"] in duplicate_number_values:
            reasons.append("duplicate snapshot number")
        record["exclusion_reasons"] = reasons
        if reasons:
            exclusions[record["file"]] = reasons

    bcc3_max = np.asarray([record["stats"]["Bcc3"]["max_abs"] for record in records])
    summary = {
        "snapshot_count": len(records),
        "unreadable_file_count": len(read_errors),
        "snapshot_number_start": int(numbers.min()),
        "snapshot_number_end": int(numbers.max()),
        "missing_snapshot_numbers": missing_numbers,
        "duplicate_snapshot_numbers": duplicate_number_values,
        "time_start": float(times[0]),
        "time_end": float(times[-1]),
        "time_strictly_increasing": bool(np.all(deltas > 0)),
        "duplicate_times": duplicate_times,
        "dt_min": float(deltas.min()) if len(deltas) else None,
        "dt_max": float(deltas.max()) if len(deltas) else None,
        "dt_mean": float(deltas.mean()) if len(deltas) else None,
        "dt_std": float(deltas.std()) if len(deltas) else None,
        "dt_median": median_dt,
        "dt_breakpoint_rule": "abs(dt-median_dt) > 1e-3*abs(median_dt)",
        "dt_breakpoint_target_indices": sorted(dt_breakpoints),
        "recommended_fixed_dt_subsequence": longest_constant_dt_run(times),
        "all_variables_consistent": all(r["variables_consistent"] for r in records),
        "all_shapes_consistent": all(r["shapes_consistent"] for r in records),
        "all_coordinates_consistent": all(r["coordinates_consistent"] for r in records),
        "all_levels_consistent": all(r["levels_consistent"] for r in records),
        "all_grids_identical": all(r["grid_consistent"] for r in records),
        "total_nan_count": sum(r["nan_count"] for r in records),
        "total_inf_count": sum(r["inf_count"] for r in records),
        "all_rho_strictly_positive": all(r["rho_strictly_positive"] for r in records),
        "all_press_strictly_positive": all(r["press_strictly_positive"] for r in records),
        "bcc3": {
            "initial_max_abs": float(bcc3_max[0]),
            "nonzero_snapshot_count": int(np.count_nonzero(bcc3_max > 0)),
            "first_nonzero_index": int(np.flatnonzero(bcc3_max > 0)[0]) if np.any(bcc3_max > 0) else None,
            "max_abs_over_all_snapshots": float(bcc3_max.max()),
        },
        "inner_r_max": args.inner_r_max,
        "early_middle_late": segment_statistics(records),
        "anomaly_snapshot_candidates": {
            record["file"]: record["anomaly_flags"] for record in records if record["anomaly_flags"]
        },
        "manifest": {
            "included": [record["file"] for record in records if not record["exclusion_reasons"]],
            "excluded": exclusions,
        },
        "interpretation_warning": (
            "Statistics and inner-r activity use stored coordinate components and sampled cells; "
            "they are not metric/volume-weighted invariants and do not establish steady state."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = [flatten_record(record) for record in records]
    with (args.out_dir / "data_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.out_dir / "data_audit.json").write_text(
        json.dumps({"summary": summary, "records": records}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_time_sequence(records, dt_breakpoints, args.out_dir / "time_sequence.csv")
    with (args.out_dir / "temporal_diagnostics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    plot_temporal_diagnostics(rows, args.out_dir / "temporal_diagnostics.png")
    write_summary_markdown(summary, args.out_dir / "summary.md")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote complete audit to {args.out_dir}")


if __name__ == "__main__":
    main()
