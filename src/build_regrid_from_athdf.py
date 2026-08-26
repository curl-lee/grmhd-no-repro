#!/usr/bin/env python
"""Leaf-local AMR-to-uniform regridding for GRMHD ATHDF snapshots.

The output axis order is ``(snapshot, channel, phi, theta, r)``.  The mapping
selects the finest leaf meshblock that contains each target point. Available
methods are nearest cell centre and trilinear interpolation strictly within the
selected block. These are data-preparation methods: neither is conservative or
divergence preserving.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np

from grmhd import CHANNELS


SNAPSHOT_NUMBER = re.compile(r"\.(\d+)\.athdf$")


def numeric_snapshot_key(path: Path) -> tuple[int, str]:
    match = SNAPSHOT_NUMBER.search(path.name)
    if match is None:
        raise ValueError(f"Cannot extract numeric snapshot number from {path.name}")
    return int(match.group(1)), path.name


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def decode_strings(values: Iterable[Any]) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in values
    ]


def load_audit_selection(
    path: Path,
) -> tuple[set[str], dict[str, Any], dict[str, float]]:
    """Load either the legacy raw-audit or Stage S expanded-audit schema."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    summary = payload["summary"]
    records = payload["records"]
    if "manifest" in summary:
        included = set(summary["manifest"]["included"])
        excluded = dict(summary["manifest"]["excluded"])
    else:
        if summary.get("raw_merge_gate") != "passed":
            raise ValueError("Stage S raw audit merge gate did not pass")
        included = {str(record["file"]) for record in records}
        excluded = {}
    expected_times = {
        str(record["file"]): float(
            record["time"] if "time" in record else record["physical_time"]
        )
        for record in records
        if str(record["file"]) in included
    }
    return included, excluded, expected_times


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
    return mapping


def source_grid_signature(handle: h5py.File) -> str:
    digest = hashlib.sha256()
    for name in ("Levels", "LogicalLocations", "x1f", "x2f", "x3f"):
        values = np.ascontiguousarray(handle[name][...])
        digest.update(name.encode())
        digest.update(str(values.shape).encode())
        digest.update(values.view(np.uint8))
    return digest.hexdigest()


@dataclass(frozen=True)
class TargetGrid:
    r: np.ndarray
    theta: np.ndarray
    phi: np.ndarray
    r_edges: np.ndarray
    theta_edges: np.ndarray
    phi_edges: np.ndarray


@dataclass(frozen=True)
class AMRMapping:
    block: np.ndarray
    k: np.ndarray
    j: np.ndarray
    i: np.ndarray
    source_signature: str


@dataclass(frozen=True)
class AMRTrilinearMapping:
    block: np.ndarray
    k0: np.ndarray
    k1: np.ndarray
    j0: np.ndarray
    j1: np.ndarray
    i0: np.ndarray
    i1: np.ndarray
    wk: np.ndarray
    wj: np.ndarray
    wi: np.ndarray
    source_signature: str


@dataclass(frozen=True)
class TargetBinAverageMapping:
    source_flat_indices: np.ndarray
    target_flat_indices: np.ndarray
    coordinate_weights: np.ndarray
    target_weight_sums: np.ndarray
    nearest_fallback: AMRMapping
    source_signature: str


def build_target_grid(
    handle: h5py.File,
    nr: int,
    ntheta: int,
    nphi: int,
    r_min: float | None,
    r_max: float | None,
) -> TargetGrid:
    source_r_min = float(np.min(handle["x1f"][:, 0]))
    source_r_max = float(np.max(handle["x1f"][:, -1]))
    r_min = source_r_min if r_min is None else float(r_min)
    r_max = source_r_max if r_max is None else float(r_max)
    if not (0 < r_min < r_max):
        raise ValueError(f"Require 0 < r_min < r_max, received {r_min}, {r_max}")
    if r_min < source_r_min - 1e-6 or r_max > source_r_max + 1e-4:
        raise ValueError(
            f"Requested radial range [{r_min}, {r_max}] lies outside source "
            f"[{source_r_min}, {source_r_max}]"
        )
    theta_min = float(np.min(handle["x2f"][:, 0]))
    theta_max = float(np.max(handle["x2f"][:, -1]))
    phi_min = float(np.min(handle["x3f"][:, 0]))
    phi_max = float(np.max(handle["x3f"][:, -1]))

    r_edges = np.geomspace(r_min, r_max, nr + 1, dtype=np.float64)
    r = np.sqrt(r_edges[:-1] * r_edges[1:])
    theta_edges = np.linspace(theta_min, theta_max, ntheta + 1, dtype=np.float64)
    theta = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    # endpoint=False behaviour through cell-centred linear edges gives one unique
    # sample per periodic cell and never duplicates the seam.
    phi_edges = np.linspace(phi_min, phi_max, nphi + 1, dtype=np.float64)
    phi = 0.5 * (phi_edges[:-1] + phi_edges[1:])
    return TargetGrid(r, theta, phi, r_edges, theta_edges, phi_edges)


def interval_indices(values: np.ndarray, lower: float, upper: float) -> np.ndarray:
    tolerance = 16 * np.finfo(np.float32).eps * max(1.0, abs(lower), abs(upper))
    return np.flatnonzero((values >= lower - tolerance) & (values <= upper + tolerance))


def periodic_interval_indices(
    values: np.ndarray, lower: float, upper: float, origin: float, period: float
) -> np.ndarray:
    wrapped_values = np.mod(values - origin, period) + origin
    wrapped_lower = float(np.mod(lower - origin, period) + origin)
    width = float(upper - lower)
    if width >= period - 1e-6:
        return np.arange(values.size)
    wrapped_upper = wrapped_lower + width
    tolerance = 16 * np.finfo(np.float32).eps * max(1.0, period)
    if wrapped_upper <= origin + period + tolerance:
        mask = (wrapped_values >= wrapped_lower - tolerance) & (
            wrapped_values <= wrapped_upper + tolerance
        )
    else:
        mask = (wrapped_values >= wrapped_lower - tolerance) | (
            wrapped_values <= wrapped_upper - period + tolerance
        )
    return np.flatnonzero(mask)


def nearest_indices(values: np.ndarray, centres: np.ndarray) -> np.ndarray:
    return np.abs(values[:, None] - centres[None, :]).argmin(axis=1).astype(np.int16)


def periodic_nearest_indices(
    values: np.ndarray, centres: np.ndarray, period: float
) -> np.ndarray:
    distance = np.abs(values[:, None] - centres[None, :])
    distance = np.minimum(distance, period - np.minimum(distance, period))
    return distance.argmin(axis=1).astype(np.int16)


def interpolation_brackets(
    values: np.ndarray, centres: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return deterministic, convex local interpolation brackets.

    Points between the block face and its first/last cell centre are clamped to
    that edge centre. This avoids extrapolation and never reaches into another
    block or refinement level.
    """
    if centres.ndim != 1 or centres.size == 0 or not np.all(np.diff(centres) > 0):
        raise ValueError("Cell centres must be a non-empty, strictly increasing 1D array")
    if centres.size == 1:
        zeros = np.zeros(values.shape, dtype=np.int16)
        return zeros, zeros.copy(), np.zeros(values.shape, dtype=np.float32)
    upper = np.searchsorted(centres, values, side="right")
    upper = np.clip(upper, 1, centres.size - 1)
    lower = upper - 1
    weight = (values - centres[lower]) / (centres[upper] - centres[lower])
    below = values <= centres[0]
    above = values >= centres[-1]
    lower[below] = upper[below] = 0
    lower[above] = upper[above] = centres.size - 1
    weight[below | above] = 0.0
    return (
        lower.astype(np.int16),
        upper.astype(np.int16),
        np.clip(weight, 0.0, 1.0).astype(np.float32),
    )


def periodic_interpolation_brackets(
    values: np.ndarray,
    centres: np.ndarray,
    block_lower: float,
    block_upper: float,
    period: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize periodic queries to the selected block, then bracket locally."""
    midpoint = 0.5 * (block_lower + block_upper)
    adjusted = values + np.rint((midpoint - values) / period) * period
    return interpolation_brackets(adjusted, centres)


def build_mapping(handle: h5py.File, grid: TargetGrid) -> AMRMapping:
    shape = (grid.phi.size, grid.theta.size, grid.r.size)
    block_map = np.full(shape, -1, dtype=np.int32)
    k_map = np.zeros(shape, dtype=np.int16)
    j_map = np.zeros(shape, dtype=np.int16)
    i_map = np.zeros(shape, dtype=np.int16)
    levels = np.asarray(handle["Levels"][...])
    phi_origin = float(grid.phi_edges[0])
    phi_period = float(grid.phi_edges[-1] - grid.phi_edges[0])

    # Coarser blocks are assigned first. Fine leaf blocks overwrite them in any
    # overlap, making the finest available representation authoritative.
    for block in np.argsort(levels, kind="stable"):
        r_faces = np.asarray(handle["x1f"][block], dtype=np.float64)
        theta_faces = np.asarray(handle["x2f"][block], dtype=np.float64)
        phi_faces = np.asarray(handle["x3f"][block], dtype=np.float64)
        ri = interval_indices(grid.r, float(r_faces[0]), float(r_faces[-1]))
        tj = interval_indices(grid.theta, float(theta_faces[0]), float(theta_faces[-1]))
        pk = periodic_interval_indices(
            grid.phi,
            float(phi_faces[0]),
            float(phi_faces[-1]),
            phi_origin,
            phi_period,
        )
        if not (ri.size and tj.size and pk.size):
            continue
        source_r = np.asarray(handle["x1v"][block], dtype=np.float64)
        source_theta = np.asarray(handle["x2v"][block], dtype=np.float64)
        source_phi = np.asarray(handle["x3v"][block], dtype=np.float64)
        local_i = nearest_indices(grid.r[ri], source_r)
        local_j = nearest_indices(grid.theta[tj], source_theta)
        local_k = periodic_nearest_indices(grid.phi[pk], source_phi, phi_period)
        selection = np.ix_(pk, tj, ri)
        block_map[selection] = int(block)
        k_map[selection] = local_k[:, None, None]
        j_map[selection] = local_j[None, :, None]
        i_map[selection] = local_i[None, None, :]

    missing = np.argwhere(block_map < 0)
    if len(missing):
        sample = missing[:5].tolist()
        raise RuntimeError(f"AMR mapping left {len(missing)} target voxels unmapped; sample={sample}")
    return AMRMapping(
        block=block_map,
        k=k_map,
        j=j_map,
        i=i_map,
        source_signature=source_grid_signature(handle),
    )


def build_trilinear_mapping(
    handle: h5py.File, grid: TargetGrid
) -> AMRTrilinearMapping:
    """Map target points to convex trilinear weights inside one finest leaf."""
    shape = (grid.phi.size, grid.theta.size, grid.r.size)
    block_map = np.full(shape, -1, dtype=np.int32)
    index_maps = {
        name: np.zeros(shape, dtype=np.int16)
        for name in ("k0", "k1", "j0", "j1", "i0", "i1")
    }
    weight_maps = {
        name: np.zeros(shape, dtype=np.float32) for name in ("wk", "wj", "wi")
    }
    levels = np.asarray(handle["Levels"][...])
    phi_origin = float(grid.phi_edges[0])
    phi_period = float(grid.phi_edges[-1] - grid.phi_edges[0])

    # The stable ascending traversal makes the winner deterministic on an exact
    # shared face: highest refinement first by priority, then highest block id.
    for block in np.argsort(levels, kind="stable"):
        r_faces = np.asarray(handle["x1f"][block], dtype=np.float64)
        theta_faces = np.asarray(handle["x2f"][block], dtype=np.float64)
        phi_faces = np.asarray(handle["x3f"][block], dtype=np.float64)
        ri = interval_indices(grid.r, float(r_faces[0]), float(r_faces[-1]))
        tj = interval_indices(grid.theta, float(theta_faces[0]), float(theta_faces[-1]))
        pk = periodic_interval_indices(
            grid.phi,
            float(phi_faces[0]),
            float(phi_faces[-1]),
            phi_origin,
            phi_period,
        )
        if not (ri.size and tj.size and pk.size):
            continue
        source_r = np.asarray(handle["x1v"][block], dtype=np.float64)
        source_theta = np.asarray(handle["x2v"][block], dtype=np.float64)
        source_phi = np.asarray(handle["x3v"][block], dtype=np.float64)
        local_i0, local_i1, local_wi = interpolation_brackets(grid.r[ri], source_r)
        local_j0, local_j1, local_wj = interpolation_brackets(
            grid.theta[tj], source_theta
        )
        local_k0, local_k1, local_wk = periodic_interpolation_brackets(
            grid.phi[pk],
            source_phi,
            float(phi_faces[0]),
            float(phi_faces[-1]),
            phi_period,
        )
        selection = np.ix_(pk, tj, ri)
        block_map[selection] = int(block)
        index_maps["k0"][selection] = local_k0[:, None, None]
        index_maps["k1"][selection] = local_k1[:, None, None]
        index_maps["j0"][selection] = local_j0[None, :, None]
        index_maps["j1"][selection] = local_j1[None, :, None]
        index_maps["i0"][selection] = local_i0[None, None, :]
        index_maps["i1"][selection] = local_i1[None, None, :]
        weight_maps["wk"][selection] = local_wk[:, None, None]
        weight_maps["wj"][selection] = local_wj[None, :, None]
        weight_maps["wi"][selection] = local_wi[None, None, :]

    missing = np.argwhere(block_map < 0)
    if len(missing):
        raise RuntimeError(
            f"AMR trilinear mapping left {len(missing)} target voxels unmapped; "
            f"sample={missing[:5].tolist()}"
        )
    return AMRTrilinearMapping(
        block=block_map,
        source_signature=source_grid_signature(handle),
        **index_maps,
        **weight_maps,
    )


def build_target_bin_average_mapping(
    handle: h5py.File, grid: TargetGrid
) -> TargetBinAverageMapping:
    """Assign source leaf cell centres to target bins with coordinate-volume weights."""
    target_indices: list[np.ndarray] = []
    source_indices: list[np.ndarray] = []
    coordinate_weights: list[np.ndarray] = []
    cells_per_block = (
        handle["x3v"].shape[1] * handle["x2v"].shape[1] * handle["x1v"].shape[1]
    )
    phi_origin = float(grid.phi_edges[0])
    phi_period = float(grid.phi_edges[-1] - grid.phi_edges[0])
    for block in range(len(handle["Levels"])):
        r = np.asarray(handle["x1v"][block], dtype=np.float64)
        theta = np.asarray(handle["x2v"][block], dtype=np.float64)
        phi = np.asarray(handle["x3v"][block], dtype=np.float64)
        i = np.searchsorted(grid.r_edges, r, side="right") - 1
        j = np.searchsorted(grid.theta_edges, theta, side="right") - 1
        wrapped_phi = np.mod(phi - phi_origin, phi_period) + phi_origin
        k = np.searchsorted(grid.phi_edges, wrapped_phi, side="right") - 1
        valid_i = (i >= 0) & (i < grid.r.size)
        valid_j = (j >= 0) & (j < grid.theta.size)
        valid_k = (k >= 0) & (k < grid.phi.size)
        kk, jj, ii = np.meshgrid(k, j, i, indexing="ij")
        valid = (
            valid_k[:, None, None]
            & valid_j[None, :, None]
            & valid_i[None, None, :]
        )
        flat_local = np.arange(cells_per_block, dtype=np.int64).reshape(valid.shape)
        target_flat = (kk * grid.theta.size + jj) * grid.r.size + ii
        dr = np.diff(np.asarray(handle["x1f"][block], dtype=np.float64))
        dtheta = np.diff(np.asarray(handle["x2f"][block], dtype=np.float64))
        dphi = np.diff(np.asarray(handle["x3f"][block], dtype=np.float64))
        weights = dphi[:, None, None] * dtheta[None, :, None] * dr[None, None, :]
        target_indices.append(target_flat[valid].astype(np.int32))
        source_indices.append((block * cells_per_block + flat_local[valid]).astype(np.int64))
        coordinate_weights.append(weights[valid].astype(np.float64))
    target_flat_indices = np.concatenate(target_indices)
    source_flat_indices = np.concatenate(source_indices)
    weights = np.concatenate(coordinate_weights)
    target_size = grid.phi.size * grid.theta.size * grid.r.size
    weight_sums = np.bincount(
        target_flat_indices, weights=weights, minlength=target_size
    ).astype(np.float64)
    return TargetBinAverageMapping(
        source_flat_indices=source_flat_indices,
        target_flat_indices=target_flat_indices,
        coordinate_weights=weights,
        target_weight_sums=weight_sums,
        nearest_fallback=build_mapping(handle, grid),
        source_signature=source_grid_signature(handle),
    )


def regrid_channel(
    handle: h5py.File, channel: str, mapping: AMRMapping
) -> np.ndarray:
    variables = variable_map(handle)
    dataset_name, variable_index = variables[channel]
    # One variable is read at a time (roughly 11 MiB for the supplied files), so
    # neither the full source snapshot nor the output time series enters memory.
    source = np.asarray(handle[dataset_name][variable_index], dtype=np.float32)
    output = source[mapping.block, mapping.k, mapping.j, mapping.i]
    if output.shape != mapping.block.shape:
        raise RuntimeError(f"{channel} output shape {output.shape} != {mapping.block.shape}")
    return np.asarray(output, dtype=np.float32)


def regrid_channel_trilinear(
    handle: h5py.File, channel: str, mapping: AMRTrilinearMapping
) -> np.ndarray:
    """Interpolate a channel using only values from each point's chosen block."""
    dataset_name, variable_index = variable_map(handle)[channel]
    source = np.asarray(handle[dataset_name][variable_index], dtype=np.float32)
    block = mapping.block
    c000 = source[block, mapping.k0, mapping.j0, mapping.i0]
    c001 = source[block, mapping.k0, mapping.j0, mapping.i1]
    c010 = source[block, mapping.k0, mapping.j1, mapping.i0]
    c011 = source[block, mapping.k0, mapping.j1, mapping.i1]
    c100 = source[block, mapping.k1, mapping.j0, mapping.i0]
    c101 = source[block, mapping.k1, mapping.j0, mapping.i1]
    c110 = source[block, mapping.k1, mapping.j1, mapping.i0]
    c111 = source[block, mapping.k1, mapping.j1, mapping.i1]
    wi, wj, wk = mapping.wi, mapping.wj, mapping.wk
    c00 = c000 + wi * (c001 - c000)
    c01 = c010 + wi * (c011 - c010)
    c10 = c100 + wi * (c101 - c100)
    c11 = c110 + wi * (c111 - c110)
    c0 = c00 + wj * (c01 - c00)
    c1 = c10 + wj * (c11 - c10)
    output = c0 + wk * (c1 - c0)
    if output.shape != mapping.block.shape:
        raise RuntimeError(f"{channel} output shape {output.shape} != {mapping.block.shape}")
    return np.asarray(output, dtype=np.float32)


def regrid_channel_target_bin_average(
    handle: h5py.File, channel: str, mapping: TargetBinAverageMapping
) -> np.ndarray:
    """Coordinate-volume average source leaf centres per target bin.

    Empty target bins deterministically use ``nearest_leaf``. This is not a
    conservative finite-volume restriction because weights are coordinate
    ``dr*dtheta*dphi``, not Kerr-Schild proper volumes or face fluxes.
    """
    dataset_name, variable_index = variable_map(handle)[channel]
    source = np.asarray(handle[dataset_name][variable_index], dtype=np.float32).reshape(-1)
    selected = source[mapping.source_flat_indices].astype(np.float64)
    weighted_sums = np.bincount(
        mapping.target_flat_indices,
        weights=selected * mapping.coordinate_weights,
        minlength=mapping.target_weight_sums.size,
    )
    nearest = regrid_channel(handle, channel, mapping.nearest_fallback).reshape(-1)
    populated = mapping.target_weight_sums > 0
    output = nearest.astype(np.float64)
    output[populated] = weighted_sums[populated] / mapping.target_weight_sums[populated]
    return output.reshape(mapping.nearest_fallback.block.shape).astype(np.float32)


def validate_output(path: Path, expected_shape: tuple[int, ...]) -> None:
    with h5py.File(path, "r") as handle:
        if tuple(handle["snapshots"].shape) != expected_shape:
            raise RuntimeError(
                f"Output shape {handle['snapshots'].shape} != expected {expected_shape}"
            )
        if tuple(handle["times"].shape) != (expected_shape[0],):
            raise RuntimeError("times has incorrect shape")
        nan_count = inf_count = 0
        nonpositive_rho = nonpositive_press = 0
        for index in range(expected_shape[0]):
            values = handle["snapshots"][index]
            nan_count += int(np.isnan(values).sum())
            inf_count += int(np.isinf(values).sum())
            nonpositive_rho += int(np.count_nonzero(values[3] <= 0))
            nonpositive_press += int(np.count_nonzero(values[4] <= 0))
        if nan_count or inf_count:
            raise FloatingPointError(f"Output contains {nan_count} NaN and {inf_count} Inf")
        if nonpositive_rho or nonpositive_press:
            raise FloatingPointError(
                f"Output has nonpositive rho={nonpositive_rho}, press={nonpositive_press}"
            )
        print(
            f"validated {path}: shape={expected_shape}, NaN={nan_count}, Inf={inf_count}, "
            "rho/press strictly positive",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--nr", type=int, default=64)
    parser.add_argument("--ntheta", type=int, default=64)
    parser.add_argument("--nphi", type=int, default=64)
    parser.add_argument("--max_files", type=int)
    parser.add_argument("--every", type=int, default=1)
    parser.add_argument("--r_min", type=float)
    parser.add_argument("--r_max", type=float)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audit_manifest", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--method",
        choices=("nearest_leaf", "leaf_trilinear", "target_bin_weighted_average"),
        default="nearest_leaf",
    )
    parser.add_argument(
        "--snapshot_indices",
        type=int,
        nargs="+",
        help="Select audited trajectory indices explicitly, preserving the given order.",
    )
    args = parser.parse_args()

    if min(args.nr, args.ntheta, args.nphi, args.every) <= 0:
        raise ValueError("Grid sizes and --every must be positive")
    output_path = args.out.expanduser()
    files = sorted(
        args.raw_dir.glob("mad98.prim.*.athdf"), key=numeric_snapshot_key
    )
    excluded_files: dict[str, Any] = {}
    expected_times: dict[str, float] = {}
    if args.audit_manifest is not None:
        included, excluded_files, expected_times = load_audit_selection(
            args.audit_manifest
        )
        files = [path for path in files if path.name in included]
        missing_manifest_files = included - {path.name for path in files}
        if missing_manifest_files:
            raise FileNotFoundError(
                f"Audit manifest includes missing source files: {sorted(missing_manifest_files)}"
            )
    if args.snapshot_indices is not None:
        if args.every != 1 or args.max_files is not None:
            raise ValueError("--snapshot_indices cannot be combined with --every/--max_files")
        invalid = [index for index in args.snapshot_indices if not 0 <= index < len(files)]
        if invalid:
            raise IndexError(f"snapshot indices outside audited trajectory: {invalid}")
        files = [files[index] for index in args.snapshot_indices]
    else:
        files = files[:: args.every]
        if args.max_files is not None:
            files = files[: args.max_files]
    if not files:
        raise FileNotFoundError(f"No ATHDF snapshots found under {args.raw_dir}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {output_path}; pass --overwrite explicitly")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp-{os.getpid()}")
    if temporary_path.exists():
        raise FileExistsError(f"Refusing to reuse temporary path {temporary_path}")

    with h5py.File(files[0], "r") as first:
        missing = [channel for channel in CHANNELS if channel not in variable_map(first)]
        if missing:
            raise KeyError(f"Missing required channels: {missing}")
        grid = build_target_grid(
            first, args.nr, args.ntheta, args.nphi, args.r_min, args.r_max
        )
        if args.method == "nearest_leaf":
            print("building reusable finest-block nearest-neighbour mapping", flush=True)
            mapping = build_mapping(first, grid)
        elif args.method == "leaf_trilinear":
            print("building reusable finest-block local-trilinear mapping", flush=True)
            mapping = build_trilinear_mapping(first, grid)
        else:
            print("building reusable target-bin coordinate-weighted mapping", flush=True)
            mapping = build_target_bin_average_mapping(first, grid)
        reference_shapes = {
            name: tuple(first[name].shape)
            for name in decode_strings(first.attrs["DatasetNames"])
        }

    string_dtype = h5py.string_dtype(encoding="utf-8")
    output_shape = (len(files), len(CHANNELS), args.nphi, args.ntheta, args.nr)
    with h5py.File(temporary_path, "w") as output:
        snapshots = output.create_dataset(
            "snapshots",
            shape=output_shape,
            dtype=np.float32,
            chunks=(1, 1, min(16, args.nphi), min(16, args.ntheta), min(32, args.nr)),
            compression="gzip",
            compression_opts=1,
            shuffle=True,
        )
        times = output.create_dataset("times", shape=(len(files),), dtype=np.float64)
        output["source_times"] = times
        output.create_dataset("channels", data=np.asarray(CHANNELS, dtype=object), dtype=string_dtype)
        output.create_dataset(
            "source_files", data=np.asarray([path.name for path in files], dtype=object), dtype=string_dtype
        )
        output.create_dataset(
            "excluded_files",
            data=np.asarray(list(excluded_files), dtype=object),
            dtype=string_dtype,
        )
        coords = output.create_group("coords")
        coords.create_dataset("r", data=grid.r)
        coords.create_dataset("theta", data=grid.theta)
        coords.create_dataset("phi", data=grid.phi)
        metadata = output.create_group("metadata")
        metadata.attrs["axis_order"] = "N,C,Nphi,Ntheta,Nr"
        metadata.attrs["coordinates"] = "spherical Kerr-Schild: phi,theta,r"
        metadata.attrs["method_name"] = args.method
        method_descriptions = {
            "nearest_leaf": "finest-containing-leaf-block nearest cell centre",
            "leaf_trilinear": (
                "finest-containing-leaf-block local trilinear cell-centred interpolation"
            ),
            "target_bin_weighted_average": (
                "source leaf cell centres in target bin weighted by coordinate dr*dtheta*dphi; "
                "empty bins use nearest_leaf"
            ),
        }
        metadata.attrs["method"] = method_descriptions[args.method]
        metadata.attrs["interpolation_crosses_refinement_levels"] = (
            False
            if args.method != "target_bin_weighted_average"
            else "not applicable: one target bin may average non-overlapping leaf cells from multiple levels"
        )
        metadata.attrs["block_edge_rule"] = (
            "source-centre target-bin assignment; empty bin uses nearest_leaf"
            if args.method == "target_bin_weighted_average"
            else "clamp to edge cell centre; exact shared face chooses highest level then "
            "highest stable block id"
        )
        metadata.attrs["coordinate_weighted_average"] = (
            args.method == "target_bin_weighted_average"
        )
        metadata.attrs["empty_target_bin_fallback"] = (
            "nearest_leaf" if args.method == "target_bin_weighted_average" else "not_applicable"
        )
        if args.method == "target_bin_weighted_average":
            metadata.attrs["empty_target_bin_count"] = int(
                np.count_nonzero(mapping.target_weight_sums == 0)
            )
            metadata.attrs["populated_target_bin_count"] = int(
                np.count_nonzero(mapping.target_weight_sums > 0)
            )
        metadata.attrs["r_target_spacing"] = "logarithmic"
        metadata.attrs["theta_target_spacing"] = "linear"
        metadata.attrs["phi_target_spacing"] = "linear-periodic"
        metadata.attrs["phi_periodic_query"] = True
        metadata.attrs["conservative"] = False
        metadata.attrs["divergence_preserving"] = False
        metadata.attrs["source_grid_signature"] = mapping.source_signature
        metadata.attrs["mapping_reused_when_signature_matches"] = True
        metadata.attrs["created_at_utc"] = datetime.now(timezone.utc).isoformat()
        metadata.attrs["source_raw_dir"] = str(args.raw_dir.resolve())
        metadata.attrs["selection_every"] = args.every
        metadata.attrs["audit_manifest"] = (
            str(args.audit_manifest.resolve()) if args.audit_manifest is not None else ""
        )
        metadata.attrs["excluded_files_json"] = json.dumps(excluded_files, ensure_ascii=False)
        metadata.attrs["requested_r_min"] = float(grid.r_edges[0])
        metadata.attrs["requested_r_max"] = float(grid.r_edges[-1])
        metadata.attrs["warning"] = (
            "Nearest-neighbour AMR regrid is not conservative and does not preserve div(B) "
            "to machine precision. Bcc/vel components remain coordinate-basis components."
        )

        for snapshot_index, path in enumerate(files):
            print(f"[{snapshot_index + 1}/{len(files)}] regridding {path.name}", flush=True)
            with h5py.File(path, "r") as source:
                current_shapes = {
                    name: tuple(source[name].shape)
                    for name in decode_strings(source.attrs["DatasetNames"])
                }
                if current_shapes != reference_shapes:
                    raise ValueError(f"Dataset shapes changed in {path.name}: {current_shapes}")
                signature = source_grid_signature(source)
                if signature != mapping.source_signature:
                    print(f"grid changed in {path.name}; rebuilding mapping", flush=True)
                    if args.method == "nearest_leaf":
                        mapping = build_mapping(source, grid)
                    elif args.method == "leaf_trilinear":
                        mapping = build_trilinear_mapping(source, grid)
                    else:
                        mapping = build_target_bin_average_mapping(source, grid)
                source_time = float(source.attrs["Time"])
                if path.name in expected_times and source_time != expected_times[path.name]:
                    raise ValueError(
                        f"{path.name} Time changed after audit: {source_time} != {expected_times[path.name]}"
                    )
                times[snapshot_index] = source_time
                for channel_index, channel in enumerate(CHANNELS):
                    if args.method == "nearest_leaf":
                        values = regrid_channel(source, channel, mapping)
                    elif args.method == "leaf_trilinear":
                        values = regrid_channel_trilinear(source, channel, mapping)
                    else:
                        values = regrid_channel_target_bin_average(source, channel, mapping)
                    if not np.isfinite(values).all():
                        raise FloatingPointError(f"{path.name}/{channel} regrid has NaN/Inf")
                    snapshots[snapshot_index, channel_index] = values
        output.attrs["format_version"] = "grmhd-regrid-v1"
        output.attrs["metadata_json"] = json.dumps(
            {
                "channels": list(CHANNELS),
                "shape": output_shape,
                "r_range_edges": [float(grid.r_edges[0]), float(grid.r_edges[-1])],
                "not_conservative": True,
                "not_divergence_preserving": True,
                "method": args.method,
            }
        )
        output.flush()

    try:
        validate_output(temporary_path, output_shape)
        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    checksum = sha256_file(output_path)
    report = {
        "output": str(output_path.resolve()),
        "shape": list(output_shape),
        "size_bytes": output_path.stat().st_size,
        "sha256": checksum,
        "source_file_count": len(files),
        "source_files": [path.name for path in files],
        "excluded_files": excluded_files,
        "axis_order": "N,C,Nphi,Ntheta,Nr",
        "channels": list(CHANNELS),
        "method": args.method,
        "command": shlex.join([sys.executable, *sys.argv]),
        "atomic_write": True,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
