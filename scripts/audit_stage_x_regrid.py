#!/usr/bin/env python3
"""Raw-ATHDF multi-resolution and temporal-increment audit for Stage X.

All diagnostic grids use the production finest-leaf/nearest-cell-centre rule.
The 128^3 grid is only a HIGHER_RES_SAMPLING_REFERENCE, never raw truth.
No HDF5 or raw source is written.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from build_regrid_from_athdf import (
    build_mapping,
    build_target_grid,
    source_grid_signature,
    variable_map,
)
from grmhd import CHANNELS
from grmhd.stage_x_audit import (
    classify_temporal_fidelity,
    cosine_similarity,
    log_shell_indices,
    nearest_reference_indices,
    radial_profile,
    radial_region_masks,
    relative_l2,
    sample_reference_to_grid,
    shell_moments,
    spectral_energy_fractions,
)


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty audit table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def source_channel(handle: h5py.File, channel: str) -> np.ndarray:
    dataset_name, variable_index = variable_map(handle)[channel]
    return np.asarray(handle[dataset_name][variable_index], dtype=np.float32)


def sample_all(source: np.ndarray, mappings: Mapping[int, Any]) -> dict[int, np.ndarray]:
    return {
        resolution: np.asarray(
            source[mapping.block, mapping.k, mapping.j, mapping.i], dtype=np.float32
        )
        for resolution, mapping in mappings.items()
    }


def reference_on_grid(
    reference: np.ndarray, resolution: int, grids: Mapping[int, Any], reference_resolution: int,
) -> np.ndarray:
    if resolution == reference_resolution:
        return reference
    low = grids[resolution]
    high = grids[reference_resolution]
    phi = nearest_reference_indices(
        low.phi, high.phi, periodic=True,
        period=float(high.phi_edges[-1] - high.phi_edges[0]),
    )
    theta = nearest_reference_indices(low.theta, high.theta)
    radius = nearest_reference_indices(low.r, high.r)
    return sample_reference_to_grid(reference, phi, theta, radius)


def region_relative(value: np.ndarray, reference: np.ndarray, r: np.ndarray) -> dict[str, float]:
    return {
        name: relative_l2(value[..., mask], reference[..., mask])
        for name, mask in radial_region_masks(r).items()
    }


def field_rows(
    snapshot: int, channel: str, values: Mapping[int, np.ndarray], grids: Mapping[int, Any],
    reference_resolution: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    high = values[reference_resolution]
    for resolution, value in values.items():
        reference = reference_on_grid(high, resolution, grids, reference_resolution)
        means, variances = shell_moments(value, grids[resolution].r)
        ref_means, ref_variances = shell_moments(reference, grids[resolution].r)
        profile = radial_profile(value)
        ref_profile = radial_profile(reference)
        low_energy, high_energy = spectral_energy_fractions(value)
        ref_low_energy, ref_high_energy = spectral_energy_fractions(reference)
        regions = region_relative(value, reference, grids[resolution].r)
        rows.append({
            "snapshot": snapshot,
            "channel": channel,
            "resolution": resolution,
            "reference_resolution": reference_resolution,
            "reference_semantics": "HIGHER_RES_SAMPLING_REFERENCE",
            "global_relative_l2": relative_l2(value, reference),
            "global_cosine": cosine_similarity(value, reference),
            "shell_mean_relative_l2": relative_l2(means, ref_means),
            "shell_variance_relative_l2": relative_l2(variances, ref_variances),
            "radial_profile_relative_l2": relative_l2(profile, ref_profile),
            "index_grid_low_frequency_energy_fraction": low_energy,
            "index_grid_high_frequency_energy_fraction": high_energy,
            "reference_low_frequency_energy_fraction": ref_low_energy,
            "reference_high_frequency_energy_fraction": ref_high_energy,
            "high_frequency_fraction_absolute_difference": abs(high_energy - ref_high_energy),
            "minimum": float(np.min(value)),
            "maximum": float(np.max(value)),
            "q001": float(np.quantile(value, 0.001)),
            "q999": float(np.quantile(value, 0.999)),
            "reference_minimum": float(np.min(reference)),
            "reference_maximum": float(np.max(reference)),
            "reference_q001": float(np.quantile(reference, 0.001)),
            "reference_q999": float(np.quantile(reference, 0.999)),
            "field_loss_inner": regions["inner"],
            "field_loss_middle": regions["middle"],
            "field_loss_outer": regions["outer"],
        })
    return rows


def temporal_rows(
    source_snapshot: int, channel: str, left: Mapping[int, np.ndarray],
    right: Mapping[int, np.ndarray], grids: Mapping[int, Any], reference_resolution: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    shell_rows: list[dict[str, Any]] = []
    radial_rows: list[dict[str, Any]] = []
    high_delta = right[reference_resolution] - left[reference_resolution]
    for resolution in left:
        delta = right[resolution] - left[resolution]
        reference = reference_on_grid(high_delta, resolution, grids, reference_resolution)
        means, _ = shell_moments(delta, grids[resolution].r)
        ref_means, _ = shell_moments(reference, grids[resolution].r)
        profile = radial_profile(delta)
        ref_profile = radial_profile(reference)
        low_energy, high_energy = spectral_energy_fractions(delta)
        ref_low_energy, ref_high_energy = spectral_energy_fractions(reference)
        regions = region_relative(delta, reference, grids[resolution].r)
        rows.append({
            "source_snapshot": source_snapshot,
            "target_snapshot": source_snapshot + 1,
            "channel": channel,
            "resolution": resolution,
            "reference_resolution": reference_resolution,
            "reference_semantics": "HIGHER_RES_SAMPLING_REFERENCE",
            "residual_relative_difference": relative_l2(delta, reference),
            "residual_cosine": cosine_similarity(delta, reference),
            "shell_increment_relative_l2": relative_l2(means, ref_means),
            "radial_increment_relative_l2": relative_l2(profile, ref_profile),
            "index_grid_low_frequency_energy_fraction": low_energy,
            "index_grid_high_frequency_energy_fraction": high_energy,
            "reference_low_frequency_energy_fraction": ref_low_energy,
            "reference_high_frequency_energy_fraction": ref_high_energy,
            "high_frequency_fraction_absolute_difference": abs(high_energy - ref_high_energy),
            "temporal_increment_loss_inner": regions["inner"],
            "temporal_increment_loss_middle": regions["middle"],
            "temporal_increment_loss_outer": regions["outer"],
        })
        shell_indices, shell_edges = log_shell_indices(grids[resolution].r, 8)
        shell_cosine = cosine_similarity(means, ref_means)
        for shell in range(8):
            shell_rows.append({
                "source_snapshot": source_snapshot,
                "target_snapshot": source_snapshot + 1,
                "channel": channel,
                "resolution": resolution,
                "shell": shell,
                "r_edge_left": float(shell_edges[shell]),
                "r_edge_right": float(shell_edges[shell + 1]),
                "radial_cells": int(np.count_nonzero(shell_indices == shell)),
                "increment_mean": float(means[shell]),
                "reference_increment_mean": float(ref_means[shell]),
                "absolute_difference": float(abs(means[shell] - ref_means[shell])),
                "shell_vector_relative_l2": relative_l2(means, ref_means),
                "shell_vector_cosine": shell_cosine,
                "reference_semantics": "HIGHER_RES_SAMPLING_REFERENCE",
            })
        for radial_index, (radius, value, ref_value) in enumerate(
            zip(grids[resolution].r, profile, ref_profile, strict=True)
        ):
            radial_rows.append({
                "source_snapshot": source_snapshot,
                "target_snapshot": source_snapshot + 1,
                "channel": channel,
                "resolution": resolution,
                "radial_index": radial_index,
                "r": float(radius),
                "increment_profile": float(value),
                "reference_increment_profile": float(ref_value),
                "absolute_difference": float(abs(value - ref_value)),
                "profile_relative_l2": relative_l2(profile, ref_profile),
                "profile_cosine": cosine_similarity(profile, ref_profile),
                "reference_semantics": "HIGHER_RES_SAMPLING_REFERENCE",
            })
    return rows, shell_rows, radial_rows


def metadata_summary(handle: h5py.File) -> dict[str, Any]:
    levels = np.asarray(handle["Levels"], dtype=np.int64)
    block_size = [int(value) for value in handle.attrs["MeshBlockSize"]]
    result: dict[str, Any] = {
        "coordinates": handle.attrs["Coordinates"].decode() if isinstance(handle.attrs["Coordinates"], bytes) else str(handle.attrs["Coordinates"]),
        "root_grid_size_x1_x2_x3": [int(value) for value in handle.attrs["RootGridSize"]],
        "mesh_block_size_x1_x2_x3": block_size,
        "max_level": int(handle.attrs["MaxLevel"]),
        "num_mesh_blocks": int(handle.attrs["NumMeshBlocks"]),
        "leaf_blocks_by_level": {str(level): int(np.count_nonzero(levels == level)) for level in np.unique(levels)},
        "leaf_cells_by_level": {
            str(level): int(np.count_nonzero(levels == level) * np.prod(block_size)) for level in np.unique(levels)
        },
        "actual_leaf_cell_count": int(len(levels) * np.prod(block_size)),
        "r_bounds": [float(np.min(handle["x1f"][:, 0])), float(np.max(handle["x1f"][:, -1]))],
        "theta_bounds": [float(np.min(handle["x2f"][:, 0])), float(np.max(handle["x2f"][:, -1]))],
        "phi_bounds": [float(np.min(handle["x3f"][:, 0])), float(np.max(handle["x3f"][:, -1]))],
    }
    spacing: dict[str, Any] = {}
    for level in np.unique(levels):
        selected = levels == level
        spacing[str(level)] = {}
        for name, dataset in (("dr", "x1f"), ("dtheta", "x2f"), ("dphi", "x3f")):
            values = np.diff(np.asarray(handle[dataset][selected], dtype=np.float64), axis=1)
            spacing[str(level)][name] = [float(np.min(values)), float(np.max(values))]
    result["coordinate_spacing_by_level"] = spacing
    root = np.asarray(result["root_grid_size_x1_x2_x3"], dtype=int)
    result["finest_effective_resolution_x1_x2_x3"] = (root * (2 ** result["max_level"])).tolist()
    return result


def make_figures(out_dir: Path, field: list[dict[str, Any]], temporal: list[dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    focus = ("Bcc1", "Bcc2", "Bcc3", "rho", "press", "vel3")
    fig, axis = plt.subplots(figsize=(8, 5))
    for channel in focus:
        xs, ys = [], []
        for resolution in (32, 64, 96):
            vals = [float(row["global_relative_l2"]) for row in field if row["channel"] == channel and row["resolution"] == resolution]
            xs.append(resolution); ys.append(float(np.median(vals)))
        axis.plot(xs, ys, marker="o", label=channel)
    axis.set(xlabel="diagnostic sampling resolution", ylabel="median relative L2 vs 128 sampling reference", yscale="log")
    axis.legend(ncol=2); axis.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "resolution_convergence_by_channel.png", dpi=160); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for resolution in (32, 64, 96):
        selected = [row for row in temporal if row["resolution"] == resolution]
        axes[0].scatter([resolution], [np.median([float(row["residual_relative_difference"]) for row in selected])], label=str(resolution))
        axes[1].scatter([resolution], [np.median([float(row["residual_cosine"]) for row in selected])], label=str(resolution))
    axes[0].set(xlabel="resolution", ylabel="median increment relative difference")
    axes[1].set(xlabel="resolution", ylabel="median increment cosine", ylim=(-0.05, 1.05))
    for axis in axes: axis.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "temporal_increment_fidelity.png", dpi=160); plt.close(fig)

    selected_field = [row for row in field if row["resolution"] == 64]
    selected_temporal = [row for row in temporal if row["resolution"] == 64]
    regions = ("inner", "middle", "outer")
    field_values = [np.median([float(row[f"field_loss_{name}"]) for row in selected_field]) for name in regions]
    temporal_values = [np.median([float(row[f"temporal_increment_loss_{name}"]) for row in selected_temporal]) for name in regions]
    x = np.arange(3)
    fig, axis = plt.subplots(figsize=(7, 4))
    axis.bar(x - 0.18, field_values, width=0.36, label="field")
    axis.bar(x + 0.18, temporal_values, width=0.36, label="increment")
    axis.set_xticks(x, regions); axis.set_ylabel("median relative difference vs 128 reference")
    axis.legend(); axis.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_dir / "regional_loss_64.png", dpi=160); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/stage_x/method_gap_audit.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/stage_x/regrid"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    raw_dir = Path(config["raw"]["directory"])
    field_snapshots = [int(value) for value in config["raw"]["field_snapshots"]]
    pair_sources = [int(value) for value in config["raw"]["adjacent_pair_sources"]]
    resolutions = [int(value) for value in config["regrid"]["resolutions"]]
    reference_resolution = int(config["regrid"]["reference_resolution"])
    r_min, r_max = [float(value) for value in config["regrid"]["r_edge_range"]]
    paths = {index: raw_dir / f"mad98.prim.{index:05d}.athdf" for index in sorted(set(field_snapshots + pair_sources + [i + 1 for i in pair_sources]))}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing fixed Stage X raw files: {missing}")

    with h5py.File(paths[min(paths)], "r") as first:
        grids = {resolution: build_target_grid(first, resolution, resolution, resolution, r_min, r_max) for resolution in resolutions}
        mappings = {resolution: build_mapping(first, grids[resolution]) for resolution in resolutions}
        signature = source_grid_signature(first)
        raw_metadata = metadata_summary(first)
    signatures: dict[int, str] = {}
    for index, path in paths.items():
        with h5py.File(path, "r") as handle:
            signatures[index] = source_grid_signature(handle)
    if any(value != signature for value in signatures.values()):
        raise RuntimeError("Stage X selected raw grid layouts differ; reusable mapping is unsafe")

    processed = Path("data_proc/grmhd_regrid_inner_r200_64_expanded.h5")
    field_records: list[dict[str, Any]] = []
    temporal_records: list[dict[str, Any]] = []
    shell_records: list[dict[str, Any]] = []
    radial_records: list[dict[str, Any]] = []
    production_checks: list[dict[str, Any]] = []
    completed_fields: set[int] = set()
    with h5py.File(processed, "r") as production:
        for pair_number, source_index in enumerate(pair_sources, start=1):
            target_index = source_index + 1
            print(f"[{pair_number}/{len(pair_sources)}] raw pair {source_index:05d}->{target_index:05d}", flush=True)
            with h5py.File(paths[source_index], "r") as left_handle, h5py.File(paths[target_index], "r") as right_handle:
                for channel_index, channel in enumerate(CHANNELS):
                    left = sample_all(source_channel(left_handle, channel), mappings)
                    right = sample_all(source_channel(right_handle, channel), mappings)
                    if source_index in field_snapshots:
                        field_records.extend(field_rows(source_index, channel, left, grids, reference_resolution))
                        stored = np.asarray(production["snapshots"][source_index, channel_index], dtype=np.float32)
                        production_checks.append({
                            "snapshot": source_index,
                            "channel": channel,
                            "array_equal": bool(np.array_equal(left[64], stored)),
                            "max_absolute_difference": float(np.max(np.abs(left[64] - stored))),
                        })
                    if target_index in field_snapshots and target_index not in completed_fields:
                        field_records.extend(field_rows(target_index, channel, right, grids, reference_resolution))
                        stored = np.asarray(production["snapshots"][target_index, channel_index], dtype=np.float32)
                        production_checks.append({
                            "snapshot": target_index,
                            "channel": channel,
                            "array_equal": bool(np.array_equal(right[64], stored)),
                            "max_absolute_difference": float(np.max(np.abs(right[64] - stored))),
                        })
                    temporal, shell, radial = temporal_rows(
                        source_index, channel, left, right, grids, reference_resolution
                    )
                    temporal_records.extend(temporal)
                    shell_records.extend(shell)
                    radial_records.extend(radial)
                if source_index in field_snapshots:
                    completed_fields.add(source_index)
                if target_index in field_snapshots:
                    completed_fields.add(target_index)

    if completed_fields != set(field_snapshots):
        raise RuntimeError(f"field audit incomplete: {completed_fields} != {set(field_snapshots)}")
    if not all(bool(row["array_equal"]) for row in production_checks):
        raise RuntimeError("diagnostic 64^3 mapping does not reproduce production HDF5 exactly")

    write_csv(args.out_dir / "resolution_convergence.csv", field_records)
    write_csv(args.out_dir / "temporal_increment_fidelity.csv", temporal_records)
    write_csv(args.out_dir / "shell_increment_fidelity.csv", shell_records)
    write_csv(args.out_dir / "radial_increment_fidelity.csv", radial_records)
    write_csv(args.out_dir / "production64_equivalence.csv", production_checks)
    make_figures(args.out_dir / "figures", field_records, temporal_records)

    at64 = [row for row in temporal_records if int(row["resolution"]) == 64]
    thresholds = config["regrid"]["fidelity_thresholds"]
    classification = classify_temporal_fidelity(
        [float(row["residual_relative_difference"]) for row in at64],
        [float(row["residual_cosine"]) for row in at64],
        good_relative_max=float(thresholds["good_relative_difference_median_max"]),
        good_cosine_min=float(thresholds["good_cosine_median_min"]),
        moderate_relative_max=float(thresholds["moderate_relative_difference_median_max"]),
        moderate_cosine_min=float(thresholds["moderate_cosine_median_min"]),
    )
    field64 = [row for row in field_records if int(row["resolution"]) == 64]
    summary = {
        "schema_version": "stage-x-raw-amr-audit-v1",
        "raw_directory": str(raw_dir),
        "selected_paths": {str(index): str(path) for index, path in paths.items()},
        "grid_layout_static": True,
        "source_grid_signature": signature,
        "raw_metadata": raw_metadata,
        "diagnostic_resolutions": resolutions,
        "reference_resolution": reference_resolution,
        "reference_semantics": "HIGHER_RES_SAMPLING_REFERENCE",
        "sampling_rule": "finest-leaf nearest-cell-center",
        "production64_exactly_reproduced": True,
        "temporal_increment_regrid_fidelity": classification,
        "median_64_increment_relative_difference": float(np.median([row["residual_relative_difference"] for row in at64])),
        "median_64_increment_cosine": float(np.median([row["residual_cosine"] for row in at64])),
        "median_64_field_relative_l2": float(np.median([row["global_relative_l2"] for row in field64])),
        "field_loss_by_region_64": {name: float(np.median([row[f"field_loss_{name}"] for row in field64])) for name in ("inner", "middle", "outer")},
        "temporal_increment_loss_by_region_64": {name: float(np.median([row[f"temporal_increment_loss_{name}"] for row in at64])) for name in ("inner", "middle", "outer")},
        "strict_divb_audit_not_authorized": True,
        "conservative": False,
        "divergence_preserving": False,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "raw_amr_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = f"""# Raw AMR and diagnostic regrid audit

- Raw directory: `{raw_dir}`
- Fixed field snapshots: `{field_snapshots}`
- Fixed adjacent pairs: `{[(value, value + 1) for value in pair_sources]}`
- Grid signatures across all {len(paths)} files: identical (`{signature}`); `GRID_LAYOUT_STATIC = true`.
- Raw root grid `(x1,x2,x3)=(r,theta,phi)`: `{raw_metadata['root_grid_size_x1_x2_x3']}`.
- MeshBlock size `(x1,x2,x3)`: `{raw_metadata['mesh_block_size_x1_x2_x3']}`; leaf blocks by level: `{raw_metadata['leaf_blocks_by_level']}`.
- Actual leaf-cell count per snapshot: `{raw_metadata['actual_leaf_cell_count']}`; finest all-domain-equivalent resolution `(r,theta,phi)`: `{raw_metadata['finest_effective_resolution_x1_x2_x3']}`.
- Diagnostic grids: `{resolutions}` on `r=[{r_min},{r_max}]`; every grid uses the production finest-leaf nearest-cell-center rule.
- `128^3` is `HIGHER_RES_SAMPLING_REFERENCE`, not raw-AMR truth.
- The independently recomputed `64^3` arrays are bitwise equal to the existing processed HDF5 for all {len(production_checks)} audited snapshot/channel arrays.
- Median 64-vs-128-sampling field relative L2: `{summary['median_64_field_relative_l2']:.6g}`.
- Median 64-vs-128-sampling temporal-increment relative difference/cosine: `{summary['median_64_increment_relative_difference']:.6g}` / `{summary['median_64_increment_cosine']:.6g}`.
- `TEMPORAL_INCREMENT_REGRID_FIDELITY = {classification}` under the thresholds frozen in `{args.config}`.
- Field loss inner/middle/outer: `{summary['field_loss_by_region_64']}`.
- Temporal-increment loss inner/middle/outer: `{summary['temporal_increment_loss_by_region_64']}`.

The production mapping is non-conservative and not divergence preserving (`src/build_regrid_from_athdf.py:265-314,446-457,685-713`). A plain Cartesian divergence would not be the GRMHD constraint in this spherical Kerr--Schild coordinate representation, so `STRICT_DIVB_AUDIT_NOT_AUTHORIZED = true`.
"""
    (args.out_dir / "raw_amr_audit.md").write_text(md, encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
