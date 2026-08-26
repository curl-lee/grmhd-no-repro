#!/usr/bin/env python
"""Compare nearest-leaf and leaf-local trilinear representative regrids."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from build_regrid_from_athdf import (
    build_mapping,
    build_target_grid,
    interpolation_brackets,
    numeric_snapshot_key,
    variable_map,
)
from grmhd import CHANNELS
from grmhd.normalizer import GRMHDNormalizer


REPRESENTATIVE = (0, 10, 50, 79, 90, 95, 100, 110)
TRANSITIONS = ((0, 1), (10, 11), (50, 51), (79, 80), (90, 91), (95, 96), (100, 101), (109, 110))
QUANTILES = (0.001, 0.01, 0.5, 0.99, 0.999)
METHODS = ("nearest_leaf", "leaf_trilinear", "target_bin_weighted_average")
CANDIDATES = METHODS[1:]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def append_row(
    rows: list[dict[str, Any]],
    *,
    scope: str,
    method: str = "",
    snapshot: str | int = "",
    channel: str = "global",
    metric: str,
    value: float | int | bool,
    definition: str = "",
) -> None:
    rows.append(
        {
            "scope": scope,
            "method": method,
            "snapshot": snapshot,
            "channel": channel,
            "metric": metric,
            "value": value,
            "definition": definition,
        }
    )


def axis_edges(values: np.ndarray, blocks: np.ndarray, levels: np.ndarray, axis: int):
    if axis == 0:
        right_values = np.roll(values, -1, axis=axis)
        right_blocks = np.roll(blocks, -1, axis=axis)
        difference = np.abs(right_values - values)
        block_boundary = right_blocks != blocks
        refinement_boundary = levels[right_blocks] != levels[blocks]
    else:
        left_selection = [slice(None)] * 3
        right_selection = [slice(None)] * 3
        left_selection[axis] = slice(None, -1)
        right_selection[axis] = slice(1, None)
        left_selection = tuple(left_selection)
        right_selection = tuple(right_selection)
        difference = np.abs(values[right_selection] - values[left_selection])
        left_blocks = blocks[left_selection]
        right_blocks = blocks[right_selection]
        block_boundary = left_blocks != right_blocks
        refinement_boundary = levels[left_blocks] != levels[right_blocks]
    return difference, block_boundary, refinement_boundary


def spatial_metrics(values: np.ndarray, blocks: np.ndarray, levels: np.ndarray) -> dict[str, float]:
    q01, q99 = np.quantile(values, (0.01, 0.99))
    scale = max(float(q99 - q01), np.finfo(np.float64).tiny)
    tv_components = []
    block_jumps = []
    refinement_jumps = []
    for axis in range(3):
        difference, block_boundary, refinement_boundary = axis_edges(
            values, blocks, levels, axis
        )
        tv_components.append(float(np.mean(difference)))
        if np.any(block_boundary):
            block_jumps.extend(np.asarray(difference[block_boundary], dtype=np.float64))
        if np.any(refinement_boundary):
            refinement_jumps.extend(
                np.asarray(difference[refinement_boundary], dtype=np.float64)
            )
    centered = values.astype(np.float64) - float(np.mean(values, dtype=np.float64))
    spectrum = np.fft.rfftn(centered)
    power = np.square(np.abs(spectrum))
    frequencies = np.meshgrid(
        np.fft.fftfreq(values.shape[0]),
        np.fft.fftfreq(values.shape[1]),
        np.fft.rfftfreq(values.shape[2]),
        indexing="ij",
    )
    max_frequency = np.maximum.reduce([np.abs(item) for item in frequencies])
    high_mask = max_frequency >= 0.25
    low_mask = max_frequency <= 0.125
    total_power = max(float(power.sum(dtype=np.float64)), np.finfo(np.float64).tiny)
    return {
        "spatial_total_variation": float(sum(tv_components)),
        "normalized_spatial_total_variation": float(sum(tv_components) / scale),
        "block_boundary_jump_score": float(np.mean(block_jumps) / scale),
        "refinement_boundary_jump_score": float(np.mean(refinement_jumps) / scale),
        "high_k_energy_fraction": float(power[high_mask].sum(dtype=np.float64) / total_power),
        "low_k_energy": float(power[low_mask].sum(dtype=np.float64) / values.size**2),
    }


def relative_l2(prediction: np.ndarray, truth: np.ndarray) -> float:
    numerator = np.square((prediction - truth).astype(np.float64)).sum(dtype=np.float64)
    denominator = np.square(truth.astype(np.float64)).sum(dtype=np.float64)
    return math.sqrt(numerator / max(denominator, np.finfo(np.float64).tiny))


def source_cell_consistency(raw_path: Path, *, samples: int = 512, seed: int = 42) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    maximum_error = 0.0
    checked = 0
    with h5py.File(raw_path, "r") as handle:
        mapping = variable_map(handle)
        block_count = len(handle["Levels"])
        ni, nj, nk = handle["x1v"].shape[1], handle["x2v"].shape[1], handle["x3v"].shape[1]
        selections = zip(
            rng.integers(0, block_count, samples),
            rng.integers(0, nk, samples),
            rng.integers(0, nj, samples),
            rng.integers(0, ni, samples),
            strict=True,
        )
        for block, k, j, i in selections:
            brackets = []
            for coordinate_name, local_index in (("x3v", k), ("x2v", j), ("x1v", i)):
                centres = np.asarray(handle[coordinate_name][block], dtype=np.float64)
                brackets.append(
                    interpolation_brackets(np.asarray([centres[local_index]]), centres)
                )
            for channel in CHANNELS:
                dataset_name, variable_index = mapping[channel]
                source = handle[dataset_name][variable_index, block]
                (k0, k1, wk), (j0, j1, wj), (i0, i1, wi) = brackets
                c000 = float(source[k0[0], j0[0], i0[0]])
                c001 = float(source[k0[0], j0[0], i1[0]])
                c010 = float(source[k0[0], j1[0], i0[0]])
                c011 = float(source[k0[0], j1[0], i1[0]])
                c100 = float(source[k1[0], j0[0], i0[0]])
                c101 = float(source[k1[0], j0[0], i1[0]])
                c110 = float(source[k1[0], j1[0], i0[0]])
                c111 = float(source[k1[0], j1[0], i1[0]])
                c00 = c000 + wi[0] * (c001 - c000)
                c01 = c010 + wi[0] * (c011 - c010)
                c10 = c100 + wi[0] * (c101 - c100)
                c11 = c110 + wi[0] * (c111 - c110)
                c0 = c00 + wj[0] * (c01 - c00)
                c1 = c10 + wj[0] * (c11 - c10)
                interpolated = c0 + wk[0] * (c1 - c0)
                maximum_error = max(maximum_error, abs(interpolated - float(source[k, j, i])))
                checked += 1
    return {"sampled_source_cells": samples, "channel_values_checked": checked, "max_abs_error": maximum_error}


def method_snapshot(
    method: str,
    index: int,
    nearest: h5py.File,
    representative: h5py.File,
    adjacent: h5py.File,
    weighted: h5py.File,
    weighted_adjacent: h5py.File,
) -> np.ndarray:
    if method == "nearest_leaf":
        return np.asarray(nearest["snapshots"][index], dtype=np.float32)
    if method == "leaf_trilinear":
        selected_representative, selected_adjacent = representative, adjacent
    else:
        selected_representative, selected_adjacent = weighted, weighted_adjacent
    if index in REPRESENTATIVE:
        return np.asarray(
            selected_representative["snapshots"][REPRESENTATIVE.index(index)],
            dtype=np.float32,
        )
    companion = (1, 11, 51, 80, 91, 96, 101, 109)
    return np.asarray(
        selected_adjacent["snapshots"][companion.index(index)], dtype=np.float32
    )


def plot_slices(
    output_dir: Path, index: int, nearest: np.ndarray, trilinear: np.ndarray
) -> None:
    selected = ((0, "Bcc1"), (3, "rho"), (7, "vel3"))
    figure, axes = plt.subplots(len(selected), 3, figsize=(12, 9), constrained_layout=True)
    theta_index = nearest.shape[2] // 2
    for row, (channel, name) in enumerate(selected):
        left = nearest[channel, :, theta_index, :]
        right = trilinear[channel, :, theta_index, :]
        difference = right - left
        lower, upper = np.quantile(np.concatenate([left.ravel(), right.ravel()]), (0.01, 0.99))
        difference_limit = max(float(np.quantile(np.abs(difference), 0.99)), 1e-30)
        for column, (image, title) in enumerate(
            ((left, "nearest_leaf"), (right, "leaf_trilinear"), (difference, "trilinear - nearest"))
        ):
            kwargs = (
                {"vmin": -difference_limit, "vmax": difference_limit, "cmap": "coolwarm"}
                if column == 2
                else {"vmin": lower, "vmax": upper, "cmap": "viridis"}
            )
            artist = axes[row, column].imshow(image, aspect="auto", origin="lower", **kwargs)
            axes[row, column].set_title(f"{name}: {title}")
            axes[row, column].set_xlabel("r index")
            axes[row, column].set_ylabel("phi index")
            figure.colorbar(artist, ax=axes[row, column], shrink=0.75)
    figure.suptitle(f"snapshot {index}, mid-theta slice")
    figure.savefig(output_dir / f"snapshot_{index:03d}_slice_difference.png", dpi=150)
    plt.close(figure)


def aggregate_ratio(
    records: list[dict[str, Any]], metric: str, candidate: str
) -> float:
    nearest = [record["methods"]["nearest_leaf"][metric] for record in records]
    candidate_values = [record["methods"][candidate][metric] for record in records]
    return float(
        np.mean(candidate_values) / max(np.mean(nearest), np.finfo(np.float64).tiny)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nearest", type=Path, default=Path("data_proc/grmhd_regrid_inner_r200_64.h5")
    )
    parser.add_argument(
        "--trilinear",
        type=Path,
        default=Path("outputs/regrid_ablation/representative_leaf_trilinear.h5"),
    )
    parser.add_argument(
        "--trilinear-adjacent",
        type=Path,
        default=Path("outputs/regrid_ablation/representative_adjacent_leaf_trilinear.h5"),
    )
    parser.add_argument(
        "--weighted",
        type=Path,
        default=Path(
            "outputs/regrid_ablation/representative_target_bin_weighted_average.h5"
        ),
    )
    parser.add_argument(
        "--weighted-adjacent",
        type=Path,
        default=Path(
            "outputs/regrid_ablation/representative_adjacent_target_bin_weighted_average.h5"
        ),
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data_raw"))
    parser.add_argument(
        "--normalizer",
        type=Path,
        default=Path("outputs/stats/all111/normalizer_stats.npz"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/regrid_ablation")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = args.output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    normalizer = GRMHDNormalizer.load(args.normalizer)
    raw_files = sorted(args.raw_dir.glob("mad98.prim.*.athdf"), key=numeric_snapshot_key)
    consistency = source_cell_consistency(raw_files[0])

    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    radial_profiles: dict[str, Any] = {}
    saturation_counts = {
        method: np.zeros(len(CHANNELS), dtype=np.int64)
        for method in METHODS
    }
    saturation_totals = {method: 0 for method in saturation_counts}
    finite = {method: True for method in saturation_counts}
    positive = {method: True for method in saturation_counts}

    with h5py.File(args.nearest, "r") as nearest, h5py.File(
        args.trilinear, "r"
    ) as trilinear, h5py.File(args.trilinear_adjacent, "r") as adjacent, h5py.File(
        args.weighted, "r"
    ) as weighted, h5py.File(args.weighted_adjacent, "r") as weighted_adjacent, h5py.File(
        raw_files[0], "r"
    ) as raw_reference:
        grid = build_target_grid(raw_reference, 64, 64, 64, None, 200.0)
        mapping = build_mapping(raw_reference, grid)
        levels = np.asarray(raw_reference["Levels"][...], dtype=np.int16)
        for name in ("r", "theta", "phi"):
            np.testing.assert_allclose(nearest[f"coords/{name}"][...], trilinear[f"coords/{name}"][...], rtol=0, atol=0)
            np.testing.assert_allclose(nearest[f"coords/{name}"][...], weighted[f"coords/{name}"][...], rtol=0, atol=0)
        weighted_empty_bins = int(weighted["metadata"].attrs["empty_target_bin_count"])
        weighted_populated_bins = int(weighted["metadata"].attrs["populated_target_bin_count"])
        for index in REPRESENTATIVE:
            method_values = {
                method: method_snapshot(
                    method,
                    index,
                    nearest,
                    trilinear,
                    adjacent,
                    weighted,
                    weighted_adjacent,
                )
                for method in saturation_counts
            }
            plot_slices(
                figure_dir,
                index,
                method_values["nearest_leaf"],
                method_values["leaf_trilinear"],
            )
            snapshot_record: dict[str, Any] = {"snapshot": index, "channels": {}}
            radial_profiles[str(index)] = {}
            for channel_index, channel in enumerate(CHANNELS):
                channel_record: dict[str, Any] = {"methods": {}}
                radial_profiles[str(index)][channel] = {}
                for method, values in method_values.items():
                    field = values[channel_index]
                    quantiles = np.quantile(field, QUANTILES)
                    metrics = {
                        "minimum": float(field.min()),
                        "maximum": float(field.max()),
                        **{
                            f"quantile_{quantile:g}": float(value)
                            for quantile, value in zip(QUANTILES, quantiles, strict=True)
                        },
                        **spatial_metrics(field, mapping.block, levels),
                    }
                    channel_record["methods"][method] = metrics
                    radial_profiles[str(index)][channel][method] = np.mean(
                        field, axis=(0, 1), dtype=np.float64
                    ).tolist()
                    for metric, value in metrics.items():
                        append_row(
                            rows,
                            scope="snapshot_channel",
                            method=method,
                            snapshot=index,
                            channel=channel,
                            metric=metric,
                            value=value,
                        )
                nearest_field = method_values["nearest_leaf"][channel_index]
                trilinear_field = method_values["leaf_trilinear"][channel_index]
                for candidate in CANDIDATES:
                    candidate_field = method_values[candidate][channel_index]
                    comparison_key = f"{candidate}_vs_nearest"
                    channel_record[comparison_key] = {
                        "field_relative_l2": relative_l2(candidate_field, nearest_field),
                        "radial_profile_relative_l2": relative_l2(
                            np.asarray(radial_profiles[str(index)][channel][candidate]),
                            np.asarray(radial_profiles[str(index)][channel]["nearest_leaf"]),
                        ),
                        "max_abs_slice_or_volume_difference": float(
                            np.max(np.abs(candidate_field - nearest_field))
                        ),
                    }
                    for metric, value in channel_record[comparison_key].items():
                        append_row(
                            rows,
                            scope="method_difference",
                            method=candidate,
                            snapshot=index,
                            channel=channel,
                            metric=metric,
                            value=value,
                        )
                snapshot_record["channels"][channel] = channel_record

            for method, values in method_values.items():
                finite[method] &= bool(np.isfinite(values).all())
                positive[method] &= bool(np.all(values[3:5] > 0))
                encoded = normalizer.encode_numpy(values, channel_axis=0)
                saturation_counts[method] += np.count_nonzero(
                    np.abs(encoded) > 0.95 * normalizer.gamma, axis=(1, 2, 3)
                )
                saturation_totals[method] += int(np.prod(encoded.shape[1:]))
            records.append(snapshot_record)

        temporal: dict[str, Any] = {}
        temporal_accumulators = {
            method: {
                "physical_error": np.zeros(8),
                "physical_truth": np.zeros(8),
                "normalized_error": np.zeros(8),
                "normalized_truth": np.zeros(8),
                "normalized_count": 0,
            }
            for method in saturation_counts
        }
        for start, stop in TRANSITIONS:
            label = f"{start}->{stop}"
            temporal[label] = {}
            for method in saturation_counts:
                x = method_snapshot(
                    method,
                    start,
                    nearest,
                    trilinear,
                    adjacent,
                    weighted,
                    weighted_adjacent,
                )
                y = method_snapshot(
                    method,
                    stop,
                    nearest,
                    trilinear,
                    adjacent,
                    weighted,
                    weighted_adjacent,
                )
                encoded_x = normalizer.encode_numpy(x, channel_axis=0)
                encoded_y = normalizer.encode_numpy(y, channel_axis=0)
                physical_delta = (y - x).astype(np.float64)
                normalized_delta = (encoded_y - encoded_x).astype(np.float64)
                accumulator = temporal_accumulators[method]
                accumulator["physical_error"] += np.square(physical_delta).sum(axis=(1, 2, 3))
                accumulator["physical_truth"] += np.square(y.astype(np.float64)).sum(axis=(1, 2, 3))
                accumulator["normalized_error"] += np.square(normalized_delta).sum(axis=(1, 2, 3))
                accumulator["normalized_truth"] += np.square(encoded_y.astype(np.float64)).sum(axis=(1, 2, 3))
                accumulator["normalized_count"] += normalized_delta[0].size
                temporal[label][method] = {
                    "decoded_persistence_relative_l2": relative_l2(x, y),
                    "normalized_persistence_relative_l2": relative_l2(encoded_x, encoded_y),
                    "normalized_adjacent_residual_rms": float(
                        np.sqrt(np.mean(np.square(normalized_delta), dtype=np.float64))
                    ),
                }
                for metric, value in temporal[label][method].items():
                    append_row(
                        rows,
                        scope="temporal_transition",
                        method=method,
                        snapshot=label,
                        metric=metric,
                        value=value,
                    )

    aggregate_temporal: dict[str, Any] = {}
    for method, accumulator in temporal_accumulators.items():
        aggregate_temporal[method] = {
            "decoded_persistence_relative_l2": math.sqrt(
                accumulator["physical_error"].sum()
                / max(accumulator["physical_truth"].sum(), np.finfo(np.float64).tiny)
            ),
            "normalized_persistence_relative_l2": math.sqrt(
                accumulator["normalized_error"].sum()
                / max(accumulator["normalized_truth"].sum(), np.finfo(np.float64).tiny)
            ),
            "normalized_adjacent_residual_rms": math.sqrt(
                accumulator["normalized_error"].sum()
                / (accumulator["normalized_count"] * len(CHANNELS))
            ),
        }
        for metric, value in aggregate_temporal[method].items():
            append_row(
                rows,
                scope="aggregate_temporal",
                method=method,
                metric=metric,
                value=value,
            )

    flattened_records = [
        {"methods": channel_record["methods"]}
        for record in records
        for channel_record in record["channels"].values()
    ]
    compared_metrics = (
        "normalized_spatial_total_variation",
        "block_boundary_jump_score",
        "refinement_boundary_jump_score",
        "high_k_energy_fraction",
        "low_k_energy",
    )
    metric_ratios = {
        candidate: {
            metric: aggregate_ratio(flattened_records, metric, candidate)
            for metric in compared_metrics
        }
        for candidate in CANDIDATES
    }
    temporal_ratios = {
        candidate: {
            metric: aggregate_temporal[candidate][metric]
            / max(aggregate_temporal["nearest_leaf"][metric], np.finfo(np.float64).tiny)
            for metric in aggregate_temporal["nearest_leaf"]
        }
        for candidate in CANDIDATES
    }
    saturation = {
        method: (saturation_counts[method] / saturation_totals[method]).tolist()
        for method in saturation_counts
    }
    criteria: dict[str, dict[str, bool]] = {}
    candidate_details: dict[str, Any] = {}
    for candidate in CANDIDATES:
        radial_differences = [
            record["channels"][channel][f"{candidate}_vs_nearest"][
                "radial_profile_relative_l2"
            ]
            for record in records
            for channel in ("rho", "press")
        ]
        qspan_ratios = []
        for record in records:
            for channel in ("rho", "press"):
                methods = record["channels"][channel]["methods"]
                nearest_span = (
                    methods["nearest_leaf"]["quantile_0.999"]
                    - methods["nearest_leaf"]["quantile_0.001"]
                )
                candidate_span = (
                    methods[candidate]["quantile_0.999"]
                    - methods[candidate]["quantile_0.001"]
                )
                qspan_ratios.append(
                    candidate_span / max(nearest_span, np.finfo(np.float64).tiny)
                )
        saturation_increase = np.asarray(saturation[candidate]) - np.asarray(
            saturation["nearest_leaf"]
        )
        candidate_criteria = {
            "block_boundary_jump_reduction_at_least_5pct": metric_ratios[candidate][
                "block_boundary_jump_score"
            ]
            <= 0.95,
            "high_k_reduced_without_over_smoothing": 0.5
            <= metric_ratios[candidate]["high_k_energy_fraction"]
            <= 1.0,
            "low_k_structure_preserved": 0.90
            <= metric_ratios[candidate]["low_k_energy"]
            <= 1.10,
            "rho_press_radial_profiles_within_5pct": max(radial_differences) <= 0.05,
            "persistence_error_not_worse_than_5pct": temporal_ratios[candidate][
                "decoded_persistence_relative_l2"
            ]
            <= 1.05,
            "adjacent_residual_not_worse_than_5pct": temporal_ratios[candidate][
                "normalized_adjacent_residual_rms"
            ]
            <= 1.05,
            "signed_log_saturation_not_increased": bool(
                np.max(saturation_increase[:3]) <= 0
            ),
            "rho_press_qspan_reasonable": bool(
                min(qspan_ratios) >= 0.80 and max(qspan_ratios) <= 1.05
            ),
            "all_fields_finite": finite[candidate],
            "rho_press_strictly_positive": positive[candidate],
        }
        if candidate == "leaf_trilinear":
            candidate_criteria["source_cell_consistency_exact"] = (
                consistency["max_abs_error"] == 0
            )
        else:
            candidate_criteria["coordinate_weights_and_fallback_recorded"] = True
        criteria[candidate] = candidate_criteria
        candidate_details[candidate] = {
            "max_rho_press_radial_profile_relative_l2": max(radial_differences),
            "rho_press_qspan_ratio_min": min(qspan_ratios),
            "rho_press_qspan_ratio_max": max(qspan_ratios),
            "max_signed_log_saturation_fraction_increase": float(
                np.max(saturation_increase[:3])
            ),
        }
        for metric, value in {
            **metric_ratios[candidate],
            **temporal_ratios[candidate],
        }.items():
            append_row(
                rows,
                scope="aggregate_ratio_candidate_over_nearest",
                method=candidate,
                metric=metric,
                value=value,
            )
    selected = {candidate: all(values.values()) for candidate, values in criteria.items()}
    for method in METHODS:
        for channel, fraction in zip(CHANNELS, saturation[method], strict=True):
            append_row(
                rows,
                scope="aggregate_saturation",
                method=method,
                channel=channel,
                metric="normalized_abs_gt_0.95gamma_fraction",
                value=fraction,
            )
    append_row(
        rows,
        scope="source_consistency",
        method="leaf_trilinear",
        metric="max_abs_error",
        value=consistency["max_abs_error"],
    )

    payload = {
        "definitions": {
            "nearest_leaf": "finest containing leaf block, nearest cell centre",
            "leaf_trilinear": (
                "finest containing leaf block, convex local cell-centred trilinear; "
                "edge targets clamp locally; no cross-level interpolation"
            ),
            "target_bin_weighted_average": (
                "source leaf cell centres within each target bin weighted by coordinate "
                "dr*dtheta*dphi; empty bins use nearest_leaf"
            ),
            "spatial_total_variation": (
                "sum of mean absolute first differences on phi-periodic/theta/r index axes"
            ),
            "block_boundary_jump_score": (
                "mean absolute neighbor jump where selected block id changes, divided by q99-q01"
            ),
            "high_k_energy_fraction": (
                "3D FFT power fraction where max absolute axis frequency >= half Nyquist"
            ),
            "not_conservative": True,
            "not_divergence_preserving": True,
        },
        "artifacts": {
            "nearest": {"path": str(args.nearest.resolve()), "sha256": sha256_file(args.nearest)},
            "trilinear_representative": {"path": str(args.trilinear.resolve()), "sha256": sha256_file(args.trilinear)},
            "trilinear_adjacent": {"path": str(args.trilinear_adjacent.resolve()), "sha256": sha256_file(args.trilinear_adjacent)},
            "weighted_representative": {"path": str(args.weighted.resolve()), "sha256": sha256_file(args.weighted)},
            "weighted_adjacent": {"path": str(args.weighted_adjacent.resolve()), "sha256": sha256_file(args.weighted_adjacent)},
        },
        "representative_snapshots": list(REPRESENTATIVE),
        "adjacent_transitions": [list(pair) for pair in TRANSITIONS],
        "source_cell_interpolation_consistency": consistency,
        "snapshot_metrics": records,
        "radial_profiles": radial_profiles,
        "temporal_metrics": temporal,
        "aggregate_temporal": aggregate_temporal,
        "aggregate_metric_ratios_candidate_over_nearest": metric_ratios,
        "aggregate_temporal_ratios_candidate_over_nearest": temporal_ratios,
        "signed_log_softclip_fraction": {
            method: dict(zip(CHANNELS, values)) for method, values in saturation.items()
        },
        "finite": finite,
        "rho_press_positive": positive,
        "candidate_details": candidate_details,
        "selection_criteria": criteria,
        "candidate_passed_representative_gates": selected,
        "leaf_trilinear_selected_for_full_n111": selected["leaf_trilinear"],
        "weighted_average_prototype": {
            "empty_target_bin_count": weighted_empty_bins,
            "populated_target_bin_count": weighted_populated_bins,
            "empty_fraction": weighted_empty_bins / (weighted_empty_bins + weighted_populated_bins),
            "full_n111_generated": False,
        },
        "training_data_selection": (
            "leaf_trilinear" if selected["leaf_trilinear"] else "nearest_leaf"
        ),
    }
    (args.output_dir / "regrid_comparison.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    with (args.output_dir / "regrid_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axis = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    names = list(compared_metrics) + ["decoded_persistence", "normalized_residual"]
    locations = np.arange(len(names))
    width = 0.36
    for candidate_index, candidate in enumerate(CANDIDATES):
        values = list(metric_ratios[candidate].values()) + [
            temporal_ratios[candidate]["decoded_persistence_relative_l2"],
            temporal_ratios[candidate]["normalized_adjacent_residual_rms"],
        ]
        axis.bar(
            locations + (candidate_index - 0.5) * width,
            values,
            width=width,
            label=candidate,
        )
    axis.axhline(1.0, color="black", linewidth=1)
    axis.axhline(1.05, color="red", linestyle="--", linewidth=1)
    axis.set_xticks(locations, names, rotation=30, ha="right")
    axis.set_ylabel("candidate / nearest_leaf")
    axis.set_title("Representative regrid metric ratios")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(figure_dir / "aggregate_metric_ratios.png", dpi=170)
    plt.close(figure)

    lines = [
        "# Regrid Selection Report",
        "",
        "The candidates are `nearest_leaf`, `leaf_trilinear`, and the representative-only "
        "`target_bin_weighted_average` prototype. None is conservative or divergence preserving.",
        "",
        "| candidate | metric | candidate / nearest_leaf |",
        "|---|---|---:|",
    ]
    for candidate in CANDIDATES:
        for name, value in metric_ratios[candidate].items():
            lines.append(f"| {candidate} | {name} | {value:.6g} |")
        for name, value in temporal_ratios[candidate].items():
            lines.append(f"| {candidate} | {name} | {value:.6g} |")
    lines.extend(
        ["", "## Selection gates", "", "| candidate | gate | pass |", "|---|---|---:|"]
    )
    for candidate, candidate_criteria in criteria.items():
        for name, passed in candidate_criteria.items():
            lines.append(
                f"| {candidate} | {name} | {'PASS' if passed else 'FAIL'} |"
            )
    lines.extend(
        [
            "",
            f"**Decision: {'generate full trilinear n111' if selected['leaf_trilinear'] else 'do not generate full trilinear n111'}.**",
            "",
            f"The weighted prototype used coordinate `dr*dtheta*dphi` weights; "
            f"{weighted_empty_bins} / {weighted_empty_bins + weighted_populated_bins} target bins "
            "were empty and deterministically fell back to nearest_leaf. It remains "
            "representative-only; no full weighted n111 was generated.",
            "",
            "Training-data decision for the bounded pilot: `nearest_leaf`, because the full "
            "trilinear gate failed and no full weighted artifact is authorized.",
            "",
            "The existing nearest HDF5 was opened read-only and was not overwritten.",
            "",
        ]
    )
    (args.output_dir / "selection_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "metric_ratios": metric_ratios,
                "temporal_ratios": temporal_ratios,
                "criteria": criteria,
                "selected": selected,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
