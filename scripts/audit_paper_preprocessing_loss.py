#!/usr/bin/env python
"""Audit information loss in the canonical paper preprocessing path.

The canonical path remains gamma=6 with a 0.99*gamma inverse clamp.  Gamma
8/12 and no-soft-clip paths are diagnostic statistics only and are never saved
as training normalizers.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/grmhd-paper-audit-matplotlib")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from grmhd import CHANNELS
from grmhd.paper_preprocessing import PAPER_TRANSFORMS, PaperPreprocessor
from grmhd.paper_protocol import PaperReduced100Protocol


QUANTILES = (0.001, 0.01, 0.5, 0.99, 0.999)
FOCUS_CHANNELS = ("Bcc3", "vel3")
GAMMA_MODES: tuple[tuple[str, float | None, bool], ...] = (
    ("gamma_6_paper_canonical", 6.0, True),
    ("gamma_8_diagnostic_extension", 8.0, False),
    ("gamma_12_diagnostic_extension", 12.0, False),
    ("no_soft_clip_diagnostic_extension", None, False),
)


def git_output(args: list[str], cwd: Path) -> str:
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def finite_summary(values: np.ndarray) -> dict[str, float | int]:
    flat = np.asarray(values).reshape(-1)
    finite = np.isfinite(flat)
    finite_values = flat[finite]
    if not finite_values.size:
        return {
            "count": int(flat.size),
            "finite_count": 0,
            "nonfinite_fraction": 1.0,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            **{f"q{quantile:g}": None for quantile in QUANTILES},
        }
    quantile_values = np.quantile(finite_values, QUANTILES)
    return {
        "count": int(flat.size),
        "finite_count": int(finite_values.size),
        "nonfinite_fraction": float(1.0 - finite_values.size / flat.size),
        "min": float(np.min(finite_values)),
        "max": float(np.max(finite_values)),
        "mean": float(np.mean(finite_values, dtype=np.float64)),
        "std": float(np.std(finite_values, dtype=np.float64)),
        **{
            f"q{quantile:g}": float(value)
            for quantile, value in zip(QUANTILES, quantile_values, strict=True)
        },
    }


def error_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float | int | None]:
    truth_flat = np.asarray(truth, dtype=np.float64).reshape(-1)
    prediction_flat = np.asarray(prediction, dtype=np.float64).reshape(-1)
    selected = np.ones(truth_flat.shape, dtype=bool) if mask is None else mask.reshape(-1).copy()
    selected &= np.isfinite(truth_flat) & np.isfinite(prediction_flat)
    nonfinite_selected = (
        np.count_nonzero(mask) - np.count_nonzero(selected)
        if mask is not None
        else truth_flat.size - np.count_nonzero(selected)
    )
    if not np.any(selected):
        return {
            "count": 0,
            "nonfinite_count": int(nonfinite_selected),
            "mae": None,
            "rmse": None,
            "relative_l2": None,
            "q99_absolute_error": None,
            "max_absolute_error": None,
            "error_squared_sum": 0.0,
            "truth_squared_sum": 0.0,
        }
    selected_truth = truth_flat[selected]
    selected_prediction = prediction_flat[selected]
    absolute_error = np.abs(selected_prediction - selected_truth)
    error_squared_sum = float(np.sum(np.square(absolute_error), dtype=np.float64))
    truth_squared_sum = float(np.sum(np.square(selected_truth), dtype=np.float64))
    return {
        "count": int(selected_truth.size),
        "nonfinite_count": int(nonfinite_selected),
        "mae": float(np.mean(absolute_error, dtype=np.float64)),
        "rmse": float(np.sqrt(error_squared_sum / selected_truth.size)),
        "relative_l2": (
            float(np.sqrt(error_squared_sum / truth_squared_sum))
            if truth_squared_sum > 0
            else None
        ),
        "q99_absolute_error": float(np.quantile(absolute_error, 0.99)),
        "max_absolute_error": float(np.max(absolute_error)),
        "error_squared_sum": error_squared_sum,
        "truth_squared_sum": truth_squared_sum,
    }


def transform_channel(physical: np.ndarray, channel: int, epsilon: float) -> np.ndarray:
    values = np.asarray(physical, dtype=np.float64).copy()
    kind = PAPER_TRANSFORMS[channel]
    if kind == "positive_log":
        if np.any(values < 0):
            raise ValueError(f"{CHANNELS[channel]} contains negative values")
        np.add(values, epsilon, out=values)
        np.log10(values, out=values)
    elif kind == "signed_log":
        signs = np.sign(values).astype(np.int8, copy=False)
        np.abs(values, out=values)
        np.divide(values, epsilon, out=values)
        np.add(values, 1.0, out=values)
        np.log10(values, out=values)
        np.multiply(values, signs, out=values)
    return values


def inverse_transformed_channel(
    transformed: np.ndarray,
    channel: int,
    epsilon: float,
    *,
    output_dtype: np.dtype | type = np.float64,
    finite_guard: bool,
) -> np.ndarray:
    values = np.asarray(transformed, dtype=np.float64)
    kind = PAPER_TRANSFORMS[channel]
    dtype = np.dtype(output_dtype)
    physical_limit = float(np.finfo(dtype).max)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        if kind == "positive_log":
            exponent = values
            if finite_guard:
                exponent = np.minimum(exponent, np.log10(physical_limit + epsilon))
            decoded = np.power(10.0, exponent) - epsilon
            if finite_guard:
                decoded = np.maximum(decoded, np.finfo(dtype).tiny)
        elif kind == "signed_log":
            magnitude = np.abs(values)
            if finite_guard:
                magnitude = np.minimum(
                    magnitude,
                    np.log10(physical_limit) - np.log10(epsilon),
                )
            decoded = np.sign(values) * epsilon * (np.power(10.0, magnitude) - 1.0)
        else:
            decoded = np.clip(values, -physical_limit, physical_limit) if finite_guard else values
    return np.asarray(decoded, dtype=dtype)


def decode_soft_clipped_channel(
    soft_clipped: np.ndarray,
    channel: int,
    preprocessor: PaperPreprocessor,
    *,
    gamma: float,
    inverse_clamp_fraction: float | None,
    output_dtype: np.dtype | type,
    finite_guard: bool,
) -> np.ndarray:
    values = np.asarray(soft_clipped)
    if inverse_clamp_fraction is not None:
        limit = inverse_clamp_fraction * gamma
        values = np.clip(values, -limit, limit)
    with np.errstate(divide="ignore", invalid="ignore"):
        robust_z = gamma * np.arctanh(np.asarray(values, dtype=np.float64) / gamma)
    transformed = robust_z * preprocessor.scale[channel] + preprocessor.median[channel]
    return inverse_transformed_channel(
        transformed,
        channel,
        preprocessor.epsilon[channel],
        output_dtype=output_dtype,
        finite_guard=finite_guard,
    )


def decode_robust_z_channel(
    robust_z: np.ndarray,
    channel: int,
    preprocessor: PaperPreprocessor,
) -> np.ndarray:
    transformed = (
        np.asarray(robust_z, dtype=np.float64) * preprocessor.scale[channel]
        + preprocessor.median[channel]
    )
    return inverse_transformed_channel(
        transformed,
        channel,
        preprocessor.epsilon[channel],
        output_dtype=np.float64,
        finite_guard=False,
    )


def spatial_clamp_statistics(
    positive_mask: np.ndarray,
    negative_mask: np.ndarray,
    *,
    snapshot_indices: tuple[int, ...],
    times: np.ndarray,
    r: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    shell_indices: np.ndarray,
    shell_edges: np.ndarray,
) -> dict[str, Any]:
    total_mask = positive_mask | negative_mask
    phi_index = 0
    equatorial_index = int(np.argmin(np.abs(theta - np.pi / 2)))
    snapshot_records = []
    for slot, snapshot_index in enumerate(snapshot_indices):
        snapshot_records.append(
            {
                "snapshot_index": snapshot_index,
                "time": float(times[snapshot_index]),
                "positive_fraction": float(np.mean(positive_mask[slot])),
                "negative_fraction": float(np.mean(negative_mask[slot])),
                "total_fraction": float(np.mean(total_mask[slot])),
                "hit_count": int(np.count_nonzero(total_mask[slot])),
            }
        )
    shell_records = []
    for shell in range(len(shell_edges) - 1):
        radial_selection = shell_indices == shell
        radial_count = int(np.count_nonzero(radial_selection))
        positive_fraction = (
            float(np.mean(positive_mask[..., radial_selection])) if radial_count else None
        )
        negative_fraction = (
            float(np.mean(negative_mask[..., radial_selection])) if radial_count else None
        )
        total_fraction = (
            float(np.mean(total_mask[..., radial_selection])) if radial_count else None
        )
        shell_records.append(
            {
                "shell": shell,
                "r_lower": float(shell_edges[shell]),
                "r_upper": float(shell_edges[shell + 1]),
                "voxel_count": int(
                    total_mask.shape[0]
                    * total_mask.shape[1]
                    * total_mask.shape[2]
                    * radial_count
                ),
                "positive_fraction": positive_fraction,
                "negative_fraction": negative_fraction,
                "total_fraction": total_fraction,
            }
        )
    spatial_voxels = int(np.prod(total_mask.shape[1:]))
    equatorial_theta = np.abs(theta - np.pi / 2) <= np.pi / 8
    polar_theta = (theta <= np.pi / 4) | (theta >= 3 * np.pi / 4)
    inner_r = shell_indices < 2

    def concentration(theta_selection: np.ndarray | None, r_selection: np.ndarray | None) -> dict[str, Any]:
        spatial_selection = np.ones(total_mask.shape[1:], dtype=bool)
        if theta_selection is not None:
            spatial_selection &= theta_selection[None, :, None]
        if r_selection is not None:
            spatial_selection &= r_selection[None, None, :]
        region_voxel_fraction = float(np.count_nonzero(spatial_selection) / spatial_voxels)
        total_hits = int(np.count_nonzero(total_mask))
        region_hits = int(np.count_nonzero(total_mask & spatial_selection[None, ...]))
        hit_share = float(region_hits / total_hits) if total_hits else 0.0
        return {
            "region_voxel_fraction": region_voxel_fraction,
            "clamp_hit_count": region_hits,
            "clamp_hit_share": hit_share,
            "enrichment_over_uniform": (
                float(hit_share / region_voxel_fraction) if region_voxel_fraction else None
            ),
        }

    hit_counts = np.asarray([record["hit_count"] for record in snapshot_records], dtype=np.int64)
    sorted_counts = np.sort(hit_counts)[::-1]
    total_hits = int(np.sum(hit_counts))
    cumulative = np.cumsum(sorted_counts)
    snapshots_for_half = (
        int(np.searchsorted(cumulative, 0.5 * total_hits, side="left") + 1)
        if total_hits
        else 0
    )
    return {
        "positive_fraction": float(np.mean(positive_mask)),
        "negative_fraction": float(np.mean(negative_mask)),
        "total_fraction": float(np.mean(total_mask)),
        "positive_count": int(np.count_nonzero(positive_mask)),
        "negative_count": int(np.count_nonzero(negative_mask)),
        "total_count": int(np.count_nonzero(total_mask)),
        "snapshot_records": snapshot_records,
        "shell_records": shell_records,
        "fixed_phi": {
            "phi_index": phi_index,
            "phi_value": float(phi[phi_index]),
            "axes": ["theta", "r"],
            "total_fraction_map": np.mean(total_mask[:, phi_index], axis=0).tolist(),
            "positive_fraction_map": np.mean(positive_mask[:, phi_index], axis=0).tolist(),
            "negative_fraction_map": np.mean(negative_mask[:, phi_index], axis=0).tolist(),
        },
        "equatorial": {
            "theta_index": equatorial_index,
            "theta_value": float(theta[equatorial_index]),
            "axes": ["phi", "r"],
            "total_fraction_map": np.mean(total_mask[:, :, equatorial_index], axis=0).tolist(),
            "positive_fraction_map": np.mean(positive_mask[:, :, equatorial_index], axis=0).tolist(),
            "negative_fraction_map": np.mean(negative_mask[:, :, equatorial_index], axis=0).tolist(),
        },
        "region_concentration": {
            "equatorial_band_abs_theta_minus_pi_over_2_le_pi_over_8": concentration(
                equatorial_theta, None
            ),
            "polar_caps_theta_le_pi_over_4_or_ge_3pi_over_4": concentration(
                polar_theta, None
            ),
            "inner_two_log_shells": concentration(None, inner_r),
        },
        "snapshot_concentration": {
            "top_1_snapshot_hit_share": (
                float(sorted_counts[0] / total_hits) if total_hits else 0.0
            ),
            "top_5_snapshot_hit_share": (
                float(np.sum(sorted_counts[:5]) / total_hits) if total_hits else 0.0
            ),
            "snapshots_for_50_percent_of_hits": snapshots_for_half,
            "snapshot_count": len(snapshot_records),
        },
    }


def append_metric_rows(
    rows: list[dict[str, Any]],
    *,
    record_type: str,
    split: str,
    channel: str,
    metrics: dict[str, Any],
    stage: str = "",
    scope: str = "",
    index: int | str = "",
    shell: int | str = "",
    gamma_mode: str = "",
    paper_canonical: bool | str = "",
) -> None:
    for metric, value in metrics.items():
        if isinstance(value, (int, float)) or value is None:
            rows.append(
                {
                    "record_type": record_type,
                    "split": split,
                    "channel": channel,
                    "stage": stage,
                    "scope": scope,
                    "index": index,
                    "shell": shell,
                    "gamma_mode": gamma_mode,
                    "paper_canonical": paper_canonical,
                    "metric": metric,
                    "value": value,
                }
            )


def audit_split(
    handle: h5py.File,
    preprocessor: PaperPreprocessor,
    *,
    split_name: str,
    snapshot_indices: tuple[int, ...],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    times = np.asarray(handle["times"][...], dtype=np.float64)
    r = np.asarray(handle["coords/r"][...], dtype=np.float64)
    theta = np.asarray(handle["coords/theta"][...], dtype=np.float64)
    phi = np.asarray(handle["coords/phi"][...], dtype=np.float64)
    shell_edges = np.geomspace(float(r[0]), float(r[-1]), 9)
    shell_indices = np.digitize(r, shell_edges[1:-1], right=False)
    channels: dict[str, Any] = {}
    global_truth_sq = 0.0
    global_error_sq = 0.0
    global_clamped_error_sq = 0.0
    global_unclamped_error_sq = 0.0
    gamma_global = {
        mode: {"truth_squared_sum": 0.0, "error_squared_sum": 0.0}
        for mode, _, _ in GAMMA_MODES
    }
    for channel, name in enumerate(CHANNELS):
        physical = np.asarray(
            handle["snapshots"][list(snapshot_indices), channel], dtype=np.float64
        )
        stages = {"physical": finite_summary(physical)}
        transformed = transform_channel(physical, channel, preprocessor.epsilon[channel])
        stages["transformed_x_hat"] = finite_summary(transformed)
        robust_z = transformed
        np.subtract(robust_z, preprocessor.median[channel], out=robust_z)
        np.divide(robust_z, preprocessor.scale[channel], out=robust_z)
        stages["robust_z"] = finite_summary(robust_z)
        soft64 = preprocessor.gamma * np.tanh(robust_z / preprocessor.gamma)
        canonical_soft = np.asarray(soft64, dtype=np.float32)
        stages["soft_clipped_z_tilde_float32"] = finite_summary(canonical_soft)
        for stage_name, stage_metrics in stages.items():
            append_metric_rows(
                rows,
                record_type="stage_summary",
                split=split_name,
                channel=name,
                stage=stage_name,
                metrics=stage_metrics,
                paper_canonical=True,
            )

        limit = preprocessor.inverse_clamp_fraction * preprocessor.gamma
        positive_mask = canonical_soft > limit
        negative_mask = canonical_soft < -limit
        clamp_mask = positive_mask | negative_mask
        spatial = spatial_clamp_statistics(
            positive_mask,
            negative_mask,
            snapshot_indices=snapshot_indices,
            times=times,
            r=r,
            theta=theta,
            phi=phi,
            shell_indices=shell_indices,
            shell_edges=shell_edges,
        )
        append_metric_rows(
            rows,
            record_type="clamp_summary",
            split=split_name,
            channel=name,
            metrics={
                key: spatial[key]
                for key in ("positive_fraction", "negative_fraction", "total_fraction")
            },
            paper_canonical=True,
        )
        for record in spatial["snapshot_records"]:
            append_metric_rows(
                rows,
                record_type="snapshot_clamp",
                split=split_name,
                channel=name,
                index=record["snapshot_index"],
                metrics=record,
                paper_canonical=True,
            )
        for record in spatial["shell_records"]:
            append_metric_rows(
                rows,
                record_type="shell_clamp",
                split=split_name,
                channel=name,
                shell=record["shell"],
                metrics=record,
                paper_canonical=True,
            )
        for region, metrics in spatial["region_concentration"].items():
            append_metric_rows(
                rows,
                record_type="region_concentration",
                split=split_name,
                channel=name,
                scope=region,
                metrics=metrics,
                paper_canonical=True,
            )

        canonical_decode = decode_soft_clipped_channel(
            canonical_soft,
            channel,
            preprocessor,
            gamma=preprocessor.gamma,
            inverse_clamp_fraction=preprocessor.inverse_clamp_fraction,
            output_dtype=np.float32,
            finite_guard=True,
        )
        roundtrip = {
            "all_voxels": error_metrics(physical, canonical_decode),
            "clamped_voxels": error_metrics(physical, canonical_decode, clamp_mask),
            "unclamped_voxels": error_metrics(physical, canonical_decode, ~clamp_mask),
        }
        for scope, metrics in roundtrip.items():
            append_metric_rows(
                rows,
                record_type="paper_roundtrip",
                split=split_name,
                channel=name,
                scope=scope,
                metrics=metrics,
                paper_canonical=True,
            )
        global_truth_sq += float(roundtrip["all_voxels"]["truth_squared_sum"])
        global_error_sq += float(roundtrip["all_voxels"]["error_squared_sum"])
        global_clamped_error_sq += float(roundtrip["clamped_voxels"]["error_squared_sum"])
        global_unclamped_error_sq += float(roundtrip["unclamped_voxels"]["error_squared_sum"])

        diagnostic_decode = decode_soft_clipped_channel(
            soft64,
            channel,
            preprocessor,
            gamma=preprocessor.gamma,
            inverse_clamp_fraction=None,
            output_dtype=np.float64,
            finite_guard=False,
        )
        diagnostic_finite = np.isfinite(diagnostic_decode)
        diagnostic = {
            "role": "diagnostic_only_not_training_or_paper_evaluation",
            "float32_exact_positive_gamma_fraction": float(np.mean(canonical_soft == preprocessor.gamma)),
            "float32_exact_negative_gamma_fraction": float(np.mean(canonical_soft == -preprocessor.gamma)),
            "float64_exact_positive_gamma_fraction": float(np.mean(soft64 == preprocessor.gamma)),
            "float64_exact_negative_gamma_fraction": float(np.mean(soft64 == -preprocessor.gamma)),
            "float64_unclamped_decode_nonfinite_fraction": float(np.mean(~diagnostic_finite)),
            "float64_unclamped_roundtrip_on_finite": error_metrics(
                physical, diagnostic_decode, diagnostic_finite
            ),
            "paper_clamped_vs_float64_unclamped_on_finite": error_metrics(
                diagnostic_decode, canonical_decode, diagnostic_finite
            ),
        }
        for scope in (
            "float64_unclamped_roundtrip_on_finite",
            "paper_clamped_vs_float64_unclamped_on_finite",
        ):
            append_metric_rows(
                rows,
                record_type="diagnostic_float64_decode",
                split=split_name,
                channel=name,
                scope=scope,
                metrics=diagnostic[scope],
                paper_canonical=False,
            )

        gamma_sensitivity: dict[str, Any] = {}
        for mode, gamma, canonical in GAMMA_MODES:
            if gamma is None:
                sensitivity_decode = decode_robust_z_channel(robust_z, channel, preprocessor)
                sensitivity_clamp = np.zeros(robust_z.shape, dtype=bool)
            else:
                sensitivity_soft = np.asarray(
                    gamma * np.tanh(robust_z / gamma), dtype=np.float32
                )
                sensitivity_limit = preprocessor.inverse_clamp_fraction * gamma
                sensitivity_clamp = np.abs(sensitivity_soft) > sensitivity_limit
                sensitivity_decode = decode_soft_clipped_channel(
                    sensitivity_soft,
                    channel,
                    preprocessor,
                    gamma=gamma,
                    inverse_clamp_fraction=preprocessor.inverse_clamp_fraction,
                    output_dtype=np.float32,
                    finite_guard=True,
                )
            metrics = error_metrics(physical, sensitivity_decode)
            gamma_sensitivity[mode] = {
                "gamma": gamma,
                "paper_canonical": canonical,
                "classification": (
                    "paper_canonical"
                    if canonical
                    else "diagnostic_extension_not_paper_reproduction"
                ),
                "clamp_hit_fraction": float(np.mean(sensitivity_clamp)),
                "roundtrip": metrics,
            }
            gamma_global[mode]["truth_squared_sum"] += float(metrics["truth_squared_sum"])
            gamma_global[mode]["error_squared_sum"] += float(metrics["error_squared_sum"])
            append_metric_rows(
                rows,
                record_type="gamma_sensitivity",
                split=split_name,
                channel=name,
                gamma_mode=mode,
                metrics={
                    "clamp_hit_fraction": float(np.mean(sensitivity_clamp)),
                    "relative_l2": metrics["relative_l2"],
                    "rmse": metrics["rmse"],
                    "max_absolute_error": metrics["max_absolute_error"],
                },
                paper_canonical=canonical,
            )
            del sensitivity_decode, sensitivity_clamp

        channels[name] = {
            "stages": stages,
            "canonical_inverse_clamp": spatial,
            "paper_roundtrip": roundtrip,
            "diagnostic_float64_unclamped_decode": diagnostic,
            "gamma_sensitivity": gamma_sensitivity,
        }
        del physical, robust_z, soft64, canonical_soft, canonical_decode, diagnostic_decode

    all_channel = {
        "physical_relative_l2": (
            float(np.sqrt(global_error_sq / global_truth_sq)) if global_truth_sq else None
        ),
        "error_squared_sum": global_error_sq,
        "truth_squared_sum": global_truth_sq,
        "clamped_voxel_error_squared_sum": global_clamped_error_sq,
        "unclamped_voxel_error_squared_sum": global_unclamped_error_sq,
        "clamped_voxel_error_contribution_fraction": (
            float(global_clamped_error_sq / global_error_sq) if global_error_sq else 0.0
        ),
        "unclamped_voxel_error_contribution_fraction": (
            float(global_unclamped_error_sq / global_error_sq) if global_error_sq else 0.0
        ),
        "clamped_voxel_relative_l2_component": (
            float(np.sqrt(global_clamped_error_sq / global_truth_sq))
            if global_truth_sq
            else None
        ),
    }
    for name, channel in channels.items():
        channel_error = float(
            channel["paper_roundtrip"]["clamped_voxels"]["error_squared_sum"]
        )
        channel["paper_roundtrip"][
            "clamped_voxel_contribution_to_all_channel_error_squared"
        ] = float(channel_error / global_error_sq) if global_error_sq else 0.0
    gamma_global_report = {}
    for mode, sums in gamma_global.items():
        gamma_global_report[mode] = {
            **sums,
            "physical_relative_l2": (
                float(np.sqrt(sums["error_squared_sum"] / sums["truth_squared_sum"]))
                if sums["truth_squared_sum"]
                else None
            ),
            "paper_canonical": mode == "gamma_6_paper_canonical",
        }
    append_metric_rows(
        rows,
        record_type="all_channel_roundtrip",
        split=split_name,
        channel="all",
        metrics=all_channel,
        paper_canonical=True,
    )
    return {
        "snapshot_indices": list(snapshot_indices),
        "snapshot_count": len(snapshot_indices),
        "channels": channels,
        "all_channel_paper_roundtrip": all_channel,
        "gamma_sensitivity_all_channel": gamma_global_report,
        "spatial_definition": {
            "shell_edges": shell_edges.tolist(),
            "shell_mode": "8 log-spaced spherical-r shells",
            "equatorial_band": "abs(theta-pi/2) <= pi/8",
            "polar_caps": "theta <= pi/4 or theta >= 3pi/4",
            "inner_region": "first two spherical-r shells",
        },
    }


def focus_comparison(splits: dict[str, Any]) -> dict[str, Any]:
    comparison = {}
    for name in FOCUS_CHANNELS:
        train = splits["train"]["channels"][name]["canonical_inverse_clamp"]
        validation = splits["validation"]["channels"][name]["canonical_inverse_clamp"]
        train_fraction = float(train["total_fraction"])
        validation_fraction = float(validation["total_fraction"])
        comparison[name] = {
            "train_clamp_fraction": train_fraction,
            "validation_clamp_fraction": validation_fraction,
            "validation_minus_train": validation_fraction - train_fraction,
            "validation_to_train_ratio": (
                validation_fraction / train_fraction if train_fraction else None
            ),
            "train_region_concentration": train["region_concentration"],
            "validation_region_concentration": validation["region_concentration"],
            "train_snapshot_concentration": train["snapshot_concentration"],
            "validation_snapshot_concentration": validation["snapshot_concentration"],
        }
    return comparison


def plot_focus(report: dict[str, Any], figure_dir: Path) -> list[str]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    r = np.asarray(report["coordinates"]["r"], dtype=np.float64)
    theta = np.asarray(report["coordinates"]["theta"], dtype=np.float64)
    phi = np.asarray(report["coordinates"]["phi"], dtype=np.float64)
    paths = []
    for split_name in ("train", "validation"):
        for channel_name in FOCUS_CHANNELS:
            clamp = report["splits"][split_name]["channels"][channel_name][
                "canonical_inverse_clamp"
            ]
            fixed = np.asarray(clamp["fixed_phi"]["total_fraction_map"])
            equatorial = np.asarray(clamp["equatorial"]["total_fraction_map"])
            snapshot_records = clamp["snapshot_records"]
            shell_records = clamp["shell_records"]
            fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
            fig.patch.set_facecolor("white")
            image = axes[0, 0].pcolormesh(r, theta, fixed, shading="auto", vmin=0, vmax=1)
            axes[0, 0].set_xscale("log")
            axes[0, 0].set_title(
                f"fixed-phi theta-r clamp fraction (phi={clamp['fixed_phi']['phi_value']:.3g})"
            )
            axes[0, 0].set_xlabel("spherical r")
            axes[0, 0].set_ylabel("theta")
            fig.colorbar(image, ax=axes[0, 0])
            image = axes[0, 1].pcolormesh(r, phi, equatorial, shading="auto", vmin=0, vmax=1)
            axes[0, 1].set_xscale("log")
            axes[0, 1].set_title(
                f"equatorial phi-r clamp fraction (theta={clamp['equatorial']['theta_value']:.3g})"
            )
            axes[0, 1].set_xlabel("spherical r")
            axes[0, 1].set_ylabel("phi")
            fig.colorbar(image, ax=axes[0, 1])
            axes[1, 0].plot(
                [record["snapshot_index"] for record in snapshot_records],
                [record["total_fraction"] for record in snapshot_records],
                marker="o",
                markersize=2,
            )
            axes[1, 0].set_title("per-snapshot clamp fraction")
            axes[1, 0].set_xlabel("snapshot index")
            axes[1, 0].set_ylabel("fraction")
            axes[1, 0].set_ylim(0, 1)
            axes[1, 1].bar(
                [record["shell"] for record in shell_records],
                [
                    0.0 if record["total_fraction"] is None else record["total_fraction"]
                    for record in shell_records
                ],
            )
            axes[1, 1].set_title("per spherical-r shell clamp fraction")
            axes[1, 1].set_xlabel("shell")
            axes[1, 1].set_ylabel("fraction")
            axes[1, 1].set_ylim(0, 1)
            fig.suptitle(f"{split_name} {channel_name}: canonical gamma=6 / 0.99 gamma")
            path = figure_dir / f"{split_name}_{channel_name}_clamp_distribution.png"
            fig.savefig(path, dpi=150, facecolor="white", transparent=False)
            plt.close(fig)
            paths.append(str(path))
    return paths


def plot_gamma_sensitivity(report: dict[str, Any], figure_dir: Path) -> list[str]:
    paths = []
    mode_labels = [mode for mode, _, _ in GAMMA_MODES]
    for split_name in ("train", "validation"):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
        fig.patch.set_facecolor("white")
        for channel_name in CHANNELS:
            metrics = report["splits"][split_name]["channels"][channel_name][
                "gamma_sensitivity"
            ]
            axes[0].plot(
                range(len(mode_labels)),
                [metrics[mode]["clamp_hit_fraction"] for mode in mode_labels],
                marker="o",
                label=channel_name,
            )
            axes[1].plot(
                range(len(mode_labels)),
                [metrics[mode]["roundtrip"]["relative_l2"] for mode in mode_labels],
                marker="o",
                label=channel_name,
            )
        for axis in axes:
            axis.set_xticks(range(len(mode_labels)), ["gamma=6", "gamma=8", "gamma=12", "no clip"])
            axis.tick_params(axis="x", rotation=20)
            axis.grid(True, alpha=0.3)
        axes[0].set_ylabel("inverse clamp hit fraction")
        axes[0].set_title("Saturation/clamp sensitivity")
        axes[1].set_yscale("symlog", linthresh=1e-8)
        axes[1].set_ylabel("physical round-trip relative L2")
        axes[1].set_title("Round-trip sensitivity")
        axes[1].legend(ncol=2, fontsize=8)
        fig.suptitle(
            f"{split_name}: gamma=8/12/no-clip are diagnostic extensions, not paper reproduction"
        )
        path = figure_dir / f"{split_name}_gamma_sensitivity.png"
        fig.savefig(path, dpi=150, facecolor="white", transparent=False)
        plt.close(fig)
        paths.append(str(path))
    return paths


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Paper preprocessing clamp-loss audit",
        "",
        f"- Status: `{report['status']}`",
        f"- Protocol: `{report['protocol_name']}`",
        f"- HDF5 SHA-256: `{report['source_hdf5_checksum']}`",
        "- Canonical path: `gamma=6`, inverse clamp `0.99*gamma`.",
        "- `gamma=8`, `gamma=12`, and no-soft-clip are diagnostic extensions only.",
        "- No training, checkpoint writing, or canonical-normalizer modification occurred.",
        "",
        "## Canonical inverse-clamp and round-trip",
        "",
        "| split | channel | + clamp | - clamp | total clamp | round-trip rel L2 | clamped error contribution to all-channel error² |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split_name in ("train", "validation"):
        split = report["splits"][split_name]
        for channel_name in CHANNELS:
            channel = split["channels"][channel_name]
            clamp = channel["canonical_inverse_clamp"]
            roundtrip = channel["paper_roundtrip"]
            lines.append(
                f"| {split_name} | {channel_name} | {clamp['positive_fraction']:.6g} | "
                f"{clamp['negative_fraction']:.6g} | {clamp['total_fraction']:.6g} | "
                f"{roundtrip['all_voxels']['relative_l2']:.6g} | "
                f"{roundtrip['clamped_voxel_contribution_to_all_channel_error_squared']:.6g} |"
            )
    lines.extend(["", "## Bcc3 and vel3 concentration", ""])
    for channel_name, comparison in report["focus_comparison"].items():
        lines.append(f"### {channel_name}")
        lines.append("")
        lines.append(
            f"Train/validation clamp fractions are `{comparison['train_clamp_fraction']:.6g}` / "
            f"`{comparison['validation_clamp_fraction']:.6g}`."
        )
        for split_name in ("train", "validation"):
            regions = comparison[f"{split_name}_region_concentration"]
            snapshots = comparison[f"{split_name}_snapshot_concentration"]
            eq = regions["equatorial_band_abs_theta_minus_pi_over_2_le_pi_over_8"]
            polar = regions["polar_caps_theta_le_pi_over_4_or_ge_3pi_over_4"]
            inner = regions["inner_two_log_shells"]
            lines.append(
                f"- {split_name}: equatorial/polar/inner enrichment = "
                f"`{eq['enrichment_over_uniform']:.4g}` / "
                f"`{polar['enrichment_over_uniform']:.4g}` / "
                f"`{inner['enrichment_over_uniform']:.4g}`; top-5 snapshots contain "
                f"`{snapshots['top_5_snapshot_hit_share']:.4g}` of clamp hits."
            )
        lines.append("")
    lines.extend(
        [
            "## Diagnostic float64 unclamped decode",
            "",
            "| split | channel | float32 exact ±gamma | float64 exact ±gamma | diagnostic nonfinite | canonical-vs-diagnostic rel L2 (finite subset) |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for split_name in ("train", "validation"):
        for channel_name in CHANNELS:
            diagnostic = report["splits"][split_name]["channels"][channel_name][
                "diagnostic_float64_unclamped_decode"
            ]
            exact32 = (
                diagnostic["float32_exact_positive_gamma_fraction"]
                + diagnostic["float32_exact_negative_gamma_fraction"]
            )
            exact64 = (
                diagnostic["float64_exact_positive_gamma_fraction"]
                + diagnostic["float64_exact_negative_gamma_fraction"]
            )
            relative = diagnostic["paper_clamped_vs_float64_unclamped_on_finite"]["relative_l2"]
            lines.append(
                f"| {split_name} | {channel_name} | {exact32:.6g} | {exact64:.6g} | "
                f"{diagnostic['float64_unclamped_decode_nonfinite_fraction']:.6g} | "
                f"{relative if relative is not None else 'null'} |"
            )
    lines.extend(
        [
            "",
            "## Gamma sensitivity (diagnostic extensions)",
            "",
            "| split | mode | all-channel physical round-trip relative L2 | paper canonical |",
            "| --- | --- | ---: | --- |",
        ]
    )
    for split_name in ("train", "validation"):
        for mode, metrics in report["splits"][split_name][
            "gamma_sensitivity_all_channel"
        ].items():
            lines.append(
                f"| {split_name} | {mode} | {metrics['physical_relative_l2']:.6g} | "
                f"{metrics['paper_canonical']} |"
            )
    lines.extend(
        [
            "",
            "The no-soft-clip path is a numerical diagnostic, not an alternative selected by",
            "validation. Canonical training and evaluation remain fixed at gamma=6 and 0.99 gamma.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/data/paper_reduced100.yaml"))
    parser.add_argument(
        "--normalizer",
        type=Path,
        default=Path("outputs/paper_reduced100/stats/normalizer.npz"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_reduced100"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    normalizer_path = args.normalizer if args.normalizer.is_absolute() else project_root / args.normalizer
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol = PaperReduced100Protocol.from_yaml(config_path, project_root=project_root)
    preprocessor = PaperPreprocessor.load(
        normalizer_path,
        h5_path=protocol.dataset_path,
        expected_training_indices=protocol.train_indices,
        expected_protocol_name=protocol.protocol_name,
    )
    if preprocessor.gamma != 6 or preprocessor.inverse_clamp_fraction != 0.99:
        raise ValueError("Canonical paper gamma/inverse clamp changed")
    upstream_root = project_root / "external/neuraloperator"
    upstream_commit = git_output(["git", "rev-parse", "HEAD"], upstream_root)
    if upstream_commit != protocol.expected_upstream_commit:
        raise ValueError("Fixed upstream commit changed")
    if git_output(["git", "status", "--short"], upstream_root):
        raise RuntimeError("Refusing audit with a dirty upstream worktree")

    rows: list[dict[str, Any]] = []
    with h5py.File(protocol.dataset_path, "r") as handle:
        coordinates = {
            axis: np.asarray(handle[f"coords/{axis}"][...], dtype=np.float64).tolist()
            for axis in ("r", "theta", "phi")
        }
        splits = {
            "train": audit_split(
                handle,
                preprocessor,
                split_name="train",
                snapshot_indices=protocol.train_indices,
                rows=rows,
            ),
            "validation": audit_split(
                handle,
                preprocessor,
                split_name="validation",
                snapshot_indices=protocol.validation_indices,
                rows=rows,
            ),
        }
    report = {
        "schema_version": "paper-preprocessing-loss-audit-v1",
        "status": "completed_with_canonical_information_loss",
        "protocol_name": protocol.protocol_name,
        "source_hdf5": str(protocol.dataset_path),
        "source_hdf5_checksum": preprocessor.source_hdf5_checksum,
        "normalizer_path": str(normalizer_path),
        "normalizer_protocol": preprocessor.protocol_name,
        "thermal_channel": protocol.thermal_channel,
        "paper_adaptation": protocol.paper_adaptation,
        "eos_conversion": protocol.eos_conversion,
        "canonical": {
            "gamma": preprocessor.gamma,
            "inverse_clamp_fraction": preprocessor.inverse_clamp_fraction,
            "inverse_clamp_limit": preprocessor.gamma * preprocessor.inverse_clamp_fraction,
            "modified_by_audit": False,
        },
        "diagnostic_extensions": {
            "float64_unclamped_atanh_decode": "diagnostic_only_not_training_or_paper_evaluation",
            "gamma_8": "diagnostic_extension_not_paper_reproduction",
            "gamma_12": "diagnostic_extension_not_paper_reproduction",
            "no_soft_clip": "diagnostic_extension_not_paper_reproduction",
            "validation_used_for_selection": False,
        },
        "coordinates": coordinates,
        "splits": splits,
        "focus_comparison": focus_comparison(splits),
        "project_git_commit": git_output(["git", "rev-parse", "HEAD"], project_root),
        "project_git_dirty": bool(git_output(["git", "status", "--short"], project_root)),
        "upstream_commit": upstream_commit,
        "upstream_worktree_clean": True,
        "command": shlex.join([sys.executable, *sys.argv]),
    }
    figure_dir = output_dir / "preprocessing_loss_figures"
    report["figures"] = plot_focus(report, figure_dir) + plot_gamma_sensitivity(
        report, figure_dir
    )

    json_path = output_dir / "preprocessing_loss_audit.json"
    csv_path = output_dir / "preprocessing_loss_audit.csv"
    markdown_path = output_dir / "preprocessing_loss_audit.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    fieldnames = (
        "record_type",
        "split",
        "channel",
        "stage",
        "scope",
        "index",
        "shell",
        "gamma_mode",
        "paper_canonical",
        "metric",
        "value",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "json": str(json_path),
                "csv": str(csv_path),
                "markdown": str(markdown_path),
                "figures": report["figures"],
                "train_all_channel_relative_l2": splits["train"][
                    "all_channel_paper_roundtrip"
                ]["physical_relative_l2"],
                "validation_all_channel_relative_l2": splits["validation"][
                    "all_channel_paper_roundtrip"
                ]["physical_relative_l2"],
                "focus_comparison": report["focus_comparison"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
