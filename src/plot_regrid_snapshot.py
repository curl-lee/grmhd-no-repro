#!/usr/bin/env python
"""Create GRMHD regrid slice comparisons and radial profiles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
os.environ.setdefault("MPLCONFIGDIR", str((Path("outputs") / ".matplotlib").resolve()))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from grmhd import CHANNELS


POSITIVE_CHANNELS = {"rho", "press"}


def decode(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


def display_values(values: np.ndarray, channel: str) -> tuple[np.ndarray, object, str]:
    if channel in POSITIVE_CHANNELS:
        positive = values[values > 0]
        floor = float(positive.min()) if positive.size else np.finfo(np.float32).tiny
        shown = np.log10(np.maximum(values, floor))
        return shown, None, f"log10({channel})"
    limit = float(np.nanpercentile(np.abs(values), 99.5))
    if not np.isfinite(limit) or limit <= 0:
        limit = 1.0
    return values, TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), channel


def plot_single_snapshot(
    snapshot: np.ndarray,
    time: float,
    channels: list[str],
    r: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    out_dir: Path,
    index: int,
) -> None:
    phi_index = len(phi) // 2
    theta_index = int(np.argmin(np.abs(theta - np.pi / 2)))
    for kind, slice_values, vertical, vertical_label in (
        ("theta_r", snapshot[:, phi_index, :, :], theta / np.pi, r"$\theta/\pi$"),
        ("phi_r", snapshot[:, :, theta_index, :], phi / (2 * np.pi), r"$\phi/(2\pi)$"),
    ):
        fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
        for channel_index, (channel, axis) in enumerate(zip(channels, axes.flat, strict=True)):
            values, norm, label = display_values(slice_values[channel_index], channel)
            mesh = axis.pcolormesh(r, vertical, values, shading="auto", cmap="coolwarm", norm=norm)
            axis.set_xscale("log")
            axis.set_xlabel("r")
            axis.set_ylabel(vertical_label)
            axis.set_title(channel)
            fig.colorbar(mesh, ax=axis, label=label)
        fig.suptitle(f"snapshot {index}, t={time:.9g}: {kind}")
        fig.savefig(out_dir / f"snapshot_{index:04d}_{kind}.png", dpi=150)
        plt.close(fig)


def plot_time_comparison(
    snapshots: list[np.ndarray],
    indices: list[int],
    times: list[float],
    channels: list[str],
    r: np.ndarray,
    theta: np.ndarray,
    out_dir: Path,
) -> None:
    phi_index = snapshots[0].shape[1] // 2
    fig, axes = plt.subplots(
        len(channels), len(snapshots), figsize=(5 * len(snapshots), 2.7 * len(channels)), squeeze=False,
        constrained_layout=True,
    )
    for column, snapshot in enumerate(snapshots):
        for row, channel in enumerate(channels):
            values, norm, label = display_values(snapshot[row, phi_index], channel)
            mesh = axes[row, column].pcolormesh(
                r, theta / np.pi, values, shading="auto", cmap="coolwarm", norm=norm
            )
            axes[row, column].set_xscale("log")
            axes[row, column].set_xlabel("r")
            axes[row, column].set_ylabel(r"$\theta/\pi$")
            axes[row, column].set_title(f"{channel}: i={indices[column]}, t={times[column]:.6g}")
            fig.colorbar(mesh, ax=axes[row, column], label=label)
    fig.savefig(out_dir / "time_comparison_theta_r.png", dpi=140)
    plt.close(fig)


def radial_profiles(
    snapshots: list[np.ndarray],
    indices: list[int],
    times: list[float],
    channels: list[str],
    r: np.ndarray,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    activity = {"rho": np.zeros_like(r), "press": np.zeros_like(r), "B": np.zeros_like(r), "v": np.zeros_like(r)}
    for snapshot, index, time in zip(snapshots, indices, times, strict=True):
        for channel_index, (channel, axis) in enumerate(zip(channels, axes.flat, strict=True)):
            values = snapshot[channel_index]
            median = np.median(values, axis=(0, 1))
            low, high = np.percentile(values, [10, 90], axis=(0, 1))
            if channel in POSITIVE_CHANNELS:
                floor = np.finfo(np.float32).tiny
                median, low, high = [np.maximum(item, floor) for item in (median, low, high)]
                axis.set_yscale("log")
            axis.plot(r, median, label=f"i={index}, t={time:.4g}")
            axis.fill_between(r, low, high, alpha=0.12)
            axis.set_xscale("log")
            axis.set_title(channel)
            axis.set_xlabel("r")
            axis.legend(fontsize=7)
        activity["rho"] += np.mean(np.abs(snapshot[channels.index("rho")]), axis=(0, 1))
        activity["press"] += np.mean(np.abs(snapshot[channels.index("press")]), axis=(0, 1))
        activity["B"] += np.mean(np.sqrt(np.sum(snapshot[0:3] ** 2, axis=0)), axis=(0, 1))
        activity["v"] += np.mean(np.sqrt(np.sum(snapshot[5:8] ** 2, axis=0)), axis=(0, 1))
    fig.savefig(out_dir / "radial_profiles.png", dpi=150)
    plt.close(fig)

    cutoffs: dict[str, dict[str, float | int | None]] = {}
    for cutoff in (100.0, 200.0):
        mask = r <= cutoff
        cutoffs[str(int(cutoff))] = {}
        for name, profile in activity.items():
            total = float(profile.sum())
            cutoffs[str(int(cutoff))][f"{name}_sampled_activity_fraction"] = (
                float(profile[mask].sum() / total) if total > 0 else None
            )
        cutoffs[str(int(cutoff))]["radial_bins"] = int(mask.sum())
    summary = {
        "r_min": float(r.min()),
        "r_max": float(r.max()),
        "snapshots_used": indices,
        "cutoffs": cutoffs,
        "interpretation_warning": (
            "These are fractions of mean-absolute activity sampled on a log-r grid, not "
            "metric/volume-weighted physical integrals. Use them with slice morphology to "
            "choose an inner-domain cutoff."
        ),
    }
    (out_dir / "radial_profile_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def parse_indices(specification: str, length: int) -> list[int]:
    result: list[int] = []
    for item in specification.split(","):
        index = int(item)
        index = length + index if index < 0 else index
        if not 0 <= index < length:
            raise IndexError(f"snapshot index {item} is outside [0, {length})")
        if index not in result:
            result.append(index)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/figures"))
    parser.add_argument("--indices", default="0,-1")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.input, "r") as handle:
        channels = decode(handle["channels"][...])
        if channels != list(CHANNELS):
            raise ValueError(f"Unexpected channel order: {channels}")
        indices = parse_indices(args.indices, len(handle["times"]))
        r = handle["coords/r"][...]
        theta = handle["coords/theta"][...]
        phi = handle["coords/phi"][...]
        times = [float(handle["times"][index]) for index in indices]
        snapshots = [np.asarray(handle["snapshots"][index], dtype=np.float32) for index in indices]
    for snapshot, index, time in zip(snapshots, indices, times, strict=True):
        if not np.isfinite(snapshot).all():
            raise FloatingPointError(f"snapshot {index} contains NaN/Inf")
        plot_single_snapshot(snapshot, time, channels, r, theta, phi, args.out_dir, index)
    plot_time_comparison(snapshots, indices, times, channels, r, theta, args.out_dir)
    radial_profiles(snapshots, indices, times, channels, r, args.out_dir)
    print(f"wrote regrid diagnostics to {args.out_dir}")


if __name__ == "__main__":
    main()
