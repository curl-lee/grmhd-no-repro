"""Physical-space statistics for Stage G validation and long rollouts."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import torch

from . import CHANNELS


def radial_shell_indices(r: np.ndarray, n_shells: int = 8) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(r, dtype=np.float64)
    if r.ndim != 1 or r.size == 0 or np.any(r <= 0) or n_shells <= 0:
        raise ValueError("Stage G radial shells require positive one-dimensional radii")
    edges = np.geomspace(
        float(r.min()), float(r.max()) * (1.0 + 1.0e-12), n_shells + 1
    )
    indices = np.clip(np.searchsorted(edges, r, side="right") - 1, 0, n_shells - 1)
    return edges, indices


def physical_state_statistics(
    state: torch.Tensor, r: np.ndarray, *, n_shells: int = 8
) -> dict[str, Any]:
    """Summarize stored spherical-coordinate fields without metric weighting."""

    if state.ndim != 5 or state.shape[0] != 1 or state.shape[1] != len(CHANNELS):
        raise ValueError("Stage G state statistics expect shape (1,8,Nphi,Ntheta,Nr)")
    values = state[0].detach().float().cpu().numpy().astype(np.float64, copy=False)
    if not np.isfinite(values).all():
        raise FloatingPointError("Stage G physical state contains NaN/Inf")
    edges, shell_index = radial_shell_indices(r, n_shells)
    quantiles = np.asarray([0.01, 0.1, 0.5, 0.9, 0.99])
    channels: dict[str, Any] = {}
    for channel, name in enumerate(CHANNELS):
        quantity = values[channel]
        shells = []
        for shell in range(n_shells):
            selected = quantity[..., shell_index == shell]
            shells.append(
                {
                    "mean": float(selected.mean()),
                    "std": float(selected.std()),
                    "minimum": float(selected.min()),
                    "maximum": float(selected.max()),
                }
            )
        channels[name] = {
            "mean": float(quantity.mean()),
            "std": float(quantity.std()),
            "minimum": float(quantity.min()),
            "maximum": float(quantity.max()),
            "quantiles": np.quantile(quantity, quantiles).tolist(),
            "radial_mean": quantity.mean(axis=(0, 1)).tolist(),
            "radial_std": quantity.std(axis=(0, 1)).tolist(),
            "shells": shells,
            "outermost_two_shells": shells[-2:],
        }
    return {
        "coordinate_semantics": (
            "native spherical Kerr-Schild stored components; unweighted grid statistics"
        ),
        "shell_edges": edges.tolist(),
        "quantile_levels": quantiles.tolist(),
        "channels": channels,
    }


def total_variation_and_high_k(state: torch.Tensor) -> dict[str, Any]:
    """Compute compact grid-space roughness diagnostics for every channel."""

    if state.ndim != 5 or state.shape[0] != 1 or state.shape[1] != len(CHANNELS):
        raise ValueError("Stage G spectral diagnostics expect shape (1,8,...)")
    values = state[0].detach().float()
    axes = (-3, -2, -1)
    total_variation = torch.stack(
        [
            torch.stack(
                [torch.diff(values[channel], dim=axis).abs().mean() for axis in axes]
            )
            for channel in range(len(CHANNELS))
        ]
    )
    spectrum = torch.fft.rfftn(values, dim=axes)
    energy = spectrum.abs().square()
    nphi, ntheta, nr = values.shape[-3:]
    fphi = torch.fft.fftfreq(nphi, device=values.device).abs().reshape(-1, 1, 1)
    ftheta = torch.fft.fftfreq(ntheta, device=values.device).abs().reshape(1, -1, 1)
    fr = torch.fft.rfftfreq(nr, device=values.device).abs().reshape(1, 1, -1)
    high_mask = torch.maximum(torch.maximum(fphi, ftheta), fr) >= (1.0 / 3.0)
    high = energy[:, high_mask].sum(dim=1)
    total = energy.flatten(start_dim=1).sum(dim=1).clamp_min(1e-30)
    high_fraction = high / total
    return {
        "per_channel": {
            name: {
                "total_variation_by_axis": total_variation[index].cpu().tolist(),
                "total_variation_mean": float(total_variation[index].mean().cpu()),
                "axis_anisotropy": float(
                    total_variation[index].max()
                    / total_variation[index].min().clamp_min(1e-30)
                ),
                "high_k_energy_fraction": float(high_fraction[index].cpu()),
            }
            for index, name in enumerate(CHANNELS)
        }
    }


def artifact_diagnostics(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    reference_input: torch.Tensor,
    *,
    reference_kind: str,
) -> dict[str, Any]:
    """Flag collapse/ripple/stripe against an explicit GT or initial reference."""

    if prediction.shape != reference.shape or prediction.shape != reference_input.shape:
        raise ValueError("Stage G artifact diagnostic states must align")
    diagnostics: dict[str, Any] = {
        "reference_kind": reference_kind,
        "channels": {},
        "flags": [],
    }
    for channel, name in enumerate(CHANNELS):
        pred = prediction[0, channel]
        truth = reference[0, channel]
        prior = reference_input[0, channel]
        pred_std = float(pred.std().cpu())
        truth_std = float(truth.std().cpu())
        std_ratio = pred_std / max(truth_std, 1e-12)
        pred_diff = [
            float(torch.mean(torch.diff(pred, dim=axis).square()).cpu())
            for axis in range(3)
        ]
        truth_diff = [
            float(torch.mean(torch.diff(truth, dim=axis).square()).cpu())
            for axis in range(3)
        ]
        prior_diff = [
            float(torch.mean(torch.diff(prior, dim=axis).square()).cpu())
            for axis in range(3)
        ]
        roughness_ratio = sum(pred_diff) / max(sum(truth_diff), 1e-12)
        creation_ratio = sum(pred_diff) / max(
            sum(truth_diff), sum(prior_diff), 1e-12
        )
        anisotropy = max(pred_diff) / max(min(pred_diff), 1e-12)
        prior_anisotropy = max(prior_diff) / max(min(prior_diff), 1e-12)
        anisotropy_excess = anisotropy / max(prior_anisotropy, 1e-12)
        diagnostics["channels"][name] = {
            "std_ratio_prediction_over_reference": std_ratio,
            "first_difference_energy_ratio": roughness_ratio,
            "created_roughness_ratio": creation_ratio,
            "axis_anisotropy_ratio": anisotropy,
            "axis_anisotropy_excess_over_input": anisotropy_excess,
        }
        if std_ratio < 0.05:
            diagnostics["flags"].append(f"{name}:possible_field_collapse")
        if creation_ratio > 5.0:
            diagnostics["flags"].append(f"{name}:possible_high_frequency_ripple")
        if anisotropy_excess > 10.0:
            diagnostics["flags"].append(f"{name}:possible_stripe_anisotropy")
    return diagnostics


def temporal_series_statistics(
    series: Iterable[Iterable[float]],
) -> dict[str, Any]:
    """Summarize channel time series with autocorrelation and temporal PSD."""

    values = np.asarray(list(series), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(CHANNELS):
        raise ValueError("Temporal Stage G series must have shape (steps,8)")
    if not np.isfinite(values).all():
        raise FloatingPointError("Temporal Stage G series contains NaN/Inf")
    output: dict[str, Any] = {}
    for channel, name in enumerate(CHANNELS):
        item = values[:, channel]
        centered = item - item.mean()
        denominator = float(np.dot(centered, centered))
        autocorrelation = {}
        for lag in (1, 5, 10, 19):
            autocorrelation[str(lag)] = (
                None
                if lag >= len(item) or denominator == 0
                else float(np.dot(centered[:-lag], centered[lag:]) / denominator)
            )
        spectrum = np.fft.rfft(centered)
        power = np.abs(spectrum) ** 2
        frequencies = np.fft.rfftfreq(len(item))
        nonzero_power = power[1:]
        if nonzero_power.size and float(nonzero_power.sum()) > 0:
            dominant_index = int(np.argmax(nonzero_power)) + 1
            dominant_frequency = float(frequencies[dominant_index])
            high_temporal_fraction = float(
                power[frequencies >= 0.25].sum() / power.sum()
            )
        else:
            dominant_frequency = None
            high_temporal_fraction = 0.0
        output[name] = {
            "mean": float(item.mean()),
            "std": float(item.std()),
            "minimum": float(item.min()),
            "maximum": float(item.max()),
            "autocorrelation": autocorrelation,
            "dominant_temporal_frequency": dominant_frequency,
            "high_temporal_frequency_power_fraction": high_temporal_fraction,
        }
    return {"steps": int(values.shape[0]), "channels": output}
