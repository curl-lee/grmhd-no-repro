"""Leakage-safe Fold B temporal baselines and sampling policies."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

from . import CHANNELS
from .dataset import GRMHDPairedDataset


def recency_weights(indices: Sequence[int], tau: float = 20.0) -> torch.Tensor:
    """Exponentially weight history, normalized to mean one."""
    if tau <= 0:
        raise ValueError("tau must be positive")
    if not indices:
        raise ValueError("indices must not be empty")
    values = torch.as_tensor(indices, dtype=torch.float64)
    weights = torch.exp((values - torch.max(values)) / tau)
    return weights / torch.mean(weights)


def recent_window_indices(
    training_indices: Sequence[int], start: int = 40, stop: int = 80
) -> tuple[int, ...]:
    selected = tuple(index for index in training_indices if start <= index < stop)
    if not selected:
        raise ValueError("recent window has no training snapshots")
    return selected


def fit_trend_alpha(
    trajectory: torch.Tensor,
) -> tuple[float, tuple[float, ...]]:
    """Fit global and channel-wise trend coefficients on one training trajectory."""
    if trajectory.ndim < 3 or trajectory.shape[1] != len(CHANNELS) or trajectory.shape[0] < 3:
        raise ValueError("trajectory must have shape (time,8,...) with at least three states")
    previous_delta = (trajectory[1:-1] - trajectory[:-2]).double()
    next_delta = (trajectory[2:] - trajectory[1:-1]).double()
    denominator_global = float(previous_delta.square().sum())
    global_alpha = (
        float((previous_delta * next_delta).sum()) / denominator_global
        if denominator_global > 0
        else 0.0
    )
    spatial_dims = (0,) + tuple(range(2, previous_delta.ndim))
    numerator = (previous_delta * next_delta).sum(dim=spatial_dims)
    denominator = previous_delta.square().sum(dim=spatial_dims)
    channel_alpha = torch.where(
        denominator > 0, numerator / denominator, torch.zeros_like(denominator)
    )
    return global_alpha, tuple(float(value) for value in channel_alpha)


def fit_trend_alpha_streaming(
    dataset: GRMHDPairedDataset,
) -> tuple[float, tuple[float, ...]]:
    """Fit using only snapshots owned by a split named train."""
    if dataset.split.name != "train":
        raise ValueError("Trend alpha may only be fit on a split named 'train'")
    indices = dataset.owned_snapshot_indices
    if len(indices) < 3:
        raise ValueError("Trend alpha needs at least three training snapshots")
    global_numerator = 0.0
    global_denominator = 0.0
    channel_numerator = np.zeros(len(CHANNELS), dtype=np.float64)
    channel_denominator = np.zeros(len(CHANNELS), dtype=np.float64)
    previous = dataset.load_snapshot(indices[0]).double()
    current = dataset.load_snapshot(indices[1]).double()
    for index in indices[2:]:
        following = dataset.load_snapshot(index).double()
        past_delta = current - previous
        future_delta = following - current
        global_numerator += float((past_delta * future_delta).sum())
        global_denominator += float(past_delta.square().sum())
        reduction = tuple(range(1, past_delta.ndim))
        channel_numerator += (past_delta * future_delta).sum(dim=reduction).numpy()
        channel_denominator += past_delta.square().sum(dim=reduction).numpy()
        previous, current = current, following
    global_alpha = global_numerator / global_denominator if global_denominator > 0 else 0.0
    channel_alpha = np.divide(
        channel_numerator,
        channel_denominator,
        out=np.zeros_like(channel_numerator),
        where=channel_denominator > 0,
    )
    return float(global_alpha), tuple(float(value) for value in channel_alpha)
