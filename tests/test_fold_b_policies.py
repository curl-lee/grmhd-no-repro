from __future__ import annotations

import torch

from grmhd.fold_b import fit_trend_alpha, recency_weights, recent_window_indices


def test_trend_alpha_uses_only_supplied_training_trajectory():
    base = torch.arange(6, dtype=torch.float32).reshape(6, 1, 1).repeat(1, 8, 2)
    global_alpha, channel_alpha = fit_trend_alpha(base[:4])
    mutated_test = base.clone()
    mutated_test[4:] = 1.0e6
    repeated_global, repeated_channel = fit_trend_alpha(mutated_test[:4])
    assert global_alpha == repeated_global == 1.0
    assert channel_alpha == repeated_channel == (1.0,) * 8


def test_recency_weights_and_recent_window_are_train_only():
    indices = tuple(range(80))
    weights = recency_weights(indices, tau=20)
    assert weights.shape == (80,)
    assert torch.mean(weights) == 1.0
    assert torch.all(torch.diff(weights) > 0)
    assert recent_window_indices(indices) == tuple(range(40, 80))
