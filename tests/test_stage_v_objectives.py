from __future__ import annotations

import torch
import pytest

from grmhd.stage_v_objectives import (
    ObjectiveWeights,
    classify_gradient_conflict,
    combined_objective,
    normalized_radial_transport_loss,
    relative_transport_loss,
    residual_direction_loss,
    shell_variance,
)


def test_direction_loss_is_zero_for_matching_nonzero_residual() -> None:
    target = torch.randn(2, 8, 3, 4, 5, dtype=torch.float64)
    value = residual_direction_loss(target, target, epsilon=1e-24)
    assert float(value) < 1e-12


def test_direction_loss_has_finite_gradient_at_zero_prediction() -> None:
    predicted = torch.zeros(1, 8, 2, 2, 2, requires_grad=True)
    target = torch.randn_like(predicted)
    value = residual_direction_loss(predicted, target)
    value.backward()
    assert torch.isfinite(value)
    assert predicted.grad is not None and torch.isfinite(predicted.grad).all()
    assert torch.linalg.vector_norm(predicted.grad) > 0


def test_shell_variance_uses_frozen_radial_membership() -> None:
    state = torch.zeros(1, 1, 1, 1, 16)
    state[..., 1::2] = 2
    shell_index = torch.arange(16) // 2
    result = shell_variance(state, shell_index)
    assert result.shape == (1, 1, 8)
    assert torch.allclose(result, torch.ones_like(result))


def test_transport_losses_are_zero_for_exact_state() -> None:
    input_state = torch.randn(1, 8, 2, 3, 4)
    target = input_state + 0.1 * torch.randn_like(input_state)
    assert relative_transport_loss(input_state.flatten(2), target.flatten(2), target.flatten(2)) == 0
    assert normalized_radial_transport_loss(input_state, target, target) == 0


def test_combined_objective_applies_each_weight_once() -> None:
    components = {
        "plain": torch.tensor(2.0),
        "direction": torch.tensor(3.0),
        "transport": torch.tensor(5.0),
    }
    value = combined_objective(components, ObjectiveWeights(direction=0.1, transport=0.2))
    assert float(value) == pytest.approx(3.3)


def test_conflict_classification_is_predeclared() -> None:
    assert classify_gradient_conflict([-0.2, 0.1, -0.1]) == "STRONG"
    assert classify_gradient_conflict([0.1, 0.2, 0.3]) == "MODERATE"
    assert classify_gradient_conflict([0.3, 0.4, 0.6]) == "WEAK"
    assert classify_gradient_conflict([0.6, 0.7, 0.8]) == "NONE"
