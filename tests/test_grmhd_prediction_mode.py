from __future__ import annotations

import torch
from torch import nn

from grmhd.models import apply_prediction_mode, zero_initialize_residual_head


class TinyProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fcs = nn.ModuleList([nn.Conv3d(16, 8, kernel_size=1)])


class TinyOperator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = TinyProjection()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.projection.fcs[-1](inputs)


def test_residual_output_shape_zero_model_and_shell_exclusion(tmp_path):
    state = torch.randn(2, 8, 4, 3, 2)
    shells = torch.full((2, 8, 4, 3, 2), 999.0)
    operator_input = torch.cat([state, shells], dim=1)
    zero_delta = torch.zeros_like(state)
    prediction = apply_prediction_mode(
        zero_delta, operator_input, predict_residual=True
    )
    assert prediction.shape == state.shape
    assert torch.equal(prediction, state)
    assert not torch.any(prediction == 999.0)


def test_direct_mode_and_checkpoint_prediction_mode_roundtrip(tmp_path):
    state = torch.randn(1, 8, 2, 2, 2)
    direct = torch.randn_like(state)
    assert torch.equal(
        apply_prediction_mode(direct, state, predict_residual=False), direct
    )
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {"config": {"model": {"predict_residual": True}}, "model_state_dict": {}},
        path,
    )
    loaded = torch.load(path, weights_only=False)
    assert loaded["config"]["model"]["predict_residual"] is True


def test_zero_initialized_head_is_exact_persistence_with_shells():
    model = TinyOperator()
    head = zero_initialize_residual_head(model)
    assert head == "projection.fcs.0"
    state = torch.randn(2, 8, 3, 2, 2)
    shells = torch.randn_like(state)
    operator_input = torch.cat([state, shells], dim=1)
    raw_delta = model(operator_input)
    prediction = apply_prediction_mode(
        raw_delta,
        operator_input,
        predict_residual=True,
        bounded_residual=True,
        residual_scale=torch.linspace(0.01, 0.08, 8),
    )
    assert torch.equal(raw_delta, torch.zeros_like(raw_delta))
    assert torch.equal(prediction, state)


def test_bounded_residual_broadcast_bound_shell_exclusion_and_gradient():
    state = torch.randn(1, 8, 2, 3, 4)
    shells = torch.full_like(state, 999.0)
    operator_input = torch.cat([state, shells], dim=1)
    raw_delta = torch.full_like(state, 0.25, requires_grad=True)
    alpha = torch.linspace(0.01, 0.08, 8)
    prediction = apply_prediction_mode(
        raw_delta,
        operator_input,
        predict_residual=True,
        bounded_residual=True,
        residual_scale=alpha,
    )
    bounded_delta = prediction - state
    expected = alpha.reshape(1, 8, 1, 1, 1) * torch.tanh(
        raw_delta.detach() / alpha.reshape(1, 8, 1, 1, 1)
    )
    assert torch.allclose(bounded_delta, expected)
    measured = torch.amax(torch.abs(bounded_delta), dim=(0, 2, 3, 4))
    assert torch.all(measured <= alpha + 4 * torch.finfo(alpha.dtype).eps)
    assert not torch.any(prediction == 999.0)
    prediction.sum().backward()
    assert raw_delta.grad is not None
    assert torch.isfinite(raw_delta.grad).all()


def test_bounded_residual_checkpoint_roundtrip(tmp_path):
    model = TinyOperator()
    zero_initialize_residual_head(model)
    inputs = torch.randn(1, 16, 2, 2, 2)
    alpha = torch.linspace(0.1, 0.8, 8)
    before = apply_prediction_mode(
        model(inputs),
        inputs,
        predict_residual=True,
        bounded_residual=True,
        residual_scale=alpha,
    )
    path = tmp_path / "bounded.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "residual_wrapper": {
                "bounded_residual": True,
                "zero_init_residual_head": True,
                "alpha": alpha.tolist(),
            },
        },
        path,
    )
    loaded = torch.load(path, weights_only=False)
    restored = TinyOperator()
    restored.load_state_dict(loaded["model_state_dict"], strict=True)
    after = apply_prediction_mode(
        restored(inputs),
        inputs,
        predict_residual=True,
        bounded_residual=loaded["residual_wrapper"]["bounded_residual"],
        residual_scale=loaded["residual_wrapper"]["alpha"],
    )
    assert torch.equal(before, after)
