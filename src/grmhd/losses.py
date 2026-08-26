"""Training losses and metrics for normalized GRMHD states."""

from __future__ import annotations

import torch
from torch import nn

from .priors import QuantileBounds, ResidualEnvelope


def index_grid_h1(error: torch.Tensor) -> torch.Tensor:
    """Unweighted H1 norm on array axes (phi, theta, log-r index grid).

    This is not the covariant gradient norm for the Kerr-Schild metric.
    """
    value = torch.mean(error.square())
    gradients = []
    for dimension in range(-3, 0):
        gradients.append(torch.mean(torch.diff(error, dim=dimension).square()))
    return value + torch.stack(gradients).mean()


class WeightedGRMHDLoss(nn.Module):
    def __init__(
        self,
        *,
        lambda_B: float = 1.2,
        lambda_velocity: float = 1.0,
        lambda_H1: float = 0.05,
        quantile_bounds: QuantileBounds | None = None,
        lambda_quantile_bounds: float = 0.0,
        residual_envelope: ResidualEnvelope | None = None,
        lambda_residual_envelope: float = 0.0,
        lambda_velocity_roi: float = 0.0,
        velocity_roi_quantile: float = 0.80,
        lambda_dissipative: float = 0.0,
    ) -> None:
        super().__init__()
        self.lambda_B = float(lambda_B)
        self.lambda_velocity = float(lambda_velocity)
        self.lambda_H1 = float(lambda_H1)
        self.quantile_bounds = quantile_bounds
        self.lambda_quantile_bounds = float(lambda_quantile_bounds)
        self.residual_envelope = residual_envelope
        self.lambda_residual_envelope = float(lambda_residual_envelope)
        self.lambda_velocity_roi = float(lambda_velocity_roi)
        self.velocity_roi_quantile = float(velocity_roi_quantile)
        self.lambda_dissipative = float(lambda_dissipative)
        if not 0 <= self.velocity_roi_quantile < 1:
            raise ValueError("velocity_roi_quantile must lie in [0, 1)")

    def _velocity_roi(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        magnitude = torch.sqrt(torch.sum(target[:, 5:8].square(), dim=1))
        thresholds = torch.quantile(
            magnitude.flatten(start_dim=1), self.velocity_roi_quantile, dim=1
        ).view(-1, 1, 1, 1)
        mask = magnitude >= thresholds
        squared = torch.mean((prediction[:, 5:8] - target[:, 5:8]).square(), dim=1)
        return squared[mask].mean() if torch.any(mask) else squared.new_zeros(())

    @staticmethod
    def _dissipative_excess(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        penalties = []
        for dimension in range(-3, 0):
            prediction_energy = torch.mean(torch.diff(prediction, dim=dimension).square())
            target_energy = torch.mean(torch.diff(target, dim=dimension).square())
            penalties.append(torch.relu(prediction_energy - target_energy))
        return torch.stack(penalties).mean()

    def components(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        input_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if prediction.shape != target.shape or prediction.shape[1] != 8:
            raise ValueError(f"Expected matching (B,8,...) tensors, got {prediction.shape}, {target.shape}")
        error = prediction - target
        b = torch.mean(error[:, 0:3].square())
        rho = torch.mean(error[:, 3:4].square())
        press = torch.mean(error[:, 4:5].square())
        velocity = torch.mean(error[:, 5:8].square())
        weighted_l2 = self.lambda_B * b + rho + press + self.lambda_velocity * velocity
        h1 = index_grid_h1(error)
        zero = prediction.new_zeros(())
        quantile_penalty = (
            self.quantile_bounds.penalty(prediction)
            if self.quantile_bounds is not None and self.lambda_quantile_bounds != 0
            else zero
        )
        if self.residual_envelope is not None and self.lambda_residual_envelope != 0:
            if input_state is None:
                raise ValueError("input_state is required for residual-envelope loss")
            residual_penalty = self.residual_envelope.penalty(prediction, input_state)
        else:
            residual_penalty = zero
        velocity_roi = (
            self._velocity_roi(prediction, target) if self.lambda_velocity_roi != 0 else zero
        )
        dissipative = (
            self._dissipative_excess(prediction, target)
            if self.lambda_dissipative != 0
            else zero
        )
        total = (
            weighted_l2
            + self.lambda_H1 * h1
            + self.lambda_quantile_bounds * quantile_penalty
            + self.lambda_residual_envelope * residual_penalty
            + self.lambda_velocity_roi * velocity_roi
            + self.lambda_dissipative * dissipative
        )
        return {
            "loss": total,
            "weighted_l2": weighted_l2,
            "h1_index_grid": h1,
            "B_mse": b,
            "rho_mse": rho,
            "press_mse": press,
            "velocity_mse": velocity,
            "quantile_bounds_penalty": quantile_penalty,
            "residual_envelope_penalty": residual_penalty,
            "velocity_roi_mse": velocity_roi,
            "dissipative_excess_index_grid": dissipative,
        }

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.components(prediction, target)["loss"]


def channel_relative_l2(
    prediction: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-12
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes differ")
    dimensions = tuple(index for index in range(prediction.ndim) if index != 1)
    numerator = torch.sqrt(torch.sum((prediction - target).square(), dim=dimensions))
    denominator = torch.sqrt(torch.sum(target.square(), dim=dimensions)).clamp_min(epsilon)
    return numerator / denominator
