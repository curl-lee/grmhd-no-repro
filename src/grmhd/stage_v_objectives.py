"""Frozen differentiable objectives and gradient diagnostics for Stage V.

The Plain term is the exact Stage-T normalized-residual objective. Direction,
shell, and radial objectives stay in normalized space. The shell objective
keeps the frozen eight-shell variance-evolution structure without importing
the known nonlinear decoder-tail amplification into the training objective.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import torch

from .paper_losses import PlainL2Loss


OBJECTIVE_NAMES = ("plain", "direction", "shell", "radial", "transport")


def _validate_residuals(predicted: torch.Tensor, target: torch.Tensor) -> None:
    if predicted.shape != target.shape or predicted.ndim != 5 or predicted.shape[1] != 8:
        raise ValueError("Stage V residuals must share shape (B,8,Nphi,Ntheta,Nr)")
    if not predicted.is_floating_point() or not target.is_floating_point():
        raise TypeError("Stage V objectives require floating-point tensors")


def residual_direction_loss(
    predicted: torch.Tensor, target: torch.Tensor, *, epsilon: float = 1e-12
) -> torch.Tensor:
    """Mean per-sample, per-channel ``1-cosine`` with stable near-zero norms."""

    _validate_residuals(predicted, target)
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    axes = (2, 3, 4)
    dot = torch.sum(predicted * target, dim=axes)
    left = torch.sum(predicted.square(), dim=axes)
    right = torch.sum(target.square(), dim=axes)
    denominator = torch.sqrt(left + epsilon) * torch.sqrt(right + epsilon)
    cosine = dot / denominator
    # Numerical roundoff should not create a negative perfect-match loss.
    return (1.0 - cosine.clamp(-1.0, 1.0)).mean()


def shell_variance(state: torch.Tensor, shell_index: torch.Tensor) -> torch.Tensor:
    """Return population variance for each frozen radial shell as ``(B,C,8)``."""

    if state.ndim != 5:
        raise ValueError("state must have shape (B,C,Nphi,Ntheta,Nr)")
    shell_index = shell_index.to(device=state.device, dtype=torch.long)
    if shell_index.ndim != 1 or shell_index.numel() != state.shape[-1]:
        raise ValueError("shell_index must align with the radial axis")
    # Frozen evaluation uses NumPy float64.  Physical rho/press decoder tails
    # can legitimately exceed the float32 squaring range, so preserve the
    # evaluation arithmetic and its gradient in float64 here.
    working = state.double()
    output = []
    for shell in range(8):
        mask = shell_index == shell
        if not bool(mask.any()):
            raise ValueError(f"Frozen shell {shell + 1} is empty")
        values = working[..., mask]
        output.append(values.var(dim=(2, 3, 4), unbiased=False))
    return torch.stack(output, dim=-1)


def radial_profile(state: torch.Tensor) -> torch.Tensor:
    """Return normalized-state radial profiles as ``(B,C,Nr)``."""

    if state.ndim != 5:
        raise ValueError("state must have shape (B,C,Nphi,Ntheta,Nr)")
    return state.mean(dim=(2, 3))


def relative_transport_loss(
    input_statistic: torch.Tensor,
    target_statistic: torch.Tensor,
    prediction_statistic: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Mean squared relative error of a predicted temporal statistic change."""

    if not (
        input_statistic.shape == target_statistic.shape == prediction_statistic.shape
        and input_statistic.ndim == 3
    ):
        raise ValueError("Stage V transport statistics must share shape (B,C,K)")
    true_delta = target_statistic - input_statistic
    predicted_delta = prediction_statistic - input_statistic
    numerator = torch.sum((predicted_delta - true_delta).square(), dim=-1)
    denominator = torch.sum(true_delta.square(), dim=-1)
    # Frozen evaluation marks an exactly-zero transport denominator undefined.
    # Training excludes only that exact/near-zero channel rather than assigning
    # it an arbitrarily large relative loss.
    valid = denominator > epsilon
    ratios = numerator / (denominator + epsilon)
    if bool(valid.any()):
        return ratios[valid].mean()
    # Keep a differentiable zero in the exceptionally all-zero case.
    return prediction_statistic.sum() * 0.0


def normalized_radial_transport_loss(
    normalized_input: torch.Tensor,
    normalized_target: torch.Tensor,
    normalized_prediction: torch.Tensor,
    *,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    return relative_transport_loss(
        radial_profile(normalized_input),
        radial_profile(normalized_target),
        radial_profile(normalized_prediction),
        epsilon=epsilon,
    )


def physical_shell_transport_loss(
    normalized_input: torch.Tensor,
    normalized_target: torch.Tensor,
    normalized_prediction: torch.Tensor,
    *,
    preprocessor: object,
    shell_index: torch.Tensor,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Physical shell-variance surrogate matching the frozen evaluation metric."""

    decode = getattr(preprocessor, "decode_tensor", None)
    if decode is None:
        raise TypeError("preprocessor must expose decode_tensor")
    physical_input = decode(normalized_input, channel_axis=1)
    physical_target = decode(normalized_target, channel_axis=1)
    physical_prediction = decode(normalized_prediction, channel_axis=1)
    return relative_transport_loss(
        shell_variance(physical_input, shell_index),
        shell_variance(physical_target, shell_index),
        shell_variance(physical_prediction, shell_index),
        epsilon=epsilon,
    )


def normalized_shell_transport_loss(
    normalized_input: torch.Tensor,
    normalized_target: torch.Tensor,
    normalized_prediction: torch.Tensor,
    *,
    shell_index: torch.Tensor,
    epsilon: float = 1e-12,
) -> torch.Tensor:
    """Eight-shell normalized-state variance-evolution relative loss."""

    return relative_transport_loss(
        shell_variance(normalized_input, shell_index),
        shell_variance(normalized_target, shell_index),
        shell_variance(normalized_prediction, shell_index),
        epsilon=epsilon,
    )


@dataclass(frozen=True)
class ObjectiveWeights:
    direction: float = 0.0
    transport: float = 0.0

    def __post_init__(self) -> None:
        if not all(math.isfinite(value) and value >= 0 for value in (self.direction, self.transport)):
            raise ValueError("Stage V objective weights must be finite and nonnegative")


def objective_components(
    *,
    predicted_residual: torch.Tensor,
    residual_target: torch.Tensor,
    normalized_input: torch.Tensor,
    normalized_target: torch.Tensor,
    preprocessor: object,
    shell_index: torch.Tensor,
    epsilon: float = 1e-12,
    enabled: Iterable[str] = OBJECTIVE_NAMES,
) -> dict[str, torch.Tensor]:
    _validate_residuals(predicted_residual, residual_target)
    selected = set(enabled)
    unknown = selected - set(OBJECTIVE_NAMES)
    if unknown:
        raise ValueError(f"Unknown Stage V objectives: {sorted(unknown)}")
    selected.add("plain")
    normalized_prediction = normalized_input + predicted_residual
    output = {"plain": PlainL2Loss()(predicted_residual, residual_target)}
    if "direction" in selected:
        output["direction"] = residual_direction_loss(
            predicted_residual, residual_target, epsilon=epsilon
        )
    if selected & {"shell", "transport"}:
        output["shell"] = normalized_shell_transport_loss(
            normalized_input,
            normalized_target,
            normalized_prediction,
            shell_index=shell_index,
            epsilon=epsilon,
        )
    if selected & {"radial", "transport"}:
        output["radial"] = normalized_radial_transport_loss(
            normalized_input, normalized_target, normalized_prediction, epsilon=epsilon
        )
    if "transport" in selected:
        output["transport"] = 0.5 * (output["shell"] + output["radial"])
    return output


def combined_objective(
    components: Mapping[str, torch.Tensor], weights: ObjectiveWeights
) -> torch.Tensor:
    if "plain" not in components:
        raise ValueError("Missing Stage V Plain objective")
    total = components["plain"]
    if weights.direction:
        if "direction" not in components:
            raise ValueError("Missing enabled Stage V direction objective")
        total = total + weights.direction * components["direction"]
    if weights.transport:
        if "transport" not in components:
            raise ValueError("Missing enabled Stage V transport objective")
        total = total + weights.transport * components["transport"]
    return total


def flattened_gradient(
    gradients: Sequence[torch.Tensor | None], parameters: Sequence[torch.nn.Parameter]
) -> torch.Tensor:
    values = []
    for gradient, parameter in zip(gradients, parameters, strict=True):
        value = torch.zeros_like(parameter) if gradient is None else gradient
        if value.is_complex():
            value = torch.view_as_real(value)
        values.append(value.reshape(-1))
    return torch.cat(values) if values else torch.empty(0)


def gradient_cosine(left: torch.Tensor, right: torch.Tensor, *, epsilon: float = 1e-30) -> float | None:
    left_norm = torch.linalg.vector_norm(left.float())
    right_norm = torch.linalg.vector_norm(right.float())
    denominator = left_norm * right_norm
    if not torch.isfinite(denominator) or float(denominator) <= epsilon:
        return None
    return float(torch.dot(left.float(), right.float()).div(denominator).detach().cpu())


def gradient_norm_ratio(numerator: torch.Tensor, denominator: torch.Tensor, *, epsilon: float = 1e-30) -> float | None:
    top = torch.linalg.vector_norm(numerator.float())
    bottom = torch.linalg.vector_norm(denominator.float())
    if not torch.isfinite(top) or not torch.isfinite(bottom) or float(bottom) <= epsilon:
        return None
    return float((top / bottom).detach().cpu())


def classify_gradient_conflict(cosines: Iterable[float | None]) -> str:
    values = [float(value) for value in cosines if value is not None and math.isfinite(float(value))]
    if not values:
        return "STRONG"
    median = float(torch.tensor(values).median())
    nonpositive = sum(value <= 0 for value in values) / len(values)
    if median <= 0 or nonpositive >= 0.5:
        return "STRONG"
    if median < 0.25 or nonpositive >= 0.25:
        return "MODERATE"
    if median < 0.5:
        return "WEAK"
    return "NONE"
