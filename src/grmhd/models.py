"""Project-local wrappers around upstream neuraloperator models."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class PersistenceBaseline(nn.Module):
    """Identity one-step forecast in the eight normalized physical channels."""

    def __init__(self, out_channels: int = 8) -> None:
        super().__init__()
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < self.out_channels:
            raise ValueError(f"Need at least {self.out_channels} channels, found {x.shape[1]}")
        return x[:, : self.out_channels]


def apply_prediction_mode(
    model_output: torch.Tensor,
    model_input: torch.Tensor,
    *,
    predict_residual: bool,
    physical_channels: int = 8,
    bounded_residual: bool = False,
    residual_scale: torch.Tensor | tuple[float, ...] | list[float] | None = None,
) -> torch.Tensor:
    """Resolve direct or delta forecasts in normalized space.

    Only the first eight state channels participate in the shortcut. Any shell
    embedding channels are inputs to the operator but are never added to output.
    """
    if model_output.ndim != model_input.ndim:
        raise ValueError("Model output and input must have equal rank")
    if model_output.shape[1] != physical_channels:
        raise ValueError(
            f"Expected {physical_channels} model output channels, found {model_output.shape[1]}"
        )
    if model_input.shape[1] < physical_channels:
        raise ValueError(
            f"Need at least {physical_channels} state input channels, found {model_input.shape[1]}"
        )
    if model_output.shape[0] != model_input.shape[0] or model_output.shape[2:] != model_input.shape[2:]:
        raise ValueError("Model output and state input batch/spatial shapes differ")
    if not predict_residual:
        if bounded_residual:
            raise ValueError("bounded_residual requires predict_residual=true")
        return model_output
    delta = model_output
    if bounded_residual:
        if residual_scale is None:
            raise ValueError("bounded_residual requires a residual_scale")
        alpha = torch.as_tensor(
            residual_scale, dtype=model_output.dtype, device=model_output.device
        )
        if alpha.shape != (physical_channels,):
            raise ValueError(
                f"Expected residual_scale shape ({physical_channels},), found {tuple(alpha.shape)}"
            )
        if not torch.isfinite(alpha).all() or torch.any(alpha <= 0):
            raise ValueError("residual_scale must be finite and strictly positive")
        alpha = alpha.reshape(1, physical_channels, *((1,) * (model_output.ndim - 2)))
        delta = alpha * torch.tanh(model_output / alpha)
    return model_input[:, :physical_channels] + delta


def zero_initialize_residual_head(model: nn.Module) -> str:
    """Zero the final neuraloperator projection layer and return its module path."""
    projection = getattr(model, "projection", None)
    fcs = getattr(projection, "fcs", None)
    if fcs is None or len(fcs) == 0:
        raise ValueError("Model does not expose projection.fcs for zero initialization")
    head = fcs[-1]
    weight = getattr(head, "weight", None)
    if weight is None:
        raise ValueError("Final projection module has no weight")
    with torch.no_grad():
        weight.zero_()
        bias = getattr(head, "bias", None)
        if bias is not None:
            bias.zero_()
    return f"projection.fcs.{len(fcs) - 1}"


def build_model(
    model_type: str,
    *,
    in_channels: int,
    out_channels: int = 8,
    default_in_shape: tuple[int, int, int] = (64, 64, 64),
    n_modes: tuple[int, int, int] = (8, 8, 8),
    hidden_channels: int = 16,
    n_layers: int = 4,
    positional_embedding: str | None = None,
    **kwargs: Any,
) -> nn.Module:
    model_type = model_type.lower()
    if model_type == "persistence":
        return PersistenceBaseline(out_channels=out_channels)
    try:
        from neuralop.models import FNO
    except ImportError as exc:
        raise ImportError(
            "neuralop is required for FNO/LocalNO; install external/neuraloperator in editable mode"
        ) from exc
    common = dict(
        in_channels=in_channels,
        out_channels=out_channels,
        n_modes=tuple(n_modes),
        hidden_channels=hidden_channels,
        n_layers=n_layers,
        positional_embedding=positional_embedding,
    )
    if model_type == "fno":
        return FNO(**common, **kwargs)
    try:
        # Preferred public API documented by upstream and requested by this
        # reproduction. Commit 86a8bc7 contains LocalNO but omits this export.
        from neuralop.models import LocalNO
    except ImportError:
        from neuralop.models.local_no import LocalNO
    if model_type in {
        "localno_diff",
        "localno_diff_only",
        "localno_differential_3d",
    }:
        if model_type == "localno_differential_3d" and (
            len(n_modes) != 3 or len(default_in_shape) != 3
        ):
            raise ValueError("localno_differential_3d requires exactly three spatial dimensions")
        return LocalNO(
            **common,
            default_in_shape=tuple(default_in_shape),
            disco_layers=False,
            diff_layers=True,
            **kwargs,
        )
    if model_type in {"localno_disco", "localno"}:
        if len(n_modes) == 3 or len(default_in_shape) == 3:
            raise ValueError(
                "volumetric 3D DISCO is unavailable in pinned upstream"
            )
        return LocalNO(
            **common,
            default_in_shape=tuple(default_in_shape),
            disco_layers=True,
            diff_layers=True,
            **kwargs,
        )
    raise ValueError(f"Unknown model_type {model_type!r}")


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def parameters_without_grad(model: nn.Module) -> list[str]:
    return [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and parameter.grad is None
    ]
