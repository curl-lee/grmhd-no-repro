"""Auditable spherical-grid geometry helpers for Stage U.

The operators in this module deliberately stop at scalar spherical-coordinate
scale factors.  They are not Kerr--Schild covariant derivatives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


PaddingMode = Literal["periodic", "reflect", "replicate", "zeros"]


def inferred_faces(centers: Sequence[float], *, periodic_extent: float | None = None) -> np.ndarray:
    """Infer cell faces from monotone centers without changing stored coordinates."""

    values = np.asarray(centers, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or np.any(np.diff(values) <= 0):
        raise ValueError("Coordinate centers must be a strictly increasing vector")
    if periodic_extent is not None:
        spacing = float(periodic_extent) / values.size
        if not np.allclose(np.diff(values), spacing, rtol=1e-7, atol=1e-12):
            raise ValueError("Periodic coordinate is not uniformly spaced")
        return values[0] - 0.5 * spacing + spacing * np.arange(values.size + 1)
    if np.allclose(np.diff(np.log(values)), np.diff(np.log(values))[0], rtol=1e-10, atol=1e-12):
        ratio = float(np.exp(np.diff(np.log(values))[0]))
        return np.concatenate(([values[0] / np.sqrt(ratio)], np.sqrt(values[:-1] * values[1:]), [values[-1] * np.sqrt(ratio)]))
    faces = np.empty(values.size + 1, dtype=np.float64)
    faces[1:-1] = 0.5 * (values[:-1] + values[1:])
    faces[0] = values[0] - 0.5 * (values[1] - values[0])
    faces[-1] = values[-1] + 0.5 * (values[-1] - values[-2])
    return faces


def padded_centered_derivative_1d(
    values: np.ndarray,
    coordinates: np.ndarray,
    *,
    axis: int,
    padding: PaddingMode,
) -> np.ndarray:
    """Centered coordinate derivative with explicit one-cell boundary policy."""

    field = np.asarray(values, dtype=np.float64)
    coords = np.asarray(coordinates, dtype=np.float64)
    moved = np.moveaxis(field, axis, -1)
    if moved.shape[-1] != coords.size:
        raise ValueError("Coordinate length does not match derivative axis")
    left = moved[..., :1]
    right = moved[..., -1:]
    if padding == "periodic":
        left, right = moved[..., -1:], moved[..., :1]
    elif padding == "reflect":
        left, right = moved[..., 1:2], moved[..., -2:-1]
    elif padding == "replicate":
        pass
    elif padding == "zeros":
        left, right = np.zeros_like(left), np.zeros_like(right)
    else:
        raise ValueError(f"Unsupported padding {padding!r}")
    padded = np.concatenate((left, moved, right), axis=-1)
    denominator = np.empty(coords.size, dtype=np.float64)
    denominator[1:-1] = coords[2:] - coords[:-2]
    denominator[0] = 2.0 * (coords[1] - coords[0])
    denominator[-1] = 2.0 * (coords[-1] - coords[-2])
    derivative = (padded[..., 2:] - padded[..., :-2]) / denominator
    return np.moveaxis(derivative, -1, axis)


def coordinate_gradient(values: np.ndarray, coordinates: np.ndarray, *, axis: int) -> np.ndarray:
    """Second-order stored-coordinate derivative with nonperiodic edge stencils."""

    return np.gradient(
        np.asarray(values, dtype=np.float64),
        np.asarray(coordinates, dtype=np.float64),
        axis=axis,
        edge_order=2,
    )


def relative_l2(predicted: np.ndarray, expected: np.ndarray, *, epsilon: float = 1e-30) -> float:
    numerator = float(np.linalg.norm(np.asarray(predicted, dtype=np.float64) - expected))
    denominator = float(np.linalg.norm(np.asarray(expected, dtype=np.float64)))
    return numerator / max(denominator, epsilon)


@dataclass(frozen=True)
class CoordinateFDPolicy:
    phi_padding: PaddingMode = "periodic"
    theta_padding: PaddingMode = "replicate"
    r_padding: PaddingMode = "replicate"
    spherical_proxy: bool = False


class StoredCoordinateFiniteDifferenceConvolution3D(nn.Module):
    """Axis-separated learned FD paths using stored ``(phi, theta, r)`` spacing.

    Each learned three-point stencil is centered by subtracting its coefficient
    sum, as in the pinned upstream differential convolution.  Unlike upstream,
    spacing and padding are axis-specific.  The optional angular scale factors
    are only a spherical-coordinate proxy and are not a covariant derivative.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        phi: Sequence[float],
        theta: Sequence[float],
        r: Sequence[float],
        policy: CoordinateFDPolicy,
        groups: int = 1,
    ) -> None:
        super().__init__()
        if groups != 1:
            raise NotImplementedError("Stage U geometry operator freezes mix_derivatives=True")
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.groups = int(groups)
        self.policy = policy
        self.phi_weight = nn.Parameter(torch.empty(out_channels, in_channels, 3))
        self.theta_weight = nn.Parameter(torch.empty(out_channels, in_channels, 3))
        self.r_weight = nn.Parameter(torch.empty(out_channels, in_channels, 3))
        for weight in (self.phi_weight, self.theta_weight, self.r_weight):
            nn.init.kaiming_uniform_(weight, a=5**0.5)
        self.register_buffer("phi_spacing", self._local_spacing(phi).reshape(1, 1, -1, 1, 1))
        self.register_buffer("theta_spacing", self._local_spacing(theta).reshape(1, 1, 1, -1, 1))
        self.register_buffer("r_spacing", self._local_spacing(r).reshape(1, 1, 1, 1, -1))
        r_tensor = torch.as_tensor(np.asarray(r), dtype=torch.float32).reshape(1, 1, 1, 1, -1)
        theta_tensor = torch.as_tensor(np.asarray(theta), dtype=torch.float32).reshape(1, 1, 1, -1, 1)
        self.register_buffer("inverse_r", 1.0 / r_tensor)
        self.register_buffer("inverse_r_sin_theta", 1.0 / (r_tensor * torch.sin(theta_tensor)))

    @staticmethod
    def _local_spacing(values: Sequence[float]) -> torch.Tensor:
        coords = np.asarray(values, dtype=np.float64)
        widths = np.gradient(coords)
        return torch.as_tensor(widths, dtype=torch.float32)

    @staticmethod
    def _pad(x: torch.Tensor, axis: int, mode: PaddingMode) -> torch.Tensor:
        pads = [0, 0, 0, 0, 0, 0]
        pair = {-1: 0, -2: 2, -3: 4}[axis]
        pads[pair] = pads[pair + 1] = 1
        if mode == "periodic":
            return F.pad(x, pads, mode="circular")
        if mode in {"reflect", "replicate"}:
            return F.pad(x, pads, mode=mode)
        if mode == "zeros":
            return F.pad(x, pads, mode="constant", value=0.0)
        raise ValueError(f"Unsupported padding {mode!r}")

    def _axis_conv(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        *,
        axis: int,
        padding: PaddingMode,
        spacing: torch.Tensor,
    ) -> torch.Tensor:
        shape = (self.out_channels, self.in_channels, 1, 1, 1)
        kernel_shape = list(shape)
        spatial_index = {-3: 2, -2: 3, -1: 4}[axis]
        kernel_shape[spatial_index] = 3
        kernel = weight.reshape(kernel_shape)
        padded = self._pad(x, axis, padding)
        convolved = F.conv3d(padded, kernel, groups=self.groups)
        center = F.conv3d(x, weight.sum(dim=-1).reshape(shape), groups=self.groups)
        return (convolved - center) / spacing.to(dtype=x.dtype)

    def forward(self, x: torch.Tensor, grid_width: float | torch.Tensor | None = None) -> torch.Tensor:
        del grid_width  # accepted for compatibility with LocalNOBlocks
        phi = self._axis_conv(
            x, self.phi_weight, axis=-3, padding=self.policy.phi_padding,
            spacing=self.phi_spacing,
        )
        theta = self._axis_conv(
            x, self.theta_weight, axis=-2, padding=self.policy.theta_padding,
            spacing=self.theta_spacing,
        )
        radial = self._axis_conv(
            x, self.r_weight, axis=-1, padding=self.policy.r_padding,
            spacing=self.r_spacing,
        )
        if self.policy.spherical_proxy:
            theta = theta * self.inverse_r.to(dtype=x.dtype)
            phi = phi * self.inverse_r_sin_theta.to(dtype=x.dtype)
        return phi + theta + radial


def project_upstream_weight(
    upstream_weight: torch.Tensor,
    module: StoredCoordinateFiniteDifferenceConvolution3D,
) -> None:
    """Deterministically project a 3^3 upstream kernel onto three axial stencils."""

    if tuple(upstream_weight.shape[-3:]) != (3, 3, 3):
        raise ValueError("Stage U projection requires the frozen 3x3x3 kernel")
    with torch.no_grad():
        # Marginalize the two orthogonal dimensions, then split the common
        # contribution equally across the three additive paths.
        module.phi_weight.copy_(upstream_weight.sum(dim=(-2, -1)) / 3.0)
        module.theta_weight.copy_(upstream_weight.sum(dim=(-3, -1)) / 3.0)
        module.r_weight.copy_(upstream_weight.sum(dim=(-3, -2)) / 3.0)
