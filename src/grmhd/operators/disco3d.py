"""Adapted isotropic radial DISCO convolution on a regular 3-D grid.

This is a project-local extension.  It is intentionally not presented as the
paper's unavailable volumetric DISCO implementation.  The continuous kernel is
expanded in five radial piecewise-linear hats and projected at cell-centre
offsets in a spherical compact support.  Evaluation uses PyTorch's
cross-correlation convention with explicit circular padding.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RadialHatSupport:
    """Frozen geometric tensors used by :class:`AdaptedRadialDISCO3d`."""

    delta_cell: float
    radius_cutoff: float
    stencil_shape: tuple[int, int, int]
    quadrature_weight: float
    centers: torch.Tensor
    width: float
    rho: torch.Tensor
    raw_basis: torch.Tensor
    normalized_basis: torch.Tensor
    normalizations: torch.Tensor


def _triple(values: Sequence[int | float], *, name: str, integer: bool) -> tuple:
    result = tuple(int(value) if integer else float(value) for value in values)
    if len(result) != 3 or any(value <= 0 for value in result):
        raise ValueError(f"{name} must contain three positive values")
    return result


def radial_hat_support(
    *,
    grid_shape: Sequence[int],
    domain_length: Sequence[float],
    radius_cells: int = 3,
    basis_count: int = 5,
    eps: float = 1.0e-12,
    dtype: torch.dtype = torch.float64,
) -> RadialHatSupport:
    """Construct the point-sampled, quadrature-normalized radial hats.

    ``radius_cells=3`` is the Stage AD scientific contract.  Other positive
    integer values are accepted only so the deterministic m=1,2,3 support audit
    can demonstrate minimality without constructing or training another model.
    """

    shape = _triple(grid_shape, name="grid_shape", integer=True)
    lengths = _triple(domain_length, name="domain_length", integer=False)
    if int(radius_cells) != radius_cells or radius_cells <= 0:
        raise ValueError("radius_cells must be a positive integer")
    if basis_count < 2:
        raise ValueError("basis_count must be at least two")
    if eps <= 0:
        raise ValueError("eps must be positive")
    if not dtype.is_floating_point:
        raise TypeError("radial hat construction requires a floating dtype")

    spacings = tuple(length / count for length, count in zip(lengths, shape, strict=True))
    delta_cell = max(spacings)
    radius = int(radius_cells) * delta_cell
    stencil_shape = tuple(
        math.floor(2.0 * radius * count / length) + 1
        for count, length in zip(shape, lengths, strict=True)
    )
    if any(size % 2 != 1 for size in stencil_shape):
        raise ValueError(
            "the frozen centred dense implementation requires an odd stencil in every direction"
        )

    axes = []
    for size, spacing in zip(stencil_shape, spacings, strict=True):
        half = size // 2
        axes.append(torch.arange(-half, half + 1, dtype=dtype) * spacing)
    mesh = torch.meshgrid(*axes, indexing="ij")
    rho = torch.sqrt(sum(axis.square() for axis in mesh))
    centers = torch.linspace(0.0, radius, basis_count, dtype=dtype)
    width = radius / (basis_count - 1)
    raw = torch.clamp(1.0 - torch.abs(rho.unsqueeze(0) - centers[:, None, None, None]) / width, min=0.0)
    raw = torch.where(rho.unsqueeze(0) <= radius, raw, torch.zeros_like(raw))
    quadrature = math.prod(lengths) / math.prod(shape)
    normalizations = quadrature * raw.sum(dim=(1, 2, 3))
    normalized = raw / (normalizations[:, None, None, None] + eps)
    return RadialHatSupport(
        delta_cell=delta_cell,
        radius_cutoff=radius,
        stencil_shape=stencil_shape,
        quadrature_weight=quadrature,
        centers=centers,
        width=width,
        rho=rho,
        raw_basis=raw,
        normalized_basis=normalized,
        normalizations=normalizations,
    )


class AdaptedRadialDISCO3d(nn.Module):
    """Five-hat adapted 3-D DISCO local integral with circular boundaries."""

    execution_backend = "DENSE_CONV3D"
    correlation_convention = "cross-correlation"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        grid_shape: Sequence[int] = (64, 64, 64),
        domain_length: Sequence[float] = (2.0, 2.0, 2.0),
        radius_cells: int = 3,
        basis_count: int = 5,
        groups: int = 1,
        bias: bool = True,
        eps: float = 1.0e-12,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or groups <= 0:
            raise ValueError("channel and group counts must be positive")
        if in_channels % groups or out_channels % groups:
            raise ValueError("input and output channels must be divisible by groups")
        if radius_cells != 3:
            raise ValueError("the trainable Stage AD operator freezes radius_cells=3")
        if basis_count != 5:
            raise ValueError("the trainable Stage AD operator freezes basis_count=5")

        support = radial_hat_support(
            grid_shape=grid_shape,
            domain_length=domain_length,
            radius_cells=radius_cells,
            basis_count=basis_count,
            eps=eps,
        )
        if support.stencil_shape != (7, 7, 7):
            raise ValueError(
                f"Stage AD requires a [7,7,7] stencil, found {support.stencil_shape}"
            )
        if torch.any(support.normalizations <= eps):
            raise ValueError("all five Stage AD radial hats must have positive support")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.groups = int(groups)
        self.groupsize = self.in_channels // self.groups
        self.grid_shape = tuple(int(value) for value in grid_shape)
        self.domain_length = tuple(float(value) for value in domain_length)
        self.radius_cells = int(radius_cells)
        self.basis_count = int(basis_count)
        self.eps = float(eps)
        self.delta_cell = float(support.delta_cell)
        self.radius_cutoff = float(support.radius_cutoff)
        self.stencil_shape = support.stencil_shape
        self.quadrature_weight = float(support.quadrature_weight)
        self.hat_width = float(support.width)

        scale = math.sqrt(1.0 / self.groupsize)
        self.weight = nn.Parameter(
            scale * torch.randn(self.out_channels, self.groupsize, self.basis_count)
        )
        self.bias = nn.Parameter(torch.zeros(self.out_channels)) if bias else None
        self.register_buffer("centers", support.centers, persistent=True)
        self.register_buffer("rho", support.rho, persistent=True)
        self.register_buffer("raw_basis", support.raw_basis, persistent=True)
        self.register_buffer("normalized_basis", support.normalized_basis, persistent=True)
        self.register_buffer("normalizations", support.normalizations, persistent=True)

    @property
    def padding(self) -> tuple[int, int, int]:
        return tuple(size // 2 for size in self.stencil_shape)

    def dense_kernel(self, weight: torch.Tensor | None = None) -> torch.Tensor:
        """Synthesize the grouped Conv3d kernel including quadrature weights."""

        coefficient = self.weight if weight is None else weight
        if coefficient.shape != self.weight.shape:
            raise ValueError("DISCO coefficient shape changed")
        basis = self.normalized_basis.to(dtype=coefficient.dtype)
        return self.quadrature_weight * torch.einsum("oik,kxyz->oixyz", coefficient, basis)

    def forward_with_parameters(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != self.in_channels:
            raise ValueError("DISCO3D input must have shape (B,C,D,H,W)")
        if tuple(x.shape[-3:]) != self.grid_shape:
            raise ValueError(
                f"DISCO3D expected spatial shape {self.grid_shape}, found {tuple(x.shape[-3:])}"
            )
        if bias is not None and bias.shape != (self.out_channels,):
            raise ValueError("DISCO bias shape changed")
        p0, p1, p2 = self.padding
        padded = F.pad(x, (p2, p2, p1, p1, p0, p0), mode="circular")
        return F.conv3d(
            padded,
            self.dense_kernel(weight),
            bias=bias,
            stride=1,
            padding=0,
            groups=self.groups,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_parameters(x, self.weight, self.bias)
