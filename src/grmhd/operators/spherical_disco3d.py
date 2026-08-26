"""Spherical-coordinate-aware adapted DISCO3D geometry and operator.

Distances use a Euclidean embedding of the stored spherical coordinate centres.
This is not a Kerr--Schild proper-distance construction and does not transform
stored vector components.  Phi is periodic; theta and radius are truncated and
renormalized at their boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SphericalProxyGeometry:
    r: torch.Tensor
    theta: torch.Tensor
    phi: torch.Tensor
    r_faces: torch.Tensor
    theta_faces: torch.Tensor
    phi_faces: torch.Tensor
    radial_face_method: str
    theta_face_method: str
    phi_face_method: str
    volume_proxy: torch.Tensor
    offsets: tuple[tuple[int, int, int], ...]
    distances: torch.Tensor
    valid_mask: torch.Tensor
    local_scale: torch.Tensor
    radius: torch.Tensor
    raw_basis: torch.Tensor
    normalizations: torch.Tensor
    integration_weights: torch.Tensor
    active_neighbor_count: torch.Tensor
    active_basis_count: torch.Tensor
    partition_max_abs_error: float


def _vector(values: Sequence[float] | torch.Tensor, *, name: str) -> torch.Tensor:
    result = torch.as_tensor(values, dtype=torch.float64).detach().clone()
    if result.ndim != 1 or result.numel() < 2:
        raise ValueError(f"{name} must be a one-dimensional coordinate vector")
    if not torch.all(torch.isfinite(result)) or not torch.all(torch.diff(result) > 0):
        raise ValueError(f"{name} must be finite and strictly increasing")
    return result


def reconstruct_faces(
    centers: Sequence[float] | torch.Tensor,
    *,
    radial: bool,
    uniform_rtol: float = 1.0e-8,
) -> tuple[torch.Tensor, str]:
    """Reconstruct cell faces from actual centres using the Stage AE contract."""

    values = _vector(centers, name="centers")
    if radial:
        if torch.any(values <= 0):
            raise ValueError("geometric radial face reconstruction requires positive centres")
        ratios = values[1:] / values[:-1]
        if not torch.allclose(
            ratios,
            torch.full_like(ratios, float(torch.mean(ratios))),
            rtol=uniform_rtol,
            atol=uniform_rtol,
        ):
            raise ValueError("radial centres are not geometric; no silent fallback is allowed")
        ratio = torch.exp(torch.mean(torch.log(ratios)))
        root = torch.sqrt(ratio)
        faces = torch.empty(values.numel() + 1, dtype=values.dtype)
        faces[1:-1] = torch.sqrt(values[:-1] * values[1:])
        faces[0] = values[0] / root
        faces[-1] = values[-1] * root
        return faces, "geometric_midpoint_with_ratio_extrapolation"

    differences = torch.diff(values)
    if not torch.allclose(
        differences,
        torch.full_like(differences, float(torch.mean(differences))),
        rtol=uniform_rtol,
        atol=uniform_rtol,
    ):
        raise ValueError("angular centres are not uniform; no silent approximation is allowed")
    faces = torch.empty(values.numel() + 1, dtype=values.dtype)
    faces[1:-1] = 0.5 * (values[:-1] + values[1:])
    faces[0] = values[0] - 0.5 * differences[0]
    faces[-1] = values[-1] + 0.5 * differences[-1]
    return faces, "uniform_center_midpoint_with_boundary_extrapolation"


def spherical_embedding(
    r: torch.Tensor, theta: torch.Tensor, phi: torch.Tensor
) -> torch.Tensor:
    """Return the Euclidean spherical-coordinate embedding ``(..., 3)``."""

    return torch.stack(
        (
            r * torch.sin(theta) * torch.cos(phi),
            r * torch.sin(theta) * torch.sin(phi),
            r * torch.cos(theta),
        ),
        dim=-1,
    )


def _offset_distance(
    r: torch.Tensor,
    theta: torch.Tensor,
    dphi_value: float,
    dtheta: int,
    dr: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    ntheta, nr = theta.numel(), r.numel()
    target_theta = torch.arange(ntheta)
    target_r = torch.arange(nr)
    source_theta = target_theta + int(dtheta)
    source_r = target_r + int(dr)
    valid_theta = (source_theta >= 0) & (source_theta < ntheta)
    valid_r = (source_r >= 0) & (source_r < nr)
    source_theta = source_theta.clamp(0, ntheta - 1)
    source_r = source_r.clamp(0, nr - 1)

    theta_i = theta[:, None]
    theta_j = theta[source_theta][:, None]
    r_i = r[None, :]
    r_j = r[source_r][None, :]
    angular_cosine = (
        torch.cos(theta_i) * torch.cos(theta_j)
        + torch.sin(theta_i) * torch.sin(theta_j) * math.cos(float(dphi_value))
    )
    squared = torch.clamp(
        r_i.square() + r_j.square() - 2.0 * r_i * r_j * angular_cosine,
        min=0.0,
    )
    valid = valid_theta[:, None] & valid_r[None, :]
    distance = torch.where(valid, torch.sqrt(squared), torch.full_like(squared, torch.inf))
    return distance, valid


def build_spherical_proxy_geometry(
    *,
    r: Sequence[float] | torch.Tensor,
    theta: Sequence[float] | torch.Tensor,
    phi: Sequence[float] | torch.Tensor,
    radius_multiplier: float = 3.0,
    basis_count: int = 5,
    candidate_radius: int = 3,
    eps: float = 1.0e-12,
) -> SphericalProxyGeometry:
    """Build fixed target-dependent geometry for the Stage AE proxy contract."""

    radial = _vector(r, name="r")
    polar = _vector(theta, name="theta")
    azimuth = _vector(phi, name="phi")
    if radius_multiplier != 3.0 or basis_count != 5 or candidate_radius != 3:
        raise ValueError("Stage AE freezes multiplier=3, K=5, and a 7^3 candidate box")
    if eps <= 0:
        raise ValueError("eps must be positive")

    r_faces, r_method = reconstruct_faces(radial, radial=True)
    theta_faces, theta_method = reconstruct_faces(polar, radial=False)
    phi_faces, phi_method = reconstruct_faces(azimuth, radial=False)
    dphi = torch.diff(phi_faces)
    if not torch.allclose(dphi, torch.full_like(dphi, float(torch.mean(dphi))), rtol=1e-8, atol=1e-8):
        raise ValueError("phi grid must be uniform for rotational reduction")
    if not math.isclose(
        float(phi_faces[-1] - phi_faces[0]),
        2.0 * math.pi,
        rel_tol=1e-7,
        abs_tol=1e-6,
    ):
        raise ValueError("phi faces must span one periodic 2pi domain")

    dr_width = torch.diff(r_faces)
    dtheta_width = torch.diff(theta_faces)
    dphi_width = torch.diff(phi_faces)
    volume_proxy = (
        dphi_width[0]
        * dtheta_width[:, None]
        * radial[None, :].square()
        * torch.sin(polar)[:, None]
        * dr_width[None, :]
    )
    if not torch.all(torch.isfinite(volume_proxy)) or not torch.all(volume_proxy > 0):
        raise FloatingPointError("spherical coordinate-volume proxy must be finite and positive")

    phi_spacing = float(torch.mean(dphi))
    nearest_offsets = ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
    nearest = []
    for dphi_index, dtheta_index, dr_index in nearest_offsets:
        distance, _ = _offset_distance(
            radial, polar, dphi_index * phi_spacing, dtheta_index, dr_index
        )
        nearest.append(distance)
    nearest_stack = torch.stack(nearest)
    finite_count = torch.isfinite(nearest_stack).sum(dim=0)
    ordered = torch.sort(nearest_stack, dim=0).values
    lower_index = ((finite_count - 1) // 2).unsqueeze(0)
    upper_index = (finite_count // 2).unsqueeze(0)
    local_scale = 0.5 * (
        torch.gather(ordered, 0, lower_index).squeeze(0)
        + torch.gather(ordered, 0, upper_index).squeeze(0)
    )
    radius = radius_multiplier * local_scale
    if not torch.all(torch.isfinite(local_scale)) or not torch.all(local_scale > 0):
        raise FloatingPointError("local characteristic scale must be finite and positive")

    offsets = tuple(
        (dphi_index, dtheta_index, dr_index)
        for dphi_index in range(-candidate_radius, candidate_radius + 1)
        for dtheta_index in range(-candidate_radius, candidate_radius + 1)
        for dr_index in range(-candidate_radius, candidate_radius + 1)
    )
    distances = []
    valid_masks = []
    source_quadratures = []
    for dphi_index, dtheta_index, dr_index in offsets:
        distance, valid = _offset_distance(
            radial, polar, dphi_index * phi_spacing, dtheta_index, dr_index
        )
        source_theta = (torch.arange(polar.numel()) + dtheta_index).clamp(0, polar.numel() - 1)
        source_r = (torch.arange(radial.numel()) + dr_index).clamp(0, radial.numel() - 1)
        source_q = volume_proxy[source_theta[:, None], source_r[None, :]]
        source_q = torch.where(valid, source_q, torch.zeros_like(source_q))
        distances.append(distance)
        valid_masks.append(valid)
        source_quadratures.append(source_q)
    distance_tensor = torch.stack(distances)
    valid_tensor = torch.stack(valid_masks)
    source_q_tensor = torch.stack(source_quadratures)
    normalized_radius = distance_tensor / radius.unsqueeze(0)
    centers = torch.linspace(0.0, 1.0, basis_count, dtype=torch.float64)
    width = 1.0 / (basis_count - 1)
    raw_basis = torch.clamp(
        1.0
        - torch.abs(normalized_radius.unsqueeze(0) - centers[:, None, None, None]) / width,
        min=0.0,
    )
    support = valid_tensor & (normalized_radius <= 1.0)
    raw_basis = torch.where(support.unsqueeze(0), raw_basis, torch.zeros_like(raw_basis))
    normalizations = torch.sum(
        raw_basis * source_q_tensor.unsqueeze(0), dim=1
    )
    integration_weights = (
        raw_basis
        * source_q_tensor.unsqueeze(0)
        / (normalizations.unsqueeze(1) + eps)
    )
    normalized_integrals = torch.sum(integration_weights, dim=1)
    positive = normalizations > eps
    partition_mask = support
    partition_error = torch.max(
        torch.abs(raw_basis.sum(dim=0)[partition_mask] - 1.0)
    )
    return SphericalProxyGeometry(
        r=radial,
        theta=polar,
        phi=azimuth,
        r_faces=r_faces,
        theta_faces=theta_faces,
        phi_faces=phi_faces,
        radial_face_method=r_method,
        theta_face_method=theta_method,
        phi_face_method=phi_method,
        volume_proxy=volume_proxy,
        offsets=offsets,
        distances=distance_tensor,
        valid_mask=valid_tensor,
        local_scale=local_scale,
        radius=radius,
        raw_basis=raw_basis,
        normalizations=normalizations,
        integration_weights=integration_weights,
        active_neighbor_count=support.sum(dim=0),
        active_basis_count=positive.sum(dim=0),
        partition_max_abs_error=float(partition_error),
    )


class SphericalAwareDISCO3d(nn.Module):
    """Position-dependent spherical-proxy DISCO3D with frozen 7^3 candidates."""

    execution_backend = "SHIFT_WINDOW_EINSUM"

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        r: Sequence[float] | torch.Tensor,
        theta: Sequence[float] | torch.Tensor,
        phi: Sequence[float] | torch.Tensor,
        groups: int = 1,
        bias: bool = True,
        eps: float = 1.0e-12,
        require_valid_normalization: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or groups <= 0:
            raise ValueError("channel and group counts must be positive")
        if in_channels % groups or out_channels % groups:
            raise ValueError("channels must be divisible by groups")
        geometry = build_spherical_proxy_geometry(r=r, theta=theta, phi=phi, eps=eps)
        zero_count = int(torch.sum(geometry.normalizations <= eps))
        if require_valid_normalization and zero_count:
            raise ValueError(
                f"Stage AE normalization invalid: ZERO_Z_COUNT={zero_count}; training prohibited"
            )
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.groups = int(groups)
        self.groupsize = self.in_channels // self.groups
        self.basis_count = 5
        self.candidate_radius = 3
        self.grid_shape = (geometry.phi.numel(), geometry.theta.numel(), geometry.r.numel())
        self.zero_z_count = zero_count
        scale = math.sqrt(1.0 / self.groupsize)
        self.weight = nn.Parameter(
            scale * torch.randn(self.out_channels, self.groupsize, self.basis_count)
        )
        self.bias = nn.Parameter(torch.zeros(self.out_channels)) if bias else None
        self.register_buffer(
            "integration_weights", geometry.integration_weights, persistent=False
        )

    def forward_with_parameters(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
    ) -> torch.Tensor:
        if x.ndim != 5 or x.shape[1] != self.in_channels:
            raise ValueError("input must have shape (B,C,Nphi,Ntheta,Nr)")
        if tuple(x.shape[-3:]) != self.grid_shape:
            raise ValueError(f"expected spatial shape {self.grid_shape}")
        if weight.shape != self.weight.shape:
            raise ValueError("weight shape changed")
        if bias is not None and bias.shape != (self.out_channels,):
            raise ValueError("bias shape changed")

        radius = self.candidate_radius
        padded = F.pad(x, (radius, radius, radius, radius, 0, 0), mode="constant")
        windows = padded.unfold(3, 2 * radius + 1, 1).unfold(4, 2 * radius + 1, 1)
        phi_windows = torch.stack(
            [torch.roll(windows, shifts=-offset, dims=2) for offset in range(-radius, radius + 1)],
            dim=2,
        )
        ntheta, nr = self.grid_shape[1:]
        geometry = self.integration_weights.to(dtype=x.dtype).reshape(
            self.basis_count, 7, 7, 7, ntheta, nr
        )
        filtered = torch.einsum("bcdptruv,kduvtr->bckptr", phi_windows, geometry)
        outputs_per_group = self.out_channels // self.groups
        grouped_fields = filtered.reshape(
            x.shape[0], self.groups, self.groupsize, self.basis_count, *x.shape[-3:]
        )
        grouped_weights = weight.reshape(
            self.groups, outputs_per_group, self.groupsize, self.basis_count
        )
        output = torch.einsum("bgckptr,gock->bgoptr", grouped_fields, grouped_weights)
        output = output.reshape(x.shape[0], self.out_channels, *x.shape[-3:])
        if bias is not None:
            output = output + bias.reshape(1, -1, 1, 1, 1)
        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_with_parameters(x, self.weight, self.bias)
