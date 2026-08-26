"""Anisotropic local-spherical-tangent DISCO3D geometry.

The construction uses Euclidean spherical-coordinate positions and the target's
local orthonormal spherical frame.  It is a geometry proxy, not a Kerr--Schild
tetrad or a transformation of stored vector components.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import nn

from grmhd.operators.spherical_disco3d import reconstruct_faces


@dataclass(frozen=True)
class AnisotropicSphericalGeometry:
    r: torch.Tensor
    theta: torch.Tensor
    phi: torch.Tensor
    r_faces: torch.Tensor
    theta_faces: torch.Tensor
    phi_faces: torch.Tensor
    volume_proxy: torch.Tensor
    local_frames: torch.Tensor
    offsets: tuple[tuple[int, int, int], ...]
    displacement_components: torch.Tensor
    valid_mask: torch.Tensor
    directional_scales: torch.Tensor
    normalized_radius: torch.Tensor
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


def local_spherical_frame(theta: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """Return ``(..., 3, 3)`` with rows ``(e_r,e_theta,e_phi)``."""

    theta, phi = torch.broadcast_tensors(theta, phi)
    sin_theta, cos_theta = torch.sin(theta), torch.cos(theta)
    sin_phi, cos_phi = torch.sin(phi), torch.cos(phi)
    e_r = torch.stack((sin_theta * cos_phi, sin_theta * sin_phi, cos_theta), dim=-1)
    e_theta = torch.stack((cos_theta * cos_phi, cos_theta * sin_phi, -sin_theta), dim=-1)
    e_phi = torch.stack((-sin_phi, cos_phi, torch.zeros_like(phi)), dim=-1)
    return torch.stack((e_r, e_theta, e_phi), dim=-2)


def _components(
    r: torch.Tensor,
    theta: torch.Tensor,
    dphi_value: float,
    dtheta: int,
    dr: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project source-target displacement into the target frame at phi=0."""

    ntheta, nr = theta.numel(), r.numel()
    target_theta = torch.arange(ntheta)
    target_r = torch.arange(nr)
    source_theta = target_theta + int(dtheta)
    source_r = target_r + int(dr)
    valid_theta = (source_theta >= 0) & (source_theta < ntheta)
    valid_r = (source_r >= 0) & (source_r < nr)
    source_theta = source_theta.clamp(0, ntheta - 1)
    source_r = source_r.clamp(0, nr - 1)

    ti = theta[:, None]
    tj = theta[source_theta][:, None]
    ri = r[None, :]
    rj = r[source_r][None, :]
    cosine, sine = math.cos(float(dphi_value)), math.sin(float(dphi_value))
    delta_r = rj * (
        torch.cos(tj) * torch.cos(ti)
        + torch.sin(tj) * torch.sin(ti) * cosine
    ) - ri
    delta_theta = rj * (
        torch.sin(tj) * cosine * torch.cos(ti)
        - torch.cos(tj) * torch.sin(ti)
    )
    delta_phi = rj * torch.sin(tj) * sine
    valid = valid_theta[:, None] & valid_r[None, :]
    components = torch.stack((delta_r, delta_theta, delta_phi))
    components = torch.where(valid.unsqueeze(0), components, torch.full_like(components, torch.nan))
    return components, valid


def _median_valid(values: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(values)
    count = finite.sum(dim=0)
    if torch.any(count == 0):
        raise FloatingPointError("directional scale has no valid nearest neighbor")
    ordered = torch.sort(torch.where(finite, values, torch.full_like(values, torch.inf)), dim=0).values
    lower = ((count - 1) // 2).unsqueeze(0)
    upper = (count // 2).unsqueeze(0)
    return 0.5 * (
        torch.gather(ordered, 0, lower).squeeze(0)
        + torch.gather(ordered, 0, upper).squeeze(0)
    )


def build_anisotropic_spherical_geometry(
    *,
    r: Sequence[float] | torch.Tensor,
    theta: Sequence[float] | torch.Tensor,
    phi: Sequence[float] | torch.Tensor,
    radius_multiplier: float = 3.0,
    basis_count: int = 5,
    candidate_radius: int = 3,
) -> AnisotropicSphericalGeometry:
    radial = _vector(r, name="r")
    polar = _vector(theta, name="theta")
    azimuth = _vector(phi, name="phi")
    if radius_multiplier != 3.0 or basis_count != 5 or candidate_radius != 3:
        raise ValueError("Stage AF freezes multiplier=3, K=5, and a 7^3 candidate box")

    r_faces, _ = reconstruct_faces(radial, radial=True)
    theta_faces, _ = reconstruct_faces(polar, radial=False)
    phi_faces, _ = reconstruct_faces(azimuth, radial=False)
    dphi_width = torch.diff(phi_faces)
    if not torch.allclose(dphi_width, torch.full_like(dphi_width, float(torch.mean(dphi_width))), rtol=1e-8, atol=1e-8):
        raise ValueError("phi spacing must be uniform")
    if not math.isclose(float(phi_faces[-1] - phi_faces[0]), 2 * math.pi, rel_tol=1e-7, abs_tol=1e-6):
        raise ValueError("phi faces do not span the periodic domain")
    phi_spacing = float(torch.mean(dphi_width))

    volume_proxy = (
        dphi_width[0]
        * torch.diff(theta_faces)[:, None]
        * radial[None, :].square()
        * torch.sin(polar)[:, None]
        * torch.diff(r_faces)[None, :]
    )
    if not torch.all(torch.isfinite(volume_proxy)) or not torch.all(volume_proxy > 0):
        raise FloatingPointError("spherical volume proxy must be finite and positive")

    frames = local_spherical_frame(polar[:, None], azimuth[None, :])
    identity = torch.eye(3, dtype=torch.float64)
    frame_error = torch.max(torch.abs(torch.einsum("...ai,...bi->...ab", frames, frames) - identity))
    if float(frame_error) >= 1e-12:
        raise FloatingPointError("local spherical frame failed orthonormality")

    radial_nearest = []
    for offset in (-1, 1):
        value, _ = _components(radial, polar, 0.0, 0, offset)
        radial_nearest.append(torch.abs(value[0]))
    theta_nearest = []
    for offset in (-1, 1):
        value, _ = _components(radial, polar, 0.0, offset, 0)
        theta_nearest.append(torch.abs(value[1]))
    phi_nearest = []
    for offset in (-1, 1):
        value, _ = _components(radial, polar, offset * phi_spacing, 0, 0)
        phi_nearest.append(torch.abs(value[2]))
    directional_scales = torch.stack((
        _median_valid(torch.stack(radial_nearest)),
        _median_valid(torch.stack(theta_nearest)),
        _median_valid(torch.stack(phi_nearest)),
    ))
    if not torch.all(torch.isfinite(directional_scales)) or not torch.all(directional_scales > 0):
        raise FloatingPointError("all directional scales must be finite and strictly positive")

    offsets = tuple(
        (dphi, dtheta, dr)
        for dphi in range(-candidate_radius, candidate_radius + 1)
        for dtheta in range(-candidate_radius, candidate_radius + 1)
        for dr in range(-candidate_radius, candidate_radius + 1)
    )
    components, valid_masks, source_q = [], [], []
    for dphi, dtheta, dr in offsets:
        value, valid = _components(radial, polar, dphi * phi_spacing, dtheta, dr)
        source_theta = (torch.arange(polar.numel()) + dtheta).clamp(0, polar.numel() - 1)
        source_r = (torch.arange(radial.numel()) + dr).clamp(0, radial.numel() - 1)
        quadrature = volume_proxy[source_theta[:, None], source_r[None, :]]
        quadrature = torch.where(valid, quadrature, torch.zeros_like(quadrature))
        components.append(value)
        valid_masks.append(valid)
        source_q.append(quadrature)
    component_tensor = torch.stack(components, dim=1)
    valid_tensor = torch.stack(valid_masks)
    source_q_tensor = torch.stack(source_q)
    scaled = component_tensor / (radius_multiplier * directional_scales[:, None])
    normalized_radius = torch.sqrt(torch.nansum(scaled.square(), dim=0))
    normalized_radius = torch.where(valid_tensor, normalized_radius, torch.full_like(normalized_radius, torch.inf))

    centers = torch.linspace(0.0, 1.0, basis_count, dtype=torch.float64)
    width = 1.0 / (basis_count - 1)
    raw_basis = torch.clamp(
        1.0 - torch.abs(normalized_radius.unsqueeze(0) - centers[:, None, None, None]) / width,
        min=0.0,
    )
    support = valid_tensor & (normalized_radius <= 1.0)
    raw_basis = torch.where(support.unsqueeze(0), raw_basis, torch.zeros_like(raw_basis))
    normalizations = torch.sum(raw_basis * source_q_tensor.unsqueeze(0), dim=1)
    zero_count = int(torch.sum(normalizations <= 0))
    if zero_count:
        raise ValueError(f"Stage AF geometry remains degenerate: ZERO_Z_COUNT={zero_count}")
    integration_weights = raw_basis * source_q_tensor.unsqueeze(0) / normalizations.unsqueeze(1)
    partition_error = torch.max(torch.abs(raw_basis.sum(dim=0)[support] - 1.0))
    return AnisotropicSphericalGeometry(
        r=radial, theta=polar, phi=azimuth,
        r_faces=r_faces, theta_faces=theta_faces, phi_faces=phi_faces,
        volume_proxy=volume_proxy, local_frames=frames, offsets=offsets,
        displacement_components=component_tensor, valid_mask=valid_tensor,
        directional_scales=directional_scales, normalized_radius=normalized_radius,
        raw_basis=raw_basis, normalizations=normalizations,
        integration_weights=integration_weights,
        active_neighbor_count=support.sum(dim=0),
        active_basis_count=(normalizations > 0).sum(dim=0),
        partition_max_abs_error=float(partition_error),
    )


class AnisotropicSphericalDISCO3d(nn.Module):
    execution_backend = "OFFSET_VECTORIZED_ACCUMULATION"

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
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or groups <= 0:
            raise ValueError("channel and group counts must be positive")
        if in_channels % groups or out_channels % groups:
            raise ValueError("channels must be divisible by groups")
        geometry = build_anisotropic_spherical_geometry(r=r, theta=theta, phi=phi)
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.groups = int(groups)
        self.groupsize = self.in_channels // self.groups
        self.basis_count = 5
        self.candidate_radius = 3
        self.grid_shape = (geometry.phi.numel(), geometry.theta.numel(), geometry.r.numel())
        scale = math.sqrt(1.0 / self.groupsize)
        self.weight = nn.Parameter(scale * torch.randn(out_channels, self.groupsize, 5))
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.register_buffer("integration_weights", geometry.integration_weights, persistent=False)

    def forward_with_parameters(self, x, weight, bias):
        if x.ndim != 5 or x.shape[1] != self.in_channels or tuple(x.shape[-3:]) != self.grid_shape:
            raise ValueError("input shape does not match anisotropic DISCO3D contract")
        if weight.shape != self.weight.shape:
            raise ValueError("weight shape changed")
        ntheta, nr = self.grid_shape[1:]
        geometry = self.integration_weights.to(dtype=x.dtype).reshape(5, 7, 7, 7, ntheta, nr)
        # Keep phi vectorized and accumulate the 7x7 theta/r offset planes.
        # Materializing a full [dphi, dtheta, dr] window at 64^3 would require
        # several GiB per layer. This representation has identical semantics,
        # no voxel loop, and truncates theta/r through source/target slices.
        phi_sources = torch.stack(
            [torch.roll(x, shifts=-offset, dims=2) for offset in range(-3, 4)], dim=2
        )
        filtered = x.new_zeros((x.shape[0], self.in_channels, 5, *x.shape[-3:]))

        def source_target_slices(offset: int, size: int) -> tuple[slice, slice]:
            if offset < 0:
                return slice(0, size + offset), slice(-offset, size)
            if offset > 0:
                return slice(offset, size), slice(0, size - offset)
            return slice(0, size), slice(0, size)

        for theta_offset in range(-3, 4):
            source_theta, target_theta = source_target_slices(theta_offset, ntheta)
            for radial_offset in range(-3, 4):
                source_r, target_r = source_target_slices(radial_offset, nr)
                source = phi_sources[:, :, :, :, source_theta, source_r]
                coefficients = geometry[
                    :, :, theta_offset + 3, radial_offset + 3, target_theta, target_r
                ]
                contribution = torch.einsum("bcdptr,kdtr->bckptr", source, coefficients)
                current = filtered[:, :, :, :, target_theta, target_r]
                filtered[:, :, :, :, target_theta, target_r] = current + contribution
        outputs_per_group = self.out_channels // self.groups
        grouped_fields = filtered.reshape(x.shape[0], self.groups, self.groupsize, 5, *x.shape[-3:])
        grouped_weights = weight.reshape(self.groups, outputs_per_group, self.groupsize, 5)
        output = torch.einsum("bgckptr,gock->bgoptr", grouped_fields, grouped_weights)
        output = output.reshape(x.shape[0], self.out_channels, *x.shape[-3:])
        if bias is not None:
            output = output + bias.reshape(1, -1, 1, 1, 1)
        return output

    def forward(self, x):
        return self.forward_with_parameters(x, self.weight, self.bias)
