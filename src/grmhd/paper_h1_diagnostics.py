"""Post-hoc H1 diagnostics for the paper-adapted reduced protocol.

This module is intentionally independent of ``PaperCompositeLoss`` and the
training entry point.  It exposes the frozen Stage E index-grid seminorm and
coordinate-aware diagnostic variants, but does not provide an optimizer or a
training configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

import torch
from neuralop.losses.differentiation import FiniteDiff

from .paper_losses import PaperH1GradientLoss


AXIS_NAMES = ("phi", "theta", "r")
DEFAULT_POLE_SIN_FLOOR = 1.0e-3
H1Variant = Literal[
    "H0_current_upstream",
    "H1_unit_index",
    "H2_normalized_axis",
    "H3_stored_r",
    "H3_stored_logr",
    "H4_spherical_metric_proxy",
]
Reduction = Literal["uniform", "volume_proxy"]
ChannelMode = Literal["all_channels_naive_proxy", "scalar_channel_proxy_only"]


@dataclass(frozen=True)
class H1Diagnostic:
    """Differentiable H1 diagnostic with additive component densities."""

    variant: str
    total: torch.Tensor
    per_channel: torch.Tensor
    per_direction: Mapping[str, torch.Tensor]
    direction_density: Mapping[str, torch.Tensor]
    spatial_weights: torch.Tensor
    metadata: Mapping[str, object]


def normalized_axis_spacings(shape: Sequence[int]) -> tuple[float, float, float]:
    """Return the exact default spacings used by pinned upstream H1Loss."""

    if len(shape) != 3 or any(int(size) < 4 for size in shape):
        raise ValueError("H1 diagnostics require three spatial dimensions of size >= 4")
    return tuple(1.0 / int(size) for size in shape)  # type: ignore[return-value]


def _validate_error(error: torch.Tensor) -> None:
    if error.ndim != 5:
        raise ValueError("error must have shape (batch, channel, phi, theta, r)")
    if not error.is_floating_point():
        raise TypeError("error must be floating point")
    if min(error.shape[-3:]) < 4:
        raise ValueError("each spatial axis must contain at least four cells")


def _uniform_derivatives(
    error: torch.Tensor,
    spacings: Sequence[float],
    *,
    periodic: Sequence[bool] = (True, True, True),
) -> dict[str, torch.Tensor]:
    _validate_error(error)
    if len(spacings) != 3 or len(periodic) != 3:
        raise ValueError("spacings and periodic must have length three")
    fd = FiniteDiff(
        dim=3,
        h=tuple(float(value) for value in spacings),
        periodic_in_x=bool(periodic[0]),
        periodic_in_y=bool(periodic[1]),
        periodic_in_z=bool(periodic[2]),
    )
    return {
        "phi": fd.dx(error),
        "theta": fd.dy(error),
        "r": fd.dz(error),
    }


def _three_point_weights(
    coordinate: torch.Tensor,
    evaluation_index: int,
    node_indices: Sequence[int],
) -> torch.Tensor:
    """Derivative weights for a three-point quadratic interpolant."""

    x_eval = coordinate[evaluation_index]
    nodes = coordinate[list(node_indices)]
    weights = []
    for j in range(3):
        others = [index for index in range(3) if index != j]
        first = 1.0 / (nodes[j] - nodes[others[0]])
        first *= (x_eval - nodes[others[1]]) / (nodes[j] - nodes[others[1]])
        second = 1.0 / (nodes[j] - nodes[others[1]])
        second *= (x_eval - nodes[others[0]]) / (nodes[j] - nodes[others[0]])
        weights.append(first + second)
    return torch.stack(weights)


def coordinate_derivative(
    field: torch.Tensor,
    coordinate: torch.Tensor,
    *,
    dim: int,
    periodic: bool = False,
) -> torch.Tensor:
    """Differentiate along a center-coordinate array.

    Periodic differentiation uses the same centered wrap stencil as upstream
    and therefore requires uniform centers.  Non-periodic differentiation uses
    a three-point, second-order Lagrange stencil at every point, including
    one-sided boundary stencils.  This supports genuinely nonuniform radii
    without replacing them by an average spacing.
    """

    if coordinate.ndim != 1 or coordinate.numel() != field.shape[dim]:
        raise ValueError("coordinate must be one-dimensional and match field axis")
    if coordinate.numel() < 4:
        raise ValueError("coordinate derivative requires at least four centers")
    coordinate = coordinate.to(device=field.device, dtype=field.dtype)
    differences = torch.diff(coordinate)
    if not bool(torch.all(differences > 0)):
        raise ValueError("coordinate centers must be strictly increasing")
    if periodic:
        if not torch.allclose(
            differences,
            differences[0].expand_as(differences),
            rtol=1.0e-5,
            atol=1.0e-6,
        ):
            raise ValueError("periodic coordinate derivative requires uniform centers")
        return (
            torch.roll(field, shifts=-1, dims=dim)
            - torch.roll(field, shifts=1, dims=dim)
        ) / (2.0 * differences[0])

    moved = field.movedim(dim, -1)
    derivative = torch.empty_like(moved)
    count = coordinate.numel()
    for index in range(count):
        if index == 0:
            nodes = (0, 1, 2)
        elif index == count - 1:
            nodes = (count - 3, count - 2, count - 1)
        else:
            nodes = (index - 1, index, index + 1)
        weights = _three_point_weights(coordinate, index, nodes)
        derivative[..., index] = sum(
            weights[position] * moved[..., node]
            for position, node in enumerate(nodes)
        )
    return derivative.movedim(-1, dim)


def coordinate_derivatives(
    error: torch.Tensor,
    coordinates: Mapping[str, torch.Tensor],
    *,
    radial_coordinate: Literal["r", "logr"] = "r",
) -> dict[str, torch.Tensor]:
    """Coordinate derivatives with periodic phi and open theta/r boundaries."""

    _validate_error(error)
    phi = coordinates["phi"]
    theta = coordinates["theta"]
    radius = coordinates["r"]
    radial = radius.log() if radial_coordinate == "logr" else radius
    return {
        "phi": coordinate_derivative(error, phi, dim=-3, periodic=True),
        "theta": coordinate_derivative(error, theta, dim=-2, periodic=False),
        "r": coordinate_derivative(error, radial, dim=-1, periodic=False),
    }


def center_cell_widths(
    coordinate: torch.Tensor,
    *,
    lower_edge: float | None = None,
    upper_edge: float | None = None,
) -> torch.Tensor:
    """Construct positive cell widths from monotonically increasing centers."""

    if coordinate.ndim != 1 or coordinate.numel() < 2:
        raise ValueError("coordinate must contain at least two centers")
    coordinate = coordinate.to(dtype=torch.float64)
    if not bool(torch.all(torch.diff(coordinate) > 0)):
        raise ValueError("coordinate centers must be strictly increasing")
    edges = torch.empty(
        coordinate.numel() + 1, dtype=coordinate.dtype, device=coordinate.device
    )
    edges[1:-1] = 0.5 * (coordinate[:-1] + coordinate[1:])
    edges[0] = (
        float(lower_edge)
        if lower_edge is not None
        else coordinate[0] - 0.5 * (coordinate[1] - coordinate[0])
    )
    edges[-1] = (
        float(upper_edge)
        if upper_edge is not None
        else coordinate[-1] + 0.5 * (coordinate[-1] - coordinate[-2])
    )
    widths = torch.diff(edges)
    if not bool(torch.all(widths > 0)):
        raise ValueError("derived cell widths must be positive")
    return widths


def spherical_proxy_volume_weights(
    coordinates: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Normalized ``r^2 sin(theta) dphi dtheta dr`` coordinate-volume proxy."""

    phi = coordinates["phi"].to(dtype=torch.float64)
    theta = coordinates["theta"].to(dtype=torch.float64)
    radius = coordinates["r"].to(dtype=torch.float64)
    dphi = center_cell_widths(phi)
    dtheta = center_cell_widths(theta, lower_edge=0.0, upper_edge=torch.pi)
    dr = center_cell_widths(radius)
    weights = (
        dphi[:, None, None]
        * (theta.sin().clamp_min(0.0) * dtheta)[None, :, None]
        * (radius.square() * dr)[None, None, :]
    )
    weights /= weights.sum()
    return weights


def uniform_spatial_weights(
    shape: Sequence[int],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    count = int(shape[0]) * int(shape[1]) * int(shape[2])
    return torch.full(
        tuple(int(value) for value in shape),
        1.0 / count,
        device=device,
        dtype=dtype,
    )


def _reduce_direction_densities(
    direction_density: Mapping[str, torch.Tensor],
    spatial_weights: torch.Tensor,
    *,
    variant: str,
    metadata: Mapping[str, object],
) -> H1Diagnostic:
    first = next(iter(direction_density.values()))
    weights = spatial_weights.to(device=first.device, dtype=first.dtype)
    if tuple(weights.shape) != tuple(first.shape[-3:]):
        raise ValueError("spatial weights do not match density")
    if not torch.isclose(weights.sum(), weights.new_tensor(1.0), atol=1.0e-6):
        raise ValueError("spatial weights must sum to one")
    per_direction_channel = {
        name: torch.sum(density * weights, dim=(-3, -2, -1))
        for name, density in direction_density.items()
    }
    per_channel_batch = sum(per_direction_channel.values())
    per_channel = per_channel_batch.mean(dim=0)
    per_direction = {
        name: values.sum(dim=1).mean(dim=0)
        for name, values in per_direction_channel.items()
    }
    total = per_channel_batch.sum(dim=1).mean(dim=0)
    return H1Diagnostic(
        variant=variant,
        total=total,
        per_channel=per_channel,
        per_direction=per_direction,
        direction_density=dict(direction_density),
        spatial_weights=weights,
        metadata=dict(metadata),
    )


def _weights_for_reduction(
    error: torch.Tensor,
    coordinates: Mapping[str, torch.Tensor] | None,
    reduction: Reduction,
) -> torch.Tensor:
    if reduction == "uniform":
        return uniform_spatial_weights(
            error.shape[-3:], device=error.device, dtype=error.dtype
        )
    if coordinates is None:
        raise ValueError("volume_proxy reduction requires coordinates")
    return spherical_proxy_volume_weights(coordinates).to(
        device=error.device, dtype=error.dtype
    )


def current_upstream_h1(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Exact frozen Stage E/Stage G H1-minus-L2 wrapper."""

    return PaperH1GradientLoss()(prediction, target)


def h1_diagnostic(
    error: torch.Tensor,
    *,
    variant: H1Variant,
    coordinates: Mapping[str, torch.Tensor] | None = None,
    reduction: Reduction = "uniform",
    channel_mode: ChannelMode = "all_channels_naive_proxy",
    pole_sin_floor: float = DEFAULT_POLE_SIN_FLOOR,
) -> H1Diagnostic:
    """Evaluate one additive H1 diagnostic variant on an error field."""

    _validate_error(error)
    shape = error.shape[-3:]
    metadata: dict[str, object] = {
        "tensor_axes": list(AXIS_NAMES),
        "reduction": reduction,
        "includes_l2": False,
        "squared": True,
    }
    if variant == "H0_current_upstream":
        spacings = normalized_axis_spacings(shape)
        derivatives = _uniform_derivatives(error, spacings)
        metadata.update(
            {
                "spacings": list(spacings),
                "boundary": "periodic_wrap_all_three_axes",
                "stencil": "second_order_centered",
                "source": "pinned upstream H1Loss defaults",
            }
        )
    elif variant == "H1_unit_index":
        spacings = (1.0, 1.0, 1.0)
        derivatives = _uniform_derivatives(error, spacings)
        metadata.update(
            {
                "spacings": list(spacings),
                "boundary": "periodic_wrap_all_three_axes",
                "stencil": "second_order_centered",
            }
        )
    elif variant == "H2_normalized_axis":
        spacings = normalized_axis_spacings(shape)
        derivatives = _uniform_derivatives(error, spacings)
        metadata.update(
            {
                "spacings": list(spacings),
                "boundary": "periodic_wrap_all_three_axes",
                "stencil": "second_order_centered",
                "expected_h0_parity": True,
            }
        )
    elif variant in {"H3_stored_r", "H3_stored_logr"}:
        if coordinates is None:
            raise ValueError(f"{variant} requires stored coordinates")
        radial_coordinate = "logr" if variant.endswith("logr") else "r"
        derivatives = coordinate_derivatives(
            error, coordinates, radial_coordinate=radial_coordinate
        )
        metadata.update(
            {
                "radial_coordinate": radial_coordinate,
                "boundary": "phi_periodic_theta_radial_one_sided",
                "stencil": "centered_or_three_point_coordinate_lagrange",
                "coordinate_gradient_only_not_covariant": True,
            }
        )
    elif variant == "H4_spherical_metric_proxy":
        if coordinates is None:
            raise ValueError("H4 requires stored coordinates")
        if pole_sin_floor <= 0:
            raise ValueError("pole_sin_floor must be positive")
        derivatives = coordinate_derivatives(error, coordinates, radial_coordinate="r")
        radius = coordinates["r"].to(device=error.device, dtype=error.dtype)
        theta = coordinates["theta"].to(device=error.device, dtype=error.dtype)
        radius_grid = radius[None, None, None, None, :]
        sin_grid = theta.sin().abs().clamp_min(float(pole_sin_floor))[
            None, None, None, :, None
        ]
        derivatives = {
            "phi": derivatives["phi"] / (radius_grid * sin_grid),
            "theta": derivatives["theta"] / radius_grid,
            "r": derivatives["r"],
        }
        metadata.update(
            {
                "pole_sin_floor": float(pole_sin_floor),
                "boundary": "phi_periodic_theta_radial_one_sided",
                "scalar_metric_formula": True,
                "not_kerr_schild_proper_metric": True,
                "not_vector_covariant_derivative": True,
                "channel_mode": channel_mode,
            }
        )
    else:
        raise ValueError(f"Unknown H1 variant: {variant}")

    if channel_mode == "scalar_channel_proxy_only":
        derivatives = {name: value[:, 3:5] for name, value in derivatives.items()}
        metadata["selected_channels"] = ["rho", "press"]
    elif channel_mode != "all_channels_naive_proxy":
        raise ValueError(f"Unknown channel mode: {channel_mode}")
    elif variant == "H4_spherical_metric_proxy":
        metadata["warning"] = (
            "B and velocity stored components are treated naively as scalars; "
            "this is not a covariant vector derivative."
        )

    densities = {name: derivative.square() for name, derivative in derivatives.items()}
    weights = _weights_for_reduction(error, coordinates, reduction)
    return _reduce_direction_densities(
        densities, weights, variant=variant, metadata=metadata
    )


def h0_wrapper_parity(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Stage E wrapper and additive-density H0 values for parity tests."""

    wrapper = current_upstream_h1(prediction, target)
    diagnostic = h1_diagnostic(
        prediction - target, variant="H0_current_upstream"
    ).total
    return wrapper, diagnostic


def flatten_optional_gradients(
    gradients: Sequence[torch.Tensor | None],
    references: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Flatten autograd outputs, replacing structurally unused entries by zero."""

    if len(gradients) != len(references):
        raise ValueError("gradient/reference lengths differ")
    values = []
    for gradient, reference in zip(gradients, references):
        value = torch.zeros_like(reference) if gradient is None else gradient
        if value.is_complex():
            value = torch.view_as_real(value)
        values.append(value.reshape(-1))
    if not values:
        raise ValueError("at least one gradient is required")
    return torch.cat(values)


def gradient_pair_metrics(
    first: torch.Tensor,
    second: torch.Tensor,
    *,
    epsilon: float = 1.0e-30,
) -> dict[str, torch.Tensor]:
    """Dot, cosine, and elementwise sign-conflict diagnostics."""

    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
        raise ValueError("gradient vectors must be aligned one-dimensional tensors")
    dot = torch.dot(first, second)
    first_norm = torch.linalg.vector_norm(first)
    second_norm = torch.linalg.vector_norm(second)
    denominator = (first_norm * second_norm).clamp_min(float(epsilon))
    products = first * second
    active = (first != 0) & (second != 0)
    conflicts = active & (products < 0)
    active_count = active.sum().clamp_min(1)
    absolute_products = products.abs()
    conflict_weight = torch.where(conflicts, absolute_products, 0).sum()
    return {
        "dot": dot,
        "cosine": dot / denominator,
        "first_norm": first_norm,
        "second_norm": second_norm,
        "opposite_sign_parameter_fraction": conflicts.sum() / active_count,
        "conflict_fraction": conflict_weight
        / absolute_products.sum().clamp_min(float(epsilon)),
    }


def simulate_global_norm_clip(
    component_vectors: Mapping[str, torch.Tensor],
    *,
    total_name: str = "total",
    max_norm: float = 1.0,
    epsilon: float = 1.0e-30,
) -> dict[str, object]:
    """Mathematically simulate PyTorch-style global norm clipping.

    No parameter or ``.grad`` tensor is modified.
    """

    if max_norm <= 0 or total_name not in component_vectors:
        raise ValueError("valid max_norm and total component are required")
    total = component_vectors[total_name]
    total_norm = torch.linalg.vector_norm(total)
    scale = torch.clamp(
        total.new_tensor(float(max_norm)) / total_norm.clamp_min(float(epsilon)),
        max=1.0,
    )
    unit_total = total / total_norm.clamp_min(float(epsilon))
    projections = {}
    for name, vector in component_vectors.items():
        before = torch.dot(vector, unit_total)
        projections[name] = {
            "before": before,
            "after": scale * before,
        }
    clipped = scale * total
    return {
        "scale_factor": scale,
        "total_norm_before": total_norm,
        "total_norm_after": torch.linalg.vector_norm(clipped),
        "direction_cosine": torch.dot(total, clipped)
        / (
            total_norm
            * torch.linalg.vector_norm(clipped).clamp_min(float(epsilon))
        ),
        "component_projections": projections,
    }


def additive_region_rows(
    diagnostic: H1Diagnostic,
    region_masks: Mapping[str, torch.Tensor],
    *,
    channel_names: Sequence[str],
    gradient_density: torch.Tensor | None = None,
    overlap_masks: Mapping[str, torch.Tensor] | None = None,
) -> list[dict[str, object]]:
    """Decompose additive direction/channel density over spatial regions."""

    sample_density = next(iter(diagnostic.direction_density.values()))
    if sample_density.shape[0] != 1:
        raise ValueError("region decomposition requires batch size one")
    if len(channel_names) != sample_density.shape[1]:
        raise ValueError("channel names do not match density")
    total_density = sum(diagnostic.direction_density.values())
    global_total = total_density.sum().clamp_min(1.0e-30)
    nvoxels = total_density.shape[-3] * total_density.shape[-2] * total_density.shape[-1]
    overlap_masks = overlap_masks or {}
    rows: list[dict[str, object]] = []
    for region_name, original_mask in region_masks.items():
        mask = original_mask.to(device=sample_density.device, dtype=torch.bool)
        if tuple(mask.shape) != tuple(sample_density.shape[-3:]):
            raise ValueError(f"region {region_name!r} has the wrong shape")
        voxel_count = int(mask.sum())
        voxel_fraction = voxel_count / nvoxels
        if voxel_count == 0:
            raise ValueError(f"region {region_name!r} is empty")
        region_total = total_density[..., mask].sum()
        contribution_fraction = region_total / global_total
        aggregate: dict[str, object] = {
            "region": region_name,
            "channel": "all",
            "direction": "all",
            "voxel_fraction": voxel_fraction,
            "contribution_fraction": contribution_fraction,
            "per_voxel_mean": region_total / (voxel_count * total_density.shape[1]),
            "enrichment": contribution_fraction / voxel_fraction,
        }
        if gradient_density is not None:
            if gradient_density.shape != total_density.shape:
                raise ValueError("gradient density must match total density")
            aggregate["gradient_norm_fraction"] = (
                gradient_density[..., mask].sum()
                / gradient_density.sum().clamp_min(1.0e-30)
            )
        for overlap_name, overlap in overlap_masks.items():
            overlap = overlap.to(device=sample_density.device, dtype=torch.bool)
            if overlap.ndim == 4:
                overlap = torch.any(overlap, dim=0)
            if tuple(overlap.shape) != tuple(mask.shape):
                raise ValueError(f"overlap mask {overlap_name!r} has the wrong shape")
            aggregate[f"{overlap_name}_voxel_overlap"] = (
                (mask & overlap).sum() / mask.sum()
            )
            aggregate[f"{overlap_name}_h1_overlap"] = (
                total_density[..., mask & overlap].sum()
                / region_total.clamp_min(1.0e-30)
            )
        rows.append(aggregate)
        for direction, density in diagnostic.direction_density.items():
            for channel_index, channel_name in enumerate(channel_names):
                selected = density[0, channel_index][mask]
                value = selected.sum()
                rows.append(
                    {
                        "region": region_name,
                        "channel": channel_name,
                        "direction": direction,
                        "voxel_fraction": voxel_fraction,
                        "contribution_fraction": value / global_total,
                        "per_voxel_mean": selected.mean(),
                        "enrichment": (value / global_total) / voxel_fraction,
                    }
                )
    return rows


__all__ = [
    "AXIS_NAMES",
    "DEFAULT_POLE_SIN_FLOOR",
    "H1Diagnostic",
    "center_cell_widths",
    "coordinate_derivative",
    "coordinate_derivatives",
    "current_upstream_h1",
    "additive_region_rows",
    "flatten_optional_gradients",
    "gradient_pair_metrics",
    "h0_wrapper_parity",
    "h1_diagnostic",
    "normalized_axis_spacings",
    "simulate_global_norm_clip",
    "spherical_proxy_volume_weights",
]
