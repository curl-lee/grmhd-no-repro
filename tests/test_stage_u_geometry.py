from __future__ import annotations

import numpy as np
import torch

from grmhd.stage_u_geometry import (
    CoordinateFDPolicy,
    StoredCoordinateFiniteDifferenceConvolution3D,
    coordinate_gradient,
    padded_centered_derivative_1d,
)


def test_stored_coordinate_gradient_is_exact_for_linear_nonuniform_grid() -> None:
    r = np.geomspace(1.1, 200.0, 64)
    field = np.broadcast_to(r.reshape(1, 1, -1), (4, 5, 64))
    result = coordinate_gradient(field, r, axis=2)
    np.testing.assert_allclose(result, 1.0, rtol=1e-12, atol=1e-12)


def test_periodic_phi_derivative_closes_seam() -> None:
    phi = (np.arange(64) + 0.5) * 2 * np.pi / 64
    field = np.sin(phi)
    result = padded_centered_derivative_1d(field, phi, axis=0, padding="periodic")
    np.testing.assert_allclose(result, np.cos(phi), rtol=2e-3, atol=2e-3)


def test_coordinate_fd_shape_parameters_and_backward() -> None:
    phi = (np.arange(8) + 0.5) * 2 * np.pi / 8
    theta = (np.arange(8) + 0.5) * np.pi / 8
    r = np.geomspace(1.1, 20.0, 8)
    module = StoredCoordinateFiniteDifferenceConvolution3D(
        2, 3, phi=phi, theta=theta, r=r,
        policy=CoordinateFDPolicy(spherical_proxy=True),
    )
    assert sum(parameter.numel() for parameter in module.parameters()) == 3 * 3 * 2 * 3
    x = torch.randn(1, 2, 8, 8, 8, requires_grad=True)
    output = module(x)
    assert output.shape == (1, 3, 8, 8, 8)
    output.square().mean().backward()
    assert torch.isfinite(output).all()
    assert torch.isfinite(x.grad).all()
