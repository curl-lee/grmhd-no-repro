import math

import torch

from grmhd.paper_h1_diagnostics import (
    coordinate_derivative,
    coordinate_derivatives,
    h1_diagnostic,
    spherical_proxy_volume_weights,
)


def coordinates(nphi=16, ntheta=12, nr=10):
    return {
        "phi": (torch.arange(nphi, dtype=torch.float64) + 0.5)
        * (2 * math.pi / nphi),
        "theta": (torch.arange(ntheta, dtype=torch.float64) + 0.5)
        * (math.pi / ntheta),
        "r": torch.exp(
            torch.linspace(
                math.log(1.2), math.log(20.0), nr, dtype=torch.float64
            )
        ),
    }


def mesh(coords):
    return torch.meshgrid(
        coords["phi"], coords["theta"], coords["r"], indexing="ij"
    )


def test_constant_has_zero_gradient_for_every_variant():
    coords = coordinates()
    error = torch.ones(1, 2, 16, 12, 10, dtype=torch.float64)
    for variant in (
        "H0_current_upstream",
        "H1_unit_index",
        "H2_normalized_axis",
        "H3_stored_r",
        "H3_stored_logr",
        "H4_spherical_metric_proxy",
    ):
        result = h1_diagnostic(error, variant=variant, coordinates=coords)
        torch.testing.assert_close(result.total, torch.tensor(0.0, dtype=error.dtype))


def test_unit_index_linear_field_has_unit_interior_gradient_and_documented_wrap():
    nphi = 8
    index = torch.arange(nphi, dtype=torch.float64)
    field = index[None, None, :, None, None].expand(1, 1, nphi, 5, 6)
    result = h1_diagnostic(field, variant="H1_unit_index")
    derivative = result.direction_density["phi"].sqrt()
    torch.testing.assert_close(derivative[..., 1:-1, :, :], torch.ones_like(derivative[..., 1:-1, :, :]))
    expected_boundary_magnitude = (nphi - 2) / 2
    torch.testing.assert_close(
        derivative[..., (0, -1), :, :],
        torch.full_like(derivative[..., (0, -1), :, :], expected_boundary_magnitude),
    )


def test_normalized_linear_field_has_expected_n_scale_interior():
    nphi = 10
    normalized = torch.arange(nphi, dtype=torch.float64) / nphi
    field = normalized[None, None, :, None, None].expand(1, 1, nphi, 5, 6)
    result = h1_diagnostic(field, variant="H2_normalized_axis")
    derivative = result.direction_density["phi"].sqrt()
    torch.testing.assert_close(
        derivative[..., 1:-1, :, :],
        torch.ones_like(derivative[..., 1:-1, :, :]),
    )


def test_nonuniform_radial_linear_field_derivative_is_one_including_boundaries():
    coords = coordinates(nphi=8, ntheta=7, nr=9)
    _, _, radius = mesh(coords)
    field = radius[None, None]
    derivatives = coordinate_derivatives(field, coords, radial_coordinate="r")
    torch.testing.assert_close(
        derivatives["r"], torch.ones_like(derivatives["r"]), rtol=1.0e-12, atol=1.0e-12
    )


def test_periodic_phi_sine_wrap_matches_centered_analytic_discretization():
    coords = coordinates(nphi=32, ntheta=6, nr=5)
    phi, _, _ = mesh(coords)
    field = phi.sin()[None, None]
    derivative = coordinate_derivative(
        field, coords["phi"], dim=-3, periodic=True
    )
    spacing = float(coords["phi"][1] - coords["phi"][0])
    expected = phi.cos() * (math.sin(spacing) / spacing)
    torch.testing.assert_close(
        derivative, expected[None, None], rtol=1.0e-12, atol=1.0e-12
    )


def test_theta_cosine_and_separable_fields_are_finite_with_open_boundaries():
    coords = coordinates()
    phi, theta, radius = mesh(coords)
    for field in (theta.cos(), radius * theta.sin() * phi.cos()):
        derivatives = coordinate_derivatives(field[None, None], coords)
        assert all(torch.isfinite(value).all() for value in derivatives.values())


def test_volume_proxy_weights_are_positive_normalized_and_outer_weighted():
    coords = coordinates()
    weights = spherical_proxy_volume_weights(coords)
    assert torch.isfinite(weights).all()
    assert torch.all(weights > 0)
    torch.testing.assert_close(
        weights.sum(), torch.tensor(1.0, dtype=weights.dtype), rtol=0, atol=1.0e-14
    )
    assert weights[..., -1].sum() > weights[..., 0].sum()


def test_h4_scalar_and_naive_all_channel_modes_are_finite():
    coords = coordinates()
    torch.manual_seed(19)
    error = torch.randn(1, 8, 16, 12, 10, dtype=torch.float64)
    scalar = h1_diagnostic(
        error,
        variant="H4_spherical_metric_proxy",
        coordinates=coords,
        reduction="volume_proxy",
        channel_mode="scalar_channel_proxy_only",
    )
    naive = h1_diagnostic(
        error,
        variant="H4_spherical_metric_proxy",
        coordinates=coords,
        reduction="volume_proxy",
        channel_mode="all_channels_naive_proxy",
    )
    assert torch.isfinite(scalar.total)
    assert torch.isfinite(naive.total)
    assert scalar.per_channel.shape == (2,)
    assert naive.per_channel.shape == (8,)
    assert "not a covariant vector derivative" in str(naive.metadata["warning"])
