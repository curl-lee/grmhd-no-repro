import math

import pytest
import torch

from grmhd.operators.spherical_disco3d import (
    SphericalAwareDISCO3d,
    build_spherical_proxy_geometry,
    reconstruct_faces,
    spherical_embedding,
)


def geomspace(start, stop, steps):
    return torch.exp(
        torch.linspace(math.log(start), math.log(stop), steps, dtype=torch.float64)
    )


def tiny_coordinates():
    phi = (torch.arange(8, dtype=torch.float64) + 0.5) * (2.0 * math.pi / 8)
    theta = torch.linspace(0.4, math.pi - 0.4, 6, dtype=torch.float64)
    r = geomspace(1.0, 3.0, 5)
    return r, theta, phi


def production_coordinates():
    radial_faces = geomspace(1.100000023841858, 200.0, 65)
    r = torch.sqrt(radial_faces[:-1] * radial_faces[1:])
    theta = (torch.arange(64, dtype=torch.float64) + 0.5) * math.pi / 64
    phi = (torch.arange(64, dtype=torch.float64) + 0.5) * 2.0 * math.pi / 64
    return r, theta, phi


def make_operator(dtype=torch.float64):
    r, theta, phi = tiny_coordinates()
    return SphericalAwareDISCO3d(
        1, 1, r=r, theta=theta, phi=phi
    ).to(dtype=dtype)


def brute_force(operator, x, weight=None, bias=None):
    weight = operator.weight if weight is None else weight
    bias = operator.bias if bias is None else bias
    result = x.new_zeros((x.shape[0], operator.out_channels, *x.shape[-3:]))
    geometry = operator.integration_weights.to(dtype=x.dtype)
    coordinate_geometry = build_spherical_proxy_geometry(
        r=tiny_coordinates()[0], theta=tiny_coordinates()[1], phi=tiny_coordinates()[2]
    )
    nphi, ntheta, nr = x.shape[-3:]
    for batch in range(x.shape[0]):
        for output_channel in range(operator.out_channels):
            for iphi in range(nphi):
                for itheta in range(ntheta):
                    for ir in range(nr):
                        value = x.new_zeros(())
                        for input_channel in range(operator.in_channels):
                            for basis in range(operator.basis_count):
                                filtered = x.new_zeros(())
                                for offset_index, (dphi, dtheta, dr) in enumerate(
                                    coordinate_geometry.offsets
                                ):
                                    source_theta = itheta + dtheta
                                    source_r = ir + dr
                                    if 0 <= source_theta < ntheta and 0 <= source_r < nr:
                                        filtered = filtered + geometry[
                                            basis, offset_index, itheta, ir
                                        ] * x[
                                            batch,
                                            input_channel,
                                            (iphi + dphi) % nphi,
                                            source_theta,
                                            source_r,
                                        ]
                                value = value + weight[
                                    output_channel, input_channel, basis
                                ] * filtered
                        if bias is not None:
                            value = value + bias[output_channel]
                        result[batch, output_channel, iphi, itheta, ir] = value
    return result


def test_coordinate_embedding_known_equatorial_points():
    r = torch.tensor([2.0, 2.0], dtype=torch.float64)
    theta = torch.tensor([math.pi / 2, math.pi / 2], dtype=torch.float64)
    phi = torch.tensor([0.0, math.pi / 2], dtype=torch.float64)
    observed = spherical_embedding(r, theta, phi)
    expected = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]], dtype=torch.float64)
    torch.testing.assert_close(observed, expected, rtol=0, atol=1e-14)


def test_face_reconstruction_uses_geometric_r_and_uniform_angles():
    r_faces = geomspace(1.1, 200.0, 65)
    r = torch.sqrt(r_faces[:-1] * r_faces[1:])
    observed_r, radial_method = reconstruct_faces(r, radial=True)
    torch.testing.assert_close(observed_r, r_faces, rtol=1e-12, atol=1e-12)
    phi = (torch.arange(64, dtype=torch.float64) + 0.5) * 2 * math.pi / 64
    observed_phi, angular_method = reconstruct_faces(phi, radial=False)
    torch.testing.assert_close(
        observed_phi, torch.linspace(0, 2 * math.pi, 65, dtype=torch.float64),
        rtol=1e-12, atol=1e-12,
    )
    assert radial_method.startswith("geometric")
    assert angular_method.startswith("uniform")


def test_phi_periodicity_and_rotational_equivariance():
    first = spherical_embedding(
        torch.tensor(2.0), torch.tensor(math.pi / 3), torch.tensor(0.0)
    )
    seam = spherical_embedding(
        torch.tensor(2.0), torch.tensor(math.pi / 3), torch.tensor(2 * math.pi)
    )
    torch.testing.assert_close(first, seam, rtol=0, atol=1e-6)

    torch.manual_seed(4)
    operator = make_operator(dtype=torch.float32)
    x = torch.randn(1, 1, *operator.grid_shape)
    observed = operator(torch.roll(x, 2, dims=2))
    expected = torch.roll(operator(x), 2, dims=2)
    relative = torch.linalg.vector_norm(observed - expected) / torch.linalg.vector_norm(expected)
    assert float(relative.detach()) < 1e-5


def test_theta_and_radial_boundaries_truncate_without_wrap():
    geometry = build_spherical_proxy_geometry(
        r=tiny_coordinates()[0], theta=tiny_coordinates()[1], phi=tiny_coordinates()[2]
    )
    theta_minus = geometry.offsets.index((0, -1, 0))
    theta_plus = geometry.offsets.index((0, 1, 0))
    radial_minus = geometry.offsets.index((0, 0, -1))
    radial_plus = geometry.offsets.index((0, 0, 1))
    assert not torch.any(geometry.valid_mask[theta_minus, 0])
    assert not torch.any(geometry.valid_mask[theta_plus, -1])
    assert not torch.any(geometry.valid_mask[radial_minus, :, 0])
    assert not torch.any(geometry.valid_mask[radial_plus, :, -1])
    assert torch.all(geometry.valid_mask[theta_minus, 1:])
    assert torch.all(geometry.valid_mask[radial_minus, :, 1:])


def test_quadrature_no_pole_singularity_and_valid_tiny_normalization():
    geometry = build_spherical_proxy_geometry(
        r=tiny_coordinates()[0], theta=tiny_coordinates()[1], phi=tiny_coordinates()[2]
    )
    assert torch.isfinite(geometry.distances[geometry.valid_mask]).all()
    assert torch.isfinite(geometry.local_scale).all()
    assert torch.isfinite(geometry.raw_basis).all()
    assert torch.isfinite(geometry.volume_proxy).all()
    assert torch.all(geometry.volume_proxy > 0)
    assert torch.all(geometry.normalizations > 0)
    normalized = geometry.integration_weights.sum(dim=1)
    torch.testing.assert_close(normalized, torch.ones_like(normalized), rtol=1e-9, atol=1e-9)


def test_production_contract_exposes_four_zero_normalizations_and_blocks_operator():
    r, theta, phi = production_coordinates()
    geometry = build_spherical_proxy_geometry(r=r, theta=theta, phi=phi)
    zero = torch.nonzero(geometry.normalizations <= 1e-12, as_tuple=False).tolist()
    assert zero == [[4, 0, 0], [4, 0, 63], [4, 63, 0], [4, 63, 63]]
    with pytest.raises(ValueError, match="ZERO_Z_COUNT=4"):
        SphericalAwareDISCO3d(1, 1, r=r, theta=theta, phi=phi)


def test_optimized_forward_matches_brute_force_and_constant_field():
    torch.manual_seed(5)
    operator = make_operator()
    x = torch.randn(1, 1, *operator.grid_shape, dtype=torch.float64)
    observed = operator(x)
    expected = brute_force(operator, x)
    difference = observed - expected
    relative = torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(expected)
    assert float(relative.detach()) < 1e-8
    assert float(torch.max(torch.abs(difference)).detach()) < 1e-8

    constant = torch.ones_like(x)
    constant_output = operator(constant)
    spatial_variation = torch.std(constant_output) / torch.mean(torch.abs(constant_output))
    assert float(spatial_variation.detach()) < 1e-5


def test_gradcheck_input_weight_and_bias():
    torch.manual_seed(6)
    operator = make_operator()
    x = torch.randn(1, 1, *operator.grid_shape, dtype=torch.float64, requires_grad=True)
    weight = operator.weight.detach().clone().requires_grad_(True)
    bias = operator.bias.detach().clone().requires_grad_(True)
    assert torch.autograd.gradcheck(
        operator.forward_with_parameters,
        (x, weight, bias),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
        fast_mode=True,
    )
