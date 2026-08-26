import math

import pytest
import torch

from grmhd.anisotropic_spherical_disco_localno import (
    anisotropic_disco_parameter_names,
    attach_anisotropic_spherical_disco3d,
)
from grmhd.models import build_model, trainable_parameter_count
from grmhd.operators.anisotropic_spherical_disco3d import (
    AnisotropicSphericalDISCO3d,
    build_anisotropic_spherical_geometry,
    local_spherical_frame,
)


def geomspace(start, stop, steps):
    return torch.exp(torch.linspace(math.log(start), math.log(stop), steps, dtype=torch.float64))


def tiny_coordinates():
    return (
        geomspace(1.0, 3.0, 5),
        torch.linspace(0.4, math.pi - 0.4, 6, dtype=torch.float64),
        (torch.arange(8, dtype=torch.float64) + 0.5) * 2 * math.pi / 8,
    )


def production_coordinates():
    faces = geomspace(1.100000023841858, 200.0, 65)
    return (
        torch.sqrt(faces[:-1] * faces[1:]),
        (torch.arange(64, dtype=torch.float64) + 0.5) * math.pi / 64,
        (torch.arange(64, dtype=torch.float64) + 0.5) * 2 * math.pi / 64,
    )


@pytest.fixture(scope="module")
def production_geometry():
    r, theta, phi = production_coordinates()
    return build_anisotropic_spherical_geometry(r=r, theta=theta, phi=phi)


def make_operator(dtype=torch.float64):
    r, theta, phi = tiny_coordinates()
    return AnisotropicSphericalDISCO3d(1, 1, r=r, theta=theta, phi=phi).to(dtype=dtype)


def explicit_reference(operator, x):
    r, theta, phi = tiny_coordinates()
    geometry = build_anisotropic_spherical_geometry(r=r, theta=theta, phi=phi)
    result = x.new_zeros((1, operator.out_channels, *x.shape[-3:]))
    for iphi in range(x.shape[-3]):
        for itheta in range(x.shape[-2]):
            for ir in range(x.shape[-1]):
                fields = x.new_zeros((operator.in_channels, 5))
                for index, (dphi, dtheta, dr) in enumerate(geometry.offsets):
                    source_theta, source_r = itheta + dtheta, ir + dr
                    if 0 <= source_theta < x.shape[-2] and 0 <= source_r < x.shape[-1]:
                        fields += (
                            x[0, :, (iphi + dphi) % x.shape[-3], source_theta, source_r].unsqueeze(1)
                            * operator.integration_weights[:, index, itheta, ir].to(x.dtype).unsqueeze(0)
                        )
                result[0, :, iphi, itheta, ir] = torch.einsum("ock,ck->o", operator.weight, fields) + operator.bias
    return result


def test_local_frame_orthonormality_below_1e_minus_12():
    theta = torch.linspace(0.01, math.pi - 0.01, 64, dtype=torch.float64)[:, None]
    phi = torch.linspace(0, 2 * math.pi, 65, dtype=torch.float64)[:-1][None, :]
    frames = local_spherical_frame(theta, phi)
    gram = torch.einsum("...ai,...bi->...ab", frames, frames)
    assert float(torch.max(torch.abs(gram - torch.eye(3)))) < 1e-12


def test_production_directional_scales_positive_and_bad_corners_repaired(production_geometry):
    scales = production_geometry.directional_scales
    assert torch.isfinite(scales).all()
    assert torch.all(scales > 0)
    assert int(torch.sum(scales <= 0)) == 0
    for theta_index, radial_index in ((0, 0), (0, 63), (63, 0), (63, 63)):
        assert torch.all(scales[:, theta_index, radial_index] > 0)
        assert int(production_geometry.active_basis_count[theta_index, radial_index]) == 5
        assert torch.all(production_geometry.normalizations[:, theta_index, radial_index] > 0)


def test_full_production_normalization_partition_quadrature_and_compact_support(production_geometry):
    assert int(torch.sum(production_geometry.normalizations <= 0)) == 0
    assert torch.all(production_geometry.active_basis_count == 5)
    assert torch.all(torch.isfinite(production_geometry.volume_proxy))
    assert torch.all(production_geometry.volume_proxy > 0)
    assert production_geometry.partition_max_abs_error < 1e-12
    normalized = production_geometry.integration_weights.sum(dim=1)
    torch.testing.assert_close(normalized, torch.ones_like(normalized), rtol=0, atol=1e-12)
    outside = production_geometry.normalized_radius > 1
    assert torch.max(torch.abs(production_geometry.raw_basis[:, outside])) == 0


def test_theta_and_r_truncate_while_phi_is_periodic(production_geometry):
    offsets = production_geometry.offsets
    assert not torch.any(production_geometry.valid_mask[offsets.index((0, -1, 0)), 0])
    assert not torch.any(production_geometry.valid_mask[offsets.index((0, 1, 0)), -1])
    assert not torch.any(production_geometry.valid_mask[offsets.index((0, 0, -1)), :, 0])
    assert not torch.any(production_geometry.valid_mask[offsets.index((0, 0, 1)), :, -1])
    assert torch.all(production_geometry.valid_mask[offsets.index((-1, 0, 0))])
    assert torch.all(production_geometry.valid_mask[offsets.index((1, 0, 0))])


def test_phi_rotational_equivariance_and_constant_field():
    torch.manual_seed(3)
    operator = make_operator(dtype=torch.float32)
    x = torch.randn(1, 1, *operator.grid_shape)
    for shift in (1, 7, 17, 31):
        shift = shift % operator.grid_shape[0]
        observed = operator(torch.roll(x, shift, dims=2))
        expected = torch.roll(operator(x), shift, dims=2)
        relative = torch.linalg.vector_norm(observed - expected) / torch.linalg.vector_norm(expected)
        assert float(relative.detach()) < 1e-5
    constant = operator(torch.ones_like(x))
    variation = torch.std(constant) / torch.mean(torch.abs(constant))
    assert float(variation.detach()) < 1e-5


def test_optimized_forward_matches_tiny_brute_force():
    torch.manual_seed(4)
    operator = make_operator()
    x = torch.randn(1, 1, *operator.grid_shape, dtype=torch.float64)
    observed = operator(x)
    expected = explicit_reference(operator, x)
    relative = torch.linalg.vector_norm(observed - expected) / torch.linalg.vector_norm(expected)
    assert float(relative.detach()) < 1e-8
    assert float(torch.max(torch.abs(observed - expected)).detach()) < 1e-8


def test_gradcheck_input_weight_and_bias():
    torch.manual_seed(5)
    operator = make_operator()
    x = torch.randn(1, 1, *operator.grid_shape, dtype=torch.float64, requires_grad=True)
    weight = operator.weight.detach().clone().requires_grad_(True)
    bias = operator.bias.detach().clone().requires_grad_(True)
    assert torch.autograd.gradcheck(
        operator.forward_with_parameters, (x, weight, bias),
        eps=1e-6, atol=1e-5, rtol=1e-3, fast_mode=True,
    )


def test_localno_attachment_preserves_trainable_contract():
    model = build_model(
        "localno_differential_3d", in_channels=16, out_channels=8,
        n_modes=(8, 8, 8), hidden_channels=16, n_layers=4,
        default_in_shape=(64, 64, 64), positional_embedding=None,
        fin_diff_kernel_size=3, mix_derivatives=True,
        conv_padding_mode="periodic", use_channel_mlp=False,
        local_no_skip="linear", norm=None, enforce_hermitian_symmetry=True,
    )
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    r, theta, phi = tiny_coordinates()
    attach_anisotropic_spherical_disco3d(model, r=r, theta=theta, phi=phi, seed=42)
    assert trainable_parameter_count(model) == 363_480
    assert len(anisotropic_disco_parameter_names(model)) == 8
    for name, expected in before.items():
        torch.testing.assert_close(dict(model.named_parameters())[name], expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_cuda_forward_agreement():
    torch.manual_seed(8)
    cpu = make_operator(dtype=torch.float32)
    cuda = make_operator(dtype=torch.float32).cuda()
    cuda.load_state_dict(cpu.state_dict(), strict=True)
    x = torch.randn(1, 1, *cpu.grid_shape)
    torch.testing.assert_close(cpu(x), cuda(x.cuda()).cpu(), rtol=2e-5, atol=2e-5)
