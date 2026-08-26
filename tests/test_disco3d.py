import math

import pytest
import torch
import torch.nn.functional as F

from grmhd.models import build_model, trainable_parameter_count
from grmhd.localno3d_disco import (
    attach_adapted_disco3d,
    disco_parameter_names,
)
from grmhd.operators.disco3d import AdaptedRadialDISCO3d, radial_hat_support


def make_operator(
    in_channels=2, out_channels=3, *, groups=1, bias=True, dtype=torch.float64
):
    return AdaptedRadialDISCO3d(
        in_channels,
        out_channels,
        grid_shape=(7, 7, 7),
        domain_length=(2.0, 2.0, 2.0),
        groups=groups,
        bias=bias,
    ).to(dtype=dtype)


def explicit_grouped_correlation(operator, x, weight=None, bias=None):
    kernel = operator.dense_kernel(operator.weight if weight is None else weight)
    bias = operator.bias if bias is None else bias
    output = x.new_zeros((x.shape[0], operator.out_channels, *x.shape[-3:]))
    half = tuple(size // 2 for size in kernel.shape[-3:])
    outputs_per_group = operator.out_channels // operator.groups
    for batch in range(x.shape[0]):
        for group in range(operator.groups):
            input_start = group * operator.groupsize
            output_start = group * outputs_per_group
            for local_output in range(outputs_per_group):
                output_channel = output_start + local_output
                for i in range(x.shape[-3]):
                    for j in range(x.shape[-2]):
                        for k in range(x.shape[-1]):
                            value = x.new_zeros(())
                            for local_input in range(operator.groupsize):
                                input_channel = input_start + local_input
                                for di in range(kernel.shape[-3]):
                                    for dj in range(kernel.shape[-2]):
                                        for dk in range(kernel.shape[-1]):
                                            value = value + kernel[
                                                output_channel, local_input, di, dj, dk
                                            ] * x[
                                                batch,
                                                input_channel,
                                                (i + di - half[0]) % x.shape[-3],
                                                (j + dj - half[1]) % x.shape[-2],
                                                (k + dk - half[2]) % x.shape[-1],
                                            ]
                            if bias is not None:
                                value = value + bias[output_channel]
                            output[batch, output_channel, i, j, k] = value
    return output


def test_radius_three_is_minimum_integer_support_for_all_five_hats():
    active = {}
    for radius_cells in (1, 2, 3):
        support = radial_hat_support(
            grid_shape=(64, 64, 64),
            domain_length=(2.0, 2.0, 2.0),
            radius_cells=radius_cells,
            basis_count=5,
        )
        active[radius_cells] = (support.raw_basis > 0).sum(dim=(1, 2, 3)).tolist()
    assert active[1] == [1, 0, 0, 0, 6]
    assert active[2][1] == 0
    assert all(count > 0 for count in active[3])


def test_support_geometry_partition_normalization_and_compact_cutoff():
    support = radial_hat_support(
        grid_shape=(64, 64, 64), domain_length=(2.0, 2.0, 2.0)
    )
    assert support.delta_cell == pytest.approx(2.0 / 64.0)
    assert support.radius_cutoff == pytest.approx(3.0 * 2.0 / 64.0)
    assert support.stencil_shape == (7, 7, 7)
    inside = support.rho <= support.radius_cutoff
    outside = ~inside
    assert int(inside.sum()) == 123
    assert int(outside.sum()) == 220
    assert torch.max(torch.abs(support.raw_basis[:, outside])) == 0
    assert torch.max(torch.abs(support.raw_basis.sum(dim=0)[inside] - 1.0)) < 1e-6
    normalized_integrals = (
        support.quadrature_weight * support.normalized_basis.sum(dim=(1, 2, 3))
    )
    assert torch.max(torch.abs(normalized_integrals - 1.0)) < 1e-6
    assert torch.all(torch.isfinite(support.normalizations))
    assert torch.all(support.normalizations > 0)


def test_shape_finite_backward_bias_and_constant_field_response():
    torch.manual_seed(3)
    operator = make_operator()
    x = torch.randn(1, 2, 7, 7, 7, dtype=torch.float64, requires_grad=True)
    output = operator(x)
    assert output.shape == (1, 3, 7, 7, 7)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert operator.weight.grad is not None and torch.isfinite(operator.weight.grad).all()
    assert operator.bias.grad is not None and torch.isfinite(operator.bias.grad).all()

    constant = torch.ones_like(x.detach())
    observed = operator(constant)
    expected = operator.dense_kernel().sum(dim=(1, 2, 3, 4)) + operator.bias
    expected = expected.reshape(1, -1, 1, 1, 1).expand_as(observed)
    torch.testing.assert_close(observed, expected, rtol=1e-12, atol=1e-12)


def test_circular_translation_equivariance():
    torch.manual_seed(4)
    operator = make_operator()
    x = torch.randn(1, 2, 7, 7, 7, dtype=torch.float64)
    shift = (2, -1, 3)
    observed = operator(torch.roll(x, shift, dims=(-3, -2, -1)))
    expected = torch.roll(operator(x), shift, dims=(-3, -2, -1))
    torch.testing.assert_close(observed, expected, rtol=1e-12, atol=1e-12)


def test_groups_and_no_bias_match_explicit_dense_reference():
    torch.manual_seed(5)
    operator = make_operator(4, 4, groups=2, bias=False)
    x = torch.randn(1, 4, 7, 7, 7, dtype=torch.float64)
    observed = operator(x)
    expected = explicit_grouped_correlation(operator, x)
    relative = torch.linalg.vector_norm(observed - expected) / torch.linalg.vector_norm(expected)
    assert float(relative.detach()) < 1e-8
    assert float(torch.max(torch.abs(observed - expected)).detach()) < 1e-8


def test_kernel_orientation_uses_cross_correlation():
    x = torch.arange(5**3, dtype=torch.float64).reshape(1, 1, 5, 5, 5)
    kernel = torch.zeros(1, 1, 3, 3, 3, dtype=torch.float64)
    kernel[0, 0, 0, 1, 1] = 2.0
    kernel[0, 0, 2, 1, 1] = -0.5
    padded = F.pad(x, (1, 1, 1, 1, 1, 1), mode="circular")
    observed = F.conv3d(padded, kernel)
    expected = 2.0 * torch.roll(x, 1, dims=-3) - 0.5 * torch.roll(x, -1, dims=-3)
    flipped = -0.5 * torch.roll(x, 1, dims=-3) + 2.0 * torch.roll(x, -1, dims=-3)
    torch.testing.assert_close(observed, expected, rtol=0, atol=0)
    assert not torch.equal(observed, flipped)


def test_autograd_gradcheck_input_weight_and_bias():
    torch.manual_seed(6)
    operator = make_operator(1, 1)
    x = torch.randn(1, 1, 7, 7, 7, dtype=torch.float64, requires_grad=True)
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


def test_localno_attachment_preserves_common_parameters_and_adds_four_branches():
    torch.manual_seed(17)
    model = build_model(
        "localno_differential_3d",
        in_channels=16,
        out_channels=8,
        n_modes=(8, 8, 8),
        hidden_channels=16,
        n_layers=4,
        default_in_shape=(64, 64, 64),
        positional_embedding=None,
        fin_diff_kernel_size=3,
        mix_derivatives=True,
        conv_padding_mode="periodic",
        use_channel_mlp=False,
        local_no_skip="linear",
        norm=None,
        enforce_hermitian_symmetry=True,
    )
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    assert trainable_parameter_count(model) == 358_296
    attach_adapted_disco3d(model)
    assert trainable_parameter_count(model) == 363_480
    assert len(model.local_no_blocks.local_convs) == 4
    assert model.local_no_blocks.disco_idx_list == [0, 1, 2, 3]
    assert len(disco_parameter_names(model)) == 8
    for name, expected in before.items():
        torch.testing.assert_close(dict(model.named_parameters())[name], expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cpu_cuda_agreement():
    torch.manual_seed(7)
    cpu = make_operator(dtype=torch.float32)
    cuda = make_operator(dtype=torch.float32).cuda()
    cuda.load_state_dict(cpu.state_dict(), strict=True)
    x = torch.randn(1, 2, 7, 7, 7)
    observed = cuda(x.cuda()).cpu()
    expected = cpu(x)
    torch.testing.assert_close(observed, expected, rtol=2e-5, atol=2e-5)
