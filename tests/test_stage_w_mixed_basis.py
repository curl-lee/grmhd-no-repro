import math
from pathlib import Path

import torch
import yaml

from grmhd.models import trainable_parameter_count
from grmhd.paper_stage_g import tensor_state_sha256
from grmhd.stage_w_mixed_basis import (
    AxisBoundaryFiniteDifferenceConvolution3d,
    MixedBasisSpectralConv3d,
    dct_lowpass_1d,
    mixed_basis_energy,
    mixed_basis_inverse,
    mixed_basis_transform,
    orthonormal_dct_ii,
    orthonormal_idct_ii,
)
from grmhd.stage_w_training import build_stage_w_model, stage_w_decision


ROOT = Path(__file__).resolve().parents[1]


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(actual - expected) / torch.linalg.vector_norm(expected))


def test_dct_roundtrip_float32_and_float64() -> None:
    generator = torch.Generator().manual_seed(42)
    for dtype, tolerance in ((torch.float32, 1e-6), (torch.float64, 1e-10)):
        x = torch.randn(3, 7, 64, generator=generator, dtype=dtype)
        reconstructed = orthonormal_idct_ii(orthonormal_dct_ii(x), dim=-1)
        assert relative_l2(reconstructed, x) < tolerance


def test_mixed_transform_roundtrip_and_parseval() -> None:
    x = torch.randn(2, 3, 16, 12, 10, generator=torch.Generator().manual_seed(7))
    coefficients = mixed_basis_transform(x)
    reconstructed = mixed_basis_inverse(coefficients, n_phi=x.shape[-3])
    assert relative_l2(reconstructed, x) < 1e-6
    relative_energy_error = abs(float(mixed_basis_energy(coefficients, n_phi=16) - x.square().sum())) / float(x.square().sum())
    assert relative_energy_error < 1e-6


def test_mixed_mode_localization() -> None:
    n_phi = n_theta = n_r = 32
    m_phi, m_theta, m_r = 3, 4, 5
    phi = 2 * math.pi * torch.arange(n_phi) / n_phi
    theta_nodes = (torch.arange(n_theta) + 0.5) / n_theta
    radial_nodes = (torch.arange(n_r) + 0.5) / n_r
    x = (
        torch.sin(m_phi * phi).reshape(n_phi, 1, 1)
        * torch.cos(math.pi * m_theta * theta_nodes).reshape(1, n_theta, 1)
        * torch.cos(math.pi * m_r * radial_nodes).reshape(1, 1, n_r)
    )
    coefficients = mixed_basis_transform(x)
    energy = coefficients.abs().square()
    localized = float(energy[m_phi, m_theta, m_r] / energy.sum())
    assert localized > 0.999999


def test_phi_circular_shift_has_fourier_phase() -> None:
    x = torch.randn(4, 32, 20, 18, generator=torch.Generator().manual_seed(11))
    shift = 5
    original = mixed_basis_transform(x)
    shifted = mixed_basis_transform(torch.roll(x, shifts=shift, dims=-3))
    modes = torch.arange(original.shape[-3], dtype=x.dtype)
    phase = torch.exp(-2j * math.pi * modes * shift / x.shape[-3]).reshape(1, -1, 1, 1)
    assert relative_l2(shifted, original * phase) < 1e-6


def test_nonperiodic_cosine_lowpass_does_not_wrap_boundary_pulse() -> None:
    pulse = torch.zeros(64)
    pulse[0] = 1
    cosine_response = dct_lowpass_1d(pulse, keep=8)
    fourier = torch.fft.rfft(pulse, norm="ortho")
    fourier_low = torch.zeros_like(fourier)
    fourier_low[:5] = fourier[:5]
    periodic_response = torch.fft.irfft(fourier_low, n=64, norm="ortho")
    cosine_outer = torch.linalg.vector_norm(cosine_response[-4:])
    periodic_outer = torch.linalg.vector_norm(periodic_response[-4:])
    assert cosine_outer < 0.2 * periodic_outer


def test_mixed_spectral_conv_forward_backward() -> None:
    module = MixedBasisSpectralConv3d(3, 4, (8, 8, 8))
    x = torch.randn(2, 3, 16, 16, 16, requires_grad=True)
    output = module(x)
    assert output.shape == (2, 4, 16, 16, 16)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(module.weight.grad).all()


def test_stage_w_variants_have_identical_tensors_and_parameter_count() -> None:
    stage_s = yaml.safe_load((ROOT / "configs/stage_s/expanded_localno_p3_residual.yaml").read_text())
    initial = torch.load(
        ROOT / stage_s["frozen_pairing"]["initial_state"], map_location="cpu", weights_only=True
    )
    w1, init_w1 = build_stage_w_model(stage_s, "mixed_basis", initial_state=initial)
    w2, init_w2 = build_stage_w_model(stage_s, "mixed_basis_boundary", initial_state=initial)
    assert trainable_parameter_count(w1) == trainable_parameter_count(w2) == 358_296
    assert init_w1["parameter_count_relative_difference"] == 0
    assert init_w1["full_initial_tensor_state_sha256"] == init_w2["full_initial_tensor_state_sha256"]
    assert tensor_state_sha256(w1.state_dict()) == tensor_state_sha256(w2.state_dict())
    for index in range(4):
        source = initial[f"local_no_blocks.convs.{index}.weight.tensor"]
        assert torch.equal(w1.local_no_blocks.convs[index].weight, source.permute(0, 1, 4, 3, 2))


def test_axis_boundary_fd_only_changes_theta_and_r_boundary_cells() -> None:
    from neuralop.layers.differential_conv import FiniteDifferenceConvolution

    periodic = FiniteDifferenceConvolution(2, 3, 3, kernel_size=3, groups=1, padding="periodic")
    boundary = AxisBoundaryFiniteDifferenceConvolution3d(2, 3)
    boundary.load_state_dict(periodic.state_dict(), strict=True)
    x = torch.randn(1, 2, 8, 8, 8, generator=torch.Generator().manual_seed(19))
    expected = periodic(x, 1.0)
    actual = boundary(x, 1.0)
    assert torch.allclose(actual[:, :, :, 1:-1, 1:-1], expected[:, :, :, 1:-1, 1:-1])
    assert not torch.allclose(actual[:, :, :, 0], expected[:, :, :, 0])
    assert not torch.allclose(actual[:, :, :, :, 0], expected[:, :, :, :, 0])


def test_stage_w_decision_contract() -> None:
    base = {
        "state_l2": 0.27, "residual_l2": 0.90, "cosine": 0.74,
        "shell_skill": -1.50, "radial_skill": -0.30, "first10x": 1,
    }
    assert stage_w_decision(
        {**base, "shell_skill": 0.1}, base, boundary_benefit=False
    ) == "A"
    assert stage_w_decision(base, base, boundary_benefit=False) == "B"
    assert stage_w_decision(
        {**base, "shell_skill": -1.65, "radial_skill": -0.34},
        base, boundary_benefit=True,
    ) == "C"
    assert stage_w_decision(
        {**base, "shell_skill": -2.0, "radial_skill": -0.6},
        {**base, "shell_skill": -2.1, "radial_skill": -0.7},
        boundary_benefit=False,
    ) == "D"
