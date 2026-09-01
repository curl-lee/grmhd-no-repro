from __future__ import annotations

import torch

from grmhd.stage_ai import (
    ParameterMatchedUNet3d,
    SphericalTensorPad3d,
    build_canonical_cnn,
    first_landmark,
    parameter_match_pass,
)


def test_unet_shape_and_parameter_budget():
    model, identity = build_canonical_cnn(seed=42)
    assert isinstance(model, ParameterMatchedUNet3d)
    assert identity["parameter_count"] == 346488
    assert parameter_match_pass(identity["parameter_count"])
    value = torch.randn(1, 16, 16, 16, 16)
    assert model(x=value).shape == (1, 8, 16, 16, 16)


def test_cnn_initialization_is_reproducible():
    _, left = build_canonical_cnn(seed=42)
    _, right = build_canonical_cnn(seed=42)
    assert left["trainable_state_sha256"] == right["trainable_state_sha256"]


def test_spherical_tensor_padding_is_periodic_only_in_phi():
    value = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(1, 1, 2, 3, 4)
    padded = SphericalTensorPad3d(1)(value)
    assert padded.shape == (1, 1, 4, 5, 6)
    torch.testing.assert_close(padded[:, :, 0, 1:-1, 1:-1], value[:, :, -1])
    torch.testing.assert_close(padded[:, :, -1, 1:-1, 1:-1], value[:, :, 0])
    assert torch.count_nonzero(padded[:, :, :, 0, :]) == 0
    assert torch.count_nonzero(padded[:, :, :, :, 0]) == 0


def test_first_landmark_uses_not_reached_not_stable():
    assert first_landmark([False, True, False]) == 2
    assert first_landmark([False] * 100) == "NOT_REACHED"
