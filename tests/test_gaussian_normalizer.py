from __future__ import annotations

import numpy as np
import torch

from grmhd.gaussian_normalizer import TransformedGaussianNormalizer


def test_transformed_gaussian_round_trip_has_no_soft_clip():
    normalizer = TransformedGaussianNormalizer(
        mean=np.zeros(8),
        std=np.ones(8),
        epsilon=np.asarray([1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 0, 0, 0]),
    )
    physical = torch.randn(2, 8, 3, 4, 5) * 0.01
    physical[:, 3] = torch.exp(physical[:, 3])
    physical[:, 4] = 0.1 * torch.exp(physical[:, 4])
    encoded = normalizer.encode_tensor(physical, channel_axis=1)
    decoded = normalizer.decode_tensor(encoded, channel_axis=1)
    torch.testing.assert_close(decoded, physical, rtol=2e-5, atol=1e-7)
    assert torch.max(torch.abs(encoded)) < 6.0
    extreme = encoded.clone()
    extreme[:, 0] = 7.0
    assert torch.all(normalizer.decode_tensor(extreme)[:, 0] > 0)
