from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from grmhd import CHANNELS
from grmhd.dataset import make_temporal_datasets
from grmhd.normalizer import GRMHDNormalizer
from grmhd.priors import QuantileBounds, ResidualEnvelope


def make_file(path):
    rng = np.random.default_rng(11)
    values = rng.normal(0, 0.02, size=(10, 8, 3, 3, 4)).astype(np.float32)
    values[:, 3] = np.exp(values[:, 3])
    values[:, 4] = 1e-2 * np.exp(values[:, 4])
    with h5py.File(path, "w") as handle:
        handle.create_dataset("snapshots", data=values)
        handle.create_dataset("times", data=np.arange(10, dtype=np.float64))
        handle.create_dataset("channels", data=np.asarray(CHANNELS, dtype=h5py.string_dtype()))
        coords = handle.create_group("coords")
        coords.create_dataset("r", data=np.geomspace(1, 20, 4))
        coords.create_dataset("theta", data=np.linspace(0.1, 3.0, 3))
        coords.create_dataset("phi", data=np.linspace(0.1, 6.0, 3))


def test_quantile_bounds_fit_penalty_and_clamp(tmp_path):
    path = tmp_path / "priors.h5"
    make_file(path)
    train = make_temporal_datasets(path, train_fraction=0.6, val_fraction=0.2)["train"]
    normalizer = GRMHDNormalizer.fit(train, max_samples_per_channel=1000)
    bounds = QuantileBounds.fit(train, normalizer, max_samples_per_channel=1000)
    assert bounds.channels == (3, 4)
    prediction = normalizer.encode_tensor(train[0]["y"]).unsqueeze(0)
    extreme = prediction.clone()
    extreme[:, 3] = bounds.upper[0] + 100
    assert bounds.penalty(extreme) > 0
    clamped = bounds.clamp(extreme)
    assert float(clamped[:, 3].max()) <= bounds.upper[0]
    restored = QuantileBounds.from_dict(bounds.as_dict())
    assert restored == bounds


def test_residual_envelope_fit_penalty_and_clamp(tmp_path):
    path = tmp_path / "priors.h5"
    make_file(path)
    train = make_temporal_datasets(path, train_fraction=0.6, val_fraction=0.2)["train"]
    normalizer = GRMHDNormalizer.fit(train, max_samples_per_channel=1000)
    envelope = ResidualEnvelope.fit(train, normalizer, max_samples_per_channel=1000)
    pair = train[0]
    x = normalizer.encode_tensor(pair["x"]).unsqueeze(0)
    prediction = x + 100
    assert envelope.penalty(prediction, x) > 0
    clamped = envelope.clamp(prediction, x)
    residual = clamped - x
    for channel in range(8):
        assert float(residual[:, channel].max()) <= envelope.upper[channel] + 1e-6
        assert float(residual[:, channel].min()) >= envelope.lower[channel] - 1e-6
    restored = ResidualEnvelope.from_dict(envelope.as_dict())
    assert restored == envelope
