from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from grmhd import CHANNELS
from grmhd.dataset import make_temporal_datasets
from grmhd.hybrid import HybridTargetStats


def _make_file(path) -> None:
    rng = np.random.default_rng(31)
    values = rng.normal(0, 0.01, size=(8, 8, 2, 2, 3)).astype(np.float32)
    values[:, 3] = np.exp(values[:, 3])
    values[:, 4] = 0.01 * np.exp(values[:, 4])
    with h5py.File(path, "w") as handle:
        handle.create_dataset("snapshots", data=values)
        handle.create_dataset("times", data=np.arange(8, dtype=np.float64))
        handle.create_dataset("channels", data=np.asarray(CHANNELS, dtype=h5py.string_dtype()))
        coords = handle.create_group("coords")
        coords.create_dataset("r", data=np.geomspace(1, 3, 3))
        coords.create_dataset("theta", data=np.linspace(0.2, 2.9, 2))
        coords.create_dataset("phi", data=np.linspace(0, np.pi, 2))


def test_hybrid_target_round_trip_and_checksum_rejection(tmp_path):
    path = tmp_path / "hybrid.h5"
    _make_file(path)
    training = make_temporal_datasets(path, train_fraction=0.5, val_fraction=0.25)["train"]
    stats = HybridTargetStats.fit(training, max_samples_per_channel=1000)
    pair = training[0]
    x = pair["x"].unsqueeze(0)
    raw = torch.full_like(x, 0.1)
    y = stats.reconstruct(x, raw)
    bounded_target = stats.bounded_target_prediction(raw)
    encoded_target = stats.encode_target(x, y)
    torch.testing.assert_close(encoded_target, bounded_target, rtol=5e-5, atol=5e-6)
    reconstructed = stats.reconstruct(x, raw)
    torch.testing.assert_close(reconstructed, y, rtol=0, atol=0)

    stats_path = tmp_path / "hybrid.json"
    stats.save(stats_path)
    loaded = HybridTargetStats.load(
        stats_path, h5_path=path,
        expected_training_indices=training.owned_snapshot_indices,
    )
    assert loaded == stats
    training.close()
    with h5py.File(path, "r+") as handle:
        handle["snapshots"][0, 0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="checksum mismatch"):
        HybridTargetStats.load(
            stats_path, h5_path=path,
            expected_training_indices=stats.training_indices,
        )
