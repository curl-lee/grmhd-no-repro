from __future__ import annotations

import h5py
import numpy as np
import pytest

from grmhd import CHANNELS
from grmhd.dataset import make_temporal_datasets
from grmhd.normalizer import GRMHDNormalizer
from grmhd.residual import ResidualScaleStats, validate_residual_wrapper_metadata


def make_file(path, *, offset: float = 0.0) -> None:
    rng = np.random.default_rng(19)
    values = rng.normal(0, 0.01, size=(9, 8, 2, 2, 3)).astype(np.float32)
    values[:, 2] = 0.0
    values[:, 3] = np.exp(values[:, 3])
    values[:, 4] = 0.01 * np.exp(values[:, 4])
    values[:, 0] += offset
    with h5py.File(path, "w") as handle:
        handle.create_dataset("snapshots", data=values)
        handle.create_dataset("times", data=np.arange(9, dtype=np.float64))
        handle.create_dataset(
            "channels", data=np.asarray(CHANNELS, dtype=h5py.string_dtype())
        )
        coords = handle.create_group("coords")
        coords.create_dataset("r", data=np.geomspace(1, 3, 3))
        coords.create_dataset("theta", data=np.linspace(0.2, 2.9, 2))
        coords.create_dataset("phi", data=np.linspace(0, np.pi, 2))


def test_train_only_residual_scale_floor_roundtrip_and_checksum(tmp_path):
    path = tmp_path / "residual.h5"
    make_file(path)
    training = make_temporal_datasets(path, train_fraction=5 / 9, val_fraction=2 / 9)[
        "train"
    ]
    with pytest.warns(RuntimeWarning):
        normalizer = GRMHDNormalizer.fit(training, max_samples_per_channel=1000)
    stats = ResidualScaleStats.fit(
        training,
        normalizer,
        quantile=0.999,
        multiplier=1.0,
        min_alpha=1e-5,
        max_samples_per_channel=1000,
    )
    assert stats.training_indices == training.owned_snapshot_indices
    assert stats.pair_starts == tuple(training.pair_starts)
    assert stats.alpha[2] == pytest.approx(1e-5)
    assert all(value >= stats.min_alpha for value in stats.alpha)

    stats_path = tmp_path / "residual_scale.json"
    stats.save(stats_path)
    loaded = ResidualScaleStats.load(
        stats_path,
        h5_path=path,
        expected_training_indices=training.owned_snapshot_indices,
        expected_quantile=0.999,
        expected_multiplier=1.0,
    )
    assert loaded == stats

    training.close()
    with h5py.File(path, "r+") as handle:
        handle["snapshots"][0, 0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="checksum mismatch"):
        ResidualScaleStats.load(
            stats_path,
            h5_path=path,
            expected_training_indices=training.owned_snapshot_indices,
            expected_quantile=0.999,
            expected_multiplier=1.0,
        )


def test_residual_scale_rejects_wrong_config(tmp_path):
    path = tmp_path / "residual.h5"
    make_file(path)
    training = make_temporal_datasets(path, train_fraction=5 / 9, val_fraction=2 / 9)[
        "train"
    ]
    with pytest.warns(RuntimeWarning):
        normalizer = GRMHDNormalizer.fit(training, max_samples_per_channel=1000)
    stats = ResidualScaleStats.fit(training, normalizer, max_samples_per_channel=1000)
    stats_path = tmp_path / "residual_scale.json"
    stats.save(stats_path)
    with pytest.raises(ValueError, match="quantile mismatch"):
        ResidualScaleStats.load(
            stats_path,
            h5_path=path,
            expected_training_indices=training.owned_snapshot_indices,
            expected_quantile=0.95,
            expected_multiplier=1.0,
        )

    wrapper_config = {
        "predict_residual": True,
        "zero_init_residual_head": True,
        "bounded_residual": True,
        "residual_scale_source": "train_abs_quantile",
        "residual_scale_quantile": 0.999,
        "residual_scale_multiplier": 1.0,
    }
    metadata = {
        "config": wrapper_config,
        "alpha": list(stats.alpha),
        "stats": stats.as_dict(),
    }
    validate_residual_wrapper_metadata(
        metadata,
        h5_path=path,
        expected_training_indices=training.owned_snapshot_indices,
        expected_config=wrapper_config,
    )
    with pytest.raises(ValueError, match="config mismatch"):
        validate_residual_wrapper_metadata(
            metadata,
            h5_path=path,
            expected_training_indices=training.owned_snapshot_indices,
            expected_config={**wrapper_config, "residual_scale_multiplier": 2.0},
        )
