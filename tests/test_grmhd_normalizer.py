from __future__ import annotations

import h5py
import json
import numpy as np
import pytest
import torch

from grmhd import CHANNELS
from grmhd.dataset import make_temporal_datasets
from grmhd.normalizer import GRMHDNormalizer, validate_stats_bundle


def make_file(path):
    rng = np.random.default_rng(7)
    values = rng.normal(size=(10, 8, 3, 4, 5)).astype(np.float32) * 0.01
    values[:, 2] = 0.0  # degenerate Bcc3
    values[:, 3] = np.exp(values[:, 3])
    values[:, 4] = 1e-3 * np.exp(values[:, 4])
    with h5py.File(path, "w") as handle:
        handle.create_dataset("snapshots", data=values)
        handle.create_dataset("times", data=np.arange(10, dtype=np.float64))
        handle.create_dataset("channels", data=np.asarray(CHANNELS, dtype=h5py.string_dtype()))
        coords = handle.create_group("coords")
        coords.create_dataset("r", data=np.geomspace(1, 10, 5))
        coords.create_dataset("theta", data=np.linspace(0, np.pi, 4))
        coords.create_dataset("phi", data=np.linspace(0, 2 * np.pi, 3, endpoint=False))


def test_fit_encode_decode_and_degenerate_channel(tmp_path):
    path = tmp_path / "normalizer.h5"
    make_file(path)
    training = make_temporal_datasets(path, train_fraction=0.6, val_fraction=0.2)["train"]
    with pytest.warns(RuntimeWarning, match="Bcc3"):
        normalizer = GRMHDNormalizer.fit(training, max_samples_per_channel=2000, seed=13)
    assert normalizer.scale[2] == pytest.approx(1e-6)
    raw = training[0]["x"]
    encoded = normalizer.encode(raw)
    decoded = normalizer.decode(encoded)
    assert torch.isfinite(encoded).all() and torch.isfinite(decoded).all()
    # This synthetic sample is near the robust centre and does not hit the soft clip.
    assert torch.allclose(decoded, raw, rtol=2e-4, atol=2e-6)
    assert torch.all(decoded[3:5] > 0)
    assert torch.count_nonzero(decoded[2]) == 0


def test_stats_roundtrip(tmp_path):
    path = tmp_path / "normalizer.h5"
    make_file(path)
    training = make_temporal_datasets(path, train_fraction=0.6, val_fraction=0.2)["train"]
    with pytest.warns(RuntimeWarning):
        normalizer = GRMHDNormalizer.fit(training, max_samples_per_channel=1000, seed=5)
    stats_path = tmp_path / "stats.npz"
    normalizer.save(stats_path)
    loaded = GRMHDNormalizer.load(
        stats_path,
        h5_path=path,
        expected_training_indices=training.owned_snapshot_indices,
    )
    np.testing.assert_array_equal(loaded.median, normalizer.median)
    np.testing.assert_array_equal(loaded.scale, normalizer.scale)
    np.testing.assert_array_equal(loaded.epsilon, normalizer.epsilon)
    sample = training[1]["x"].numpy()
    np.testing.assert_allclose(loaded.encode(sample), normalizer.encode(sample), rtol=0, atol=0)


def test_stats_reject_data_checksum_or_training_index_mismatch(tmp_path):
    path = tmp_path / "normalizer.h5"
    make_file(path)
    training = make_temporal_datasets(path, train_fraction=0.6, val_fraction=0.2)["train"]
    with pytest.warns(RuntimeWarning):
        normalizer = GRMHDNormalizer.fit(training, max_samples_per_channel=1000)
    stats_path = tmp_path / "stats.npz"
    normalizer.save(stats_path)
    with pytest.raises(ValueError, match="training indices mismatch"):
        GRMHDNormalizer.load(
            stats_path,
            h5_path=path,
            expected_training_indices=tuple(range(5)),
        )
    with h5py.File(path, "r+") as handle:
        handle["snapshots"][0, 0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="checksum mismatch"):
        GRMHDNormalizer.load(stats_path, h5_path=path)


def test_stats_bundle_rejects_window_mismatch(tmp_path):
    path = tmp_path / "normalizer.h5"
    make_file(path)
    training = make_temporal_datasets(path, train_fraction=0.6, val_fraction=0.2)["train"]
    with pytest.warns(RuntimeWarning):
        normalizer = GRMHDNormalizer.fit(training, max_samples_per_channel=1000)
    stats_dir = tmp_path / "stats"
    normalizer.save(stats_dir / "normalizer_stats.npz")
    common = {
        "source_hdf5_checksum": normalizer.source_hdf5_checksum,
        "training_indices": list(normalizer.training_indices),
    }
    bundle = {
        "window_name": "all111",
        "source_hdf5_checksum": normalizer.source_hdf5_checksum,
        "train_snapshot_indices": list(normalizer.training_indices),
        "normalizer": normalizer.as_dict(),
        "radial_baseline": {},
        "quantile_bounds": common,
        "residual_envelope": common,
        "shell_metadata": {},
        "velocity_roi": {},
    }
    (stats_dir / "priors.json").write_text(json.dumps(bundle), encoding="utf-8")
    report = validate_stats_bundle(
        stats_dir,
        path,
        expected_training_indices=training.owned_snapshot_indices,
        expected_window_name="all111",
    )
    assert report["status"] == "passed"
    with pytest.raises(ValueError, match="window mismatch"):
        validate_stats_bundle(
            stats_dir,
            path,
            expected_training_indices=training.owned_snapshot_indices,
            expected_window_name="late100",
        )


def test_inverse_clip_is_bounded_and_finite():
    normalizer = GRMHDNormalizer(np.zeros(8), np.ones(8))
    encoded = np.full((8, 2, 2, 2), 1000.0, dtype=np.float32)
    decoded = normalizer.decode(encoded)
    assert np.isfinite(decoded).all()
    assert np.all(decoded[3:5] > 0)


def test_radial_baseline_is_separate_recorded_and_invertible(tmp_path):
    path = tmp_path / "radial.h5"
    r = np.geomspace(1.0, 100.0, 8)
    values = np.zeros((8, 8, 2, 3, len(r)), dtype=np.float32)
    values[:, 0:3] = 1e-4
    values[:, 3] = r[None, None, None, :] ** -1.0
    values[:, 4] = 1e-2 * r[None, None, None, :] ** -2.0
    values[:, 5:8] = 0.01
    with h5py.File(path, "w") as handle:
        handle.create_dataset("snapshots", data=values)
        handle.create_dataset("times", data=np.arange(8, dtype=np.float64))
        handle.create_dataset("channels", data=np.asarray(CHANNELS, dtype=h5py.string_dtype()))
        coords = handle.create_group("coords")
        coords.create_dataset("r", data=r)
        coords.create_dataset("theta", data=np.linspace(0.1, 3.0, 3))
        coords.create_dataset("phi", data=np.linspace(0.1, 6.0, 2))
    training = make_temporal_datasets(path, train_fraction=0.5, val_fraction=0.25)["train"]
    with pytest.warns(RuntimeWarning):
        normalizer = GRMHDNormalizer.fit(
            training, max_samples_per_channel=1000, radial_baseline=True
        )
    baseline = normalizer.radial_baseline
    assert baseline["radial_coordinate"] == "log10(r)"
    assert baseline["fit_space"] == "raw positive-log before robust normalization"
    assert baseline["fit_separately"] is True
    assert baseline["channels"]["rho"]["slope"] == pytest.approx(-1.0, abs=1e-5)
    assert baseline["channels"]["press"]["slope"] == pytest.approx(-2.0, abs=1e-5)
    raw = training[0]["x"]
    decoded = normalizer.decode(normalizer.encode(raw))
    assert torch.allclose(decoded, raw, rtol=2e-4, atol=1e-7)
