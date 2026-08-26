from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from grmhd import CHANNELS
from grmhd.dataset import sha256_file
from grmhd.paper_preprocessing import (
    PAPER_GAMMA,
    PAPER_MAD_MULTIPLIER,
    PAPER_TRANSFORMS,
    PaperPreprocessor,
    write_preprocessing_audit,
)


TRAIN_INDICES = tuple(range(5))
PROTOCOL_NAME = "paper_reduced100_press_spherical_ks"


def make_values() -> np.ndarray:
    count, shape = 8, (2, 2, 2)
    values = np.empty((count, 8, *shape), dtype=np.float32)
    spatial = np.arange(1, np.prod(shape) + 1, dtype=np.float32).reshape(shape)
    for index in range(count):
        phase = index + spatial
        values[index, 0] = (-1.0) ** spatial * (0.2 + 0.03 * phase)
        values[index, 1] = -0.03 + 0.01 * phase
        values[index, 2] = 2.0 - 0.05 * phase
        values[index, 3] = 1e-4 * (1.0 + 0.02 * phase)
        values[index, 4] = 2e-7 * (1.0 + 0.03 * phase)
        values[index, 5] = -0.4 + 0.02 * phase
        values[index, 6] = 0.1 + 0.01 * phase
        values[index, 7] = -0.2 + 0.015 * phase
    return values


def make_file(path: Path, values: np.ndarray | None = None) -> None:
    values = make_values() if values is None else values
    with h5py.File(path, "w") as handle:
        handle.create_dataset("snapshots", data=values)
        handle.create_dataset("times", data=np.arange(len(values), dtype=np.float64))
        handle.create_dataset("channels", data=np.asarray(CHANNELS, dtype=h5py.string_dtype()))
        coords = handle.create_group("coords")
        coords.create_dataset("r", data=np.geomspace(1.0, 10.0, values.shape[-1]))
        coords.create_dataset("theta", data=np.linspace(0.2, 2.9, values.shape[-2]))
        coords.create_dataset("phi", data=np.linspace(0.0, 3.0, values.shape[-3]))


def fit(path: Path) -> PaperPreprocessor:
    return PaperPreprocessor.fit_hdf5(
        path,
        training_indices=TRAIN_INDICES,
        protocol_name=PROTOCOL_NAME,
        expected_source_hdf5_checksum=sha256_file(path),
    )


def transform_reference(values: np.ndarray, channel: int, epsilon: float) -> np.ndarray:
    kind = PAPER_TRANSFORMS[channel]
    if kind == "positive_log":
        return np.log10(values + epsilon)
    if kind == "signed_log":
        return np.sign(values) * np.log10(1.0 + np.abs(values) / epsilon)
    return values


def expected_epsilon(reference: float) -> float:
    return float(10 ** (np.floor(np.log10(reference)) - 2))


def test_epsilon_transform_median_mad_roundtrip_and_tensor_parity(tmp_path):
    path = tmp_path / "paper.h5"
    values = make_values()
    make_file(path, values)
    preprocessor = fit(path)
    train = values[list(TRAIN_INDICES)]

    for channel in range(3):
        assert preprocessor.epsilon[channel] == pytest.approx(
            expected_epsilon(float(np.max(np.abs(train[:, channel]))))
        )
    for channel in (3, 4):
        positive = train[:, channel][train[:, channel] > 0]
        assert preprocessor.epsilon[channel] == pytest.approx(
            expected_epsilon(float(np.min(positive)))
        )
    np.testing.assert_array_equal(preprocessor.epsilon[5:], np.zeros(3))

    for channel in range(8):
        transformed = transform_reference(
            train[:, channel].astype(np.float64), channel, preprocessor.epsilon[channel]
        )
        expected_median = np.median(transformed)
        expected_scale = max(
            PAPER_MAD_MULTIPLIER * np.median(np.abs(transformed - expected_median)),
            1e-6,
        )
        assert preprocessor.median[channel] == pytest.approx(expected_median)
        assert preprocessor.scale[channel] == pytest.approx(expected_scale)

    assert preprocessor.gamma == PAPER_GAMMA
    assert preprocessor.inverse_clamp_fraction == pytest.approx(0.99)
    raw = values[2]
    encoded = preprocessor.encode(raw)
    decoded = preprocessor.decode(encoded)
    assert np.isfinite(encoded).all() and np.isfinite(decoded).all()
    np.testing.assert_allclose(decoded, raw, rtol=2e-4, atol=1e-7)
    assert np.all(decoded[3:5] > 0)

    tensor = torch.from_numpy(raw)
    tensor_encoded = preprocessor.encode(tensor)
    tensor_decoded = preprocessor.decode(tensor_encoded)
    torch.testing.assert_close(tensor_encoded, torch.from_numpy(encoded), rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(tensor_decoded, tensor, rtol=2e-4, atol=1e-7)


def test_validation_values_do_not_affect_train_only_fit(tmp_path):
    first_path = tmp_path / "first.h5"
    second_path = tmp_path / "second.h5"
    values = make_values()
    changed_validation = values.copy()
    changed_validation[5:, 0:3] *= 1e20
    changed_validation[5:, 3:5] *= 1e20
    changed_validation[5:, 5:8] += 1e20
    make_file(first_path, values)
    make_file(second_path, changed_validation)
    first = fit(first_path)
    second = fit(second_path)
    assert first.source_hdf5_checksum != second.source_hdf5_checksum
    np.testing.assert_array_equal(first.epsilon, second.epsilon)
    np.testing.assert_array_equal(first.median, second.median)
    np.testing.assert_array_equal(first.scale, second.scale)
    assert first.training_indices == TRAIN_INDICES
    assert second.training_indices == TRAIN_INDICES


def test_save_load_and_wrong_checksum_indices_or_protocol_are_rejected(tmp_path):
    path = tmp_path / "paper.h5"
    make_file(path)
    preprocessor = fit(path)
    json_path = tmp_path / "normalizer.json"
    npz_path = tmp_path / "normalizer.npz"
    preprocessor.save(json_path, npz_path)
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    assert metadata["thermal_channel"] == "press"
    assert metadata["paper_adaptation"] is True
    assert metadata["eos_conversion"] == "disabled_unverified_gamma"
    assert metadata["training_indices"] == list(TRAIN_INDICES)
    assert metadata["decode_finite_guard_is_adaptation"] is True
    loaded = PaperPreprocessor.load(
        npz_path,
        h5_path=path,
        expected_training_indices=TRAIN_INDICES,
        expected_protocol_name=PROTOCOL_NAME,
    )
    np.testing.assert_array_equal(loaded.epsilon, preprocessor.epsilon)
    np.testing.assert_array_equal(loaded.median, preprocessor.median)
    np.testing.assert_array_equal(loaded.scale, preprocessor.scale)
    with pytest.raises(ValueError, match="checksum mismatch"):
        PaperPreprocessor.load(npz_path, expected_source_hdf5_checksum="wrong")
    with pytest.raises(ValueError, match="training indices mismatch"):
        PaperPreprocessor.load(npz_path, expected_training_indices=(0, 1, 2, 3))
    with pytest.raises(ValueError, match="protocol name mismatch"):
        PaperPreprocessor.load(npz_path, expected_protocol_name="wrong_protocol")


def test_inverse_clamp_is_bounded_finite_and_positive(tmp_path):
    path = tmp_path / "paper.h5"
    make_file(path)
    preprocessor = fit(path)
    extreme = np.full((8, 2, 2, 2), 1e6, dtype=np.float32)
    at_limit = np.full(
        (8, 2, 2, 2),
        preprocessor.inverse_clamp_fraction * preprocessor.gamma,
        dtype=np.float32,
    )
    decoded_extreme = preprocessor.decode(extreme)
    decoded_limit = preprocessor.decode(at_limit)
    assert np.isfinite(decoded_extreme).all()
    np.testing.assert_allclose(decoded_extreme, decoded_limit, rtol=0, atol=0)
    assert np.all(decoded_extreme[3:5] > 0)


def test_invalid_epsilon_uses_recorded_fallback_and_nonfinite_is_rejected(tmp_path):
    fallback_path = tmp_path / "fallback.h5"
    values = make_values()
    values[:5, 2] = 0.0
    make_file(fallback_path, values)
    with pytest.warns(RuntimeWarning):
        preprocessor = fit(fallback_path)
    assert preprocessor.epsilon[2] == pytest.approx(1e-30)
    warning_codes = {(item["code"], item["channel"]) for item in preprocessor.fit_warnings}
    assert ("invalid_epsilon_reference", "Bcc3") in warning_codes
    assert ("mad_scale_floor", "Bcc3") in warning_codes

    nonfinite_path = tmp_path / "nonfinite.h5"
    values = make_values()
    values[0, 0, 0, 0, 0] = np.nan
    make_file(nonfinite_path, values)
    with pytest.raises(FloatingPointError, match="NaN/Inf"):
        fit(nonfinite_path)

    negative_path = tmp_path / "negative.h5"
    values = make_values()
    values[0, 3, 0, 0, 0] = -1.0
    make_file(negative_path, values)
    with pytest.raises(ValueError, match="contains negatives"):
        fit(negative_path)


def test_audit_records_saturation_fractions_roundtrip_and_train_scope(tmp_path):
    path = tmp_path / "paper.h5"
    make_file(path)
    preprocessor = fit(path)
    audit = preprocessor.audit_hdf5(path)
    assert audit["status"] == "passed"
    assert audit["fit_snapshot_indices"] == list(TRAIN_INDICES)
    assert audit["validation_excluded_from_fit"] is True
    assert audit["thermal_channel"] == "press"
    assert audit["paper_adaptation"] is True
    assert audit["all_encoded_and_decoded_finite"] is True
    assert audit["rho_press_decoded_positive"] is True
    assert audit["decode_finite_guard_is_adaptation"] is True
    for channel in CHANNELS:
        record = audit["channels"][channel]
        assert 0 <= record["saturation_fraction_abs_ge_0.90_gamma"] <= 1
        assert 0 <= record["saturation_fraction_abs_ge_0.95_gamma"] <= 1
        assert 0 <= record["saturation_fraction_abs_ge_0.99_gamma"] <= 1
        assert 0 <= record["inverse_clamp_hit_fraction"] <= 1
        assert 0 <= record["decode_finite_guard_hit_fraction"] <= 1
        assert np.isfinite(record["roundtrip_max_abs_error"])

    json_path = tmp_path / "audit.json"
    markdown_path = tmp_path / "audit.md"
    write_preprocessing_audit(audit, json_path, markdown_path)
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["source_hdf5_checksum"] == sha256_file(path)
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "paper_reduced100 preprocessing audit" in markdown
    assert "Validation excluded from fit: `True`" in markdown
