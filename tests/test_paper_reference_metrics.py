from __future__ import annotations

import numpy as np
import pytest
import torch

from grmhd import CHANNELS
from grmhd.paper_metrics import (
    ClampMaskAccumulator,
    PaperMetricAccumulator,
    compare_clamp_masks,
    compute_paper_metrics,
)
from grmhd.paper_preprocessing import PaperPreprocessor
from grmhd.paper_references import (
    REFERENCE_SEMANTICS_VERSION,
    PaperReferenceStates,
    build_paper_reference_states,
    paper_reference_metadata,
)


def make_preprocessor() -> PaperPreprocessor:
    return PaperPreprocessor(
        epsilon=np.asarray([1e-3, 1e-3, 1e-3, 1e-4, 1e-5, 0, 0, 0]),
        median=np.zeros(8),
        scale=np.ones(8),
        training_indices=(11, 12),
        source_hdf5_checksum="synthetic-checksum",
        protocol_name="paper_reduced100_press_spherical_ks",
    )


def make_raw_target() -> np.ndarray:
    raw_physical_target = np.ones((8, 2), dtype=np.float64)
    raw_physical_target[0] = [-1e-3, 2e-3]
    raw_physical_target[1] = [-2e-3, 3e-3]
    raw_physical_target[2] = [-3e-3, 4e-3]
    raw_physical_target[3] = [1e-3, 2e-3]
    raw_physical_target[4] = [1e-4, 2e-4]
    raw_physical_target[5] = [100.0, 1.0]
    raw_physical_target[6] = [-2.0, 3.0]
    raw_physical_target[7] = [4.0, -5.0]
    return raw_physical_target


def test_reference_states_name_all_three_semantics_and_metadata() -> None:
    preprocessor = make_preprocessor()
    raw_physical_target = make_raw_target()
    normalized_target = preprocessor.encode(raw_physical_target)

    references = build_paper_reference_states(
        raw_physical_target=raw_physical_target,
        normalized_prediction=normalized_target,
        preprocessor=preprocessor,
    )

    assert references.raw_physical_target is raw_physical_target
    np.testing.assert_array_equal(references.normalized_target, normalized_target)
    np.testing.assert_array_equal(
        references.oracle_physical_target, preprocessor.decode(normalized_target)
    )
    np.testing.assert_array_equal(
        references.model_physical_prediction, references.oracle_physical_target
    )
    assert set(references.as_dict()) == {
        "raw_physical_target",
        "oracle_physical_target",
        "normalized_target",
        "normalized_prediction",
        "model_physical_prediction",
        "metadata",
    }
    metadata = references.metadata
    assert metadata["reference_semantics_version"] == REFERENCE_SEMANTICS_VERSION
    assert metadata["canonical_gamma"] == 6.0
    assert metadata["inverse_clamp_fraction"] == 0.99
    assert metadata["thermal_channel"] == "press"
    assert metadata["coordinate_components"]["not_cartesian"] is True
    assert tuple(metadata["channel_order"]) == CHANNELS


def test_oracle_prediction_has_zero_normalized_and_model_oracle_error() -> None:
    preprocessor = make_preprocessor()
    raw_physical_target = make_raw_target()
    normalized_target = preprocessor.encode(raw_physical_target)
    references = build_paper_reference_states(
        raw_physical_target=raw_physical_target,
        normalized_prediction=normalized_target,
        preprocessor=preprocessor,
    )

    result = compute_paper_metrics(references)

    assert result["channel_order"] == list(CHANNELS)
    assert result["metrics"]["E_norm"]["arithmetic_average"] == 0.0
    assert result["metrics"]["E_model_oracle"]["arithmetic_average"] == 0.0
    for channel in CHANNELS:
        assert result["metrics"]["E_model_raw"]["per_channel"][channel] == pytest.approx(
            result["metrics"]["E_oracle_raw"]["per_channel"][channel]
        )
    assert result["metrics"]["E_oracle_raw"]["per_channel"]["vel1"] > 0


def test_metric_reduction_and_squared_numerator_decomposition_are_exact() -> None:
    raw_physical_target = np.tile(np.asarray([3.0, 4.0]), (8, 1))
    oracle_physical_target = np.tile(np.asarray([4.0, 4.0]), (8, 1))
    model_physical_prediction = np.tile(np.asarray([5.0, 2.0]), (8, 1))
    normalized_target = raw_physical_target.copy()
    normalized_prediction = np.tile(np.asarray([3.0, 8.0]), (8, 1))
    references = PaperReferenceStates(
        raw_physical_target=raw_physical_target,
        oracle_physical_target=oracle_physical_target,
        normalized_target=normalized_target,
        normalized_prediction=normalized_prediction,
        model_physical_prediction=model_physical_prediction,
        metadata=paper_reference_metadata(make_preprocessor()),
    )

    result = compute_paper_metrics(references)

    for channel in CHANNELS:
        assert result["metrics"]["E_norm"]["per_channel"][channel] == pytest.approx(4 / 5)
        assert result["metrics"]["E_model_oracle"]["per_channel"][channel] == pytest.approx(
            np.sqrt(5 / 32)
        )
        assert result["metrics"]["E_model_raw"]["per_channel"][channel] == pytest.approx(
            np.sqrt(8) / 5
        )
        assert result["metrics"]["E_oracle_raw"]["per_channel"][channel] == pytest.approx(
            1 / 5
        )
        assert result["metrics"]["excess_absolute"]["per_channel"][channel] == pytest.approx(
            np.sqrt(8) / 5 - 1 / 5
        )
        decomposition = result["squared_error_numerator_decomposition"]["per_channel"][
            channel
        ]
        assert decomposition["model_raw_squared_error"] == pytest.approx(8)
        assert decomposition["model_oracle_squared_error"] == pytest.approx(5)
        assert decomposition["oracle_raw_squared_error"] == pytest.approx(1)
        assert decomposition["cross_term"] == pytest.approx(2)
        assert decomposition["reconstruction_residual"] == pytest.approx(0, abs=1e-14)
    assert "not additive" in result["squared_error_numerator_decomposition"]["warning"]


def test_accumulator_reduces_across_updates_before_taking_square_root() -> None:
    metadata = paper_reference_metadata(make_preprocessor())
    first_raw_physical_target = np.ones((8, 1))
    second_raw_physical_target = np.full((8, 1), 2.0)
    first_references = PaperReferenceStates(
        raw_physical_target=first_raw_physical_target,
        oracle_physical_target=first_raw_physical_target,
        normalized_target=first_raw_physical_target,
        normalized_prediction=np.full((8, 1), 2.0),
        model_physical_prediction=np.full((8, 1), 2.0),
        metadata=metadata,
    )
    second_references = PaperReferenceStates(
        raw_physical_target=second_raw_physical_target,
        oracle_physical_target=second_raw_physical_target,
        normalized_target=second_raw_physical_target,
        normalized_prediction=np.full((8, 1), 3.0),
        model_physical_prediction=np.full((8, 1), 3.0),
        metadata=metadata,
    )
    accumulator = PaperMetricAccumulator()
    accumulator.update(first_references)
    accumulator.update(second_references)

    result = accumulator.finalize()

    expected = np.sqrt((1.0**2 + 1.0**2) / (1.0**2 + 2.0**2))
    assert result["metrics"]["E_norm"]["arithmetic_average"] == pytest.approx(expected)
    assert result["update_count"] == 2
    assert set(result["value_count_per_channel"].values()) == {2}


def test_clamp_masks_keep_target_model_overlap_and_sign_separate() -> None:
    normalized_target = np.tile(np.asarray([6.0, -6.0, 6.0, 0.0, 0.0]), (8, 1))
    normalized_prediction = np.tile(np.asarray([6.0, 6.0, 0.0, -6.0, 0.0]), (8, 1))

    result = compare_clamp_masks(
        normalized_target=normalized_target,
        normalized_prediction=normalized_prediction,
        gamma=6.0,
        inverse_clamp_fraction=0.99,
    )

    assert result["target_and_model_masks_kept_separate"] is True
    for channel in CHANNELS:
        record = result["channels"][channel]
        assert record["target_clamp_fraction"] == pytest.approx(3 / 5)
        assert record["target_positive_fraction"] == pytest.approx(2 / 5)
        assert record["target_negative_fraction"] == pytest.approx(1 / 5)
        assert record["model_clamp_fraction"] == pytest.approx(3 / 5)
        assert record["model_positive_fraction"] == pytest.approx(2 / 5)
        assert record["model_negative_fraction"] == pytest.approx(1 / 5)
        assert record["intersection_fraction"] == pytest.approx(2 / 5)
        assert record["model_only_fraction"] == pytest.approx(1 / 5)
        assert record["target_only_fraction"] == pytest.approx(1 / 5)
        assert record["neither_fraction"] == pytest.approx(1 / 5)
        assert record["jaccard"] == pytest.approx(1 / 2)
        assert record["precision"] == pytest.approx(2 / 3)
        assert record["recall"] == pytest.approx(2 / 3)
        assert record["sign_agreement_within_intersection"] == pytest.approx(1 / 2)


def test_tensor_batch_channel_axis_and_shape_errors() -> None:
    preprocessor = make_preprocessor()
    raw_physical_target = torch.from_numpy(make_raw_target()).unsqueeze(0)
    normalized_target = preprocessor.encode(raw_physical_target, channel_axis=1)
    references = build_paper_reference_states(
        raw_physical_target=raw_physical_target,
        normalized_prediction=normalized_target,
        preprocessor=preprocessor,
        channel_axis=1,
    )
    result = compute_paper_metrics(references, channel_axis=1)
    assert result["metrics"]["E_norm"]["arithmetic_average"] == 0.0

    with pytest.raises(ValueError, match="shapes differ"):
        build_paper_reference_states(
            raw_physical_target=raw_physical_target,
            normalized_prediction=normalized_target[..., :1],
            preprocessor=preprocessor,
            channel_axis=1,
        )

    clamp_accumulator = ClampMaskAccumulator(gamma=6.0, inverse_clamp_fraction=0.99)
    with pytest.raises(ValueError, match="equal shape"):
        clamp_accumulator.update(
            normalized_target=normalized_target,
            normalized_prediction=normalized_target[..., :1],
            channel_axis=1,
        )
