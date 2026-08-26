from __future__ import annotations

import numpy as np
import pytest
import torch

from grmhd.paper_bounds import PaperPhysicalBounds, fit_paper_physical_bounds


def fit_bounds(case):
    return fit_paper_physical_bounds(
        case["path"],
        training_indices=case["train_indices"],
        preprocessor=case["preprocessor"],
        provenance=case["provenance"],
    )


def test_bounds_are_train_only_ordered_and_mapped_to_canonical_space(paper_prior_factory):
    first = paper_prior_factory("bounds_first")
    changed = paper_prior_factory("bounds_changed", validation_multiplier=1e6)
    bounds = fit_bounds(first)
    changed_bounds = fit_bounds(changed)
    assert bounds.raw_quantiles == changed_bounds.raw_quantiles
    assert bounds.physical_bounds == changed_bounds.physical_bounds
    for channel, name in ((3, "rho"), (4, "press")):
        lower, upper = bounds.physical_bounds[name]
        assert 0 < lower < upper
        transformed = np.log10(
            np.asarray([lower, upper]) + first["preprocessor"].epsilon[channel]
        )
        np.testing.assert_allclose(transformed, bounds.transformed_bounds[name])
        z = (transformed - first["preprocessor"].median[channel]) / first[
            "preprocessor"
        ].scale[channel]
        expected = first["preprocessor"].gamma * np.tanh(
            z / first["preprocessor"].gamma
        )
        np.testing.assert_allclose(expected, bounds.normalized_bounds[name])


def test_bound_clamp_changes_only_rho_press_and_preserves_unclamped(paper_prior_factory):
    case = paper_prior_factory("bounds_clamp")
    bounds = fit_bounds(case)
    prediction = torch.randn(2, 8, 2, 2, 2)
    original = prediction.clone()
    prediction[:, 3] = bounds.normalized_bounds["rho"][1] + 10
    prediction[:, 4] = bounds.normalized_bounds["press"][0] - 10
    result = bounds.clamp_normalized(prediction)
    torch.testing.assert_close(result.unclamped, prediction)
    assert torch.equal(result.clamped[:, [0, 1, 2, 5, 6, 7]], prediction[:, [0, 1, 2, 5, 6, 7]])
    assert torch.equal(result.mask[:, [0, 1, 2, 5, 6, 7]], torch.zeros_like(result.mask[:, [0, 1, 2, 5, 6, 7]]))
    assert result.hit_fraction["rho"] == 1
    assert result.hit_fraction["press"] == 1
    torch.testing.assert_close(original[:, [0, 1, 2, 5, 6, 7]], result.clamped[:, [0, 1, 2, 5, 6, 7]])


def test_bounds_save_load_and_provenance_rejection(paper_prior_factory, tmp_path):
    case = paper_prior_factory("bounds_save")
    bounds = fit_bounds(case)
    path = tmp_path / "bounds.json"
    bounds.save(path)
    loaded = PaperPhysicalBounds.load(
        path,
        source_hdf5_checksum=case["provenance"].source_hdf5_checksum,
        preprocessing_stats_checksum=case["provenance"].preprocessing_stats_checksum,
        training_indices=case["train_indices"],
        protocol_name=case["provenance"].protocol_name,
        thermal_channel="press",
    )
    assert loaded.physical_bounds == bounds.physical_bounds
    with pytest.raises(ValueError, match="HDF5 checksum mismatch"):
        PaperPhysicalBounds.load(
            path,
            source_hdf5_checksum="wrong",
            preprocessing_stats_checksum=case["provenance"].preprocessing_stats_checksum,
            training_indices=case["train_indices"],
            protocol_name=case["provenance"].protocol_name,
            thermal_channel="press",
        )
