from __future__ import annotations

import pytest
import torch

from grmhd.paper_dissipation import (
    PaperDissipativeReference,
    fit_dissipative_reference,
    global_state_norm,
)


def fit(case):
    return fit_dissipative_reference(
        case["path"],
        training_indices=case["train_indices"],
        preprocessor=case["preprocessor"],
        provenance=case["provenance"],
    )


def test_global_norm_axes_and_train_only_fit(paper_prior_factory):
    state = torch.ones(2, 8, 2, 3, 4)
    expected = torch.full((2,), (8 * 2 * 3 * 4) ** 0.5)
    torch.testing.assert_close(global_state_norm(state), expected)
    torch.testing.assert_close(global_state_norm(state[0]), expected[0])
    first = paper_prior_factory("diss_first")
    changed = paper_prior_factory("diss_changed", validation_multiplier=1e6)
    reference = fit(first)
    changed_reference = fit(changed)
    assert reference.rmax == pytest.approx(changed_reference.rmax)
    assert reference.rin == pytest.approx(1.05 * reference.rmax)
    assert reference.rout == pytest.approx(1.5 * reference.rin)
    assert tuple(index for index, _ in reference.snapshot_norms) == first["train_indices"]


def test_gate_target_positive_growth_and_gradient(paper_prior_factory):
    case = paper_prior_factory("diss_gate")
    reference = fit(case)
    low = torch.zeros(1, 8, 2, 2, 2)
    high = torch.full_like(low, reference.rin)
    low_prediction = torch.ones_like(low, requires_grad=True)
    high_prediction = (2 * high).requires_grad_()
    low_result = reference.apply(low, low_prediction)
    high_result = reference.apply(high, high_prediction)
    assert low_result.gate > high_result.gate
    torch.testing.assert_close(high_result.y_target, (reference.rin / reference.rout) * high)
    assert low_result.penalty >= 0 and high_result.penalty >= 0
    shrinking = reference.apply(high, torch.zeros_like(high))
    assert shrinking.penalty == 0
    high_result.penalty.backward()
    assert high_prediction.grad is not None
    assert torch.isfinite(high_prediction.grad).all()


def test_zero_prediction_has_finite_dissipative_subgradient(paper_prior_factory):
    case = paper_prior_factory("diss_zero_gradient")
    reference = fit(case)
    normalized_input = torch.zeros(1, 8, 2, 2, 2)
    normalized_prediction = torch.zeros_like(normalized_input, requires_grad=True)
    result = reference.apply(normalized_input, normalized_prediction)
    gradient = torch.autograd.grad(result.penalty, normalized_prediction)[0]
    assert result.penalty == 0
    assert torch.isfinite(gradient).all()


def test_dissipative_save_load_and_checksum_rejection(paper_prior_factory, tmp_path):
    case = paper_prior_factory("diss_save")
    reference = fit(case)
    path = tmp_path / "dissipative.json"
    reference.save(path)
    loaded = PaperDissipativeReference.load(
        path,
        source_hdf5_checksum=case["provenance"].source_hdf5_checksum,
        preprocessing_stats_checksum=case["provenance"].preprocessing_stats_checksum,
        training_indices=case["train_indices"],
        protocol_name=case["provenance"].protocol_name,
        thermal_channel="press",
    )
    assert loaded.rmax == reference.rmax
    with pytest.raises(ValueError, match="preprocessing checksum mismatch"):
        PaperDissipativeReference.load(
            path,
            source_hdf5_checksum=case["provenance"].source_hdf5_checksum,
            preprocessing_stats_checksum="wrong",
            training_indices=case["train_indices"],
            protocol_name=case["provenance"].protocol_name,
            thermal_channel="press",
        )
