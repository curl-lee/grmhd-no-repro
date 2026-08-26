from __future__ import annotations

import pytest
import torch


def test_total_is_exact_sum_of_weighted_components(paper_loss_factory):
    case = paper_loss_factory("composite_sum", epoch=187.5)
    prediction = torch.zeros_like(case["context"].normalized_target, requires_grad=True)
    result = case["loss"].components(prediction, context=case["context"])
    expected = (
        result.base_fidelity_weighted
        + result.h1_weighted
        + result.roi_weighted
        + result.bounds_weighted
        + result.envelope_weighted
        + result.dissipation_weighted
    )
    torch.testing.assert_close(result.total, expected)
    assert result.roi_ramp == pytest.approx(0.5)
    assert result.metadata["h1_weight"] == 0.05
    assert result.metadata["roi_kappa"] == 8.0


def test_truth_equals_prediction_has_zero_fidelity_but_not_required_zero_total(
    paper_loss_factory,
):
    case = paper_loss_factory("composite_truth")
    prediction = case["context"].normalized_target.clone().requires_grad_()
    result = case["loss"].components(prediction, context=case["context"])
    assert result.base_fidelity_raw == 0
    assert result.h1_raw == 0
    assert result.roi_raw == 0
    torch.testing.assert_close(
        result.total,
        result.bounds_weighted
        + result.envelope_weighted
        + result.dissipation_weighted,
    )
    assert result.total >= 0


@pytest.mark.parametrize(
    ("epoch", "expected"),
    ((0, 0.0), (1, 1 / 375), (374, 374 / 375), (375, 1.0), (900, 1.0)),
)
def test_epoch_is_passed_to_roi_ramp(paper_loss_factory, epoch, expected):
    case = paper_loss_factory(f"composite_epoch_{epoch}", epoch=epoch)
    prediction = torch.zeros_like(case["context"].normalized_target)
    result = case["loss"].components(prediction, context=case["context"])
    assert result.roi_ramp == pytest.approx(expected)
    torch.testing.assert_close(result.roi_weighted, expected * 8 * result.roi_raw)
