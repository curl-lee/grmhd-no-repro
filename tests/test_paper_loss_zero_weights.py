from __future__ import annotations

import pytest
import torch

from grmhd.paper_losses import PaperCompositeLoss


def make_loss(case, **overrides):
    return PaperCompositeLoss(
        bounds=case["bounds"],
        envelope=case["envelope"],
        roi=case["roi"],
        dissipation=case["dissipation"],
        radial_metadata=case["radial"].metadata,
        **overrides,
    )


@pytest.mark.parametrize(
    ("overrides", "raw_name", "weighted_name"),
    (
        ({"h1_weight": 0.0}, "h1_raw", "h1_weighted"),
        ({"roi_kappa": 0.0}, "roi_raw", "roi_weighted"),
        (
            {"bounds_rho_low_weight": 0.0, "bounds_press_low_weight": 0.0},
            "bounds_rho_raw",
            "bounds_weighted",
        ),
        (
            {"envelope_rho_weight": 0.0, "envelope_press_weight": 0.0},
            "envelope_rho_raw",
            "envelope_weighted",
        ),
        ({"dissipation_alpha": 0.0}, "dissipation_raw", "dissipation_weighted"),
    ),
)
def test_zero_weight_reports_raw_but_has_exact_zero_value_and_gradient(
    paper_loss_factory, overrides, raw_name, weighted_name
):
    case = paper_loss_factory(f"zero_{weighted_name}")
    prediction = torch.full_like(
        case["context"].normalized_target, -10.0, requires_grad=True
    )
    loss = make_loss(case, **overrides)
    result = loss.components(prediction, context=case["context"])
    raw = getattr(result, raw_name)
    weighted = getattr(result, weighted_name)
    assert raw > 0
    assert weighted == 0
    gradient = torch.autograd.grad(weighted, prediction, retain_graph=True)[0]
    assert torch.equal(gradient, torch.zeros_like(gradient))
    assert torch.isfinite(result.total)


def test_zeroing_one_component_does_not_change_other_components(paper_loss_factory):
    case = paper_loss_factory("zero_isolation")
    prediction = torch.full_like(case["context"].normalized_target, -10.0)
    default = case["loss"].components(prediction, context=case["context"])
    no_roi = make_loss(case, roi_kappa=0.0).components(
        prediction, context=case["context"]
    )
    torch.testing.assert_close(no_roi.base_fidelity_weighted, default.base_fidelity_weighted)
    torch.testing.assert_close(no_roi.h1_weighted, default.h1_weighted)
    torch.testing.assert_close(no_roi.bounds_weighted, default.bounds_weighted)
    torch.testing.assert_close(no_roi.envelope_weighted, default.envelope_weighted)
    torch.testing.assert_close(no_roi.dissipation_weighted, default.dissipation_weighted)
    torch.testing.assert_close(default.total - no_roi.total, default.roi_weighted)
