from __future__ import annotations

import torch


def test_all_weighted_components_and_total_have_finite_gradients_on_same_leaf(
    paper_loss_factory,
):
    case = paper_loss_factory("gradient_components")
    target = case["context"].normalized_target
    pattern = torch.linspace(-3.0, 3.0, target[0, 0].numel()).reshape_as(target[0, 0])
    prediction = torch.full_like(target, -10.0)
    prediction = (prediction + pattern.view(1, 1, *pattern.shape)).requires_grad_()
    result = case["loss"].components(prediction, context=case["context"])
    components = {
        "base": result.base_fidelity_weighted,
        "h1": result.h1_weighted,
        "roi": result.roi_weighted,
        "bounds": result.bounds_weighted,
        "envelope": result.envelope_weighted,
        "dissipation": result.dissipation_weighted,
        "total": result.total,
    }
    for name, value in components.items():
        assert torch.isfinite(value), name
        gradient = torch.autograd.grad(value, prediction, retain_graph=True)[0]
        assert torch.isfinite(gradient).all(), name
        assert torch.linalg.vector_norm(gradient) > 0, name


def test_truth_equals_prediction_has_finite_total_gradient(paper_loss_factory):
    case = paper_loss_factory("gradient_truth")
    prediction = case["context"].normalized_target.clone().requires_grad_()
    result = case["loss"].components(prediction, context=case["context"])
    gradient = torch.autograd.grad(result.total, prediction)[0]
    assert torch.isfinite(gradient).all()
