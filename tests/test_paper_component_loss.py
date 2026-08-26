from __future__ import annotations

import pytest
import torch

from grmhd.paper_losses import PaperComponentFidelityLoss, PaperH1GradientLoss


def test_component_fidelity_matches_hand_calculation_and_weights_once():
    target = torch.zeros(2, 8, 2, 2, 2)
    prediction = target.clone()
    prediction[:, 0] = 2.0
    prediction[:, 3] = 3.0
    components = PaperComponentFidelityLoss().components(prediction, target)
    assert components["channel_raw"][0] == pytest.approx(4.0)
    assert components["channel_raw"][3] == pytest.approx(9.0)
    assert components["raw"] == pytest.approx(13.0)
    assert components["weighted"] == pytest.approx(1.2 * 4.0 + 9.0)


def test_component_fidelity_is_spatial_mean_and_batch_mean():
    target = torch.zeros(1, 8, 2, 2, 2)
    prediction = torch.zeros_like(target)
    prediction[:, 5] = 4.0
    loss = PaperComponentFidelityLoss()(prediction, target)
    duplicate = PaperComponentFidelityLoss()(
        prediction.repeat(2, 1, 1, 1, 1), target.repeat(2, 1, 1, 1, 1)
    )
    assert loss == pytest.approx(16.0)
    assert duplicate == pytest.approx(loss)


def test_h1_is_gradient_only_and_finite():
    target = torch.zeros(1, 8, 3, 3, 4)
    constant = torch.ones_like(target, requires_grad=True)
    h1 = PaperH1GradientLoss()
    torch.testing.assert_close(h1(constant, target), torch.tensor(0.0))
    varying = torch.zeros_like(target)
    varying[:, 0, :, :, :] = torch.arange(4).view(1, 1, 1, 4)
    varying.requires_grad_()
    value = h1(varying, target)
    assert value > 0 and torch.isfinite(value)
    value.backward()
    assert varying.grad is not None and torch.isfinite(varying.grad).all()
