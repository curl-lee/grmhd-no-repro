from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from grmhd.paper_losses import PlainL2Loss


def test_plain_l2_is_unit_weight_normalized_per_voxel_mse():
    target = torch.randn(2, 8, 3, 3, 4)
    prediction = torch.randn_like(target, requires_grad=True)
    loss = PlainL2Loss()
    actual = loss(prediction, target)
    torch.testing.assert_close(actual, 8 * F.mse_loss(prediction, target))
    assert loss.metadata["channel_weights"] == [1.0] * 8
    assert loss.metadata["enabled_priors"] == []


def test_plain_l2_rejects_enabled_or_unknown_prior_flags():
    prediction = torch.zeros(1, 8, 2, 2, 2)
    target = torch.ones_like(prediction)
    loss = PlainL2Loss()
    with pytest.raises(ValueError, match="rejects enabled"):
        loss(prediction, target, enabled_prior_flags={"roi": True})
    with pytest.raises(ValueError, match="Unknown"):
        loss(prediction, target, enabled_prior_flags={"round3_hybrid": False})
    assert loss(prediction, target, enabled_prior_flags={"roi": False}) == pytest.approx(8)


def test_plain_l2_api_does_not_accept_full_prior_context(paper_loss_factory):
    case = paper_loss_factory("plain_context")
    with pytest.raises(TypeError):
        PlainL2Loss()(
            case["context"].normalized_target,
            case["context"].normalized_target,
            context=case["context"],
        )
