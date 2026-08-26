from __future__ import annotations

import torch
import torch.nn.functional as F
from neuralop import H1Loss, LpLoss

from grmhd.paper_losses import (
    PaperComponentFidelityLoss,
    PaperH1GradientLoss,
    PlainL2Loss,
)


def test_unit_base_matches_torch_mse_and_upstream_absolute_squared_l2():
    torch.manual_seed(7)
    prediction = torch.randn(2, 8, 3, 3, 4)
    target = torch.randn_like(prediction)
    plain = PlainL2Loss()(prediction, target)
    torch.testing.assert_close(plain, 8 * F.mse_loss(prediction, target))
    upstream = LpLoss(d=3, p=2, reduction="mean").abs(
        prediction, target, take_root=False
    )
    torch.testing.assert_close(plain, 8 * upstream)


def test_h1_wrapper_is_exact_upstream_squared_h1_minus_l2():
    torch.manual_seed(8)
    prediction = torch.randn(2, 8, 3, 3, 4)
    target = torch.randn_like(prediction)
    actual = PaperH1GradientLoss()(prediction, target)
    upstream_h1 = H1Loss(d=3, reduction="mean").abs(
        prediction, target, take_root=False
    )
    upstream_l2 = LpLoss(d=3, p=2, reduction="mean").abs(
        prediction, target, take_root=False
    )
    torch.testing.assert_close(actual, 8 * (upstream_h1 - upstream_l2))
    assert not torch.isclose(actual, H1Loss(d=3, reduction="mean")(prediction, target))


def test_full_loss_has_trainer_compatible_scalar_call_signature(paper_loss_factory):
    case = paper_loss_factory("upstream_signature")
    prediction = case["context"].normalized_target.clone().requires_grad_()
    sample = {"context": case["context"]}
    scalar = case["loss"](prediction, **sample)
    assert scalar.ndim == 0 and scalar.requires_grad
    weighted = PaperComponentFidelityLoss()(prediction, case["context"].normalized_target)
    assert weighted.ndim == 0
