import torch
from neuralop import H1Loss, LpLoss

from grmhd.paper_h1_diagnostics import h0_wrapper_parity, h1_diagnostic
from grmhd.paper_losses import PaperH1GradientLoss


def test_h0_matches_frozen_stage_e_wrapper():
    torch.manual_seed(7)
    prediction = torch.randn(2, 8, 6, 7, 8, dtype=torch.float64)
    target = torch.randn_like(prediction)
    wrapper, diagnostic = h0_wrapper_parity(prediction, target)
    torch.testing.assert_close(diagnostic, wrapper, rtol=1.0e-12, atol=1.0e-12)


def test_stage_e_wrapper_is_squared_absolute_h1_minus_l2():
    torch.manual_seed(11)
    prediction = torch.randn(1, 8, 5, 6, 7, dtype=torch.float64)
    target = torch.randn_like(prediction)
    expected = 8 * (
        H1Loss(d=3, reduction="mean").abs(
            prediction, target, take_root=False
        )
        - LpLoss(d=3, p=2, reduction="mean").abs(
            prediction, target, take_root=False
        )
    )
    actual = PaperH1GradientLoss()(prediction, target)
    torch.testing.assert_close(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_h2_explicit_normalized_axis_has_h0_parity():
    torch.manual_seed(13)
    error = torch.randn(2, 3, 6, 7, 8, dtype=torch.float64)
    h0 = h1_diagnostic(error, variant="H0_current_upstream")
    h2 = h1_diagnostic(error, variant="H2_normalized_axis")
    torch.testing.assert_close(h2.total, h0.total, rtol=0, atol=0)
    torch.testing.assert_close(h2.per_channel, h0.per_channel, rtol=0, atol=0)
    for direction in ("phi", "theta", "r"):
        torch.testing.assert_close(
            h2.per_direction[direction],
            h0.per_direction[direction],
            rtol=0,
            atol=0,
        )
