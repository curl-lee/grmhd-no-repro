import torch

from grmhd.losses import WeightedGRMHDLoss


def test_optional_roi_and_dissipative_terms_are_finite_and_differentiable():
    target = torch.randn(2, 8, 5, 5, 5)
    prediction = (target + 0.1 * torch.randn_like(target)).requires_grad_()
    loss = WeightedGRMHDLoss(
        lambda_velocity_roi=0.2,
        velocity_roi_quantile=0.8,
        lambda_dissipative=0.1,
    )
    components = loss.components(prediction, target, input_state=target)
    assert torch.isfinite(components["loss"])
    assert components["velocity_roi_mse"] > 0
    assert components["dissipative_excess_index_grid"] >= 0
    components["loss"].backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_dissipative_penalty_is_zero_when_prediction_is_smoother():
    target = torch.randn(1, 8, 5, 5, 5)
    prediction = torch.zeros_like(target)
    loss = WeightedGRMHDLoss(lambda_dissipative=1.0)
    components = loss.components(prediction, target)
    assert components["dissipative_excess_index_grid"] == 0
