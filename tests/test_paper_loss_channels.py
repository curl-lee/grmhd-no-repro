from __future__ import annotations

import pytest
import torch


@pytest.mark.parametrize(
    ("channel", "name", "family"),
    (
        (0, "Bcc1", "magnetic"),
        (2, "Bcc3", "magnetic"),
        (3, "rho", "positive"),
        (4, "press", "positive"),
        (5, "vel1", "velocity"),
        (7, "vel3", "velocity"),
    ),
)
def test_single_channel_perturbations_are_isolated(
    paper_loss_factory, channel, name, family
):
    case = paper_loss_factory(f"channel_{name}")
    context = case["context"]
    reference_prediction = context.normalized_target.clone()
    reference = case["loss"].components(reference_prediction, context=context)
    pattern = torch.linspace(-2.0, 2.0, reference_prediction[0, channel].numel()).reshape_as(
        reference_prediction[0, channel]
    )
    prediction = reference_prediction.clone()
    prediction[0, channel] += pattern
    result = case["loss"].components(prediction, context=context)
    channel_raw = result.diagnostics["base_channel_raw"]
    assert channel_raw[name] > 0
    assert all(channel_raw[other] == 0 for other in channel_raw if other != name)
    assert result.base_fidelity_weighted > reference.base_fidelity_weighted
    assert result.h1_weighted > reference.h1_weighted
    if family == "magnetic":
        torch.testing.assert_close(result.bounds_weighted, reference.bounds_weighted)
        torch.testing.assert_close(result.envelope_weighted, reference.envelope_weighted)
        torch.testing.assert_close(result.roi_weighted, reference.roi_weighted)
    elif family == "positive":
        torch.testing.assert_close(result.roi_weighted, reference.roi_weighted)
        assert (
            result.bounds_weighted != reference.bounds_weighted
            or result.envelope_weighted != reference.envelope_weighted
        )
    else:
        torch.testing.assert_close(result.bounds_weighted, reference.bounds_weighted)
        torch.testing.assert_close(result.envelope_weighted, reference.envelope_weighted)
        assert result.roi_weighted > reference.roi_weighted
    assert torch.equal(context.canonical_roi_mask, case["context"].canonical_roi_mask)


def test_shell_or_extra_channels_cannot_enter_output_loss(paper_loss_factory):
    case = paper_loss_factory("channel_shell_guard")
    prediction = torch.zeros(1, 16, 3, 3, 4)
    with pytest.raises(ValueError, match="B,8"):
        case["loss"].components(prediction, context=case["context"])


@pytest.mark.parametrize("channel", (0, 2, 3, 4, 5, 7))
def test_global_dissipative_term_can_respond_to_any_physics_channel(
    paper_loss_factory, channel
):
    case = paper_loss_factory(f"channel_dissipation_{channel}")
    prediction = torch.zeros_like(case["context"].normalized_target)
    prediction[:, channel] = 100.0
    result = case["dissipation"].apply(case["context"].normalized_input, prediction)
    assert result.penalty > 0
