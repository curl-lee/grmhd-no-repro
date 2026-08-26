from __future__ import annotations

import pytest
import torch

from grmhd.paper_priors import PaperResidualEnvelope
from grmhd.paper_radial import fit_appendix_literal_press_proxy


def make(case):
    radial = fit_appendix_literal_press_proxy(
        str(case["path"]),
        training_indices=case["train_indices"],
        r=case["r"],
        preprocessor=case["preprocessor"],
        provenance=case["provenance"],
    )
    envelope = PaperResidualEnvelope(case["provenance"], radial.mode)
    baseline = radial.state((3, 3, 4), normalized=True, batch_size=2)
    return radial, envelope, baseline


def test_envelope_inside_zero_outside_square_and_only_positive_channels(paper_prior_factory):
    case = paper_prior_factory("envelope")
    radial, envelope, baseline = make(case)
    inside = baseline.clone().requires_grad_()
    assert envelope.penalty(inside, baseline, radial_metadata=radial.metadata) == 0
    outside = baseline.clone()
    outside[:, 3] += 2.5
    outside[:, 4] -= 3.5
    components = envelope.penalty_components(outside, baseline, radial_metadata=radial.metadata)
    assert components["rho"] == pytest.approx(0.05 * (2.5 - 1.5) ** 2)
    assert components["press"] == pytest.approx(0.05 * (3.5 - 1.5) ** 2)
    b_velocity_only = baseline.clone()
    b_velocity_only[:, [0, 1, 2, 5, 6, 7]] += 100
    assert envelope.penalty(b_velocity_only, baseline, radial_metadata=radial.metadata) == 0


def test_envelope_broadcasting_gradient_and_metadata_guards(paper_prior_factory):
    case = paper_prior_factory("envelope_gradient")
    radial, envelope, baseline = make(case)
    prediction = (baseline + 2).requires_grad_()
    loss = envelope.penalty(prediction, baseline, radial_metadata=radial.metadata)
    loss.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    fractions = envelope.violation_fractions(prediction.detach(), baseline)
    assert fractions == {"rho": 1.0, "press": 1.0}
    wrong = dict(radial.metadata)
    wrong["protocol_name"] = "wrong"
    with pytest.raises(ValueError, match="protocol mismatch"):
        envelope.penalty(prediction, baseline, radial_metadata=wrong)
