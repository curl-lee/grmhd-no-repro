from __future__ import annotations

import numpy as np
import pytest
import torch

from grmhd.paper_velocity_roi import (
    canonical_oracle_velocity_roi_mask,
    normalized_velocity_roi_relative_error,
    paper_roi_ramp,
    raw_physical_velocity_roi_diagnostic_mask,
    roi_mask_diagnostics,
    stored_component_speed_proxy,
    top_fraction_mask,
)


def test_top20_exact_batch_independent_and_ties_deterministic():
    speed = torch.zeros(2, 2, 2, 5)
    speed[0] = torch.arange(20).reshape(2, 2, 5)
    speed[1] = 1
    mask = top_fraction_mask(speed)
    assert mask.flatten(start_dim=1).sum(dim=1).tolist() == [4, 4]
    assert mask[0].flatten().nonzero().flatten().tolist() == [16, 17, 18, 19]
    assert mask[1].flatten().nonzero().flatten().tolist() == [0, 1, 2, 3]


def test_canonical_mask_source_raw_diagnostic_and_proxy_semantics(paper_prior_factory):
    case = paper_prior_factory("roi_source")
    raw = torch.from_numpy(case["values"][0])
    canonical_mask, canonical = canonical_oracle_velocity_roi_mask(
        raw, case["preprocessor"]
    )
    explicit = top_fraction_mask(stored_component_speed_proxy(canonical))
    assert torch.equal(canonical_mask, explicit)
    raw_mask = raw_physical_velocity_roi_diagnostic_mask(raw)
    assert raw_mask.sum() == canonical_mask.sum()


def test_roi_relative_error_ramp_epsilon_gradient_and_overlap():
    target = torch.zeros(2, 8, 2, 2, 5)
    prediction = torch.ones_like(target, requires_grad=True)
    mask = top_fraction_mask(torch.ones(2, 2, 2, 5))
    loss = normalized_velocity_roi_relative_error(
        prediction, target, mask, denominator_epsilon=1e-6
    )
    assert torch.isfinite(loss) and loss > 0
    loss.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    assert paper_roi_ramp(0) == 0
    assert paper_roi_ramp(187.5) == 0.5
    assert paper_roi_ramp(375) == 1
    assert paper_roi_ramp(1000) == 1
    canonical = torch.zeros(1, 2, 2, 5, dtype=torch.bool)
    raw = canonical.clone()
    canonical.flatten()[0:4] = True
    raw.flatten()[2:6] = True
    clamp = torch.zeros_like(canonical)
    clamp.flatten()[0:2] = True
    audit = roi_mask_diagnostics(
        canonical, raw, clamp, shell_index=[0, 0, 1, 1, 1], snapshot_indices=[7]
    )
    assert audit["jaccard"] == pytest.approx(2 / 6)
    assert audit["precision_canonical_against_raw"] == pytest.approx(0.5)
    assert audit["canonical_clamp_overlap"] == pytest.approx(0.5)
    assert len(audit["shells"]) == 2
