from __future__ import annotations

import numpy as np
import pytest

from grmhd.paper_radial import (
    PAPER_REFERENCE_K,
    PaperRadialBaseline,
    evaluate_radial_baseline,
    fit_appendix_literal_press_proxy,
    fit_spherical_logr_channelwise,
    select_radial_mode,
)


def fit_candidates(case):
    kwargs = {
        "h5_path": str(case["path"]),
        "training_indices": case["train_indices"],
        "r": case["r"],
        "preprocessor": case["preprocessor"],
        "provenance": case["provenance"],
    }
    return fit_appendix_literal_press_proxy(**kwargs), fit_spherical_logr_channelwise(**kwargs)


def test_literal_and_spherical_candidates_are_explicit_and_finite(paper_prior_factory):
    case = paper_prior_factory("radial")
    literal, adapted = fit_candidates(case)
    assert literal.mode == "appendix_literal_press_proxy"
    assert literal.fit["intercept_enabled"] is False
    assert literal.fit["intercept"] == 0
    assert literal.fit["paper_reference_k"] == PAPER_REFERENCE_K
    assert literal.fit["paper_reference_for_comparison_only"] is True
    assert literal.fit["epsilon_U"] > 0
    assert literal.fit["channel_partition"]["paper_adaptation"] is True
    assert adapted.mode == "spherical_logr_channelwise"
    assert adapted.exact_paper_formula is False
    assert adapted.adaptation_level == "stronger_spherical_adaptation"
    for baseline in (literal, adapted):
        state = baseline.state((3, 3, 4), normalized=True)
        assert state.shape == (8, 3, 3, 4)
        assert np.isfinite(state.numpy()).all()
        assert torch_channels_zero(state)
        restored = PaperRadialBaseline.from_dict(baseline.as_dict())
        assert restored.mode == baseline.mode
        np.testing.assert_allclose(restored.rho_normalized, baseline.rho_normalized)


def torch_channels_zero(state):
    return bool((state[[0, 1, 2, 5, 6, 7]] == 0).all())


def test_validation_change_does_not_change_train_radial_coefficients(paper_prior_factory):
    first = paper_prior_factory("radial_first")
    changed = paper_prior_factory("radial_changed", validation_multiplier=1e5)
    first_literal, first_adapted = fit_candidates(first)
    changed_literal, changed_adapted = fit_candidates(changed)
    assert first_literal.fit["k"] == pytest.approx(changed_literal.fit["k"])
    assert first_literal.fit["epsilon_U"] == pytest.approx(changed_literal.fit["epsilon_U"])
    assert first_literal.fit["channel_partition"] == changed_literal.fit["channel_partition"]
    assert first_adapted.fit["channels"] == changed_adapted.fit["channels"]


def test_candidate_audit_and_literal_selection_rule(paper_prior_factory):
    case = paper_prior_factory("radial_audit")
    literal, adapted = fit_candidates(case)
    edges = np.geomspace(case["r"][0], case["r"][-1], 3)
    literal_audit = evaluate_radial_baseline(
        literal,
        str(case["path"]),
        snapshot_indices=case["train_indices"],
        preprocessor=case["preprocessor"],
        shell_edges=edges,
    )
    adapted_audit = evaluate_radial_baseline(
        adapted,
        str(case["path"]),
        snapshot_indices=case["train_indices"],
        preprocessor=case["preprocessor"],
        shell_edges=edges,
    )
    assert literal_audit["finite"] and adapted_audit["finite"]
    for name in ("rho", "press"):
        record = literal_audit["channels"][name]
        assert len(record["shell_residual_mean"]) == 2
        assert record["residual_q0.001"] < record["residual_q0.999"]
        assert 0 <= record["envelope_violation_fraction"] <= 1
    mode, reason = select_radial_mode(literal_audit, adapted_audit)
    assert mode == "appendix_literal_press_proxy"
    assert "not a selection criterion" in reason
    mode, _ = select_radial_mode({"finite": False}, adapted_audit)
    assert mode == "spherical_logr_channelwise"
