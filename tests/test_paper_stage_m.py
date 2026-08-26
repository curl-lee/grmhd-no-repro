from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from grmhd.paper_stage_g_evaluation import artifact_diagnostics
from grmhd.paper_stage_m import (
    classify_floor_source,
    diagnostic_counterfactual,
    floor_limited_channel,
    forward_nonlinear,
    inverse_nonlinear,
    model_added_gate,
    radial_profile_vector,
    recovery_rate,
    require_target_oracle,
    trace_channel_transform,
    transport_metrics,
    validate_candidate_threshold_sources,
    variance_vector,
)


ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = {
    "kind": "signed_log",
    "epsilon": 1.0e-3,
    "median": -0.05,
    "scale": 0.12,
    "gamma": 6.0,
    "inverse_clamp_fraction": 0.99,
}


def field() -> np.ndarray:
    return np.linspace(-0.02, 0.03, 4 * 4 * 4).reshape(4, 4, 4)


def test_transform_stage_trace_identity():
    trace = trace_channel_transform(field(), **PARAMETERS)
    assert list(trace) == [
        "T0_raw_physical",
        "T1_forward_nonlinear",
        "T2_robust_normalized_unclipped",
        "T3_forward_softclip_canonical_normalized",
        "T4_decode_input_no_forward_hard_clamp",
        "T5_inverse_input_clamp_then_inverse_softclip",
        "T6_inverse_robust_normalization",
        "T7_inverse_nonlinear_canonical_oracle",
    ]
    assert all(value.shape == field().shape for value in trace.values())


def test_canonical_round_trip_parity():
    trace = trace_channel_transform(field(), **PARAMETERS)
    result = diagnostic_counterfactual(field(), label="CANONICAL_FULL", **PARAMETERS)
    assert result.finite
    np.testing.assert_allclose(result.values, trace["T7_inverse_nonlinear_canonical_oracle"])


def test_no_final_clamp_counterfactual():
    result = diagnostic_counterfactual(
        field(), label="NO_FINAL_INVERSE_CLAMP", **PARAMETERS
    )
    assert result.finite
    np.testing.assert_allclose(result.values, field(), rtol=1.0e-12, atol=1.0e-14)


def test_no_softclip_counterfactual():
    extreme = field() * 1.0e4
    result = diagnostic_counterfactual(
        extreme, label="NO_SOFTCLIP_COUNTERFACTUAL", **PARAMETERS
    )
    assert result.finite
    np.testing.assert_allclose(result.values, extreme, rtol=1.0e-12, atol=1.0e-12)


def test_normalizer_only_round_trip():
    result = diagnostic_counterfactual(
        field(), label="NORMALIZER_ONLY_ROUNDTRIP", **PARAMETERS
    )
    np.testing.assert_allclose(result.values, field(), rtol=1.0e-12, atol=1.0e-14)


def test_nonlinear_only_round_trip():
    result = diagnostic_counterfactual(
        field(), label="NONLINEAR_ONLY_ROUNDTRIP", **PARAMETERS
    )
    np.testing.assert_allclose(result.values, field(), rtol=1.0e-12, atol=1.0e-14)


def test_float64_reference_is_finite():
    result = diagnostic_counterfactual(field(), label="FLOAT64_REFERENCE", **PARAMETERS)
    assert result.finite and result.values.dtype == np.float64


def test_recovery_rate_denominator_handling():
    assert recovery_rate(1.0, 0.25, epsilon=1.0e-30) == 1.0
    assert recovery_rate(None, 0.25, epsilon=1.0e-30) is None


def test_single_component_dominated_classification():
    result = classify_floor_source(
        recovery_by_component={
            "forward_softclip": [0.9, 0.8, 0.7, 0.6],
            "final_inverse_clamp": [0.1, 0.2, 0.3, 0.4],
        },
        nonlinear_isolated_retentions=[1.0] * 4,
        normalizer_isolated_retentions=[1.0] * 4,
        float64_recoveries=[0.0] * 4,
    )
    assert result.choice == "A. FORWARD_SOFTCLIP_DOMINATED"


def test_multiple_components_classification():
    result = classify_floor_source(
        recovery_by_component={
            "forward_softclip": [0.9, 0.8, 0.1, 0.1],
            "final_inverse_clamp": [0.9, 0.8, 0.1, 0.1],
        },
        nonlinear_isolated_retentions=[1.0] * 4,
        normalizer_isolated_retentions=[1.0] * 4,
        float64_recoveries=[0.0] * 4,
    )
    assert result.choice == "E. MULTIPLE_COMPONENTS"


def test_not_isolated_classification():
    result = classify_floor_source(
        recovery_by_component={
            "forward_softclip": [0.1] * 4,
            "final_inverse_clamp": [0.1] * 4,
        },
        nonlinear_isolated_retentions=[1.0] * 4,
        normalizer_isolated_retentions=[1.0] * 4,
        float64_recoveries=[0.0] * 4,
    )
    assert result.choice == "F. NOT_ISOLATED"


def test_shell_variance_transport():
    values = np.arange(4 * 4 * 8, dtype=np.float64).reshape(4, 4, 8)
    result = variance_vector(values, np.arange(8))
    assert result.shape == (8,)


def test_radial_profile_transport():
    profile = radial_profile_vector(np.ones((3, 5, 7)))
    np.testing.assert_array_equal(profile, np.ones(7))


def test_zero_true_transport_is_explicitly_undefined():
    result = transport_metrics(
        np.ones(3), np.ones(3), np.ones(3), epsilon=1.0e-30, sign_zero_tolerance=1.0e-30
    )
    assert result["relative_error"] is None
    assert result["persistence_relative_skill"] is None


def test_persistence_transport_is_exactly_zero():
    result = transport_metrics(
        np.array([1.0, 2.0]),
        np.array([2.0, 1.0]),
        np.array([1.0, 2.0]),
        epsilon=1.0e-30,
        sign_zero_tolerance=1.0e-30,
    )
    assert result["delta_model"] == [0.0, 0.0]


def test_transport_cosine_undefined_handling():
    result = transport_metrics(
        np.ones(2), np.array([2.0, 3.0]), np.ones(2), epsilon=1.0e-30, sign_zero_tolerance=1.0e-30
    )
    assert result["transport_cosine"] is None
    assert result["transport_cosine_undefined"] is True


def test_floor_limited_channel_detection():
    assert floor_limited_channel(
        oracle_legacy_detector_triggered=False,
        median_variance_retention=0.4,
        median_shell_radial_retention=0.9,
        median_high_k_retention=0.9,
    )


def test_oracle_conditioned_degradation_gate():
    severe = {
        step: {
            "global_variance": step < 3,
            "shell_radial_variance": step < 3,
            "dynamic_span": False,
            "high_k_energy": False,
        }
        for step in (1, 2, 3)
    }
    assert model_added_gate(severe)["failed"] is True


def test_gate_thresholds_do_not_read_validation_model_outcomes():
    config = {
        "calibration_split": {
            "validation_outcomes_forbidden": True,
            "model_outcomes_forbidden": True,
        }
    }
    validate_candidate_threshold_sources(config)


def test_no_gt_cannot_request_target_oracle():
    with pytest.raises(ValueError, match="cannot request a target oracle"):
        require_target_oracle(ground_truth_available=False)


def test_legacy_detector_implementation_is_unchanged():
    source_hash = hashlib.sha256(inspect.getsource(artifact_diagnostics).encode()).hexdigest()
    assert source_hash == "3a9ea391b663df6b58a0c75942283658ee0214669d234e2c5a9594cc5f36e95d"


def test_stage_k_decision_cannot_be_overwritten():
    decision = json.loads(
        (ROOT / "outputs/paper_reduced100/stage_k/stage_k_decision.json").read_text()
    )
    assert decision["choice"] == "C"
    assert decision["decision"] == "TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE"


def test_stage_l_decision_cannot_be_overwritten():
    decision = json.loads(
        (ROOT / "outputs/paper_reduced100/stage_l/stage_l_decision.json").read_text()
    )
    assert decision["overall_decision"] == "3. MIXED_OVERALL"


def test_nonlinear_functions_are_inverse_on_moderate_values():
    original = field()
    transformed = forward_nonlinear(original, kind="signed_log", epsilon=1.0e-3)
    decoded = inverse_nonlinear(transformed, kind="signed_log", epsilon=1.0e-3)
    np.testing.assert_allclose(decoded, original, rtol=1.0e-12, atol=1.0e-14)


def test_stage_m_generated_decisions_obey_frozen_choices():
    decision = json.loads(
        (ROOT / "outputs/paper_reduced100/stage_m/stage_m_decision.json").read_text()
    )
    assert decision["transform_floor_decisions"] == {
        "Bcc2": "E. MULTIPLE_COMPONENTS",
        "Bcc3": "A. FORWARD_SOFTCLIP_DOMINATED",
        "vel3": "A. FORWARD_SOFTCLIP_DOMINATED",
    }
    assert decision["localno_transport_decision"] == (
        "4. MIXED_TRANSPORT_AND_PREPROCESSING_FAILURE"
    )
    assert decision["candidate_gate_decision"] == (
        "I. CANDIDATE_GATE_READY_FOR_FUTURE_RUNS"
    )


def test_stage_m_no_gt_output_has_no_target_oracle():
    payload = json.loads(
        (ROOT / "outputs/paper_reduced100/stage_m/no_gt_floor_corrected.json").read_text()
    )
    assert all(row["ground_truth_available"] is False for row in payload["rows"])
    assert all(row["target_oracle_available"] is False for row in payload["rows"])


def test_stage_m_replay_never_reclassifies_history():
    payload = json.loads(
        (ROOT / "outputs/paper_reduced100/stage_m/candidate_gate_replay.json").read_text()
    )
    assert all(row["historical_reclassification"] is False for row in payload["rows"])
    assert all(
        row["replay_classification"] == "counterfactual_gate_replay_only"
        for row in payload["rows"]
    )
