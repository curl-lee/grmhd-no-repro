from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from grmhd.paper_stage_l_attribution import (
    all_spectrum_metrics,
    basic_field_metrics,
    classify_channel,
    decompose_metric,
    overall_attribution,
    radial_profile,
    require_oracle,
    safe_retention,
    shell_metrics,
    spectrum_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
CORE = {"global_variance", "shell_radial_variance", "high_k_spectral_energy"}


def evidence(enabled: int) -> dict[str, bool]:
    return {name: index < enabled for index, name in enumerate(sorted(CORE))}


def test_variance_retention_toy_case():
    result = decompose_metric(4.0, 2.0, 1.0, epsilon=1.0e-30)
    assert result["preprocessing_retention"] == 0.5
    assert result["model_retention"] == 0.5
    assert result["total_retention"] == 0.25


def test_zero_denominator_returns_null():
    ratio, undefined = safe_retention(1.0, 0.0, epsilon=1.0e-30)
    assert ratio is None
    assert undefined is True


def test_shell_aggregation_does_not_mix_batch_or_channel():
    field = np.arange(64, dtype=np.float64).reshape(4, 4, 4)
    rows = shell_metrics(field, np.array([0, 0, 1, 1]), n_shells=2)
    assert [row["voxel_count"] for row in rows] == [32, 32]
    with pytest.raises(ValueError, match="exactly"):
        shell_metrics(field[None, None], np.array([0, 0, 1, 1]), n_shells=2)


def test_radial_profile_shape():
    result = radial_profile(np.ones((3, 5, 7)))
    assert result["length"] == 7
    assert np.asarray(result["profile"]).shape == (7,)


def test_fft_excludes_batch_and_channel_axes():
    rows = all_spectrum_metrics(np.ones((4, 5, 6)))
    assert len(rows) == 8
    with pytest.raises(ValueError, match="exactly"):
        spectrum_metrics(np.ones((1, 1, 4, 5, 6)), axis="combined", demean=False)


def test_parseval_consistency():
    rng = np.random.default_rng(42)
    result = spectrum_metrics(rng.normal(size=(7, 8, 9)), axis="combined", demean=True)
    assert result["parseval_relative_error"] < 1.0e-12


def test_spectral_bands_are_mutually_exclusive():
    result = spectrum_metrics(np.ones((8, 8, 8)), axis="combined", demean=False)
    assert (
        result["bin_count_low"]
        + result["bin_count_mid"]
        + result["bin_count_high"]
        == 8**3
    )


def test_raw_oracle_model_decomposition_identity():
    result = decompose_metric(8.0, 2.0, 1.0, epsilon=1.0e-30)
    assert result["total_retention"] == pytest.approx(
        result["preprocessing_retention"] * result["model_retention"]
    )


def test_no_gt_code_cannot_request_oracle():
    with pytest.raises(ValueError, match="cannot request an oracle"):
        require_oracle(ground_truth_available=False)


def test_frozen_detector_reproduction_remains_passed():
    payload = json.loads(
        (
            ROOT
            / "outputs/paper_reduced100/stage_l/detector_reproduction.json"
        ).read_text(encoding="utf-8")
    )
    assert payload["status"] == "passed"
    assert all(row["exact_all_flag_match"] for row in payload["steps"].values())


def test_attribution_a_toy_case():
    result = classify_channel(
        preprocessing_severe=evidence(2),
        model_severe=evidence(1),
        oracle_detector_triggered=True,
        localno_detector_triggered=True,
    )
    assert result.choice == "A. PREPROCESSING_DOMINATED"


def test_attribution_b_toy_case():
    result = classify_channel(
        preprocessing_severe=evidence(1),
        model_severe=evidence(2),
        oracle_detector_triggered=False,
        localno_detector_triggered=True,
    )
    assert result.choice == "B. MODEL_DOMINATED"


def test_attribution_c_toy_case():
    result = classify_channel(
        preprocessing_severe=evidence(2),
        model_severe=evidence(2),
        oracle_detector_triggered=True,
        localno_detector_triggered=True,
    )
    assert result.choice == "C. MIXED_PREPROCESSING_AND_MODEL"


def test_attribution_d_toy_case():
    result = classify_channel(
        preprocessing_severe=evidence(1),
        model_severe=evidence(1),
        oracle_detector_triggered=False,
        localno_detector_triggered=True,
    )
    assert result.choice == "D. DETECTOR_SPECIFIC_MISMATCH"


def test_attribution_e_failure_case():
    result = classify_channel(
        preprocessing_severe=evidence(3),
        model_severe=evidence(0),
        oracle_detector_triggered=False,
        localno_detector_triggered=False,
        engineering_failure=True,
    )
    assert result.choice == "E. INCONCLUSIVE_OR_ENGINEERING_FAILURE"


def test_stage_k_decision_cannot_be_overwritten():
    decision = json.loads(
        (
            ROOT / "outputs/paper_reduced100/stage_k/stage_k_decision.json"
        ).read_text(encoding="utf-8")
    )
    assert decision["choice"] == "C"
    assert decision["decision"] == "TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE"
    assert overall_attribution(
        {
            "Bcc3": "C. MIXED_PREPROCESSING_AND_MODEL",
            "vel3": "C. MIXED_PREPROCESSING_AND_MODEL",
        }
    ) == "3. MIXED_OVERALL"
