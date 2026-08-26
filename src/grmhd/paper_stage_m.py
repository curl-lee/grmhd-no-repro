"""Pure post-hoc diagnostics for the Stage M transform and transport audit."""

from __future__ import annotations

from dataclasses import dataclass
import copy
from typing import Any, Mapping, Sequence

import numpy as np


def _finite_field(values: np.ndarray) -> np.ndarray:
    field = np.asarray(values, dtype=np.float64)
    if field.ndim != 3:
        raise ValueError("Stage M field must have exactly (phi,theta,r) axes")
    if not np.isfinite(field).all():
        raise FloatingPointError("Stage M field contains NaN/Inf")
    return field


def forward_nonlinear(values: np.ndarray, *, kind: str, epsilon: float) -> np.ndarray:
    """Apply the exact per-channel paper nonlinearity in float64."""

    values = _finite_field(values)
    if kind == "positive_log":
        if np.any(values < 0):
            raise ValueError("Positive-log Stage M field contains negative values")
        return np.log10(values + float(epsilon))
    if kind == "signed_log":
        return np.sign(values) * np.log10(1.0 + np.abs(values) / float(epsilon))
    if kind == "linear":
        return values.copy()
    raise ValueError(f"Unknown Stage M channel transform: {kind}")


def inverse_nonlinear(
    values: np.ndarray,
    *,
    kind: str,
    epsilon: float,
    physical_limit: float | None = None,
) -> np.ndarray:
    """Invert a channel nonlinearity, optionally with the canonical dtype guard."""

    transformed = _finite_field(values)
    limit = float(np.finfo(np.float32).max if physical_limit is None else physical_limit)
    if kind == "positive_log":
        exponent_limit = np.log10(limit + float(epsilon))
        decoded = np.power(10.0, np.minimum(transformed, exponent_limit)) - float(epsilon)
        return np.maximum(decoded, np.finfo(np.float32).tiny)
    if kind == "signed_log":
        exponent_limit = np.log10(limit) - np.log10(float(epsilon))
        magnitude = np.minimum(np.abs(transformed), exponent_limit)
        return np.sign(transformed) * float(epsilon) * (np.power(10.0, magnitude) - 1.0)
    if kind == "linear":
        return np.clip(transformed, -limit, limit)
    raise ValueError(f"Unknown Stage M channel transform: {kind}")


def trace_channel_transform(
    raw: np.ndarray,
    *,
    kind: str,
    epsilon: float,
    median: float,
    scale: float,
    gamma: float,
    inverse_clamp_fraction: float,
) -> dict[str, np.ndarray]:
    """Expose the real canonical stage order without changing the processor."""

    raw = _finite_field(raw)
    nonlinear = forward_nonlinear(raw, kind=kind, epsilon=epsilon)
    normalized = (nonlinear - float(median)) / float(scale)
    softclipped = float(gamma) * np.tanh(normalized / float(gamma))
    decode_input = softclipped.copy()
    clamped = np.clip(
        decode_input,
        -float(gamma) * float(inverse_clamp_fraction),
        float(gamma) * float(inverse_clamp_fraction),
    )
    inverse_softclip = float(gamma) * np.arctanh(clamped / float(gamma))
    denormalized = inverse_softclip * float(scale) + float(median)
    decoded = inverse_nonlinear(
        denormalized,
        kind=kind,
        epsilon=epsilon,
        physical_limit=np.finfo(np.float32).max,
    )
    return {
        "T0_raw_physical": raw.copy(),
        "T1_forward_nonlinear": nonlinear,
        "T2_robust_normalized_unclipped": normalized,
        "T3_forward_softclip_canonical_normalized": softclipped,
        "T4_decode_input_no_forward_hard_clamp": decode_input,
        "T5_inverse_input_clamp_then_inverse_softclip": inverse_softclip,
        "T6_inverse_robust_normalization": denormalized,
        "T7_inverse_nonlinear_canonical_oracle": decoded,
    }


@dataclass(frozen=True)
class CounterfactualResult:
    label: str
    values: np.ndarray | None
    finite: bool
    nonfinite_count: int
    note: str


def diagnostic_counterfactual(
    raw: np.ndarray,
    *,
    label: str,
    kind: str,
    epsilon: float,
    median: float,
    scale: float,
    gamma: float,
    inverse_clamp_fraction: float,
) -> CounterfactualResult:
    """Evaluate a named diagnostic transform without mutating canonical artifacts."""

    raw = _finite_field(raw)
    nonlinear = forward_nonlinear(raw, kind=kind, epsilon=epsilon)
    normalized = (nonlinear - float(median)) / float(scale)
    canonical_encoded = float(gamma) * np.tanh(normalized / float(gamma))
    note = "diagnostic_counterfactual"

    if label == "CANONICAL_FULL":
        encoded = np.clip(
            canonical_encoded,
            -float(gamma) * float(inverse_clamp_fraction),
            float(gamma) * float(inverse_clamp_fraction),
        )
        inverse_z = float(gamma) * np.arctanh(encoded / float(gamma))
        representation = inverse_z * float(scale) + float(median)
    elif label == "NO_FINAL_INVERSE_CLAMP":
        with np.errstate(divide="ignore", invalid="ignore"):
            inverse_z = float(gamma) * np.arctanh(canonical_encoded / float(gamma))
        representation = inverse_z * float(scale) + float(median)
        note += "; inverse clamp bypassed and nonfinite values are never replaced"
    elif label in {"NO_SOFTCLIP_COUNTERFACTUAL", "NORMALIZER_ONLY_ROUNDTRIP"}:
        representation = normalized * float(scale) + float(median)
    elif label == "NONLINEAR_ONLY_ROUNDTRIP":
        representation = nonlinear
    elif label == "FLOAT64_REFERENCE":
        encoded = np.clip(
            canonical_encoded,
            -float(gamma) * float(inverse_clamp_fraction),
            float(gamma) * float(inverse_clamp_fraction),
        )
        inverse_z = float(gamma) * np.arctanh(encoded / float(gamma))
        representation = inverse_z * float(scale) + float(median)
        note += "; full canonical calculation in CPU float64"
    else:
        raise ValueError(f"Unknown Stage M counterfactual: {label}")

    finite_mask = np.isfinite(representation)
    nonfinite_count = int(representation.size - np.count_nonzero(finite_mask))
    if nonfinite_count:
        return CounterfactualResult(label, None, False, nonfinite_count, note)
    decoded = inverse_nonlinear(
        representation,
        kind=kind,
        epsilon=epsilon,
        physical_limit=np.finfo(np.float32).max,
    )
    return CounterfactualResult(label, decoded, True, 0, note)


def safe_ratio(
    numerator: float, denominator: float, *, epsilon: float
) -> tuple[float | None, bool]:
    numerator = float(numerator)
    denominator = float(denominator)
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        raise FloatingPointError("Stage M ratio inputs must be finite")
    if abs(denominator) <= float(epsilon):
        return None, True
    return numerator / denominator, False


def recovery_rate(
    counterfactual_retention: float | None,
    canonical_retention: float | None,
    *,
    epsilon: float,
) -> float | None:
    if counterfactual_retention is None or canonical_retention is None:
        return None
    denominator = max(1.0 - float(canonical_retention), float(epsilon))
    return (float(counterfactual_retention) - float(canonical_retention)) / denominator


@dataclass(frozen=True)
class FloorSourceDecision:
    choice: str
    recovered_metric_counts: Mapping[str, int]
    reason: str


def classify_floor_source(
    *,
    recovery_by_component: Mapping[str, Sequence[float | None]],
    nonlinear_isolated_retentions: Sequence[float | None],
    normalizer_isolated_retentions: Sequence[float | None],
    float64_recoveries: Sequence[float | None],
    recovery_threshold: float = 0.5,
    severe_threshold: float = 0.5,
) -> FloorSourceDecision:
    """Apply the predeclared A--F transform-source decision rules."""

    expected = {"forward_softclip", "final_inverse_clamp"}
    if set(recovery_by_component) != expected:
        raise ValueError("Stage M source classification requires softclip and clamp recovery")
    lengths = {
        len(values)
        for values in (
            *recovery_by_component.values(),
            nonlinear_isolated_retentions,
            normalizer_isolated_retentions,
            float64_recoveries,
        )
    }
    if lengths != {4}:
        raise ValueError("Stage M floor source requires exactly four core metrics")
    all_values = [
        value
        for values in (
            *recovery_by_component.values(),
            nonlinear_isolated_retentions,
            normalizer_isolated_retentions,
            float64_recoveries,
        )
        for value in values
    ]
    if sum(value is None for value in all_values) > len(all_values) // 2:
        return FloorSourceDecision(
            "F. NOT_ISOLATED", {}, "majority of source evidence is undefined"
        )

    counts = {
        name: sum(value is not None and value > recovery_threshold for value in values)
        for name, values in recovery_by_component.items()
    }
    counts["nonlinear_transform"] = sum(
        value is not None and value < severe_threshold
        for value in nonlinear_isolated_retentions
    )
    counts["normalizer"] = max(
        sum(
            value is not None and value < severe_threshold
            for value in normalizer_isolated_retentions
        ),
        sum(value is not None and value > recovery_threshold for value in float64_recoveries),
    )
    strong = [name for name, count in counts.items() if count >= 3]
    if len(strong) == 1 and all(
        count < 3 for name, count in counts.items() if name != strong[0]
    ):
        labels = {
            "forward_softclip": "A. FORWARD_SOFTCLIP_DOMINATED",
            "final_inverse_clamp": "B. FINAL_INVERSE_CLAMP_DOMINATED",
            "nonlinear_transform": "C. NONLINEAR_TRANSFORM_DOMINATED",
            "normalizer": "D. NORMALIZER_NUMERICAL_OR_RANGE_DOMINATED",
        }
        return FloorSourceDecision(
            labels[strong[0]], counts, f"{strong[0]} uniquely explains at least 3/4 metrics"
        )
    if sum(count >= 2 for count in counts.values()) >= 2:
        return FloorSourceDecision(
            "E. MULTIPLE_COMPONENTS",
            counts,
            "at least two components explain at least 2/4 core metrics",
        )
    return FloorSourceDecision(
        "F. NOT_ISOLATED", counts, "frozen recovery rules do not isolate a stable source"
    )


def variance_vector(values: np.ndarray, shell_index: np.ndarray) -> np.ndarray:
    field = _finite_field(values)
    shell_index = np.asarray(shell_index, dtype=np.int64)
    if shell_index.shape != (field.shape[-1],):
        raise ValueError("Stage M shell index must align with radial cells")
    output = []
    for shell in range(8):
        selected = field[..., shell_index == shell]
        if selected.size == 0:
            raise ValueError(f"Frozen Stage M shell {shell + 1} is empty")
        output.append(float(np.var(selected, ddof=0)))
    return np.asarray(output, dtype=np.float64)


def radial_profile_vector(values: np.ndarray) -> np.ndarray:
    return np.mean(_finite_field(values), axis=(0, 1), dtype=np.float64)


def _signed(values: np.ndarray, tolerance: float) -> np.ndarray:
    output = np.zeros(values.shape, dtype=np.int8)
    output[values > tolerance] = 1
    output[values < -tolerance] = -1
    return output


def transport_metrics(
    input_values: np.ndarray,
    target_values: np.ndarray,
    model_values: np.ndarray,
    *,
    epsilon: float,
    sign_zero_tolerance: float,
) -> dict[str, Any]:
    """Compare a model transport vector with oracle and persistence transports."""

    input_values = np.asarray(input_values, dtype=np.float64)
    target_values = np.asarray(target_values, dtype=np.float64)
    model_values = np.asarray(model_values, dtype=np.float64)
    if input_values.shape != target_values.shape or input_values.shape != model_values.shape:
        raise ValueError("Stage M transport vectors must have identical shapes")
    if input_values.ndim != 1 or not all(
        np.isfinite(values).all() for values in (input_values, target_values, model_values)
    ):
        raise ValueError("Stage M transport vectors must be finite one-dimensional arrays")
    true = target_values - input_values
    model = model_values - input_values
    persistence = np.zeros_like(true)
    error = model - true
    persistence_error = persistence - true
    error_norm = float(np.linalg.norm(error))
    true_norm = float(np.linalg.norm(true))
    persistence_error_norm = float(np.linalg.norm(persistence_error))
    relative_error, relative_undefined = safe_ratio(
        error_norm, true_norm, epsilon=epsilon
    )
    skill_ratio, skill_undefined = safe_ratio(
        error_norm, persistence_error_norm, epsilon=epsilon
    )
    cosine_denominator = float(np.linalg.norm(model) * np.linalg.norm(true))
    cosine = (
        None
        if cosine_denominator <= epsilon
        else float(np.dot(model, true) / cosine_denominator)
    )
    return {
        "input": input_values.tolist(),
        "target": target_values.tolist(),
        "model_state_metric": model_values.tolist(),
        "delta_true": true.tolist(),
        "delta_model": model.tolist(),
        "delta_persistence": persistence.tolist(),
        "absolute_error": np.abs(error).tolist(),
        "error_l2": error_norm,
        "persistence_error_l2": persistence_error_norm,
        "relative_error": relative_error,
        "relative_error_undefined": relative_undefined,
        "transport_cosine": cosine,
        "transport_cosine_undefined": cosine is None,
        "signed_transport_agreement": float(
            np.mean(
                _signed(model, sign_zero_tolerance)
                == _signed(true, sign_zero_tolerance)
            )
        ),
        "persistence_relative_skill": (
            None if skill_ratio is None else 1.0 - skill_ratio
        ),
        "persistence_relative_skill_undefined": skill_undefined,
    }


def require_target_oracle(*, ground_truth_available: bool) -> None:
    if not ground_truth_available:
        raise ValueError("Stage M no-GT diagnostics cannot request a target oracle")


def floor_limited_channel(
    *,
    oracle_legacy_detector_triggered: bool,
    median_variance_retention: float | None,
    median_shell_radial_retention: float | None,
    median_high_k_retention: float | None,
    threshold: float = 0.5,
) -> bool:
    values = (
        median_variance_retention,
        median_shell_radial_retention,
        median_high_k_retention,
    )
    return bool(oracle_legacy_detector_triggered) or any(
        value is not None and value < threshold for value in values
    )


def model_added_gate(
    severe_by_step: Mapping[int, Mapping[str, bool]], *, required_steps: int = 2
) -> dict[str, Any]:
    required = {
        "global_variance",
        "shell_radial_variance",
        "dynamic_span",
        "high_k_energy",
    }
    step_failures: list[int] = []
    for step, values in severe_by_step.items():
        if set(values) != required:
            raise ValueError("Stage M model gate requires exactly four core categories")
        count = sum(bool(value) for value in values.values())
        if count >= 2 and (
            values["global_variance"] or values["shell_radial_variance"]
        ):
            step_failures.append(int(step))
    return {
        "failed": len(step_failures) >= int(required_steps),
        "failure_steps": sorted(step_failures),
        "required_steps": int(required_steps),
    }


def validate_candidate_threshold_sources(config: Mapping[str, Any]) -> None:
    """Reject any gate config that admits validation/model-outcome threshold tuning."""

    calibration = config["calibration_split"]
    if not calibration.get("validation_outcomes_forbidden", False):
        raise ValueError("Stage M gate must forbid validation-driven thresholds")
    if not calibration.get("model_outcomes_forbidden", False):
        raise ValueError("Stage M gate must forbid model-outcome-driven thresholds")
    serialized = str(config).lower()
    forbidden = ("stage_k_best_epoch", "localno_ranking", "fno_ranking")
    if any(token in serialized for token in forbidden):
        raise ValueError("Stage M gate threshold source depends on frozen model outcomes")


def oracle_conditioned_gate_schema() -> dict[str, Any]:
    """Return the frozen Stage M candidate-gate reporting schema."""

    return {
        "oracle_conditioned_gate_version": "stage_m_v1",
        "reporting_only": True,
        "training_blocking_default": False,
        "required_outputs": [
            "legacy_detector",
            "gate_0_engineering_validity",
            "gate_1_floor_qualification",
            "gate_2_model_added_degradation",
            "gate_3_transport_skill",
        ],
        "thresholds": {
            "retention": 0.5,
            "model_severe_categories": 2,
            "model_failure_steps": 2,
            "transport_minimum_skill": 0.0,
            "transport_sign_agreement": 0.5,
        },
    }


def evaluate_oracle_conditioned_structure_gate(
    *,
    legacy_detector: Mapping[str, Any],
    engineering_checks: Mapping[str, bool],
    floor_metrics: Mapping[str, float | bool | None],
    severe_by_step: Mapping[int, Mapping[str, bool]],
    shell_transport_skill: float | None,
    radial_transport_skill: float | None,
    shell_sign_agreement: float | None,
    radial_sign_agreement: float | None,
    enabled: bool = True,
    training_blocking: bool = False,
) -> dict[str, Any]:
    """Report the frozen Stage M oracle-conditioned gate beside legacy output.

    The interface is intentionally read-only and reporting-only.  It does not
    call or alter the legacy detector, and it refuses to become a training
    blocker without a future contract/version.
    """

    required_engineering = {
        "finite",
        "rho_press_positive",
        "transform_counters",
        "checkpoint_provenance",
        "decoded_range",
        "Rout",
        "shape_device",
    }
    if set(engineering_checks) != required_engineering:
        raise ValueError("Oracle-conditioned Gate 0 requires the frozen engineering checks")
    if training_blocking:
        raise ValueError("stage_m_v1 is reporting-only and cannot block training")
    floor_required = {
        "oracle_legacy_detector_triggered",
        "median_variance_retention",
        "median_shell_radial_retention",
        "median_high_k_retention",
    }
    if set(floor_metrics) != floor_required:
        raise ValueError("Oracle-conditioned Gate 1 requires the frozen floor metrics")
    gate0_passed = all(bool(value) for value in engineering_checks.values())
    gate1_limited = floor_limited_channel(
        oracle_legacy_detector_triggered=bool(
            floor_metrics["oracle_legacy_detector_triggered"]
        ),
        median_variance_retention=floor_metrics["median_variance_retention"],
        median_shell_radial_retention=floor_metrics[
            "median_shell_radial_retention"
        ],
        median_high_k_retention=floor_metrics["median_high_k_retention"],
        threshold=0.5,
    )
    gate2 = model_added_gate(severe_by_step, required_steps=2)
    skills = (shell_transport_skill, radial_transport_skill)
    signs = (shell_sign_agreement, radial_sign_agreement)
    defined = all(value is not None and np.isfinite(value) for value in (*skills, *signs))
    gate3_passed = bool(
        defined
        and all(float(value) >= 0.0 for value in skills if value is not None)
        and any(float(value) > 0.0 for value in skills if value is not None)
        and all(float(value) >= 0.5 for value in signs if value is not None)
    )
    return {
        "oracle_conditioned_gate_version": "stage_m_v1",
        "enabled": bool(enabled),
        "reporting_only": True,
        "training_blocking": False,
        "legacy_detector": copy.deepcopy(dict(legacy_detector)),
        "gate_0_engineering_validity": {
            "passed": gate0_passed,
            "checks": dict(engineering_checks),
        },
        "gate_1_floor_qualification": {
            "floor_limited": gate1_limited,
            "metrics": dict(floor_metrics),
        },
        "gate_2_model_added_degradation": gate2,
        "gate_3_transport_skill": {
            "passed": gate3_passed,
            "shell_skill": shell_transport_skill,
            "radial_skill": radial_transport_skill,
            "shell_sign_agreement": shell_sign_agreement,
            "radial_sign_agreement": radial_sign_agreement,
        },
    }
