"""Pure contracts for the no-training Stage Q residual-anchor audit."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from . import CHANNELS


STAGE_Q_CLASSIFICATION = "no_training_output_contract_audit"
ALPHA_GRID = (0.0, 0.125, 0.25, 0.5, 1.0)
CANDIDATE_ALPHAS = (0.125, 0.25, 0.5)
LOCAL_GAIN_EPSILONS = (1.0e-3, 1.0e-2)
FROZEN_HISTORY = {
    "Stage K": "C. TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE",
    "Stage L": "3. MIXED_OVERALL",
    "Stage M": "4. MIXED_TRANSPORT_AND_PREPROCESSING_FAILURE",
    "Stage N": "P3 ready; mixed operator-response failure",
    "Stage O": "C. P3_PILOT_COMPLETE_BUT_ROLLOUT_UNSTABLE",
    "Stage P": "6. MIXED_CLOSED_LOOP_FAILURE",
}


def anchored_output(
    normalized_state: torch.Tensor,
    direct_output: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Apply only the frozen affine output contract; no transform or repair."""

    if normalized_state.shape != direct_output.shape:
        raise ValueError("Stage Q state/direct output shapes differ")
    if normalized_state.ndim != 5 or normalized_state.shape[1] != len(CHANNELS):
        raise ValueError("Stage Q output contract requires shape (B,8,...)")
    value = float(alpha)
    if value not in ALPHA_GRID:
        raise ValueError("Stage Q alpha is outside the frozen grid")
    if value == 0.0:
        return normalized_state
    if value == 1.0:
        return direct_output
    return normalized_state + value * (direct_output - normalized_state)


def anchored_model_step(
    model: torch.nn.Module,
    normalized_state: torch.Tensor,
    shells: torch.Tensor,
    *,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return direct and anchored outputs without feedback transforms."""

    if shells.shape != normalized_state.shape:
        raise ValueError("Stage Q fixed shell tensor must match state shape")
    with torch.no_grad():
        direct = model(x=torch.cat((normalized_state, shells), dim=1))
        anchored = anchored_output(normalized_state, direct, alpha)
    return direct, anchored


def tensor_collection_sha256(values: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        digest.update(name.encode("utf-8"))
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("utf-8"))
            digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(
                json.dumps(value, sort_keys=True, default=str).encode("utf-8")
            )
    return digest.hexdigest()


def model_state_sha256(model: torch.nn.Module) -> str:
    return tensor_collection_sha256(model.state_dict())


def relative_l2_tensor(
    prediction: torch.Tensor, reference: torch.Tensor, *, epsilon: float = 1.0e-30
) -> float:
    delta = (prediction - reference).to(torch.float64)
    denominator = torch.linalg.vector_norm(reference.to(torch.float64)).clamp_min(epsilon)
    return float((torch.linalg.vector_norm(delta) / denominator).detach().cpu())


def residual_diagnostics(
    normalized_state: torch.Tensor,
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, Any]:
    predicted = (prediction - normalized_state).to(torch.float64)
    truth = (target - normalized_state).to(torch.float64)
    predicted_norm = torch.linalg.vector_norm(predicted)
    truth_norm = torch.linalg.vector_norm(truth)
    denominator = predicted_norm * truth_norm
    cosine = (
        0.0
        if float(denominator) == 0.0
        else float((torch.sum(predicted * truth) / denominator).detach().cpu())
    )
    return {
        "residual_norm": float(predicted_norm.detach().cpu()),
        "true_residual_norm": float(truth_norm.detach().cpu()),
        "residual_over_true_residual": float(
            (predicted_norm / truth_norm.clamp_min(1.0e-30)).detach().cpu()
        ),
        "residual_cosine": cosine,
        "residual_sign_agreement": float(
            torch.mean((torch.signbit(predicted) == torch.signbit(truth)).to(torch.float64))
            .detach()
            .cpu()
        ),
    }


def _fraction_reduction(baseline: float, candidate: float) -> float:
    baseline = float(baseline)
    candidate = float(candidate)
    if baseline <= 0.0:
        return 1.0 if candidate <= 0.0 else 0.0
    return (baseline - candidate) / baseline


def readiness_predicates(
    candidate: Mapping[str, Any],
    *,
    persistence: Mapping[str, Any],
    direct: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate frozen A--F train-only readiness predicates."""

    thresholds = config["readiness"]
    engineering = candidate["engineering"]
    a = {
        "all_finite": bool(engineering["all_finite"]),
        "rho_press_positive": bool(engineering["rho_press_positive"]),
        "transform_counter_errors_zero": int(engineering["transform_counter_errors"]) == 0,
        "new_nonfinite_decoder_derivatives_zero": int(
            engineering["new_nonfinite_decoder_derivatives"]
        )
        == 0,
    }
    ood_reduction = _fraction_reduction(
        direct["normalized_q_ood_fraction_median"],
        candidate["normalized_q_ood_fraction_median"],
    )
    rout_reduction = _fraction_reduction(
        direct["rout_failure_count"], candidate["rout_failure_count"]
    )
    b = {
        "q_ood_reduction": ood_reduction
        >= float(thresholds["overshoot_reduction"]["q001_q999_ood_fraction_reduction_min"]),
        "rout_failure_reduction": rout_reduction
        >= float(thresholds["overshoot_reduction"]["rout_failure_count_reduction_min"]),
    }
    persistence_ratio = float(candidate["normalized_average"]) / max(
        float(persistence["normalized_average"]), 1.0e-30
    )
    c = {
        "candidate_over_persistence": persistence_ratio
        <= float(thresholds["persistence_bound"]["normalized_average_ratio_max"])
    }
    positive_cosines = sum(
        float(value) > 0.0
        for value in candidate["residual_cosine_channel_medians"].values()
    )
    d = {
        "residual_magnitude": float(candidate["residual_over_true_residual_median"])
        >= float(
            thresholds["nontrivial_dynamics"][
                "median_residual_over_true_residual_min"
            ]
        ),
        "positive_residual_cosine_channels": positive_cosines
        >= int(
            thresholds["nontrivial_dynamics"][
                "positive_residual_cosine_channels_min"
            ]
        ),
        "different_from_persistence": not bool(candidate["persistence_equivalent"]),
    }
    gate2_reduction = _fraction_reduction(
        direct["gate_2_severe_count"], candidate["gate_2_severe_count"]
    )
    shell_or_radial = (
        float(candidate["shell_transport_skill_median"])
        >= float(persistence["shell_transport_skill_median"])
        or float(candidate["radial_transport_skill_median"])
        >= float(persistence["radial_transport_skill_median"])
    )
    e = {
        "gate_2_reduction": gate2_reduction
        >= float(thresholds["structure"]["gate_2_severe_count_reduction_min"]),
        "shell_or_radial_not_worse_than_persistence": shell_or_radial,
        "no_new_persistent_range_failure": int(candidate["rout_failure_count"])
        <= int(direct["rout_failure_count"]),
    }
    control_names = tuple(thresholds["control_channels"]["names"])
    worsening_threshold = float(
        thresholds["control_channels"]["relative_error_worsening_threshold"]
    )
    worsened = [
        name
        for name in control_names
        if float(candidate["per_channel_relative_l2"][name])
        > (1.0 + worsening_threshold) * float(direct["per_channel_relative_l2"][name])
    ]
    f = {
        "control_channel_worsening_count": len(worsened)
        <= int(thresholds["control_channels"]["worsened_channel_count_max"])
    }
    groups = {"A": a, "B": b, "C": c, "D": d, "E": e, "F": f}
    group_pass = {name: all(values.values()) for name, values in groups.items()}
    return {
        "groups": groups,
        "group_pass": group_pass,
        "passed": all(group_pass.values()),
        "diagnostics": {
            "q_ood_reduction": ood_reduction,
            "rout_failure_reduction": rout_reduction,
            "candidate_over_persistence": persistence_ratio,
            "positive_residual_cosine_channels": positive_cosines,
            "gate_2_reduction": gate2_reduction,
            "worsened_control_channels": worsened,
        },
    }


def select_candidate_alpha(
    summaries: Mapping[float, Mapping[str, Any]],
    readiness: Mapping[float, Mapping[str, Any]],
    *,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze alpha from train-only summaries; validation is not an argument."""

    eligible = [alpha for alpha in CANDIDATE_ALPHAS if readiness[alpha]["passed"]]
    if not eligible:
        return {
            "candidate_alpha": None,
            "eligible_alphas": [],
            "reason": "no_intermediate_alpha_passed_all_train_only_readiness_groups",
            "validation_used": False,
        }
    best_metric = min(float(summaries[alpha]["normalized_average"]) for alpha in eligible)
    within = float(config["selection"]["within_relative_fraction"])
    near = [
        alpha
        for alpha in eligible
        if float(summaries[alpha]["normalized_average"])
        <= best_metric * (1.0 + within)
    ]
    if config["selection"]["within_one_percent_choose_larger_alpha"]:
        selected = max(near)
        reason = "within_one_percent_selected_larger_alpha_for_nontrivial_dynamics"
    else:
        selected = min(
            near, key=lambda alpha: float(summaries[alpha]["normalized_average"])
        )
        reason = "lowest_train_normalized_average"
    return {
        "candidate_alpha": selected,
        "eligible_alphas": sorted(eligible),
        "near_best_alphas": sorted(near),
        "best_train_normalized_average": best_metric,
        "reason": reason,
        "validation_used": False,
    }


def anchored_directional_gain(
    direct_mapping,
    state: torch.Tensor,
    direction: torch.Tensor,
    *,
    alpha: float,
    epsilon: float,
) -> dict[str, Any]:
    """Compare analytic affine-Jacobian composition with finite difference."""

    if epsilon not in LOCAL_GAIN_EPSILONS:
        raise ValueError("Stage Q local-gain epsilon is frozen to 1e-3 or 1e-2")
    if state.shape != direction.shape:
        raise ValueError("Stage Q local-gain state/direction shapes differ")
    direction_norm = torch.linalg.vector_norm(direction.to(torch.float64))
    if float(direction_norm) == 0.0:
        raise ValueError("Stage Q local-gain direction is zero")
    with torch.no_grad():
        direct_base = direct_mapping(state)
        direct_perturbed = direct_mapping(state + epsilon * direction)
        jf_response = (direct_perturbed - direct_base) / epsilon
        analytic = (1.0 - float(alpha)) * direction + float(alpha) * jf_response
        anchored_base = anchored_output(state, direct_base, alpha)
        anchored_perturbed = anchored_output(
            state + epsilon * direction, direct_perturbed, alpha
        )
        numeric = (anchored_perturbed - anchored_base) / epsilon
    difference = torch.linalg.vector_norm((numeric - analytic).to(torch.float64))
    analytic_norm = torch.linalg.vector_norm(analytic.to(torch.float64))
    return {
        "gain": float(
            (torch.linalg.vector_norm(numeric.to(torch.float64)) / direction_norm)
            .detach()
            .cpu()
        ),
        "analytic_numeric_relative_difference": float(
            (difference / analytic_norm.clamp_min(1.0e-30)).detach().cpu()
        ),
        "response": numeric.detach(),
        "analytic_response": analytic.detach(),
        "used_backward": False,
    }


def apply_anchored_map(
    direct_mapping,
    state: torch.Tensor,
    *,
    alpha: float,
    applications: int,
) -> list[torch.Tensor]:
    if applications not in (1, 2):
        raise ValueError("Stage Q fixed-state response permits at most two applications")
    outputs = []
    current = state
    with torch.no_grad():
        for _ in range(applications):
            current = anchored_output(current, direct_mapping(current), alpha)
            outputs.append(current.detach())
    return outputs


def validate_rollout_steps(steps: int) -> int:
    value = int(steps)
    if not 1 <= value <= 19:
        raise ValueError("Stage Q counterfactual rollout is limited to 19 steps")
    return value


def expected_transform_counts(steps: int) -> dict[str, int]:
    value = validate_rollout_steps(steps)
    return {"prediction_decode": value, "next_input_encode": value}


def selection_artifact_sha256(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
