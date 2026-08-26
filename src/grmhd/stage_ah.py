"""Frozen post-hoc statistics and decision rules for Stage AH."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


THETA_REGION_BOUNDS = {
    "near_north_pole": (0, 8),
    "mid_latitude_north": (8, 24),
    "equatorial": (24, 40),
    "mid_latitude_south": (40, 56),
    "near_south_pole": (56, 64),
}

RADIAL_REGION_BOUNDS = {
    "inner": (0, 21),
    "middle": (21, 42),
    "outer": (42, 64),
}


def validate_region_bounds(
    bounds: Mapping[str, tuple[int, int]], *, size: int
) -> None:
    """Require an ordered, non-overlapping, exhaustive axis partition."""

    ordered = list(bounds.values())
    if not ordered or ordered[0][0] != 0 or ordered[-1][1] != size:
        raise ValueError("Stage AH region bounds must cover the full axis")
    if any(start < 0 or stop <= start or stop > size for start, stop in ordered):
        raise ValueError("Stage AH region bounds contain an invalid interval")
    if any(left[1] != right[0] for left, right in zip(ordered, ordered[1:])):
        raise ValueError("Stage AH region bounds must be contiguous")


def field_metrics(
    predicted_state: np.ndarray,
    target_state: np.ndarray,
    predicted_residual: np.ndarray,
    target_residual: np.ndarray,
    *,
    epsilon: float = 1.0e-300,
) -> dict[str, float]:
    """Return frozen state/residual metrics for one selected field region."""

    arrays = [
        np.asarray(value, dtype=np.float64)
        for value in (predicted_state, target_state, predicted_residual, target_residual)
    ]
    if any(value.shape != arrays[0].shape for value in arrays[1:]):
        raise ValueError("Stage AH metric fields must have identical shapes")
    if any(not np.isfinite(value).all() for value in arrays):
        raise FloatingPointError("Stage AH metric field contains NaN/Inf")
    state_prediction, state_target, residual_prediction, residual_target = arrays
    state_difference = state_prediction - state_target
    residual_difference = residual_prediction - residual_target
    state_l2 = np.linalg.norm(state_difference.ravel()) / max(
        np.linalg.norm(state_target.ravel()), epsilon
    )
    residual_l2 = np.linalg.norm(residual_difference.ravel()) / max(
        np.linalg.norm(residual_target.ravel()), epsilon
    )
    residual_cosine = np.dot(residual_prediction.ravel(), residual_target.ravel()) / max(
        np.linalg.norm(residual_prediction.ravel())
        * np.linalg.norm(residual_target.ravel()),
        epsilon,
    )
    return {
        "state_l2": float(state_l2),
        "residual_l2": float(residual_l2),
        "residual_cosine": float(residual_cosine),
        "absolute_residual_error": float(np.mean(np.abs(residual_difference))),
    }


def paired_bootstrap(
    rows: Sequence[Mapping[str, Any]],
    metrics: Sequence[str],
    *,
    seed: int = 42,
    samples: int = 10_000,
) -> list[dict[str, Any]]:
    """Bootstrap paired Stage-AG minus Stage-AD validation differences."""

    if not rows or samples <= 0:
        raise ValueError("Stage AH paired bootstrap requires rows and samples")
    rng = np.random.default_rng(seed)
    output: list[dict[str, Any]] = []
    higher_is_better_metrics = {"residual_cosine", "shell_skill", "radial_skill"}
    for metric in metrics:
        delta = np.asarray(
            [float(row[f"stage_ag_{metric}"]) - float(row[f"stage_ad_{metric}"]) for row in rows],
            dtype=np.float64,
        )
        if not np.isfinite(delta).all():
            raise FloatingPointError(f"nonfinite paired delta for {metric}")
        indices = rng.integers(0, len(delta), size=(samples, len(delta)))
        boot = np.mean(delta[indices], axis=1)
        higher = metric in higher_is_better_metrics
        output.append({
            "metric": metric,
            "delta_definition": "Stage_AG_minus_Stage_AD",
            "higher_is_better": higher,
            "mean_paired_delta": float(np.mean(delta)),
            "median_paired_delta": float(np.median(delta)),
            "win_fraction": float(np.mean(delta > 0.0 if higher else delta < 0.0)),
            "bootstrap_seed": seed,
            "bootstrap_samples": samples,
            "ci95_low": float(np.quantile(boot, 0.025)),
            "ci95_high": float(np.quantile(boot, 0.975)),
        })
    return output


def primary_gates(
    stage_ad: Mapping[str, float],
    stage_ag: Mapping[str, float],
    *,
    stage_ad_shell_absolute_error: float,
    stage_ag_shell_absolute_error: float,
    stage_ad_radial_absolute_error: float,
    stage_ag_radial_absolute_error: float,
    first_10x_step: int | None,
) -> dict[str, bool]:
    """Apply the immutable G1--G6 thresholds from the Stage AH contract."""

    shell_abs_improved = stage_ag_shell_absolute_error < stage_ad_shell_absolute_error
    radial_abs_improved = stage_ag_radial_absolute_error < stage_ad_radial_absolute_error
    return {
        "STATE_RETENTION_GATE": stage_ag["state_l2"] <= 1.05 * stage_ad["state_l2"],
        "RESIDUAL_GEOMETRY_GAIN": stage_ag["residual_l2"] < stage_ad["residual_l2"],
        "DIRECTION_GEOMETRY_GAIN": stage_ag["cosine"] > stage_ad["cosine"],
        "SHELL_ABSOLUTE_ERROR_IMPROVED": shell_abs_improved,
        "SHELL_GEOMETRY_GAIN": (
            stage_ag["shell_skill"] > stage_ad["shell_skill"] + 0.10
            and shell_abs_improved
        ),
        "SHELL_STRONG_GAIN": stage_ag["shell_skill"] > 0.0 and shell_abs_improved,
        "RADIAL_ABSOLUTE_ERROR_IMPROVED": radial_abs_improved,
        "RADIAL_GEOMETRY_GAIN": (
            stage_ag["radial_skill"] > stage_ad["radial_skill"] + 0.10
            and radial_abs_improved
        ),
        "RADIAL_STRONG_GAIN": stage_ag["radial_skill"] > 0.0 and radial_abs_improved,
        "ROLLOUT_GEOMETRY_GAIN": first_10x_step is None or first_10x_step > 1,
        "ROLLOUT_STRONG_GAIN": first_10x_step is None or first_10x_step > 3,
    }


def scientific_decision(
    gates: Mapping[str, bool],
    stage_ad: Mapping[str, float],
    stage_ag: Mapping[str, float],
    *,
    integrity_valid: bool,
) -> str:
    """Choose exactly one predeclared Stage AH decision A--F."""

    if not integrity_valid:
        return "F"
    transport = gates["SHELL_GEOMETRY_GAIN"] or gates["RADIAL_GEOMETRY_GAIN"]
    both_transport = gates["SHELL_GEOMETRY_GAIN"] and gates["RADIAL_GEOMETRY_GAIN"]
    strong_transport = gates["SHELL_STRONG_GAIN"] or gates["RADIAL_STRONG_GAIN"]
    if (
        gates["STATE_RETENTION_GATE"]
        and gates["RESIDUAL_GEOMETRY_GAIN"]
        and gates["DIRECTION_GEOMETRY_GAIN"]
        and (strong_transport or both_transport)
        and gates["ROLLOUT_GEOMETRY_GAIN"]
    ):
        return "A"
    if gates["STATE_RETENTION_GATE"] and transport:
        return "B"
    one_step_wins = sum((
        stage_ag["state_l2"] < stage_ad["state_l2"],
        stage_ag["residual_l2"] < stage_ad["residual_l2"],
        stage_ag["cosine"] > stage_ad["cosine"],
    ))
    if one_step_wins >= 2 and not transport:
        return "C"
    one_step_losses = sum((
        stage_ag["state_l2"] > stage_ad["state_l2"],
        stage_ag["residual_l2"] > stage_ad["residual_l2"],
        stage_ag["cosine"] < stage_ad["cosine"],
    ))
    if not gates["STATE_RETENTION_GATE"] or one_step_losses == 3:
        return "E"
    return "D"
