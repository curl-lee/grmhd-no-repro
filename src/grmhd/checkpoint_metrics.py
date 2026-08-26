"""Transparent checkpoint-selection metrics for short GRMHD rollouts."""

from __future__ import annotations

import math


CHECKPOINT_METRIC_FILENAMES = {
    "normalized_total": "best_normalized.pt",
    "decoded_one_step": "best_decoded_one_step.pt",
    "decoded_three_step": "best_short_rollout.pt",
    "composite_stability": "best_composite.pt",
}

COMPOSITE_STABILITY_FORMULA = (
    "decoded_one_step_global + 0.5 * decoded_three_step_global "
    "+ 0.1 * magnetic_range_violation"
)


def relative_l2(error_sum_of_squares: float, truth_sum_of_squares: float) -> float:
    """Return an aggregate relative L2 from separately accumulated sums."""
    if error_sum_of_squares < 0 or truth_sum_of_squares < 0:
        raise ValueError("sum-of-squares inputs must be non-negative")
    if truth_sum_of_squares == 0:
        return 0.0 if error_sum_of_squares == 0 else math.inf
    return math.sqrt(error_sum_of_squares / truth_sum_of_squares)


def composite_stability_metric(
    decoded_one_step_global: float,
    decoded_three_step_global: float,
    magnetic_range_violation: float,
) -> float:
    """Combine decoded accuracy and magnetic-range stability for selection."""
    values = (
        decoded_one_step_global,
        decoded_three_step_global,
        magnetic_range_violation,
    )
    if any(value < 0 for value in values):
        raise ValueError("composite stability inputs must be non-negative")
    return values[0] + 0.5 * values[1] + 0.1 * values[2]
