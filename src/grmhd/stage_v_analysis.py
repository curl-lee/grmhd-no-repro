"""Predeclared Stage V scientific-decision helpers."""

from __future__ import annotations

import math
from typing import Mapping, Sequence


def distribution_shift_error_coupling(
    correlations: Sequence[float], *, moderate: float = 0.30, strong: float = 0.60
) -> str:
    values = [abs(float(value)) for value in correlations if math.isfinite(float(value))]
    strong_count = sum(value >= strong for value in values)
    moderate_count = sum(value >= moderate for value in values)
    if strong_count >= 2:
        return "STRONG"
    if strong_count >= 1 or moderate_count >= 2:
        return "MODERATE"
    return "LOW"


def stage_v_primary_decision(
    candidates: Sequence[Mapping[str, object]],
    *,
    gradient_conflict: str,
    pairwise_misalignment: bool,
    meaningful_improvement: float = 0.10,
    strong_rollout_delay_after_step: int = 3,
) -> str:
    """Classify completed V2/V3 results under the frozen decision contract."""

    if not candidates:
        return "E"
    scientific_signal = pairwise_misalignment or gradient_conflict in {"MODERATE", "STRONG"}
    def gained(row: Mapping[str, object]) -> bool:
        shell = float(row["delta_shell_skill_vs_stage_t"])
        radial = float(row["delta_radial_skill_vs_stage_t"])
        first = row.get("first_10x_step")
        delayed = first is None or int(first) > 1
        return shell >= meaningful_improvement or radial >= meaningful_improvement or delayed

    gains = [row for row in candidates if gained(row)]
    if not gains:
        if all(not bool(row["O1_state_retention"]) for row in candidates):
            return "D"
        return "C"
    if not scientific_signal:
        return "C"
    for row in gains:
        first = row.get("first_10x_step")
        strong_delay = first is None or int(first) > strong_rollout_delay_after_step
        if (
            bool(row["O1_state_retention"])
            and float(row["shell_skill"]) > 0
            and float(row["radial_skill"]) > 0
            and strong_delay
        ):
            return "A"
    return "B"
