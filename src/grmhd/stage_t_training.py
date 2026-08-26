"""Pure budget and decision contracts for the Stage T optimization audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping, Sequence


@dataclass(frozen=True)
class OptimizationBudget:
    pair_count: int
    batch_size: int
    gradient_accumulation: int
    total_epochs: int
    warmup_epochs: int
    microbatches_per_epoch: int
    updates_per_epoch: int
    total_updates: int
    warmup_updates: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def resolve_optimization_budget(
    *, pair_count: int, batch_size: int, gradient_accumulation: int,
    total_epochs: int, warmup_epochs: int,
) -> OptimizationBudget:
    """Resolve epoch quantities from the actual pair and batching contract."""

    values = (pair_count, batch_size, gradient_accumulation, total_epochs)
    if any(int(value) <= 0 for value in values):
        raise ValueError("Stage T budget values must be positive")
    if not 0 <= int(warmup_epochs) <= int(total_epochs):
        raise ValueError("Stage T warmup must lie inside the total horizon")
    microbatches = math.ceil(int(pair_count) / int(batch_size))
    updates = math.ceil(microbatches / int(gradient_accumulation))
    return OptimizationBudget(
        pair_count=int(pair_count),
        batch_size=int(batch_size),
        gradient_accumulation=int(gradient_accumulation),
        total_epochs=int(total_epochs),
        warmup_epochs=int(warmup_epochs),
        microbatches_per_epoch=microbatches,
        updates_per_epoch=updates,
        total_updates=updates * int(total_epochs),
        warmup_updates=updates * int(warmup_epochs),
    )


def validation_thirds(pair_indices: Sequence[int]) -> dict[str, list[int]]:
    """Split a fixed chronological validation block into three equal reports."""

    values = [int(value) for value in pair_indices]
    if not values or values != sorted(values) or len(set(values)) != len(values):
        raise ValueError("Validation pairs must be unique and chronological")
    if len(values) % 3:
        raise ValueError("Stage T validation thirds must have equal sizes")
    width = len(values) // 3
    return {
        "early": values[:width],
        "middle": values[width : 2 * width],
        "late": values[2 * width :],
    }


def data_effect_at_1260(
    small_normalized_l2: float, full_normalized_l2: float,
    *, relative_neutral_band: float = 0.02,
) -> tuple[str, float]:
    """Predeclared directional classification for the matched-update control.

    Differences within two percent of the small-control error are labelled
    neutral; this avoids turning normal run-to-run numerical noise into a data
    claim while leaving the frozen seed/order comparison directional.
    """

    small = float(small_normalized_l2)
    full = float(full_normalized_l2)
    if not math.isfinite(small) or not math.isfinite(full) or small <= 0:
        raise ValueError("Matched errors must be finite and positive")
    relative_change = (small - full) / small
    if relative_change > relative_neutral_band:
        return "POSITIVE", relative_change
    if relative_change < -relative_neutral_band:
        return "NEGATIVE", relative_change
    return "NEUTRAL", relative_change


def first_gate_step(records: Sequence[Mapping[str, object]], key: str) -> int | None:
    for record in records:
        if bool(record.get(key, False)):
            return int(record["step"])
    return None
