"""Deterministic data controls for the Stage S expanded-data experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ChronologicalSplit:
    """Half-open snapshot ranges with one deliberately dropped boundary pair."""

    train_start: int
    train_end: int
    validation_start: int
    validation_end: int

    @property
    def train_snapshot_count(self) -> int:
        return self.train_end - self.train_start

    @property
    def validation_snapshot_count(self) -> int:
        return self.validation_end - self.validation_start

    @property
    def train_pair_count(self) -> int:
        return max(self.train_snapshot_count - 1, 0)

    @property
    def validation_pair_count(self) -> int:
        return max(self.validation_snapshot_count - 1, 0)

    @property
    def dropped_boundary_pair(self) -> tuple[int, int]:
        return self.train_end - 1, self.validation_start

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "range_semantics": "half_open",
            "train_snapshot_count": self.train_snapshot_count,
            "train_pair_count": self.train_pair_count,
            "validation_snapshot_count": self.validation_snapshot_count,
            "validation_pair_count": self.validation_pair_count,
            "dropped_boundary_pair": list(self.dropped_boundary_pair),
        }


def chronological_split(
    snapshot_count: int, *, validation_fraction: float = 0.20
) -> ChronologicalSplit:
    """Reserve the final ceil(fraction*N) snapshots as held-out validation."""

    if snapshot_count < 4:
        raise ValueError("Stage S needs at least four snapshots")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between zero and one")
    validation_count = int(math.ceil(snapshot_count * validation_fraction))
    train_count = snapshot_count - validation_count
    if train_count < 2 or validation_count < 2:
        raise ValueError("Both Stage S splits must contain at least one adjacent pair")
    return ChronologicalSplit(0, train_count, train_count, snapshot_count)


def stratified_small_pair_indices(
    train_pair_indices: Sequence[int], *, count: int = 79, seed: int = 42
) -> list[int]:
    """Select one adjacent pair from each deterministic temporal stratum.

    The strata span the complete chronological training period. Randomness only
    selects within each non-overlapping stratum and is fixed by ``seed``.
    """

    values = np.asarray(tuple(int(value) for value in train_pair_indices), dtype=np.int64)
    if values.ndim != 1 or values.size < count or count <= 0:
        raise ValueError("Small control requires at least count distinct pair indices")
    if len(np.unique(values)) != len(values) or np.any(np.diff(values) <= 0):
        raise ValueError("Training pair indices must be unique and strictly increasing")
    rng = np.random.default_rng(seed)
    selected: list[int] = []
    for stratum in range(count):
        lower = stratum * len(values) // count
        upper = (stratum + 1) * len(values) // count
        if upper <= lower:
            raise RuntimeError("Temporal stratum is empty")
        selected.append(int(values[int(rng.integers(lower, upper))]))
    if len(selected) != count or len(set(selected)) != count:
        raise RuntimeError("Stratified Stage S pair selection duplicated a pair")
    return selected


def classify_distribution_shift(
    old: Mapping[str, Mapping[str, float]],
    new: Mapping[str, Mapping[str, float]],
) -> tuple[str, dict[str, dict[str, float]]]:
    """Classify shift with fixed, scale-relative thresholds.

    STRONG is triggered by a >3x/<1/3 spread change, a mean displacement over
    two old standard deviations, or a >3x/<1/3 central-tail-span change. MILD
    uses corresponding 1.5x, 0.5 standard-deviation, and 1.5x thresholds.
    """

    diagnostics: dict[str, dict[str, float]] = {}
    strong = mild = False
    tiny = np.finfo(np.float64).tiny
    for channel in sorted(old):
        before, after = old[channel], new[channel]
        std_ratio = float(after["std"] / max(before["std"], tiny))
        mean_shift = float(abs(after["mean"] - before["mean"]) / max(before["std"], tiny))
        old_span = before["q999"] - before["q001"]
        new_span = after["q999"] - after["q001"]
        tail_span_ratio = float(new_span / max(old_span, tiny))
        diagnostics[channel] = {
            "new_over_old_std": std_ratio,
            "mean_shift_in_old_std": mean_shift,
            "new_over_old_q001_q999_span": tail_span_ratio,
        }
        channel_strong = (
            std_ratio > 3.0
            or std_ratio < 1.0 / 3.0
            or mean_shift > 2.0
            or tail_span_ratio > 3.0
            or tail_span_ratio < 1.0 / 3.0
        )
        channel_mild = (
            std_ratio > 1.5
            or std_ratio < 1.0 / 1.5
            or mean_shift > 0.5
            or tail_span_ratio > 1.5
            or tail_span_ratio < 1.0 / 1.5
        )
        strong |= channel_strong
        mild |= channel_mild
    return ("STRONG" if strong else "MILD" if mild else "NONE"), diagnostics
