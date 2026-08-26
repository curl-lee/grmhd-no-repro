from __future__ import annotations

from grmhd.stage_s_data import (
    chronological_split,
    classify_distribution_shift,
    stratified_small_pair_indices,
)


def test_expanded_chronological_split_drops_only_boundary_pair() -> None:
    split = chronological_split(212)
    assert split.train_snapshot_count == 169
    assert split.train_pair_count == 168
    assert split.validation_snapshot_count == 43
    assert split.validation_pair_count == 42
    assert split.dropped_boundary_pair == (168, 169)


def test_small_control_is_reproducible_unique_and_period_spanning() -> None:
    first = stratified_small_pair_indices(range(168), count=79, seed=42)
    second = stratified_small_pair_indices(range(168), count=79, seed=42)
    assert first == second
    assert len(first) == len(set(first)) == 79
    assert first == sorted(first)
    assert first[0] <= 2
    assert first[-1] >= 165


def test_distribution_shift_classification_uses_fixed_thresholds() -> None:
    base = {"x": {"mean": 0.0, "std": 1.0, "q001": -2.0, "q999": 2.0}}
    none = {"x": {"mean": 0.1, "std": 1.1, "q001": -2.1, "q999": 2.1}}
    mild = {"x": {"mean": 0.6, "std": 1.6, "q001": -2.1, "q999": 2.1}}
    strong = {"x": {"mean": 2.1, "std": 1.0, "q001": -2.1, "q999": 2.1}}
    assert classify_distribution_shift(base, none)[0] == "NONE"
    assert classify_distribution_shift(base, mild)[0] == "MILD"
    assert classify_distribution_shift(base, strong)[0] == "STRONG"
