from __future__ import annotations

from grmhd.stage_t_training import (
    data_effect_at_1260,
    resolve_optimization_budget,
    validation_thirds,
)


def test_full_long_budget_is_derived_from_actual_loader_contract() -> None:
    budget = resolve_optimization_budget(
        pair_count=168, batch_size=1, gradient_accumulation=4,
        total_epochs=1200, warmup_epochs=75,
    )
    assert budget.microbatches_per_epoch == 168
    assert budget.updates_per_epoch == 42
    assert budget.total_updates == 50_400
    assert budget.warmup_updates == 3_150


def test_budget_preserves_partial_accumulation() -> None:
    budget = resolve_optimization_budget(
        pair_count=79, batch_size=1, gradient_accumulation=4,
        total_epochs=1, warmup_epochs=0,
    )
    assert budget.microbatches_per_epoch == 79
    assert budget.updates_per_epoch == 20


def test_validation_thirds_are_complete_disjoint_and_chronological() -> None:
    parts = validation_thirds(range(169, 211))
    assert {name: len(values) for name, values in parts.items()} == {
        "early": 14, "middle": 14, "late": 14,
    }
    assert [value for part in parts.values() for value in part] == list(range(169, 211))


def test_data_effect_classifier_uses_predeclared_two_percent_band() -> None:
    assert data_effect_at_1260(1.0, 0.97)[0] == "POSITIVE"
    assert data_effect_at_1260(1.0, 1.01)[0] == "NEUTRAL"
    assert data_effect_at_1260(1.0, 1.03)[0] == "NEGATIVE"
