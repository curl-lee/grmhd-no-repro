from __future__ import annotations

import pytest
import yaml

from grmhd.paper_config import load_paper_experiment_config

from grmhd.stage_s_training import (
    accumulation_groups,
    matched_microbatch_order,
    natural_epoch_orders,
    order_sha256,
    warmup_cosine_learning_rate,
)


def test_matched_plans_have_identical_update_and_sample_budgets() -> None:
    small = matched_microbatch_order(range(79), optimizer_updates=600, accumulation=4)
    full = matched_microbatch_order(range(168), optimizer_updates=600, accumulation=4)
    assert len(small) == len(full) == 2400
    assert len(accumulation_groups(small, accumulation=4)) == 600
    assert len(accumulation_groups(full, accumulation=4)) == 600
    assert all(len(group) == 4 for group in accumulation_groups(small, accumulation=4))
    assert set(small) == set(range(79))
    assert set(full) == set(range(168))


def test_pair_orders_are_reproducible_and_seeded() -> None:
    first = matched_microbatch_order(range(79), seed=42)
    second = matched_microbatch_order(range(79), seed=42)
    changed = matched_microbatch_order(range(79), seed=43)
    assert first == second and first != changed
    assert order_sha256(first) == order_sha256(second)


def test_natural_full_budget_is_30_complete_epochs() -> None:
    orders = natural_epoch_orders(range(168), epochs=30)
    assert len(orders) == 30
    assert all(len(order) == 168 and set(order) == set(range(168)) for order in orders)
    assert sum(len(accumulation_groups(order, accumulation=4)) for order in orders) == 1260


def test_update_scheduler_hits_frozen_endpoints() -> None:
    assert warmup_cosine_learning_rate(
        1, total_updates=600, warmup_updates=40,
        base_learning_rate=1e-3, min_learning_rate=1e-6,
    ) == pytest.approx(2.5e-5)
    assert warmup_cosine_learning_rate(
        40, total_updates=600, warmup_updates=40,
        base_learning_rate=1e-3, min_learning_rate=1e-6,
    ) == pytest.approx(1e-3)
    assert warmup_cosine_learning_rate(
        600, total_updates=600, warmup_updates=40,
        base_learning_rate=1e-3, min_learning_rate=1e-6,
    ) == pytest.approx(1e-6)


def test_stage_s_model_contract_matches_resolved_stage_r() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    stage_s = yaml.safe_load(
        (root / "configs/stage_s/expanded_localno_p3_residual.yaml").read_text()
    )
    stage_r = load_paper_experiment_config(
        root / "configs/paper_reduced100/stage_r_localno_p3_residual_plain.yaml",
        project_root=root,
    )
    assert stage_s["model"] == stage_r.values["model"]
    assert stage_s["optimizer"]["name"] == stage_r.values["optimizer"]["name"]
    assert stage_s["optimizer"]["learning_rate"] == stage_r.values["optimizer"]["learning_rate"]
    assert stage_s["optimizer"]["weight_decay"] == stage_r.values["optimizer"]["weight_decay"]
    assert stage_s["runtime"]["gradient_accumulation"] == stage_r.values["runtime"]["gradient_accumulation"]


def test_required_small_pair_manifest_contains_adjacent_pairs() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    rows = [
        tuple(int(value) for value in line.split())
        for line in (root / "artifacts/stage_s/small_train_pairs.txt").read_text().splitlines()
        if line.strip()
    ]
    assert len(rows) == 79
    assert all(len(row) == 2 and row[1] == row[0] + 1 for row in rows)
