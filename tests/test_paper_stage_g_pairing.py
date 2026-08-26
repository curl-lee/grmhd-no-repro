from __future__ import annotations

import copy

import pytest
import torch

from grmhd.paper_stage_g import (
    accumulation_count_for_batch,
    generate_epoch_pair_order,
    is_accumulation_step,
    tensor_state_sha256,
    validate_epoch_pair_order,
)


def test_epoch_pair_order_is_complete_reproducible_and_excludes_validation():
    first = generate_epoch_pair_order()
    second = generate_epoch_pair_order()
    assert first == second
    orders = validate_epoch_pair_order(first)
    assert len(orders) == 30
    assert all(len(order) == 79 and sorted(order) == list(range(79)) for order in orders)
    assert all(90 not in record["source_snapshots"] for record in first["epoch_orders"])
    assert first["validation_shuffle"] is False


@pytest.mark.parametrize("epoch", [0, 7, 29])
def test_epoch_pair_order_rejects_duplicate_or_missing_pair(epoch):
    payload = generate_epoch_pair_order()
    stale = copy.deepcopy(payload)
    stale["epoch_orders"][epoch]["pair_indices"][-1] = stale["epoch_orders"][epoch][
        "pair_indices"
    ][0]
    with pytest.raises(ValueError, match="complete train permutation"):
        validate_epoch_pair_order(stale)


def test_partial_accumulation_uses_three_not_four_and_produces_twenty_steps():
    counts = [
        accumulation_count_for_batch(index, total_batches=79, accumulation=4)
        for index in range(79)
    ]
    step_indices = [
        index
        for index in range(79)
        if is_accumulation_step(index, total_batches=79, accumulation=4)
    ]
    assert counts[:76] == [4] * 76
    assert counts[76:] == [3, 3, 3]
    assert step_indices == list(range(3, 76, 4)) + [78]
    assert len(step_indices) == 20


def test_partial_accumulation_actual_count_normalizes_mean_gradient():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    for value in (1.0, 2.0, 6.0):
        (parameter * value / 3).backward()
    assert parameter.grad is not None
    assert parameter.grad.item() == pytest.approx(3.0)


def test_tensor_state_hash_is_order_independent_and_value_sensitive():
    first = {"b": torch.tensor([2.0]), "a": torch.tensor([1.0])}
    second = {"a": torch.tensor([1.0]), "b": torch.tensor([2.0])}
    assert tensor_state_sha256(first) == tensor_state_sha256(second)
    second["b"] += 1
    assert tensor_state_sha256(first) != tensor_state_sha256(second)
