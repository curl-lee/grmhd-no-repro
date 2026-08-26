"""Frozen pairing utilities for the 30-epoch paper-reduced Stage G pilots."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


STAGE_G_SCHEMA_VERSION = "paper-stage-g-pairing-v1"
STAGE_G_EPOCHS = 30
STAGE_G_SEED = 42
STAGE_G_TRAIN_PAIRS = 79
STAGE_G_VALIDATION_PAIRS = 19
STAGE_G_ACCUMULATION = 4
STAGE_G_STEPS_PER_EPOCH = 20


def tensor_state_sha256(state: Mapping[str, Any]) -> str:
    """Hash tensor state identically to ``model_tensor_state_sha256``."""

    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def generate_epoch_pair_order(
    *,
    seed: int = STAGE_G_SEED,
    epochs: int = STAGE_G_EPOCHS,
    pair_count: int = STAGE_G_TRAIN_PAIRS,
    first_source_snapshot: int = 11,
) -> dict[str, Any]:
    """Generate one reproducible complete permutation for each training epoch."""

    if seed < 0 or epochs <= 0 or pair_count <= 0:
        raise ValueError("Stage G order parameters must be positive")
    generator = torch.Generator().manual_seed(int(seed))
    epoch_records = []
    expected = list(range(pair_count))
    for epoch in range(epochs):
        pair_indices = torch.randperm(pair_count, generator=generator).tolist()
        epoch_records.append(
            {
                "epoch": epoch,
                "pair_indices": pair_indices,
                "source_snapshots": [
                    first_source_snapshot + index for index in pair_indices
                ],
            }
        )
    payload = {
        "schema_version": STAGE_G_SCHEMA_VERSION,
        "seed": int(seed),
        "epochs": int(epochs),
        "train_pair_count": int(pair_count),
        "train_pair_indices": expected,
        "train_source_snapshots": [
            first_source_snapshot + index for index in expected
        ],
        "validation_pair_count": STAGE_G_VALIDATION_PAIRS,
        "validation_shuffle": False,
        "dropped_transition": [90, 91],
        "epoch_orders": epoch_records,
    }
    validate_epoch_pair_order(
        payload,
        expected_epochs=epochs,
        expected_pair_count=pair_count,
        expected_seed=seed,
        first_source_snapshot=first_source_snapshot,
    )
    return payload


def validate_epoch_pair_order(
    payload: Mapping[str, Any],
    *,
    expected_epochs: int = STAGE_G_EPOCHS,
    expected_pair_count: int = STAGE_G_TRAIN_PAIRS,
    expected_seed: int = STAGE_G_SEED,
    first_source_snapshot: int = 11,
) -> list[list[int]]:
    """Reject missing, repeated, validation, or dropped-transition entries."""

    if payload.get("schema_version") != STAGE_G_SCHEMA_VERSION:
        raise ValueError("Stage G pair-order schema changed")
    if int(payload.get("seed", -1)) != expected_seed:
        raise ValueError("Stage G pair-order seed changed")
    if int(payload.get("epochs", -1)) != expected_epochs:
        raise ValueError("Stage G pair-order epoch count changed")
    if int(payload.get("train_pair_count", -1)) != expected_pair_count:
        raise ValueError("Stage G pair-order train-pair count changed")
    if payload.get("validation_shuffle") is not False:
        raise ValueError("Stage G validation loader must not shuffle")
    if list(payload.get("dropped_transition", ())) != [90, 91]:
        raise ValueError("Stage G dropped transition changed")
    records = payload.get("epoch_orders")
    if not isinstance(records, Sequence) or len(records) != expected_epochs:
        raise ValueError("Stage G pair-order records are incomplete")
    expected = list(range(expected_pair_count))
    expected_sources = list(
        range(first_source_snapshot, first_source_snapshot + expected_pair_count)
    )
    orders: list[list[int]] = []
    for epoch, record in enumerate(records):
        if not isinstance(record, Mapping) or int(record.get("epoch", -1)) != epoch:
            raise ValueError(f"Stage G pair-order epoch {epoch} metadata is invalid")
        order = [int(value) for value in record.get("pair_indices", ())]
        sources = [int(value) for value in record.get("source_snapshots", ())]
        if sorted(order) != expected:
            raise ValueError(f"Stage G epoch {epoch} is not a complete train permutation")
        if sources != [first_source_snapshot + index for index in order]:
            raise ValueError(f"Stage G epoch {epoch} source snapshots do not match pair indices")
        if any(source not in expected_sources for source in sources):
            raise ValueError(f"Stage G epoch {epoch} contains a validation pair")
        if 90 in sources:
            raise ValueError(f"Stage G epoch {epoch} contains dropped transition 90->91")
        orders.append(order)
    return orders


def load_epoch_pair_order(path: str | Path) -> tuple[dict[str, Any], list[list[int]]]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Stage G pair-order file must contain a JSON object")
    return payload, validate_epoch_pair_order(payload)


def accumulation_count_for_batch(
    batch_index: int,
    *,
    total_batches: int,
    accumulation: int = STAGE_G_ACCUMULATION,
) -> int:
    """Return the actual denominator for this batch's accumulation group."""

    batch_index = int(batch_index)
    total_batches = int(total_batches)
    accumulation = int(accumulation)
    if total_batches <= 0 or accumulation <= 0 or not 0 <= batch_index < total_batches:
        raise ValueError("Invalid gradient-accumulation coordinates")
    group_start = (batch_index // accumulation) * accumulation
    return min(accumulation, total_batches - group_start)


def is_accumulation_step(
    batch_index: int,
    *,
    total_batches: int,
    accumulation: int = STAGE_G_ACCUMULATION,
) -> bool:
    """Return whether this microbatch closes a full or final partial group."""

    count = accumulation_count_for_batch(
        batch_index, total_batches=total_batches, accumulation=accumulation
    )
    group_start = (batch_index // accumulation) * accumulation
    return batch_index == group_start + count - 1
