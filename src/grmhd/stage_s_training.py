"""Pure scheduling and pairing contracts for Stage S training."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Iterable, Sequence

import numpy as np


def deterministic_population_orders(
    pair_indices: Sequence[int], *, passes: int, seed: int
) -> list[list[int]]:
    values = np.asarray(tuple(int(value) for value in pair_indices), dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("Pair population must be non-empty")
    if len(np.unique(values)) != len(values):
        raise ValueError("Pair population contains duplicates")
    if passes <= 0:
        raise ValueError("Pass count must be positive")
    output: list[list[int]] = []
    for population_pass in range(passes):
        rng = np.random.default_rng(int(seed) + population_pass)
        output.append([int(values[index]) for index in rng.permutation(len(values))])
    return output


def matched_microbatch_order(
    pair_indices: Sequence[int],
    *,
    optimizer_updates: int = 600,
    accumulation: int = 4,
    seed: int = 42,
) -> list[int]:
    if optimizer_updates <= 0 or accumulation <= 0:
        raise ValueError("Update and accumulation counts must be positive")
    needed = optimizer_updates * accumulation
    population = tuple(int(value) for value in pair_indices)
    passes = math.ceil(needed / len(population))
    flattened = [
        pair
        for order in deterministic_population_orders(population, passes=passes, seed=seed)
        for pair in order
    ]
    result = flattened[:needed]
    if len(result) != needed:
        raise RuntimeError("Matched microbatch plan is incomplete")
    return result


def natural_epoch_orders(
    pair_indices: Sequence[int], *, epochs: int, seed: int = 42
) -> list[list[int]]:
    return deterministic_population_orders(pair_indices, passes=epochs, seed=seed)


def accumulation_groups(
    microbatches: Sequence[int], *, accumulation: int
) -> list[list[int]]:
    if accumulation <= 0:
        raise ValueError("Accumulation must be positive")
    return [
        list(microbatches[start : start + accumulation])
        for start in range(0, len(microbatches), accumulation)
    ]


def warmup_cosine_learning_rate(
    update: int,
    *,
    total_updates: int,
    warmup_updates: int,
    base_learning_rate: float,
    min_learning_rate: float,
) -> float:
    if not 1 <= update <= total_updates:
        raise ValueError("Update index lies outside the scheduler horizon")
    if not 0 <= warmup_updates <= total_updates:
        raise ValueError("Warmup update count is invalid")
    if not 0 <= min_learning_rate <= base_learning_rate:
        raise ValueError("Learning-rate bounds are invalid")
    if warmup_updates and update <= warmup_updates:
        return base_learning_rate * update / warmup_updates
    decay_updates = max(1, total_updates - warmup_updates)
    progress = min(1.0, (update - warmup_updates) / decay_updates)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learning_rate + (base_learning_rate - min_learning_rate) * cosine


def order_sha256(values: Iterable[int]) -> str:
    encoded = json.dumps([int(value) for value in values], separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
