"""Train-only hybrid physical residual targets for Round 3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import CHANNELS
from .dataset import GRMHDPairedDataset, sha256_file


SIGNED_CHANNELS = (0, 1, 2, 5, 6, 7)
POSITIVE_CHANNELS = (3, 4)


def _canonical_checksum(values: dict[str, Any]) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class HybridTargetStats:
    """Robust train-only scales and bounds for a physical residual target."""

    scale: tuple[float, ...]
    alpha: tuple[float, ...]
    epsilon: tuple[float, ...]
    alpha_quantile: float
    min_scale: float
    min_alpha: float
    sample_count_per_channel: int
    seed: int
    source_hdf5_checksum: str
    training_indices: tuple[int, ...]
    pair_starts: tuple[int, ...]
    stats_checksum: str
    channel_order: tuple[str, ...] = CHANNELS

    @staticmethod
    def _physical_delta(
        x: np.ndarray, y: np.ndarray, channel: int, epsilon: float
    ) -> np.ndarray:
        if channel in POSITIVE_CHANNELS:
            if np.any(x < 0) or np.any(y < 0):
                raise ValueError(f"{CHANNELS[channel]} must be non-negative")
            return np.log((y + epsilon) / (x + epsilon))
        return y - x

    @classmethod
    def fit(
        cls,
        training_dataset: GRMHDPairedDataset,
        *,
        alpha_quantile: float = 0.999,
        min_scale: float = 1.0e-12,
        min_alpha: float = 1.0e-3,
        max_samples_per_channel: int = 200_000,
        seed: int = 42,
    ) -> "HybridTargetStats":
        if training_dataset.split.name != "train":
            raise ValueError("Hybrid target statistics require a split named 'train'")
        if not 0 < alpha_quantile <= 1:
            raise ValueError("alpha_quantile must lie in (0, 1]")
        if min_scale <= 0 or min_alpha <= 0 or max_samples_per_channel <= 0:
            raise ValueError("Hybrid target floors and sample count must be positive")
        if len(training_dataset) == 0:
            raise ValueError("Training dataset has no next-step pairs")

        pair_count = len(training_dataset)
        voxels_per_pair = int(np.prod(training_dataset.snapshot_shape[1:]))
        total_values = pair_count * voxels_per_pair
        sample_count = min(max_samples_per_channel, total_values)
        samples = np.empty((len(CHANNELS), sample_count), dtype=np.float64)
        choices = [
            np.sort(
                np.random.default_rng(seed + 104729 * channel).choice(
                    total_values, size=sample_count, replace=False
                )
            )
            for channel in range(len(CHANNELS))
        ]

        # Positive epsilons are derived only from positive values owned by the
        # training split.  Signed channels do not use epsilon in this target.
        positive_min = {channel: np.inf for channel in POSITIVE_CHANNELS}
        for snapshot_index in training_dataset.owned_snapshot_indices:
            snapshot = training_dataset.load_snapshot(snapshot_index).numpy()
            for channel in POSITIVE_CHANNELS:
                values = snapshot[channel]
                positive = values[values > 0]
                if positive.size:
                    positive_min[channel] = min(
                        positive_min[channel], float(np.min(positive))
                    )
        epsilons = np.zeros(len(CHANNELS), dtype=np.float64)
        for channel in POSITIVE_CHANNELS:
            if not np.isfinite(positive_min[channel]):
                raise ValueError(f"Training {CHANNELS[channel]} has no positive values")
            epsilons[channel] = max(positive_min[channel] * 1.0e-3, np.finfo(np.float64).tiny)

        for pair_slot in range(pair_count):
            lower = pair_slot * voxels_per_pair
            upper = lower + voxels_per_pair
            selected = []
            any_selected = False
            for channel_choices in choices:
                start = int(np.searchsorted(channel_choices, lower, side="left"))
                stop = int(np.searchsorted(channel_choices, upper, side="left"))
                local = channel_choices[start:stop] - lower
                selected.append((start, stop, local))
                any_selected |= local.size > 0
            if not any_selected:
                continue
            pair = training_dataset[pair_slot]
            x = pair["x"].numpy().reshape(len(CHANNELS), -1).astype(np.float64)
            y = pair["y"].numpy().reshape(len(CHANNELS), -1).astype(np.float64)
            for channel, (start, stop, local) in enumerate(selected):
                if local.size == 0:
                    continue
                samples[channel, start:stop] = cls._physical_delta(
                    x[channel, local], y[channel, local], channel, epsilons[channel]
                )

        scales = np.empty(len(CHANNELS), dtype=np.float64)
        alphas = np.empty(len(CHANNELS), dtype=np.float64)
        for channel in range(len(CHANNELS)):
            median = float(np.median(samples[channel]))
            robust_scale = float(
                1.4826 * np.median(np.abs(samples[channel] - median))
            )
            scales[channel] = max(robust_scale, min_scale)
            standardized = samples[channel] / scales[channel]
            alphas[channel] = max(
                float(np.quantile(np.abs(standardized), alpha_quantile)), min_alpha
            )
        if not np.all(np.isfinite(scales)) or not np.all(np.isfinite(alphas)):
            raise FloatingPointError("Hybrid target statistics contain NaN/Inf")

        core = {
            "scale": [float(value) for value in scales],
            "alpha": [float(value) for value in alphas],
            "epsilon": [float(value) for value in epsilons],
            "alpha_quantile": float(alpha_quantile),
            "min_scale": float(min_scale),
            "min_alpha": float(min_alpha),
            "sample_count_per_channel": int(sample_count),
            "seed": int(seed),
            "source_hdf5_checksum": sha256_file(training_dataset.h5_path),
            "training_indices": list(training_dataset.owned_snapshot_indices),
            "pair_starts": [int(value) for value in training_dataset.pair_starts],
            "channel_order": list(CHANNELS),
        }
        return cls(
            scale=tuple(core["scale"]),
            alpha=tuple(core["alpha"]),
            epsilon=tuple(core["epsilon"]),
            alpha_quantile=float(alpha_quantile),
            min_scale=float(min_scale),
            min_alpha=float(min_alpha),
            sample_count_per_channel=sample_count,
            seed=int(seed),
            source_hdf5_checksum=core["source_hdf5_checksum"],
            training_indices=tuple(core["training_indices"]),
            pair_starts=tuple(core["pair_starts"]),
            stats_checksum=_canonical_checksum(core),
        )

    def _broadcast(
        self, values: tuple[float, ...], reference: torch.Tensor
    ) -> torch.Tensor:
        return torch.as_tensor(values, dtype=reference.dtype, device=reference.device).reshape(
            1, len(CHANNELS), *((1,) * (reference.ndim - 2))
        )

    def encode_target(self, physical_input: torch.Tensor, physical_target: torch.Tensor) -> torch.Tensor:
        if physical_input.shape != physical_target.shape or physical_input.shape[1] != len(CHANNELS):
            raise ValueError("Hybrid target expects matching batched eight-channel states")
        scale = self._broadcast(self.scale, physical_input)
        epsilon = self._broadcast(self.epsilon, physical_input)
        target = torch.empty_like(physical_input)
        target[:, SIGNED_CHANNELS] = (
            physical_target[:, SIGNED_CHANNELS] - physical_input[:, SIGNED_CHANNELS]
        ) / scale[:, SIGNED_CHANNELS]
        target[:, POSITIVE_CHANNELS] = torch.log(
            (physical_target[:, POSITIVE_CHANNELS] + epsilon[:, POSITIVE_CHANNELS])
            / (physical_input[:, POSITIVE_CHANNELS] + epsilon[:, POSITIVE_CHANNELS])
        ) / scale[:, POSITIVE_CHANNELS]
        if not torch.isfinite(target).all():
            raise FloatingPointError("Hybrid target encoding produced NaN/Inf")
        return target

    def bounded_target_prediction(self, raw_output: torch.Tensor) -> torch.Tensor:
        alpha = self._broadcast(self.alpha, raw_output)
        bounded = alpha * torch.tanh(raw_output / alpha)
        if not torch.isfinite(bounded).all():
            raise FloatingPointError("Bounded hybrid target produced NaN/Inf")
        return bounded

    def reconstruct(self, physical_input: torch.Tensor, raw_output: torch.Tensor) -> torch.Tensor:
        if physical_input.shape != raw_output.shape or physical_input.shape[1] != len(CHANNELS):
            raise ValueError("Hybrid reconstruction expects matching batched eight-channel tensors")
        bounded = self.bounded_target_prediction(raw_output)
        scale = self._broadcast(self.scale, raw_output)
        epsilon = self._broadcast(self.epsilon, raw_output)
        prediction = torch.empty_like(physical_input)
        prediction[:, SIGNED_CHANNELS] = physical_input[:, SIGNED_CHANNELS] + (
            scale[:, SIGNED_CHANNELS] * bounded[:, SIGNED_CHANNELS]
        )
        positive_prediction = (
            physical_input[:, POSITIVE_CHANNELS] + epsilon[:, POSITIVE_CHANNELS]
        ) * torch.exp(
            scale[:, POSITIVE_CHANNELS] * bounded[:, POSITIVE_CHANNELS]
        ) - epsilon[:, POSITIVE_CHANNELS]
        # Preserve the zero-head initialization bit-for-bit.  Evaluating
        # ``(x + eps) - eps`` otherwise introduces a small round-off even
        # though the mathematical residual is exactly zero.
        prediction[:, POSITIVE_CHANNELS] = torch.where(
            bounded[:, POSITIVE_CHANNELS] == 0,
            physical_input[:, POSITIVE_CHANNELS],
            positive_prediction,
        )
        prediction[:, POSITIVE_CHANNELS] = torch.clamp_min(
            prediction[:, POSITIVE_CHANNELS], torch.finfo(prediction.dtype).tiny
        )
        if not torch.isfinite(prediction).all():
            raise FloatingPointError("Hybrid reconstruction produced NaN/Inf")
        return prediction

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["scale_by_channel"] = dict(zip(self.channel_order, self.scale))
        values["alpha_by_channel"] = dict(zip(self.channel_order, self.alpha))
        values["epsilon_by_channel"] = dict(zip(self.channel_order, self.epsilon))
        return values

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "HybridTargetStats":
        values = dict(values)
        values.pop("scale_by_channel", None)
        values.pop("alpha_by_channel", None)
        values.pop("epsilon_by_channel", None)
        for key in (
            "scale",
            "alpha",
            "epsilon",
            "training_indices",
            "pair_starts",
            "channel_order",
        ):
            values[key] = tuple(values[key])
        return cls(**values)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        h5_path: str | Path,
        expected_training_indices: tuple[int, ...],
    ) -> "HybridTargetStats":
        stats = cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        stats.validate(h5_path=h5_path, expected_training_indices=expected_training_indices)
        return stats

    def validate(
        self,
        *,
        h5_path: str | Path,
        expected_training_indices: tuple[int, ...],
    ) -> None:
        if self.channel_order != CHANNELS:
            raise ValueError("Hybrid target channel order mismatch")
        if self.source_hdf5_checksum != sha256_file(h5_path):
            raise ValueError("Hybrid target HDF5 checksum mismatch")
        if self.training_indices != tuple(expected_training_indices):
            raise ValueError("Hybrid target training indices mismatch")
        values = self.as_dict()
        core = {
            key: values[key]
            for key in (
                "scale",
                "alpha",
                "epsilon",
                "alpha_quantile",
                "min_scale",
                "min_alpha",
                "sample_count_per_channel",
                "seed",
                "source_hdf5_checksum",
                "training_indices",
                "pair_starts",
                "channel_order",
            )
        }
        if _canonical_checksum(core) != self.stats_checksum:
            raise ValueError("Hybrid target statistics checksum mismatch")
