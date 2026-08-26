"""Optional, training-split-only statistical priors for GRMHD forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from . import CHANNELS
from .dataset import GRMHDPairedDataset
from .normalizer import GRMHDNormalizer


def _choices_by_channel(
    population: int, sample_count: int, channels: tuple[int, ...], seed: int
) -> dict[int, np.ndarray]:
    return {
        channel: np.sort(
            np.random.default_rng(seed + 104729 * channel).choice(
                population, size=min(sample_count, population), replace=False
            )
        )
        for channel in channels
    }


@dataclass(frozen=True)
class QuantileBounds:
    channels: tuple[int, ...]
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    quantiles: tuple[float, float]
    sample_count_per_channel: int
    seed: int
    space: str = "normalized"
    source_hdf5_checksum: str | None = None
    training_indices: tuple[int, ...] = ()

    @classmethod
    def fit(
        cls,
        dataset: GRMHDPairedDataset,
        normalizer: GRMHDNormalizer,
        *,
        channels: tuple[int, ...] = (3, 4),
        quantiles: tuple[float, float] = (0.001, 0.999),
        max_samples_per_channel: int = 100_000,
        seed: int = 42,
    ) -> "QuantileBounds":
        if dataset.split.name != "train":
            raise ValueError("Quantile bounds must be fit on the train split")
        if not 0 <= quantiles[0] < quantiles[1] <= 1:
            raise ValueError("Invalid quantiles")
        voxels = int(np.prod(dataset.snapshot_shape[1:]))
        indices = dataset.owned_snapshot_indices
        total = len(indices) * voxels
        sample_count = min(max_samples_per_channel, total)
        choices = _choices_by_channel(total, sample_count, channels, seed)
        samples = {channel: np.empty(sample_count, dtype=np.float32) for channel in channels}
        offsets = {channel: 0 for channel in channels}
        for slot, snapshot_index in enumerate(indices):
            encoded = normalizer.encode_tensor(dataset.load_snapshot(snapshot_index)).numpy()
            for channel in channels:
                selection = choices[channel]
                mask = selection // voxels == slot
                local = selection[mask] % voxels
                count = int(mask.sum())
                samples[channel][offsets[channel] : offsets[channel] + count] = encoded[
                    channel
                ].reshape(-1)[local]
                offsets[channel] += count
        lower = tuple(float(np.quantile(samples[channel], quantiles[0])) for channel in channels)
        upper = tuple(float(np.quantile(samples[channel], quantiles[1])) for channel in channels)
        return cls(
            channels,
            lower,
            upper,
            quantiles,
            sample_count,
            seed,
            source_hdf5_checksum=normalizer.source_hdf5_checksum,
            training_indices=dataset.owned_snapshot_indices,
        )

    def penalty(self, prediction: torch.Tensor) -> torch.Tensor:
        penalties = []
        for channel, lower, upper in zip(self.channels, self.lower, self.upper, strict=True):
            values = prediction[:, channel]
            penalties.append(torch.mean(torch.relu(lower - values).square()))
            penalties.append(torch.mean(torch.relu(values - upper).square()))
        return torch.stack(penalties).mean()

    def clamp(self, prediction: torch.Tensor) -> torch.Tensor:
        output = prediction.clone()
        for channel, lower, upper in zip(self.channels, self.lower, self.upper, strict=True):
            output[:, channel] = torch.clamp(output[:, channel], lower, upper)
        return output

    def as_dict(self) -> dict[str, Any]:
        return {
            "channels": [CHANNELS[channel] for channel in self.channels],
            "channel_indices": list(self.channels),
            "lower": list(self.lower),
            "upper": list(self.upper),
            "quantiles": list(self.quantiles),
            "sample_count_per_channel": self.sample_count_per_channel,
            "seed": self.seed,
            "space": self.space,
            "source_hdf5_checksum": self.source_hdf5_checksum,
            "training_indices": list(self.training_indices),
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "QuantileBounds":
        return cls(
            tuple(values["channel_indices"]),
            tuple(values["lower"]),
            tuple(values["upper"]),
            tuple(values["quantiles"]),
            int(values["sample_count_per_channel"]),
            int(values["seed"]),
            str(values["space"]),
            values.get("source_hdf5_checksum"),
            tuple(int(index) for index in values.get("training_indices", ())),
        )


@dataclass(frozen=True)
class ResidualEnvelope:
    lower: tuple[float, ...]
    upper: tuple[float, ...]
    quantiles: tuple[float, float]
    sample_count_per_channel: int
    seed: int
    space: str = "normalized one-step residual y-x"
    source_hdf5_checksum: str | None = None
    training_indices: tuple[int, ...] = ()

    @classmethod
    def fit(
        cls,
        dataset: GRMHDPairedDataset,
        normalizer: GRMHDNormalizer,
        *,
        quantiles: tuple[float, float] = (0.001, 0.999),
        max_samples_per_channel: int = 100_000,
        seed: int = 42,
    ) -> "ResidualEnvelope":
        if dataset.split.name != "train":
            raise ValueError("Residual envelope must be fit on the train split")
        if not 0 <= quantiles[0] < quantiles[1] <= 1:
            raise ValueError("Invalid quantiles")
        voxels = int(np.prod(dataset.snapshot_shape[1:]))
        total = len(dataset) * voxels
        sample_count = min(max_samples_per_channel, total)
        channels = tuple(range(8))
        choices = _choices_by_channel(total, sample_count, channels, seed)
        samples = np.empty((8, sample_count), dtype=np.float32)
        offsets = np.zeros(8, dtype=np.int64)
        for slot in range(len(dataset)):
            pair = dataset[slot]
            x = normalizer.encode_tensor(pair["x"])
            y = normalizer.encode_tensor(pair["y"])
            residual = (y - x).numpy()
            for channel in channels:
                selection = choices[channel]
                mask = selection // voxels == slot
                local = selection[mask] % voxels
                count = int(mask.sum())
                start = int(offsets[channel])
                samples[channel, start : start + count] = residual[channel].reshape(-1)[local]
                offsets[channel] += count
        lower = tuple(float(np.quantile(samples[channel], quantiles[0])) for channel in channels)
        upper = tuple(float(np.quantile(samples[channel], quantiles[1])) for channel in channels)
        return cls(
            lower,
            upper,
            quantiles,
            sample_count,
            seed,
            source_hdf5_checksum=normalizer.source_hdf5_checksum,
            training_indices=dataset.owned_snapshot_indices,
        )

    def penalty(self, prediction: torch.Tensor, input_state: torch.Tensor) -> torch.Tensor:
        residual = prediction - input_state[:, :8]
        lower = torch.as_tensor(self.lower, dtype=residual.dtype, device=residual.device).view(1, 8, 1, 1, 1)
        upper = torch.as_tensor(self.upper, dtype=residual.dtype, device=residual.device).view(1, 8, 1, 1, 1)
        return 0.5 * (
            torch.mean(torch.relu(lower - residual).square())
            + torch.mean(torch.relu(residual - upper).square())
        )

    def clamp(self, prediction: torch.Tensor, input_state: torch.Tensor) -> torch.Tensor:
        lower = torch.as_tensor(self.lower, dtype=prediction.dtype, device=prediction.device).view(1, 8, 1, 1, 1)
        upper = torch.as_tensor(self.upper, dtype=prediction.dtype, device=prediction.device).view(1, 8, 1, 1, 1)
        residual = torch.maximum(torch.minimum(prediction - input_state[:, :8], upper), lower)
        return input_state[:, :8] + residual

    def as_dict(self) -> dict[str, Any]:
        return {
            "channels": list(CHANNELS),
            "lower": list(self.lower),
            "upper": list(self.upper),
            "quantiles": list(self.quantiles),
            "sample_count_per_channel": self.sample_count_per_channel,
            "seed": self.seed,
            "space": self.space,
            "source_hdf5_checksum": self.source_hdf5_checksum,
            "training_indices": list(self.training_indices),
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ResidualEnvelope":
        return cls(
            tuple(values["lower"]),
            tuple(values["upper"]),
            tuple(values["quantiles"]),
            int(values["sample_count_per_channel"]),
            int(values["seed"]),
            str(values["space"]),
            values.get("source_hdf5_checksum"),
            tuple(int(index) for index in values.get("training_indices", ())),
        )
