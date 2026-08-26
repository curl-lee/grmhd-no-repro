"""Train-only residual scales for bounded normalized-space forecasts."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import CHANNELS
from .dataset import GRMHDPairedDataset, sha256_file
from .normalizer import GRMHDNormalizer


@dataclass(frozen=True)
class ResidualScaleStats:
    alpha: tuple[float, ...]
    raw_quantiles: tuple[float, ...]
    quantile: float
    multiplier: float
    min_alpha: float
    seed: int
    sample_count_per_channel: int
    source: str
    source_hdf5_checksum: str
    training_indices: tuple[int, ...]
    pair_starts: tuple[int, ...]
    channel_order: tuple[str, ...] = CHANNELS

    @classmethod
    def fit(
        cls,
        training_dataset: GRMHDPairedDataset,
        normalizer: GRMHDNormalizer,
        *,
        quantile: float = 0.999,
        multiplier: float = 1.0,
        min_alpha: float = 1e-6,
        max_samples_per_channel: int = 200_000,
        seed: int = 42,
    ) -> "ResidualScaleStats":
        if training_dataset.split.name != "train":
            raise ValueError("Residual scale may only be fit from a split named 'train'")
        if not 0 < quantile <= 1:
            raise ValueError("quantile must lie in (0, 1]")
        if multiplier <= 0 or min_alpha <= 0:
            raise ValueError("multiplier and min_alpha must be positive")
        if max_samples_per_channel <= 0:
            raise ValueError("max_samples_per_channel must be positive")
        training_indices = training_dataset.owned_snapshot_indices
        if tuple(normalizer.training_indices) != tuple(training_indices):
            raise ValueError("Normalizer training indices do not match residual-scale split")
        checksum = sha256_file(training_dataset.h5_path)
        if normalizer.source_hdf5_checksum != checksum:
            raise ValueError("Normalizer/data checksum mismatch during residual-scale fit")

        pair_count = len(training_dataset)
        if pair_count == 0:
            raise ValueError("Training dataset has no residual pairs")
        spatial_shape = training_dataset.snapshot_shape[1:]
        voxels_per_pair = int(np.prod(spatial_shape))
        total_values = pair_count * voxels_per_pair
        sample_count = min(int(max_samples_per_channel), total_values)
        choices: list[np.ndarray] = []
        samples = np.empty((len(CHANNELS), sample_count), dtype=np.float32)
        for channel in range(len(CHANNELS)):
            rng = np.random.default_rng(seed + 104729 * channel)
            choices.append(
                np.sort(rng.choice(total_values, size=sample_count, replace=False))
            )

        for pair_slot in range(pair_count):
            selected_by_channel: list[np.ndarray] = []
            any_selected = False
            lower = pair_slot * voxels_per_pair
            upper = lower + voxels_per_pair
            for channel_choices in choices:
                start = int(np.searchsorted(channel_choices, lower, side="left"))
                stop = int(np.searchsorted(channel_choices, upper, side="left"))
                local = channel_choices[start:stop] - lower
                selected_by_channel.append(local)
                any_selected |= len(local) > 0
            if not any_selected:
                continue
            pair = training_dataset[pair_slot]
            encoded_x = normalizer.encode_numpy(pair["x"].numpy(), channel_axis=0)
            encoded_y = normalizer.encode_numpy(pair["y"].numpy(), channel_axis=0)
            absolute_residual = np.abs(encoded_y - encoded_x).reshape(len(CHANNELS), -1)
            for channel, local in enumerate(selected_by_channel):
                if len(local) == 0:
                    continue
                start = int(np.searchsorted(choices[channel], lower, side="left"))
                stop = start + len(local)
                samples[channel, start:stop] = absolute_residual[channel, local]

        raw_quantiles = np.asarray(
            np.quantile(samples, quantile, axis=1), dtype=np.float64
        )
        alpha = np.maximum(raw_quantiles * multiplier, np.float64(min_alpha))
        if not np.all(np.isfinite(alpha)):
            raise FloatingPointError("Residual alpha contains NaN/Inf")
        return cls(
            alpha=tuple(float(value) for value in alpha),
            raw_quantiles=tuple(float(value) for value in raw_quantiles),
            quantile=float(quantile),
            multiplier=float(multiplier),
            min_alpha=float(min_alpha),
            seed=int(seed),
            sample_count_per_channel=sample_count,
            source="train_abs_quantile",
            source_hdf5_checksum=checksum,
            training_indices=tuple(training_indices),
            pair_starts=tuple(int(value) for value in training_dataset.pair_starts),
        )

    def as_dict(self) -> dict[str, Any]:
        output = asdict(self)
        output["alpha_by_channel"] = dict(zip(self.channel_order, self.alpha))
        output["raw_quantile_by_channel"] = dict(
            zip(self.channel_order, self.raw_quantiles)
        )
        return output

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ResidualScaleStats":
        values = dict(values)
        values.pop("alpha_by_channel", None)
        values.pop("raw_quantile_by_channel", None)
        for key in ("alpha", "raw_quantiles", "training_indices", "pair_starts", "channel_order"):
            values[key] = tuple(values[key])
        return cls(**values)

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        h5_path: str | Path,
        expected_training_indices: tuple[int, ...],
        expected_quantile: float,
        expected_multiplier: float,
    ) -> "ResidualScaleStats":
        stats = cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        stats.validate(
            h5_path=h5_path,
            expected_training_indices=expected_training_indices,
            expected_quantile=expected_quantile,
            expected_multiplier=expected_multiplier,
        )
        return stats

    def validate(
        self,
        *,
        h5_path: str | Path,
        expected_training_indices: tuple[int, ...],
        expected_quantile: float,
        expected_multiplier: float,
    ) -> None:
        if tuple(self.channel_order) != tuple(CHANNELS):
            raise ValueError("Residual scale channel order mismatch")
        if self.source != "train_abs_quantile":
            raise ValueError("Residual scale source mismatch")
        if self.source_hdf5_checksum != sha256_file(h5_path):
            raise ValueError("Residual scale/data checksum mismatch")
        if tuple(self.training_indices) != tuple(expected_training_indices):
            raise ValueError("Residual scale training indices mismatch")
        if self.quantile != float(expected_quantile):
            raise ValueError("Residual scale quantile mismatch")
        if self.multiplier != float(expected_multiplier):
            raise ValueError("Residual scale multiplier mismatch")
        alpha = np.asarray(self.alpha)
        if alpha.shape != (len(CHANNELS),) or np.any(alpha < self.min_alpha):
            raise ValueError("Residual alpha shape/floor mismatch")


def validate_residual_wrapper_metadata(
    metadata: dict[str, Any],
    *,
    h5_path: str | Path,
    expected_training_indices: tuple[int, ...],
    expected_config: dict[str, Any],
) -> ResidualScaleStats:
    """Reject checkpoint residual wrappers with mismatched data or configuration."""
    stored_config = metadata.get("config")
    if stored_config != expected_config:
        raise ValueError("Residual wrapper checkpoint config mismatch")
    stats = ResidualScaleStats.from_dict(metadata["stats"])
    stats.validate(
        h5_path=h5_path,
        expected_training_indices=expected_training_indices,
        expected_quantile=float(expected_config["residual_scale_quantile"]),
        expected_multiplier=float(expected_config["residual_scale_multiplier"]),
    )
    if list(stats.alpha) != list(metadata.get("alpha", ())):
        raise ValueError("Residual wrapper checkpoint alpha mismatch")
    return stats
