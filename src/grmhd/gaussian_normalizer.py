"""Gaussian-after-physical-transform ablation using the upstream normalizer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neuralop.data.transforms.normalizers import UnitGaussianNormalizer

from . import CHANNELS
from .dataset import GRMHDPairedDataset, sha256_file
from .normalizer import DEFAULT_EPSILON, TRANSFORMS


class TransformedGaussianNormalizer:
    """Keep GRMHD physical transforms, then apply upstream mean/std only.

    Unlike :class:`GRMHDNormalizer`, this controlled ablation has no robust
    median/MAD and no soft clipping.  Raw rho/press/B are never Gaussianized.
    """

    def __init__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        *,
        epsilon: np.ndarray = DEFAULT_EPSILON,
        training_indices: tuple[int, ...] = (),
        source_hdf5_checksum: str | None = None,
        sample_count: int | None = None,
        seed: int = 42,
    ) -> None:
        mean = np.asarray(mean, dtype=np.float64)
        std = np.asarray(std, dtype=np.float64)
        epsilon = np.asarray(epsilon, dtype=np.float64)
        if mean.shape != (8,) or std.shape != (8,) or epsilon.shape != (8,):
            raise ValueError("mean, std and epsilon must have shape (8,)")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
            raise ValueError("Gaussian statistics must be finite with positive std")
        self.mean = mean
        self.std = std
        self.epsilon = epsilon
        self.training_indices = tuple(int(value) for value in training_indices)
        self.source_hdf5_checksum = source_hdf5_checksum
        self.sample_count = sample_count
        self.seed = int(seed)
        self.unit_normalizer = UnitGaussianNormalizer(
            mean=torch.as_tensor(mean, dtype=torch.float32).reshape(1, 8, 1, 1, 1),
            std=torch.as_tensor(std, dtype=torch.float32).reshape(1, 8, 1, 1, 1),
            eps=1.0e-7,
        )

    @classmethod
    def fit(
        cls,
        training_dataset: GRMHDPairedDataset,
        *,
        max_samples_per_channel: int = 200_000,
        epsilon: np.ndarray = DEFAULT_EPSILON,
        seed: int = 42,
    ) -> "TransformedGaussianNormalizer":
        if training_dataset.split.name != "train":
            raise ValueError("Gaussian statistics require a split named 'train'")
        indices = training_dataset.owned_snapshot_indices
        voxels = int(np.prod(training_dataset.snapshot_shape[1:]))
        total = len(indices) * voxels
        sample_count = min(max_samples_per_channel, total)
        samples = np.empty((8, sample_count), dtype=np.float64)
        for channel in range(8):
            choices = np.sort(
                np.random.default_rng(seed + 104729 * channel).choice(
                    total, size=sample_count, replace=False
                )
            )
            for slot in np.unique(choices // voxels):
                mask = choices // voxels == slot
                local = choices[mask] % voxels
                start = int(np.searchsorted(choices, slot * voxels, side="left"))
                stop = start + int(mask.sum())
                raw = training_dataset.load_snapshot(indices[int(slot)])[channel].numpy().reshape(-1)
                samples[channel, start:stop] = raw[local]
            kind = TRANSFORMS[channel]
            if kind == "positive_log":
                samples[channel] = np.log10(
                    np.maximum(samples[channel] + epsilon[channel], np.finfo(np.float64).tiny)
                )
            elif kind == "signed_log":
                samples[channel] = np.sign(samples[channel]) * np.log10(
                    1.0 + np.abs(samples[channel]) / epsilon[channel]
                )
        means = np.mean(samples, axis=1)
        stds = np.std(samples, axis=1, ddof=1)
        stds = np.maximum(stds, 1.0e-6)
        return cls(
            means,
            stds,
            epsilon=np.asarray(epsilon),
            training_indices=indices,
            source_hdf5_checksum=sha256_file(training_dataset.h5_path),
            sample_count=sample_count,
            seed=seed,
        )

    def _pretransform(self, values: torch.Tensor) -> torch.Tensor:
        output = torch.empty_like(values)
        for channel, kind in enumerate(TRANSFORMS):
            item = values[:, channel]
            epsilon = torch.as_tensor(
                self.epsilon[channel], dtype=item.dtype, device=item.device
            )
            if kind == "positive_log":
                if torch.any(item < 0):
                    raise ValueError(f"{CHANNELS[channel]} contains negative values")
                output[:, channel] = torch.log10(
                    torch.clamp_min(item + epsilon, torch.finfo(item.dtype).tiny)
                )
            elif kind == "signed_log":
                output[:, channel] = torch.sign(item) * torch.log10(
                    1.0 + torch.abs(item) / epsilon
                )
            else:
                output[:, channel] = item
        return output

    def encode_tensor(self, values: torch.Tensor, channel_axis: int | None = None) -> torch.Tensor:
        if channel_axis not in (None, 1, -4) or values.ndim != 5 or values.shape[1] != 8:
            raise ValueError("Transformed Gaussian tensor encoding expects (batch,8,phi,theta,r)")
        transformed = self._pretransform(values)
        self.unit_normalizer = self.unit_normalizer.to(values.device)
        encoded = self.unit_normalizer.transform(transformed)
        if not torch.isfinite(encoded).all():
            raise FloatingPointError("Gaussian encode produced NaN/Inf")
        return encoded

    def decode_tensor(self, values: torch.Tensor, channel_axis: int | None = None) -> torch.Tensor:
        if channel_axis not in (None, 1, -4) or values.ndim != 5 or values.shape[1] != 8:
            raise ValueError("Transformed Gaussian tensor decoding expects (batch,8,phi,theta,r)")
        self.unit_normalizer = self.unit_normalizer.to(values.device)
        transformed = self.unit_normalizer.inverse_transform(values)
        output = torch.empty_like(values)
        for channel, kind in enumerate(TRANSFORMS):
            item = transformed[:, channel]
            epsilon = torch.as_tensor(
                self.epsilon[channel], dtype=item.dtype, device=item.device
            )
            if kind == "positive_log":
                output[:, channel] = torch.clamp_min(
                    torch.pow(10.0, item) - epsilon, torch.finfo(item.dtype).tiny
                )
            elif kind == "signed_log":
                output[:, channel] = torch.sign(item) * epsilon * (
                    torch.pow(10.0, torch.abs(item)) - 1.0
                )
            else:
                output[:, channel] = item
        if not torch.isfinite(output).all():
            raise FloatingPointError("Gaussian decode produced NaN/Inf")
        return output

    def as_dict(self) -> dict[str, Any]:
        return {
            "type": "upstream_gaussian_transformed",
            "upstream_class": type(self.unit_normalizer).__module__
            + "."
            + type(self.unit_normalizer).__name__,
            "physical_transforms": list(TRANSFORMS),
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "epsilon": self.epsilon.tolist(),
            "soft_clip": False,
            "training_indices": list(self.training_indices),
            "source_hdf5_checksum": self.source_hdf5_checksum,
            "sample_count_per_channel": self.sample_count,
            "seed": self.seed,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
