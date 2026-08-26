"""Robust paper-style channel normalization for GRMHD states."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import warnings

import h5py
import numpy as np
import torch

from . import CHANNELS
from .dataset import GRMHDPairedDataset, sha256_file


TRANSFORMS = ("signed_log", "signed_log", "signed_log", "positive_log", "positive_log", "linear", "linear", "linear")
DEFAULT_EPSILON = np.asarray([1e-6, 1e-6, 1e-6, 1e-12, 1e-14, 0.0, 0.0, 0.0], dtype=np.float64)


def validate_stats_bundle(
    stats_dir: str | Path,
    h5_path: str | Path,
    *,
    expected_training_indices: tuple[int, ...],
    expected_window_name: str,
) -> dict[str, Any]:
    """Validate every train-only artifact against data, indices, and window."""
    stats_dir = Path(stats_dir)
    bundle_path = stats_dir / "priors.json"
    if not bundle_path.exists():
        raise FileNotFoundError(f"Stats bundle metadata is missing: {bundle_path}")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    actual_checksum = sha256_file(h5_path)
    expected_indices = list(expected_training_indices)
    if bundle.get("window_name") != expected_window_name:
        raise ValueError(
            f"Stats window mismatch: bundle={bundle.get('window_name')!r}, "
            f"expected={expected_window_name!r}"
        )
    if bundle.get("source_hdf5_checksum") != actual_checksum:
        raise ValueError("Stats bundle/data checksum mismatch")
    if bundle.get("train_snapshot_indices") != expected_indices:
        raise ValueError("Stats bundle training indices mismatch")
    artifact_keys = (
        "normalizer",
        "radial_baseline",
        "quantile_bounds",
        "residual_envelope",
        "shell_metadata",
        "velocity_roi",
    )
    missing = [key for key in artifact_keys if key not in bundle]
    if missing:
        raise ValueError(f"Stats bundle is missing artifacts: {missing}")
    for key in ("normalizer", "quantile_bounds", "residual_envelope"):
        artifact = bundle[key]
        if artifact.get("source_hdf5_checksum") != actual_checksum:
            raise ValueError(f"{key} checksum mismatch")
        artifact_indices = artifact.get("training_indices")
        if artifact_indices != expected_indices:
            raise ValueError(f"{key} training indices mismatch")
    normalizer_path = stats_dir / "normalizer_stats.npz"
    normalizer = GRMHDNormalizer.load(normalizer_path)
    if normalizer.source_hdf5_checksum != actual_checksum:
        raise ValueError("normalizer_stats.npz checksum metadata mismatch")
    if list(normalizer.training_indices) != expected_indices:
        raise ValueError("normalizer_stats.npz training indices mismatch")
    return {
        "status": "passed",
        "window_name": expected_window_name,
        "source_hdf5": str(Path(h5_path).resolve()),
        "source_hdf5_checksum": actual_checksum,
        "train_snapshot_indices": expected_indices,
        "normalizer_stats_path": str(normalizer_path.resolve()),
        "normalizer_stats_checksum": sha256_file(normalizer_path),
        "artifacts": {
            key: {
                "present": True,
                "source_hdf5_checksum": actual_checksum,
                "training_indices": expected_indices,
            }
            for key in artifact_keys
        },
    }


class GRMHDNormalizer:
    def __init__(
        self,
        median: np.ndarray,
        scale: np.ndarray,
        *,
        epsilon: np.ndarray = DEFAULT_EPSILON,
        gamma: float = 6.0,
        min_scale: float = 1e-6,
        seed: int = 42,
        sample_count: int | None = None,
        training_indices: tuple[int, ...] = (),
        radial_baseline: dict[str, Any] | None = None,
        source_hdf5_checksum: str | None = None,
    ) -> None:
        self.median = np.asarray(median, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        self.epsilon = np.asarray(epsilon, dtype=np.float64)
        if self.median.shape != (8,) or self.scale.shape != (8,) or self.epsilon.shape != (8,):
            raise ValueError("median, scale, and epsilon must each have shape (8,)")
        if gamma <= 0 or min_scale <= 0:
            raise ValueError("gamma and min_scale must be positive")
        if not np.all(np.isfinite(self.median)) or not np.all(np.isfinite(self.scale)):
            raise ValueError("normalizer statistics must be finite")
        if np.any(self.scale < min_scale):
            raise ValueError("scale values must respect min_scale")
        self.gamma = float(gamma)
        self.min_scale = float(min_scale)
        self.seed = int(seed)
        self.sample_count = sample_count
        self.training_indices = tuple(int(index) for index in training_indices)
        self.radial_baseline = radial_baseline
        self.source_hdf5_checksum = source_hdf5_checksum

    def _radial_baseline_numpy(self, channel: int, radial_size: int) -> np.ndarray | None:
        if self.radial_baseline is None or CHANNELS[channel] not in {"rho", "press"}:
            return None
        r = np.asarray(self.radial_baseline["r"], dtype=np.float64)
        if len(r) != radial_size:
            raise ValueError(
                f"Radial baseline was fit at Nr={len(r)} but input has Nr={radial_size}"
            )
        coefficients = self.radial_baseline["channels"][CHANNELS[channel]]
        return coefficients["slope"] * np.log10(r) + coefficients["intercept"]

    @staticmethod
    def _pretransform_numpy(values: np.ndarray, channel: int, epsilon: np.ndarray) -> np.ndarray:
        kind = TRANSFORMS[channel]
        if kind == "positive_log":
            if np.any(values < 0):
                raise ValueError(f"{CHANNELS[channel]} contains negative values")
            return np.log10(np.maximum(values + epsilon[channel], np.finfo(np.float64).tiny))
        if kind == "signed_log":
            return np.sign(values) * np.log10(1.0 + np.abs(values) / epsilon[channel])
        return values

    @classmethod
    def fit(
        cls,
        training_dataset: GRMHDPairedDataset,
        *,
        max_samples_per_channel: int = 200_000,
        epsilon: np.ndarray = DEFAULT_EPSILON,
        gamma: float = 6.0,
        min_scale: float = 1e-6,
        seed: int = 42,
        radial_baseline: bool = False,
    ) -> "GRMHDNormalizer":
        if training_dataset.split.name != "train":
            raise ValueError("Statistics may only be fit from a split named 'train'")
        if max_samples_per_channel <= 0:
            raise ValueError("max_samples_per_channel must be positive")
        indices = training_dataset.owned_snapshot_indices
        if not indices:
            raise ValueError("Training split has no snapshots")
        radial_baseline_metadata: dict[str, Any] | None = None
        source_hdf5_checksum = sha256_file(training_dataset.h5_path)
        with h5py.File(training_dataset.h5_path, "r") as handle:
            spatial_shape = tuple(int(value) for value in handle["snapshots"].shape[2:])
            voxels_per_snapshot = int(np.prod(spatial_shape))
            total_voxels = len(indices) * voxels_per_snapshot
            sample_count = min(max_samples_per_channel, total_voxels)
            if radial_baseline:
                r = np.asarray(training_dataset.coords["r"], dtype=np.float64)
                log_r = np.log10(r)
                coefficients: dict[str, dict[str, float]] = {}
                for channel in (3, 4):
                    snapshot_profiles = []
                    for snapshot_index in indices:
                        raw = np.asarray(
                            handle["snapshots"][snapshot_index, channel], dtype=np.float64
                        )
                        transformed = cls._pretransform_numpy(raw, channel, np.asarray(epsilon))
                        snapshot_profiles.append(np.median(transformed, axis=(0, 1)))
                    profile = np.median(np.stack(snapshot_profiles), axis=0)
                    slope, intercept = np.polyfit(log_r, profile, deg=1)
                    coefficients[CHANNELS[channel]] = {
                        "slope": float(slope),
                        "intercept": float(intercept),
                    }
                radial_baseline_metadata = {
                    "enabled": True,
                    "radial_coordinate": "log10(r)",
                    "fit_space": "raw positive-log before robust normalization",
                    "fit_statistic": "median over phi/theta, then median over training snapshots",
                    "fit_separately": True,
                    "r": r.tolist(),
                    "channels": coefficients,
                }
            medians = np.empty(8, dtype=np.float64)
            scales = np.empty(8, dtype=np.float64)
            for channel in range(8):
                rng = np.random.default_rng(seed + 104729 * channel)
                flat_choices = np.sort(
                    rng.choice(total_voxels, size=sample_count, replace=False)
                )
                samples = np.empty(sample_count, dtype=np.float64)
                sample_radial_indices = np.empty(sample_count, dtype=np.int64)
                destination = 0
                snapshot_slots = flat_choices // voxels_per_snapshot
                local_indices = flat_choices % voxels_per_snapshot
                for slot in np.unique(snapshot_slots):
                    mask = snapshot_slots == slot
                    selected = local_indices[mask]
                    snapshot_index = indices[int(slot)]
                    values = np.asarray(
                        handle["snapshots"][snapshot_index, channel], dtype=np.float64
                    ).reshape(-1)
                    count = int(mask.sum())
                    samples[destination : destination + count] = values[selected]
                    sample_radial_indices[destination : destination + count] = (
                        selected % spatial_shape[-1]
                    )
                    destination += count
                transformed = cls._pretransform_numpy(samples, channel, np.asarray(epsilon))
                if radial_baseline_metadata is not None and channel in (3, 4):
                    coefficients = radial_baseline_metadata["channels"][CHANNELS[channel]]
                    transformed = transformed - (
                        coefficients["slope"]
                        * np.log10(np.asarray(radial_baseline_metadata["r"])[sample_radial_indices])
                        + coefficients["intercept"]
                    )
                median = float(np.median(transformed))
                mad_scale = float(1.4826 * np.median(np.abs(transformed - median)))
                if not np.isfinite(mad_scale) or mad_scale < min_scale:
                    warnings.warn(
                        f"Degenerate/constant training channel {CHANNELS[channel]}: "
                        f"MAD scale={mad_scale!r}; using stable fallback {min_scale}.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    mad_scale = min_scale
                medians[channel] = median
                scales[channel] = mad_scale
        return cls(
            medians,
            scales,
            epsilon=np.asarray(epsilon),
            gamma=gamma,
            min_scale=min_scale,
            seed=seed,
            sample_count=sample_count,
            training_indices=indices,
            radial_baseline=radial_baseline_metadata,
            source_hdf5_checksum=source_hdf5_checksum,
        )

    @staticmethod
    def _channel_axis(shape: tuple[int, ...], channel_axis: int | None) -> int:
        if channel_axis is not None:
            axis = channel_axis % len(shape)
            if shape[axis] != 8:
                raise ValueError(f"channel_axis {channel_axis} has size {shape[axis]}, not 8")
            return axis
        if shape[0] == 8:
            return 0
        if len(shape) > 1 and shape[1] == 8:
            return 1
        raise ValueError(f"Cannot infer 8-channel axis from shape {shape}")

    def encode_numpy(self, values: np.ndarray, channel_axis: int | None = None) -> np.ndarray:
        array = np.asarray(values)
        axis = self._channel_axis(array.shape, channel_axis)
        output = np.empty_like(array, dtype=np.result_type(array.dtype, np.float32))
        for channel in range(8):
            selection = [slice(None)] * array.ndim
            selection[axis] = channel
            key = tuple(selection)
            transformed = self._pretransform_numpy(
                np.asarray(array[key], dtype=np.float64), channel, self.epsilon
            )
            baseline = self._radial_baseline_numpy(channel, transformed.shape[-1])
            if baseline is not None:
                transformed = transformed - baseline.reshape(
                    (1,) * (transformed.ndim - 1) + (len(baseline),)
                )
            z = (transformed - self.median[channel]) / self.scale[channel]
            output[key] = self.gamma * np.tanh(z / self.gamma)
        if not np.isfinite(output).all():
            raise FloatingPointError("Normalizer encode produced NaN/Inf")
        return output

    def decode_numpy(self, values: np.ndarray, channel_axis: int | None = None) -> np.ndarray:
        array = np.asarray(values)
        axis = self._channel_axis(array.shape, channel_axis)
        output = np.empty_like(array, dtype=np.result_type(array.dtype, np.float32))
        clipped = np.clip(array, -0.99 * self.gamma, 0.99 * self.gamma)
        for channel in range(8):
            selection = [slice(None)] * array.ndim
            selection[axis] = channel
            key = tuple(selection)
            z = self.gamma * np.arctanh(np.asarray(clipped[key], dtype=np.float64) / self.gamma)
            transformed = z * self.scale[channel] + self.median[channel]
            baseline = self._radial_baseline_numpy(channel, transformed.shape[-1])
            if baseline is not None:
                transformed = transformed + baseline.reshape(
                    (1,) * (transformed.ndim - 1) + (len(baseline),)
                )
            kind = TRANSFORMS[channel]
            if kind == "positive_log":
                decoded = np.power(10.0, transformed) - self.epsilon[channel]
                decoded = np.maximum(decoded, np.finfo(output.dtype).tiny)
            elif kind == "signed_log":
                decoded = np.sign(transformed) * self.epsilon[channel] * (
                    np.power(10.0, np.abs(transformed)) - 1.0
                )
            else:
                decoded = transformed
            output[key] = decoded
        if not np.isfinite(output).all():
            raise FloatingPointError("Normalizer decode produced NaN/Inf")
        return output

    def encode_tensor(self, values: torch.Tensor, channel_axis: int | None = None) -> torch.Tensor:
        axis = self._channel_axis(tuple(values.shape), channel_axis)
        output = torch.empty_like(values)
        for channel in range(8):
            selection = [slice(None)] * values.ndim
            selection[axis] = channel
            key = tuple(selection)
            item = values[key]
            kind = TRANSFORMS[channel]
            epsilon = torch.as_tensor(self.epsilon[channel], dtype=item.dtype, device=item.device)
            if kind == "positive_log":
                if torch.any(item < 0):
                    raise ValueError(f"{CHANNELS[channel]} contains negative values")
                transformed = torch.log10(torch.clamp_min(item + epsilon, torch.finfo(item.dtype).tiny))
            elif kind == "signed_log":
                transformed = torch.sign(item) * torch.log10(1.0 + torch.abs(item) / epsilon)
            else:
                transformed = item
            baseline = self._radial_baseline_numpy(channel, transformed.shape[-1])
            if baseline is not None:
                transformed = transformed - torch.as_tensor(
                    baseline,
                    dtype=transformed.dtype,
                    device=transformed.device,
                ).reshape((1,) * (transformed.ndim - 1) + (len(baseline),))
            median = torch.as_tensor(self.median[channel], dtype=item.dtype, device=item.device)
            scale = torch.as_tensor(self.scale[channel], dtype=item.dtype, device=item.device)
            z = (transformed - median) / scale
            output[key] = self.gamma * torch.tanh(z / self.gamma)
        if not torch.isfinite(output).all():
            raise FloatingPointError("Normalizer encode produced NaN/Inf")
        return output

    def decode_tensor(self, values: torch.Tensor, channel_axis: int | None = None) -> torch.Tensor:
        axis = self._channel_axis(tuple(values.shape), channel_axis)
        output = torch.empty_like(values)
        clipped = torch.clamp(values, -0.99 * self.gamma, 0.99 * self.gamma)
        for channel in range(8):
            selection = [slice(None)] * values.ndim
            selection[axis] = channel
            key = tuple(selection)
            item = clipped[key]
            median = torch.as_tensor(self.median[channel], dtype=item.dtype, device=item.device)
            scale = torch.as_tensor(self.scale[channel], dtype=item.dtype, device=item.device)
            transformed = self.gamma * torch.atanh(item / self.gamma) * scale + median
            baseline = self._radial_baseline_numpy(channel, transformed.shape[-1])
            if baseline is not None:
                transformed = transformed + torch.as_tensor(
                    baseline,
                    dtype=transformed.dtype,
                    device=transformed.device,
                ).reshape((1,) * (transformed.ndim - 1) + (len(baseline),))
            epsilon = torch.as_tensor(self.epsilon[channel], dtype=item.dtype, device=item.device)
            kind = TRANSFORMS[channel]
            if kind == "positive_log":
                decoded = torch.pow(10.0, transformed) - epsilon
                decoded = torch.clamp_min(decoded, torch.finfo(item.dtype).tiny)
            elif kind == "signed_log":
                decoded = torch.sign(transformed) * epsilon * (
                    torch.pow(10.0, torch.abs(transformed)) - 1.0
                )
            else:
                decoded = transformed
            output[key] = decoded
        if not torch.isfinite(output).all():
            raise FloatingPointError("Normalizer decode produced NaN/Inf")
        return output

    def encode(self, values: np.ndarray | torch.Tensor, channel_axis: int | None = None):
        return (
            self.encode_tensor(values, channel_axis)
            if isinstance(values, torch.Tensor)
            else self.encode_numpy(values, channel_axis)
        )

    def decode(self, values: np.ndarray | torch.Tensor, channel_axis: int | None = None):
        return (
            self.decode_tensor(values, channel_axis)
            if isinstance(values, torch.Tensor)
            else self.decode_numpy(values, channel_axis)
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "channels": list(CHANNELS),
            "transforms": list(TRANSFORMS),
            "gamma": self.gamma,
            "min_scale": self.min_scale,
            "seed": self.seed,
            "sample_count_per_channel": self.sample_count,
            "training_indices": list(self.training_indices),
            "radial_baseline": self.radial_baseline,
            "source_hdf5_checksum": self.source_hdf5_checksum,
        }
        np.savez(
            path,
            median=self.median,
            scale=self.scale,
            epsilon=self.epsilon,
            gamma=np.asarray(self.gamma),
            min_scale=np.asarray(self.min_scale),
            seed=np.asarray(self.seed),
            sample_count=np.asarray(-1 if self.sample_count is None else self.sample_count),
            training_indices=np.asarray(self.training_indices, dtype=np.int64),
            channels=np.asarray(CHANNELS),
            transforms=np.asarray(TRANSFORMS),
            metadata_json=np.asarray(json.dumps(metadata)),
            radial_baseline_json=np.asarray(json.dumps(self.radial_baseline)),
            source_hdf5_checksum=np.asarray(self.source_hdf5_checksum or ""),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        h5_path: str | Path | None = None,
        expected_training_indices: tuple[int, ...] | None = None,
    ) -> "GRMHDNormalizer":
        with np.load(path, allow_pickle=False) as values:
            channels = tuple(str(item) for item in values["channels"])
            transforms = tuple(str(item) for item in values["transforms"])
            if channels != CHANNELS or transforms != TRANSFORMS:
                raise ValueError("Saved normalizer channel schema is incompatible")
            sample_count = int(values["sample_count"])
            normalizer = cls(
                values["median"],
                values["scale"],
                epsilon=values["epsilon"],
                gamma=float(values["gamma"]),
                min_scale=float(values["min_scale"]),
                seed=int(values["seed"]),
                sample_count=None if sample_count < 0 else sample_count,
                training_indices=tuple(int(item) for item in values["training_indices"]),
                radial_baseline=json.loads(str(values["radial_baseline_json"])),
                source_hdf5_checksum=(
                    str(values["source_hdf5_checksum"])
                    if "source_hdf5_checksum" in values and str(values["source_hdf5_checksum"])
                    else None
                ),
            )
        if h5_path is not None:
            normalizer.validate_compatibility(
                h5_path, expected_training_indices=expected_training_indices
            )
        elif expected_training_indices is not None:
            if tuple(expected_training_indices) != normalizer.training_indices:
                raise ValueError("Normalizer training indices are incompatible")
        return normalizer

    def validate_compatibility(
        self,
        h5_path: str | Path,
        *,
        expected_training_indices: tuple[int, ...] | None = None,
    ) -> None:
        if self.source_hdf5_checksum is None:
            raise ValueError("Normalizer has no source HDF5 checksum; refusing silent reuse")
        actual_checksum = sha256_file(h5_path)
        if actual_checksum != self.source_hdf5_checksum:
            raise ValueError(
                "Normalizer/data checksum mismatch: "
                f"stats={self.source_hdf5_checksum}, data={actual_checksum}"
            )
        if (
            expected_training_indices is not None
            and tuple(expected_training_indices) != self.training_indices
        ):
            raise ValueError(
                "Normalizer training indices mismatch: "
                f"stats={self.training_indices}, expected={tuple(expected_training_indices)}"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "channels": list(CHANNELS),
            "transforms": list(TRANSFORMS),
            "epsilon": self.epsilon.tolist(),
            "median": self.median.tolist(),
            "scale": self.scale.tolist(),
            "gamma": self.gamma,
            "min_scale": self.min_scale,
            "seed": self.seed,
            "sample_count_per_channel": self.sample_count,
            "training_indices": list(self.training_indices),
            "radial_baseline": self.radial_baseline,
            "source_hdf5_checksum": self.source_hdf5_checksum,
        }
