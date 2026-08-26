"""Independent Appendix-C preprocessing for the reduced paper protocol.

This module deliberately does not load or subclass the Round 1--3 normalizer.
It fits epsilon, median, and MAD statistics from the explicitly supplied train
snapshot indices and stores enough provenance to reject cross-protocol reuse.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable
import warnings

import h5py
import numpy as np
import torch

from . import CHANNELS
from .dataset import sha256_file


PAPER_TRANSFORMS = (
    "signed_log",
    "signed_log",
    "signed_log",
    "positive_log",
    "positive_log",
    "linear",
    "linear",
    "linear",
)
PAPER_GAMMA = 6.0
PAPER_INVERSE_CLAMP_FRACTION = 0.99
PAPER_MAD_MULTIPLIER = 1.4826
PAPER_MIN_SCALE = 1e-6


def _decode_strings(values: np.ndarray) -> tuple[str, ...]:
    return tuple(value.decode() if isinstance(value, bytes) else str(value) for value in values)


class PaperPreprocessor:
    """Paper channel transforms, robust scaling, and reversible soft clipping."""

    def __init__(
        self,
        epsilon: np.ndarray,
        median: np.ndarray,
        scale: np.ndarray,
        *,
        gamma: float = PAPER_GAMMA,
        inverse_clamp_fraction: float = PAPER_INVERSE_CLAMP_FRACTION,
        mad_multiplier: float = PAPER_MAD_MULTIPLIER,
        min_scale: float = PAPER_MIN_SCALE,
        invalid_epsilon_fallback: float = 1e-30,
        training_indices: Iterable[int] = (),
        source_hdf5_checksum: str,
        protocol_name: str,
        thermal_channel: str = "press",
        paper_adaptation: bool = True,
        eos_conversion: str = "disabled_unverified_gamma",
        fit_scope: str = "all_train_voxels",
        fit_warnings: Iterable[dict[str, Any]] = (),
        sample_count_per_channel: int | None = None,
    ) -> None:
        self.epsilon = np.asarray(epsilon, dtype=np.float64)
        self.median = np.asarray(median, dtype=np.float64)
        self.scale = np.asarray(scale, dtype=np.float64)
        if any(values.shape != (len(CHANNELS),) for values in (self.epsilon, self.median, self.scale)):
            raise ValueError("epsilon, median, and scale must each have shape (8,)")
        if not np.all(np.isfinite(self.epsilon)) or np.any(self.epsilon[:5] <= 0):
            raise ValueError("Paper log-channel epsilons must be positive and finite")
        if np.any(self.epsilon[5:] != 0):
            raise ValueError("Linear velocity channels must have epsilon=0")
        if not np.all(np.isfinite(self.median)) or not np.all(np.isfinite(self.scale)):
            raise ValueError("Paper robust statistics must be finite")
        if gamma != PAPER_GAMMA:
            raise ValueError("Paper preprocessing requires gamma=6")
        if not 0 < inverse_clamp_fraction < 1:
            raise ValueError("inverse_clamp_fraction must lie in (0,1)")
        if min_scale <= 0 or np.any(self.scale < min_scale):
            raise ValueError("Paper scales must respect the positive min_scale")
        if invalid_epsilon_fallback <= 0 or not np.isfinite(invalid_epsilon_fallback):
            raise ValueError("invalid_epsilon_fallback must be positive and finite")
        if thermal_channel != "press" or not paper_adaptation:
            raise ValueError("This protocol requires explicit press paper adaptation")
        if eos_conversion != "disabled_unverified_gamma":
            raise ValueError("Unverified EOS conversion cannot be enabled")
        self.gamma = float(gamma)
        self.inverse_clamp_fraction = float(inverse_clamp_fraction)
        self.mad_multiplier = float(mad_multiplier)
        self.min_scale = float(min_scale)
        self.invalid_epsilon_fallback = float(invalid_epsilon_fallback)
        self.training_indices = tuple(int(index) for index in training_indices)
        if not self.training_indices:
            raise ValueError("Paper preprocessor requires non-empty training indices")
        if tuple(sorted(set(self.training_indices))) != self.training_indices:
            raise ValueError("Training indices must be unique and strictly increasing")
        self.source_hdf5_checksum = str(source_hdf5_checksum)
        if not self.source_hdf5_checksum:
            raise ValueError("Paper preprocessor requires a source HDF5 checksum")
        self.protocol_name = str(protocol_name)
        self.thermal_channel = thermal_channel
        self.paper_adaptation = bool(paper_adaptation)
        self.eos_conversion = eos_conversion
        self.fit_scope = str(fit_scope)
        self.fit_warnings = tuple(dict(item) for item in fit_warnings)
        self.sample_count_per_channel = (
            None if sample_count_per_channel is None else int(sample_count_per_channel)
        )

    @staticmethod
    def _channel_axis(shape: tuple[int, ...], channel_axis: int | None) -> int:
        if channel_axis is not None:
            axis = channel_axis % len(shape)
            if shape[axis] != len(CHANNELS):
                raise ValueError(f"channel_axis {channel_axis} has size {shape[axis]}, not 8")
            return axis
        if shape and shape[0] == len(CHANNELS):
            return 0
        if len(shape) > 1 and shape[1] == len(CHANNELS):
            return 1
        raise ValueError(f"Cannot infer 8-channel axis from shape {shape}")

    @staticmethod
    def _epsilon_from_reference(
        reference: float,
        *,
        channel: str,
        fallback: float,
        reference_name: str,
    ) -> tuple[float, dict[str, Any] | None]:
        epsilon = np.nan
        exponent: int | None = None
        if np.isfinite(reference) and reference > 0:
            exponent = int(np.floor(np.log10(reference)) - 2)
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                epsilon = float(np.power(10.0, exponent))
        if np.isfinite(epsilon) and epsilon > 0:
            return epsilon, None
        record = {
            "code": "invalid_epsilon_reference",
            "channel": channel,
            "reference_name": reference_name,
            "reference_value": None if not np.isfinite(reference) else float(reference),
            "computed_exponent": exponent,
            "fallback": float(fallback),
        }
        warnings.warn(
            f"{channel} produced an invalid epsilon from {reference_name}={reference!r}; "
            f"using explicit fallback {fallback}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return float(fallback), record

    @staticmethod
    def _transform_values_inplace(
        values: np.ndarray,
        channel: int,
        epsilon: float,
    ) -> None:
        kind = PAPER_TRANSFORMS[channel]
        if not np.isfinite(values).all():
            raise FloatingPointError(f"{CHANNELS[channel]} contains NaN/Inf")
        if kind == "positive_log":
            if np.any(values < 0):
                raise ValueError(f"{CHANNELS[channel]} contains negative values")
            np.add(values, epsilon, out=values)
            if np.any(values <= 0):
                raise FloatingPointError(f"{CHANNELS[channel]} + epsilon is not positive")
            np.log10(values, out=values)
        elif kind == "signed_log":
            signs = np.sign(values).astype(np.int8, copy=False)
            np.abs(values, out=values)
            np.divide(values, epsilon, out=values)
            np.add(values, 1.0, out=values)
            np.log10(values, out=values)
            np.multiply(values, signs, out=values)

    @classmethod
    def fit_hdf5(
        cls,
        h5_path: str | Path,
        *,
        training_indices: Iterable[int],
        protocol_name: str,
        expected_source_hdf5_checksum: str | None = None,
        gamma: float = PAPER_GAMMA,
        inverse_clamp_fraction: float = PAPER_INVERSE_CLAMP_FRACTION,
        mad_multiplier: float = PAPER_MAD_MULTIPLIER,
        min_scale: float = PAPER_MIN_SCALE,
        invalid_epsilon_fallback: float = 1e-30,
        thermal_channel: str = "press",
        paper_adaptation: bool = True,
        eos_conversion: str = "disabled_unverified_gamma",
        fit_scope: str = "all_train_voxels",
    ) -> "PaperPreprocessor":
        indices = tuple(int(index) for index in training_indices)
        if not indices or tuple(sorted(set(indices))) != indices:
            raise ValueError("Fit indices must be non-empty, unique, and increasing")
        if fit_scope != "all_train_voxels":
            raise ValueError("This implementation currently requires all_train_voxels")
        h5_path = Path(h5_path).resolve()
        checksum = sha256_file(h5_path)
        if expected_source_hdf5_checksum is not None and checksum != expected_source_hdf5_checksum:
            raise ValueError("Paper preprocessing source HDF5 checksum mismatch")
        epsilon = np.zeros(len(CHANNELS), dtype=np.float64)
        medians = np.empty(len(CHANNELS), dtype=np.float64)
        scales = np.empty(len(CHANNELS), dtype=np.float64)
        fit_warnings: list[dict[str, Any]] = []
        with h5py.File(h5_path, "r") as handle:
            if "snapshots" not in handle or "channels" not in handle:
                raise KeyError("HDF5 must contain snapshots and channels")
            snapshots = handle["snapshots"]
            channels = _decode_strings(handle["channels"][...])
            if channels != CHANNELS:
                raise ValueError(f"Expected channel order {CHANNELS}, found {channels}")
            if indices[0] < 0 or indices[-1] >= snapshots.shape[0]:
                raise IndexError("Training indices lie outside the HDF5 trajectory")
            sample_count = len(indices) * int(np.prod(snapshots.shape[2:]))
            for channel, name in enumerate(CHANNELS):
                values = np.asarray(snapshots[list(indices), channel], dtype=np.float64)
                if not np.isfinite(values).all():
                    raise FloatingPointError(f"Training channel {name} contains NaN/Inf")
                kind = PAPER_TRANSFORMS[channel]
                if kind == "positive_log":
                    if np.any(values < 0):
                        raise ValueError(f"Positive training channel {name} contains negatives")
                    positive = values[values > 0]
                    reference = float(np.min(positive)) if positive.size else np.nan
                    epsilon[channel], warning_record = cls._epsilon_from_reference(
                        reference,
                        channel=name,
                        fallback=invalid_epsilon_fallback,
                        reference_name="min_positive_train",
                    )
                    del positive
                    if warning_record is not None:
                        fit_warnings.append(warning_record)
                elif kind == "signed_log":
                    minimum = float(np.min(values))
                    maximum = float(np.max(values))
                    reference = max(abs(minimum), abs(maximum))
                    epsilon[channel], warning_record = cls._epsilon_from_reference(
                        reference,
                        channel=name,
                        fallback=invalid_epsilon_fallback,
                        reference_name="max_abs_train",
                    )
                    if warning_record is not None:
                        fit_warnings.append(warning_record)
                cls._transform_values_inplace(values, channel, epsilon[channel])
                median = float(np.median(values, overwrite_input=True))
                np.subtract(values, median, out=values)
                np.abs(values, out=values)
                mad_scale = float(mad_multiplier * np.median(values, overwrite_input=True))
                if not np.isfinite(mad_scale) or mad_scale < min_scale:
                    record = {
                        "code": "mad_scale_floor",
                        "channel": name,
                        "computed_scale": None if not np.isfinite(mad_scale) else mad_scale,
                        "fallback": float(min_scale),
                    }
                    fit_warnings.append(record)
                    warnings.warn(
                        f"{name} MAD scale={mad_scale!r}; using explicit floor {min_scale}.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    mad_scale = float(min_scale)
                medians[channel] = median
                scales[channel] = mad_scale
        return cls(
            epsilon,
            medians,
            scales,
            gamma=gamma,
            inverse_clamp_fraction=inverse_clamp_fraction,
            mad_multiplier=mad_multiplier,
            min_scale=min_scale,
            invalid_epsilon_fallback=invalid_epsilon_fallback,
            training_indices=indices,
            source_hdf5_checksum=checksum,
            protocol_name=protocol_name,
            thermal_channel=thermal_channel,
            paper_adaptation=paper_adaptation,
            eos_conversion=eos_conversion,
            fit_scope=fit_scope,
            fit_warnings=fit_warnings,
            sample_count_per_channel=sample_count,
        )

    def encode_numpy(self, values: np.ndarray, channel_axis: int | None = None) -> np.ndarray:
        array = np.asarray(values)
        axis = self._channel_axis(array.shape, channel_axis)
        output = np.empty_like(array, dtype=np.result_type(array.dtype, np.float32))
        for channel in range(len(CHANNELS)):
            selection = [slice(None)] * array.ndim
            selection[axis] = channel
            key = tuple(selection)
            transformed = np.asarray(array[key], dtype=np.float64).copy()
            self._transform_values_inplace(transformed, channel, self.epsilon[channel])
            z = (transformed - self.median[channel]) / self.scale[channel]
            output[key] = self.gamma * np.tanh(z / self.gamma)
        if not np.isfinite(output).all():
            raise FloatingPointError("Paper preprocessing encode produced NaN/Inf")
        return output

    def decode_numpy(self, values: np.ndarray, channel_axis: int | None = None) -> np.ndarray:
        array = np.asarray(values)
        axis = self._channel_axis(array.shape, channel_axis)
        output = np.empty_like(array, dtype=np.result_type(array.dtype, np.float32))
        limit = self.inverse_clamp_fraction * self.gamma
        clipped = np.clip(array, -limit, limit)
        physical_limit = float(np.finfo(output.dtype).max)
        for channel in range(len(CHANNELS)):
            selection = [slice(None)] * array.ndim
            selection[axis] = channel
            key = tuple(selection)
            item = np.asarray(clipped[key], dtype=np.float64)
            z = self.gamma * np.arctanh(item / self.gamma)
            transformed = z * self.scale[channel] + self.median[channel]
            kind = PAPER_TRANSFORMS[channel]
            if kind == "positive_log":
                exponent_limit = float(np.log10(physical_limit + self.epsilon[channel]))
                safe_transformed = np.minimum(transformed, exponent_limit)
                decoded = np.power(10.0, safe_transformed) - self.epsilon[channel]
                decoded = np.maximum(decoded, np.finfo(output.dtype).tiny)
            elif kind == "signed_log":
                exponent_limit = float(
                    np.log10(physical_limit) - np.log10(self.epsilon[channel])
                )
                safe_magnitude = np.minimum(np.abs(transformed), exponent_limit)
                decoded = np.sign(transformed) * self.epsilon[channel] * (
                    np.power(10.0, safe_magnitude) - 1.0
                )
            else:
                decoded = np.clip(transformed, -physical_limit, physical_limit)
            output[key] = decoded
        if not np.isfinite(output).all():
            raise FloatingPointError("Paper preprocessing decode produced NaN/Inf")
        return output

    def encode_tensor(self, values: torch.Tensor, channel_axis: int | None = None) -> torch.Tensor:
        axis = self._channel_axis(tuple(values.shape), channel_axis)
        output = torch.empty_like(values)
        for channel, kind in enumerate(PAPER_TRANSFORMS):
            selection = [slice(None)] * values.ndim
            selection[axis] = channel
            key = tuple(selection)
            item = values[key]
            epsilon = torch.as_tensor(self.epsilon[channel], dtype=item.dtype, device=item.device)
            if not torch.isfinite(item).all():
                raise FloatingPointError(f"{CHANNELS[channel]} contains NaN/Inf")
            if kind == "positive_log":
                if torch.any(item < 0):
                    raise ValueError(f"{CHANNELS[channel]} contains negative values")
                transformed = torch.log10(item + epsilon)
            elif kind == "signed_log":
                transformed = torch.sign(item) * torch.log10(1.0 + torch.abs(item) / epsilon)
            else:
                transformed = item
            median = torch.as_tensor(self.median[channel], dtype=item.dtype, device=item.device)
            scale = torch.as_tensor(self.scale[channel], dtype=item.dtype, device=item.device)
            z = (transformed - median) / scale
            output[key] = self.gamma * torch.tanh(z / self.gamma)
        if not torch.isfinite(output).all():
            raise FloatingPointError("Paper preprocessing encode produced NaN/Inf")
        return output

    def decode_tensor(self, values: torch.Tensor, channel_axis: int | None = None) -> torch.Tensor:
        axis = self._channel_axis(tuple(values.shape), channel_axis)
        output = torch.empty_like(values)
        limit = self.inverse_clamp_fraction * self.gamma
        clipped = torch.clamp(values, -limit, limit)
        for channel, kind in enumerate(PAPER_TRANSFORMS):
            selection = [slice(None)] * values.ndim
            selection[axis] = channel
            key = tuple(selection)
            item = clipped[key]
            median = torch.as_tensor(self.median[channel], dtype=item.dtype, device=item.device)
            scale = torch.as_tensor(self.scale[channel], dtype=item.dtype, device=item.device)
            transformed = self.gamma * torch.atanh(item / self.gamma) * scale + median
            epsilon = torch.as_tensor(self.epsilon[channel], dtype=item.dtype, device=item.device)
            physical_limit = torch.finfo(item.dtype).max
            if kind == "positive_log":
                exponent_limit = torch.log10(
                    torch.as_tensor(physical_limit, dtype=item.dtype, device=item.device)
                    + epsilon
                )
                decoded = torch.pow(10.0, torch.minimum(transformed, exponent_limit)) - epsilon
                decoded = torch.clamp_min(decoded, torch.finfo(item.dtype).tiny)
            elif kind == "signed_log":
                exponent_limit = torch.log10(
                    torch.as_tensor(physical_limit, dtype=item.dtype, device=item.device)
                ) - torch.log10(epsilon)
                decoded = torch.sign(transformed) * epsilon * (
                    torch.pow(10.0, torch.minimum(torch.abs(transformed), exponent_limit)) - 1.0
                )
            else:
                decoded = torch.clamp(transformed, -physical_limit, physical_limit)
            output[key] = decoded
        if not torch.isfinite(output).all():
            raise FloatingPointError("Paper preprocessing decode produced NaN/Inf")
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

    def validate_compatibility(
        self,
        *,
        h5_path: str | Path | None = None,
        expected_source_hdf5_checksum: str | None = None,
        expected_training_indices: Iterable[int] | None = None,
        expected_protocol_name: str | None = None,
    ) -> None:
        checksum = expected_source_hdf5_checksum
        if h5_path is not None:
            actual = sha256_file(h5_path)
            if checksum is not None and checksum != actual:
                raise ValueError("Expected checksum does not match supplied HDF5")
            checksum = actual
        if checksum is not None and checksum != self.source_hdf5_checksum:
            raise ValueError("Paper preprocessor source HDF5 checksum mismatch")
        if (
            expected_training_indices is not None
            and tuple(int(index) for index in expected_training_indices) != self.training_indices
        ):
            raise ValueError("Paper preprocessor training indices mismatch")
        if expected_protocol_name is not None and expected_protocol_name != self.protocol_name:
            raise ValueError("Paper preprocessor protocol name mismatch")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "paper-preprocessing-v1",
            "channels": list(CHANNELS),
            "transforms": list(PAPER_TRANSFORMS),
            "epsilon": self.epsilon.tolist(),
            "median": self.median.tolist(),
            "scale": self.scale.tolist(),
            "gamma": self.gamma,
            "inverse_clamp_fraction": self.inverse_clamp_fraction,
            "inverse_clamp_limit": self.inverse_clamp_fraction * self.gamma,
            "mad_multiplier": self.mad_multiplier,
            "min_scale": self.min_scale,
            "invalid_epsilon_fallback": self.invalid_epsilon_fallback,
            "fit_scope": self.fit_scope,
            "sample_count_per_channel": self.sample_count_per_channel,
            "training_indices": list(self.training_indices),
            "source_hdf5_checksum": self.source_hdf5_checksum,
            "protocol_name": self.protocol_name,
            "thermal_channel": self.thermal_channel,
            "paper_adaptation": self.paper_adaptation,
            "eos_conversion": self.eos_conversion,
            "fit_warnings": list(self.fit_warnings),
            "decode_finite_guard": "output_dtype_max_after_paper_inverse_clamp",
            "decode_finite_guard_is_adaptation": True,
            "formulas": {
                "positive": "log10(x + epsilon)",
                "signed": "sign(x) * log10(1 + abs(x)/epsilon)",
                "linear": "x",
                "robust_scale": "max(1.4826 * median(abs(x_hat-median)), 1e-6)",
                "soft_clip": "6 * tanh(z/6)",
                "inverse_clamp": "clamp(z_clip, -0.99*6, 0.99*6)",
                "decode_finite_guard": "clip inverse-transformed physical magnitude to output dtype max after the paper inverse clamp",
            },
        }

    def save(self, json_path: str | Path, npz_path: str | Path) -> None:
        json_path = Path(json_path)
        npz_path = Path(npz_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        metadata = self.as_dict()
        json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        np.savez(
            npz_path,
            epsilon=self.epsilon,
            median=self.median,
            scale=self.scale,
            gamma=np.asarray(self.gamma),
            inverse_clamp_fraction=np.asarray(self.inverse_clamp_fraction),
            mad_multiplier=np.asarray(self.mad_multiplier),
            min_scale=np.asarray(self.min_scale),
            invalid_epsilon_fallback=np.asarray(self.invalid_epsilon_fallback),
            training_indices=np.asarray(self.training_indices, dtype=np.int64),
            source_hdf5_checksum=np.asarray(self.source_hdf5_checksum),
            protocol_name=np.asarray(self.protocol_name),
            thermal_channel=np.asarray(self.thermal_channel),
            paper_adaptation=np.asarray(self.paper_adaptation),
            eos_conversion=np.asarray(self.eos_conversion),
            fit_scope=np.asarray(self.fit_scope),
            sample_count_per_channel=np.asarray(
                -1 if self.sample_count_per_channel is None else self.sample_count_per_channel
            ),
            channels=np.asarray(CHANNELS),
            transforms=np.asarray(PAPER_TRANSFORMS),
            metadata_json=np.asarray(json.dumps(metadata)),
        )

    @classmethod
    def load(
        cls,
        npz_path: str | Path,
        *,
        h5_path: str | Path | None = None,
        expected_source_hdf5_checksum: str | None = None,
        expected_training_indices: Iterable[int] | None = None,
        expected_protocol_name: str | None = None,
    ) -> "PaperPreprocessor":
        with np.load(npz_path, allow_pickle=False) as values:
            if _decode_strings(values["channels"]) != CHANNELS:
                raise ValueError("Saved paper preprocessor channel schema is incompatible")
            if _decode_strings(values["transforms"]) != PAPER_TRANSFORMS:
                raise ValueError("Saved paper transform schema is incompatible")
            metadata = json.loads(str(values["metadata_json"]))
            sample_count = int(values["sample_count_per_channel"])
            preprocessor = cls(
                values["epsilon"],
                values["median"],
                values["scale"],
                gamma=float(values["gamma"]),
                inverse_clamp_fraction=float(values["inverse_clamp_fraction"]),
                mad_multiplier=float(values["mad_multiplier"]),
                min_scale=float(values["min_scale"]),
                invalid_epsilon_fallback=float(values["invalid_epsilon_fallback"]),
                training_indices=tuple(int(index) for index in values["training_indices"]),
                source_hdf5_checksum=str(values["source_hdf5_checksum"]),
                protocol_name=str(values["protocol_name"]),
                thermal_channel=str(values["thermal_channel"]),
                paper_adaptation=bool(values["paper_adaptation"]),
                eos_conversion=str(values["eos_conversion"]),
                fit_scope=str(values["fit_scope"]),
                fit_warnings=metadata.get("fit_warnings", []),
                sample_count_per_channel=None if sample_count < 0 else sample_count,
            )
        preprocessor.validate_compatibility(
            h5_path=h5_path,
            expected_source_hdf5_checksum=expected_source_hdf5_checksum,
            expected_training_indices=expected_training_indices,
            expected_protocol_name=expected_protocol_name,
        )
        return preprocessor

    def audit_hdf5(self, h5_path: str | Path) -> dict[str, Any]:
        """Audit finite values, saturation, positivity, and inverse round-trip on train only."""

        self.validate_compatibility(h5_path=h5_path)
        thresholds = (0.90, 0.95, 0.99)
        counts = np.zeros(len(CHANNELS), dtype=np.int64)
        saturation = {threshold: np.zeros(len(CHANNELS), dtype=np.int64) for threshold in thresholds}
        inverse_clamp_hits = np.zeros(len(CHANNELS), dtype=np.int64)
        decode_finite_guard_hits = np.zeros(len(CHANNELS), dtype=np.int64)
        encoded_min = np.full(len(CHANNELS), np.inf, dtype=np.float64)
        encoded_max = np.full(len(CHANNELS), -np.inf, dtype=np.float64)
        max_abs_error = np.zeros(len(CHANNELS), dtype=np.float64)
        max_relative_error = np.zeros(len(CHANNELS), dtype=np.float64)
        decoded_positive_min = np.full(len(CHANNELS), np.nan, dtype=np.float64)
        with h5py.File(h5_path, "r") as handle:
            snapshots = handle["snapshots"]
            for index in self.training_indices:
                raw = np.asarray(snapshots[index], dtype=np.float32)
                encoded = self.encode_numpy(raw)
                decoded = self.decode_numpy(encoded)
                if not np.isfinite(encoded).all() or not np.isfinite(decoded).all():
                    raise FloatingPointError(f"Audit found NaN/Inf at training snapshot {index}")
                for channel in range(len(CHANNELS)):
                    flat_encoded = encoded[channel].reshape(-1)
                    flat_raw = raw[channel].reshape(-1).astype(np.float64)
                    flat_decoded = decoded[channel].reshape(-1).astype(np.float64)
                    counts[channel] += flat_encoded.size
                    absolute_encoded = np.abs(flat_encoded)
                    for threshold in thresholds:
                        saturation[threshold][channel] += np.count_nonzero(
                            absolute_encoded >= threshold * self.gamma
                        )
                    inverse_clamp_hits[channel] += np.count_nonzero(
                        absolute_encoded > self.inverse_clamp_fraction * self.gamma
                    )
                    clipped_channel = np.clip(
                        flat_encoded.astype(np.float64),
                        -self.inverse_clamp_fraction * self.gamma,
                        self.inverse_clamp_fraction * self.gamma,
                    )
                    inverse_z = self.gamma * np.arctanh(clipped_channel / self.gamma)
                    inverse_transformed = (
                        inverse_z * self.scale[channel] + self.median[channel]
                    )
                    physical_limit = float(np.finfo(np.float32).max)
                    if PAPER_TRANSFORMS[channel] == "positive_log":
                        guard_limit = float(
                            np.log10(physical_limit + self.epsilon[channel])
                        )
                        decode_finite_guard_hits[channel] += np.count_nonzero(
                            inverse_transformed > guard_limit
                        )
                    elif PAPER_TRANSFORMS[channel] == "signed_log":
                        guard_limit = float(
                            np.log10(physical_limit) - np.log10(self.epsilon[channel])
                        )
                        decode_finite_guard_hits[channel] += np.count_nonzero(
                            np.abs(inverse_transformed) > guard_limit
                        )
                    else:
                        decode_finite_guard_hits[channel] += np.count_nonzero(
                            np.abs(inverse_transformed) > physical_limit
                        )
                    encoded_min[channel] = min(encoded_min[channel], float(np.min(flat_encoded)))
                    encoded_max[channel] = max(encoded_max[channel], float(np.max(flat_encoded)))
                    error = np.abs(flat_decoded - flat_raw)
                    max_abs_error[channel] = max(max_abs_error[channel], float(np.max(error)))
                    denominator = np.maximum(np.abs(flat_raw), np.finfo(np.float32).tiny)
                    max_relative_error[channel] = max(
                        max_relative_error[channel],
                        float(np.max(error / denominator)),
                    )
                    if channel in (3, 4):
                        current_min = float(np.min(flat_decoded))
                        decoded_positive_min[channel] = (
                            current_min
                            if np.isnan(decoded_positive_min[channel])
                            else min(decoded_positive_min[channel], current_min)
                        )
        channel_audit: dict[str, Any] = {}
        audit_warnings: list[dict[str, Any]] = []
        for channel, name in enumerate(CHANNELS):
            inverse_hit_fraction = float(inverse_clamp_hits[channel] / counts[channel])
            finite_guard_fraction = float(
                decode_finite_guard_hits[channel] / counts[channel]
            )
            if inverse_hit_fraction > 0:
                audit_warnings.append(
                    {
                        "code": "training_values_exceed_inverse_clamp",
                        "channel": name,
                        "fraction": inverse_hit_fraction,
                        "impact": "paper encode/decode is lossy for these training voxels",
                    }
                )
            if finite_guard_fraction > 0:
                audit_warnings.append(
                    {
                        "code": "decode_finite_guard_used",
                        "channel": name,
                        "fraction": finite_guard_fraction,
                        "impact": "physical decode reached the output dtype range",
                    }
                )
            channel_audit[name] = {
                "epsilon": float(self.epsilon[channel]),
                "median": float(self.median[channel]),
                "mad_scale": float(self.scale[channel]),
                "value_count": int(counts[channel]),
                "encoded_min": float(encoded_min[channel]),
                "encoded_max": float(encoded_max[channel]),
                "saturation_fraction_abs_ge_0.90_gamma": float(
                    saturation[0.90][channel] / counts[channel]
                ),
                "saturation_fraction_abs_ge_0.95_gamma": float(
                    saturation[0.95][channel] / counts[channel]
                ),
                "saturation_fraction_abs_ge_0.99_gamma": float(
                    saturation[0.99][channel] / counts[channel]
                ),
                "inverse_clamp_hit_fraction": inverse_hit_fraction,
                "decode_finite_guard_hit_fraction": finite_guard_fraction,
                "roundtrip_max_abs_error": float(max_abs_error[channel]),
                "roundtrip_max_relative_error": float(max_relative_error[channel]),
                "decoded_positive_min": (
                    float(decoded_positive_min[channel])
                    if channel in (3, 4)
                    else None
                ),
            }
        return {
            "schema_version": "paper-preprocessing-audit-v1",
            "status": "passed_with_warnings" if audit_warnings else "passed",
            "protocol_name": self.protocol_name,
            "source_hdf5_checksum": self.source_hdf5_checksum,
            "fit_scope": self.fit_scope,
            "fit_snapshot_indices": list(self.training_indices),
            "validation_excluded_from_fit": True,
            "thermal_channel": self.thermal_channel,
            "paper_adaptation": self.paper_adaptation,
            "eos_conversion": self.eos_conversion,
            "gamma": self.gamma,
            "inverse_clamp_limit": self.inverse_clamp_fraction * self.gamma,
            "all_encoded_and_decoded_finite": True,
            "rho_press_decoded_positive": bool(
                decoded_positive_min[3] > 0 and decoded_positive_min[4] > 0
            ),
            "fit_warnings": list(self.fit_warnings),
            "audit_warnings": audit_warnings,
            "paper_roundtrip_lossless_on_training_values": not any(
                warning["code"] == "training_values_exceed_inverse_clamp"
                for warning in audit_warnings
            ),
            "decode_finite_guard": "output_dtype_max_after_paper_inverse_clamp",
            "decode_finite_guard_is_adaptation": True,
            "channels": channel_audit,
        }


def write_preprocessing_audit(
    audit: dict[str, Any],
    json_path: str | Path,
    markdown_path: str | Path,
) -> None:
    json_path = Path(json_path)
    markdown_path = Path(markdown_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# paper_reduced100 preprocessing audit",
        "",
        f"- Status: `{audit['status']}`",
        f"- Protocol: `{audit['protocol_name']}`",
        f"- Thermal channel: `{audit['thermal_channel']}` (paper adaptation)",
        f"- HDF5 SHA-256: `{audit['source_hdf5_checksum']}`",
        f"- Fit snapshots: `{audit['fit_snapshot_indices'][0]}..{audit['fit_snapshot_indices'][-1]}`",
        f"- Gamma / inverse limit: `{audit['gamma']}` / `{audit['inverse_clamp_limit']}`",
        f"- Validation excluded from fit: `{audit['validation_excluded_from_fit']}`",
        "",
        "| channel | epsilon | median | MAD scale | >=0.95 gamma | inverse clamp hits | max abs round-trip |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in CHANNELS:
        record = audit["channels"][name]
        lines.append(
            f"| {name} | {record['epsilon']:.9g} | {record['median']:.9g} | "
            f"{record['mad_scale']:.9g} | "
            f"{record['saturation_fraction_abs_ge_0.95_gamma']:.9g} | "
            f"{record['inverse_clamp_hit_fraction']:.9g} | "
            f"{record['roundtrip_max_abs_error']:.9g} |"
        )
    if audit.get("audit_warnings"):
        lines.extend(["", "## Warnings", ""])
        for warning in audit["audit_warnings"]:
            lines.append(
                f"- `{warning['code']}` on `{warning['channel']}`: "
                f"fraction `{warning['fraction']:.9g}`; {warning['impact']}."
            )
    lines.extend(
        [
            "",
            "This audit uses train snapshots only. It does not load Round 1--3 normalizer stats,",
            "fit a radial baseline, or authorize any training run.",
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
