"""Train-only paper physical bounds for the positive rho/press channels."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import numpy as np
import torch

from .paper_preprocessing import PaperPreprocessor
from .paper_priors import PriorProvenance, read_prior_json, write_prior_json


@dataclass(frozen=True)
class BoundsClampResult:
    unclamped: torch.Tensor
    clamped: torch.Tensor
    mask: torch.Tensor
    hit_fraction: dict[str, float]


@dataclass(frozen=True)
class PaperPhysicalBounds:
    """Physical quantile bounds and their canonical normalized images."""

    provenance: PriorProvenance
    raw_quantiles: Mapping[str, tuple[float, float]]
    physical_bounds: Mapping[str, tuple[float, float]]
    transformed_bounds: Mapping[str, tuple[float, float]]
    normalized_bounds: Mapping[str, tuple[float, float]]
    qlow: float = 0.001
    qhigh: float = 0.999
    margin: float = 0.05
    lambda_low_rho: float = 0.05
    lambda_low_press: float = 0.05
    lambda_high_rho: float = 0.0
    lambda_high_press: float = 0.0

    schema_version = "paper-physical-bounds-v1"

    def __post_init__(self) -> None:
        if not 0 <= self.qlow < self.qhigh <= 1:
            raise ValueError("Invalid paper bound quantiles")
        if self.margin < 0:
            raise ValueError("Paper bound margin cannot be negative")
        for collection in (
            self.raw_quantiles,
            self.physical_bounds,
            self.transformed_bounds,
            self.normalized_bounds,
        ):
            if set(collection) != {"rho", "press"}:
                raise ValueError("Paper bounds must contain only rho and press")
            for lower, upper in collection.values():
                if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                    raise ValueError("Paper bounds must be finite and ordered")

    @staticmethod
    def _axis(values: torch.Tensor, channel_axis: int | None) -> int:
        if channel_axis is None:
            if values.ndim >= 2 and values.shape[1] == 8:
                return 1
            if values.ndim >= 1 and values.shape[0] == 8:
                return 0
            raise ValueError(f"Cannot infer channel axis from {tuple(values.shape)}")
        axis = channel_axis % values.ndim
        if values.shape[axis] != 8:
            raise ValueError("Paper bound operation requires an eight-channel axis")
        return axis

    def clamp_normalized(
        self, prediction: torch.Tensor, *, channel_axis: int | None = None
    ) -> BoundsClampResult:
        """Clamp only rho/press and retain both tensors plus a full-shape hit mask."""

        axis = self._axis(prediction, channel_axis)
        unclamped = prediction.clone()
        clamped = prediction.clone()
        mask = torch.zeros_like(prediction, dtype=torch.bool)
        fractions: dict[str, float] = {}
        for channel, name in ((3, "rho"), (4, "press")):
            values = clamped.select(axis, channel)
            lower, upper = self.normalized_bounds[name]
            channel_mask = (values < lower) | (values > upper)
            values.clamp_(min=lower, max=upper)
            mask.select(axis, channel).copy_(channel_mask)
            fractions[name] = float(channel_mask.float().mean())
        fractions["combined"] = float(
            torch.cat(
                [mask.select(axis, 3).reshape(-1), mask.select(axis, 4).reshape(-1)]
            ).float().mean()
        )
        return BoundsClampResult(unclamped, clamped, mask, fractions)

    def penalty_components(
        self, prediction: torch.Tensor, *, channel_axis: int | None = None
    ) -> dict[str, torch.Tensor]:
        """Expose independent bound terms without constructing a total paper loss."""

        axis = self._axis(prediction, channel_axis)
        output: dict[str, torch.Tensor] = {}
        for channel, name, low_weight, high_weight in (
            (3, "rho", self.lambda_low_rho, self.lambda_high_rho),
            (4, "press", self.lambda_low_press, self.lambda_high_press),
        ):
            values = prediction.select(axis, channel)
            lower, upper = self.normalized_bounds[name]
            output[f"{name}_low"] = low_weight * torch.mean(
                torch.relu(values.new_tensor(lower) - values).square()
            )
            output[f"{name}_high"] = high_weight * torch.mean(
                torch.relu(values - values.new_tensor(upper)).square()
            )
        return output

    def as_dict(self) -> dict[str, Any]:
        def pairs(values: Mapping[str, tuple[float, float]]) -> dict[str, list[float]]:
            return {name: [float(bounds[0]), float(bounds[1])] for name, bounds in values.items()}

        return {
            "schema_version": self.schema_version,
            "provenance": self.provenance.as_dict(),
            "quantiles": {"qlow": self.qlow, "qhigh": self.qhigh, "margin": self.margin},
            "raw_physical_quantiles": pairs(self.raw_quantiles),
            "raw_physical_bounds": pairs(self.physical_bounds),
            "transformed_bounds": pairs(self.transformed_bounds),
            "normalized_bounds": pairs(self.normalized_bounds),
            "penalty_parameters": {
                "lambda_low_rho": self.lambda_low_rho,
                "lambda_low_press": self.lambda_low_press,
                "lambda_high_rho": self.lambda_high_rho,
                "lambda_high_press": self.lambda_high_press,
                "combined_into_total_paper_loss": False,
            },
            "evaluation_clamp": {
                "channels": ["rho", "press"],
                "magnetic_and_velocity_channels_unchanged": True,
                "returns_unclamped_tensor_and_hit_mask": True,
            },
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "PaperPhysicalBounds":
        if values.get("schema_version") != cls.schema_version:
            raise ValueError("Physical bounds schema mismatch")
        quantiles = values["quantiles"]
        penalty = values["penalty_parameters"]

        def pairs(key: str) -> dict[str, tuple[float, float]]:
            return {
                name: (float(bounds[0]), float(bounds[1]))
                for name, bounds in values[key].items()
            }

        return cls(
            provenance=PriorProvenance.from_dict(values["provenance"]),
            raw_quantiles=pairs("raw_physical_quantiles"),
            physical_bounds=pairs("raw_physical_bounds"),
            transformed_bounds=pairs("transformed_bounds"),
            normalized_bounds=pairs("normalized_bounds"),
            qlow=float(quantiles["qlow"]),
            qhigh=float(quantiles["qhigh"]),
            margin=float(quantiles["margin"]),
            lambda_low_rho=float(penalty["lambda_low_rho"]),
            lambda_low_press=float(penalty["lambda_low_press"]),
            lambda_high_rho=float(penalty["lambda_high_rho"]),
            lambda_high_press=float(penalty["lambda_high_press"]),
        )

    def save(self, path: str | Path) -> None:
        write_prior_json(path, self.as_dict())

    @classmethod
    def load(cls, path: str | Path, **expected: Any) -> "PaperPhysicalBounds":
        payload = read_prior_json(path, expected_schema=cls.schema_version, **expected)
        return cls.from_dict(payload)


def fit_paper_physical_bounds(
    h5_path: str | Path,
    *,
    training_indices: Iterable[int],
    preprocessor: PaperPreprocessor,
    provenance: PriorProvenance,
    qlow: float = 0.001,
    qhigh: float = 0.999,
    margin: float = 0.05,
) -> PaperPhysicalBounds:
    indices = tuple(int(index) for index in training_indices)
    if not indices or tuple(sorted(set(indices))) != indices:
        raise ValueError("Bound fit indices must be non-empty, unique, and increasing")
    if not 0 <= qlow < qhigh <= 1 or margin < 0:
        raise ValueError("Invalid paper bound fit parameters")
    preprocessor.validate_compatibility(
        h5_path=h5_path,
        expected_training_indices=indices,
        expected_protocol_name=provenance.protocol_name,
    )
    provenance.validate(
        source_hdf5_checksum=preprocessor.source_hdf5_checksum,
        training_indices=indices,
        protocol_name=preprocessor.protocol_name,
        thermal_channel=preprocessor.thermal_channel,
    )
    raw_quantiles: dict[str, tuple[float, float]] = {}
    physical: dict[str, tuple[float, float]] = {}
    transformed: dict[str, tuple[float, float]] = {}
    normalized: dict[str, tuple[float, float]] = {}
    with h5py.File(h5_path, "r") as handle:
        for channel, name in ((3, "rho"), (4, "press")):
            values = np.asarray(handle["snapshots"][list(indices), channel], dtype=np.float64)
            if not np.all(np.isfinite(values)) or np.any(values < 0):
                raise ValueError(f"Train {name} values must be finite and nonnegative")
            q_lower, q_upper = (float(value) for value in np.quantile(values, (qlow, qhigh)))
            lower = (1.0 - margin) * q_lower
            upper = (1.0 + margin) * q_upper
            transformed_lower = float(np.log10(lower + preprocessor.epsilon[channel]))
            transformed_upper = float(np.log10(upper + preprocessor.epsilon[channel]))
            z_lower = (transformed_lower - preprocessor.median[channel]) / preprocessor.scale[channel]
            z_upper = (transformed_upper - preprocessor.median[channel]) / preprocessor.scale[channel]
            norm_lower = float(preprocessor.gamma * np.tanh(z_lower / preprocessor.gamma))
            norm_upper = float(preprocessor.gamma * np.tanh(z_upper / preprocessor.gamma))
            raw_quantiles[name] = (q_lower, q_upper)
            physical[name] = (float(lower), float(upper))
            transformed[name] = (transformed_lower, transformed_upper)
            normalized[name] = (norm_lower, norm_upper)
    return PaperPhysicalBounds(
        provenance=provenance,
        raw_quantiles=raw_quantiles,
        physical_bounds=physical,
        transformed_bounds=transformed,
        normalized_bounds=normalized,
        qlow=qlow,
        qhigh=qhigh,
        margin=margin,
    )
