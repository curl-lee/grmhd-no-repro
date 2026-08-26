"""Oracle-aware paper metrics and target/model inverse-clamp diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from . import CHANNELS
from .paper_references import PaperReferenceStates


METRIC_SCHEMA_VERSION = "paper-oracle-metrics-v1"
CLAMP_SCHEMA_VERSION = "paper-clamp-comparison-v1"


def _as_channel_rows(values: np.ndarray | torch.Tensor, channel_axis: int | None) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        array = values.detach().cpu().numpy()
    else:
        array = np.asarray(values)
    if channel_axis is None:
        if array.ndim and array.shape[0] == len(CHANNELS):
            axis = 0
        elif array.ndim > 1 and array.shape[1] == len(CHANNELS):
            axis = 1
        else:
            raise ValueError(f"Cannot infer eight-channel axis from shape {array.shape}")
    else:
        axis = channel_axis % array.ndim
        if array.shape[axis] != len(CHANNELS):
            raise ValueError(f"channel_axis {channel_axis} has size {array.shape[axis]}, not 8")
    rows = np.moveaxis(array, axis, 0).astype(np.float64, copy=False).reshape(len(CHANNELS), -1)
    if not np.isfinite(rows).all():
        raise FloatingPointError("Paper metric input contains NaN/Inf")
    return rows


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


class PaperMetricAccumulator:
    """Accumulate exact per-channel squared sums before taking relative norms."""

    metric_names = ("E_norm", "E_model_oracle", "E_model_raw", "E_oracle_raw")

    def __init__(self) -> None:
        self.numerator_squared_sums = {
            name: np.zeros(len(CHANNELS), dtype=np.float64) for name in self.metric_names
        }
        self.denominator_squared_sums = {
            name: np.zeros(len(CHANNELS), dtype=np.float64) for name in self.metric_names
        }
        self.decomposition_model_oracle = np.zeros(len(CHANNELS), dtype=np.float64)
        self.decomposition_oracle_raw = np.zeros(len(CHANNELS), dtype=np.float64)
        self.decomposition_cross_term = np.zeros(len(CHANNELS), dtype=np.float64)
        self.value_counts = np.zeros(len(CHANNELS), dtype=np.int64)
        self.update_count = 0

    def update(
        self,
        references: PaperReferenceStates,
        *,
        channel_axis: int | None = None,
    ) -> None:
        raw_physical_target = _as_channel_rows(
            references.raw_physical_target, channel_axis
        )
        oracle_physical_target = _as_channel_rows(
            references.oracle_physical_target, channel_axis
        )
        normalized_target = _as_channel_rows(references.normalized_target, channel_axis)
        normalized_prediction = _as_channel_rows(
            references.normalized_prediction, channel_axis
        )
        model_physical_prediction = _as_channel_rows(
            references.model_physical_prediction, channel_axis
        )
        shapes = {
            item.shape
            for item in (
                raw_physical_target,
                oracle_physical_target,
                normalized_target,
                normalized_prediction,
                model_physical_prediction,
            )
        }
        if len(shapes) != 1:
            raise ValueError(f"Paper metric arrays do not align after channel reduction: {shapes}")

        differences = {
            "E_norm": normalized_prediction - normalized_target,
            "E_model_oracle": model_physical_prediction - oracle_physical_target,
            "E_model_raw": model_physical_prediction - raw_physical_target,
            "E_oracle_raw": oracle_physical_target - raw_physical_target,
        }
        denominators = {
            "E_norm": normalized_target,
            "E_model_oracle": oracle_physical_target,
            "E_model_raw": raw_physical_target,
            "E_oracle_raw": raw_physical_target,
        }
        for name in self.metric_names:
            self.numerator_squared_sums[name] += np.einsum(
                "ij,ij->i", differences[name], differences[name]
            )
            self.denominator_squared_sums[name] += np.einsum(
                "ij,ij->i", denominators[name], denominators[name]
            )

        model_oracle_difference = differences["E_model_oracle"]
        oracle_raw_difference = differences["E_oracle_raw"]
        self.decomposition_model_oracle += np.einsum(
            "ij,ij->i", model_oracle_difference, model_oracle_difference
        )
        self.decomposition_oracle_raw += np.einsum(
            "ij,ij->i", oracle_raw_difference, oracle_raw_difference
        )
        self.decomposition_cross_term += 2.0 * np.einsum(
            "ij,ij->i", model_oracle_difference, oracle_raw_difference
        )
        self.value_counts += raw_physical_target.shape[1]
        self.update_count += 1

    def _relative_record(self, name: str) -> dict[str, Any]:
        numerator = self.numerator_squared_sums[name]
        denominator = self.denominator_squared_sums[name]
        if np.any(denominator <= 0):
            missing = [CHANNELS[index] for index in np.flatnonzero(denominator <= 0)]
            raise ZeroDivisionError(f"{name} has zero target norm for channels {missing}")
        values = np.sqrt(numerator / denominator)
        return {
            "per_channel": {
                channel: float(values[index]) for index, channel in enumerate(CHANNELS)
            },
            "arithmetic_average": float(np.mean(values)),
            "global_relative_l2": float(np.sqrt(np.sum(numerator) / np.sum(denominator))),
            "numerator_squared_sum": {
                channel: float(numerator[index]) for index, channel in enumerate(CHANNELS)
            },
            "denominator_squared_sum": {
                channel: float(denominator[index]) for index, channel in enumerate(CHANNELS)
            },
        }

    def finalize(self) -> dict[str, Any]:
        if self.update_count == 0:
            raise RuntimeError("Cannot finalize paper metrics before an update")
        metrics = {name: self._relative_record(name) for name in self.metric_names}
        model_raw_values = metrics["E_model_raw"]["per_channel"]
        oracle_raw_values = metrics["E_oracle_raw"]["per_channel"]
        excess_values = {
            channel: float(model_raw_values[channel] - oracle_raw_values[channel])
            for channel in CHANNELS
        }
        metrics["excess_absolute"] = {
            "definition": "E_model_raw - E_oracle_raw; a norm difference, not an error decomposition",
            "per_channel": excess_values,
            "arithmetic_average": float(np.mean(list(excess_values.values()))),
            "global_relative_l2_difference": float(
                metrics["E_model_raw"]["global_relative_l2"]
                - metrics["E_oracle_raw"]["global_relative_l2"]
            ),
        }

        direct = self.numerator_squared_sums["E_model_raw"]
        reconstructed = (
            self.decomposition_model_oracle
            + self.decomposition_oracle_raw
            + self.decomposition_cross_term
        )
        decomposition: dict[str, Any] = {
            "identity": (
                "||model-raw||^2 = ||model-oracle||^2 + ||oracle-raw||^2 "
                "+ 2<model-oracle, oracle-raw>"
            ),
            "warning": (
                "Relative L2 norms and excess_absolute are not additive; only this squared-error "
                "numerator identity is exact."
            ),
            "per_channel": {},
        }
        for index, channel in enumerate(CHANNELS):
            residual = direct[index] - reconstructed[index]
            decomposition["per_channel"][channel] = {
                "model_raw_squared_error": float(direct[index]),
                "model_oracle_squared_error": float(self.decomposition_model_oracle[index]),
                "oracle_raw_squared_error": float(self.decomposition_oracle_raw[index]),
                "cross_term": float(self.decomposition_cross_term[index]),
                "reconstructed_model_raw_squared_error": float(reconstructed[index]),
                "reconstruction_residual": float(residual),
                "relative_reconstruction_residual": float(
                    abs(residual) / max(abs(direct[index]), np.finfo(np.float64).tiny)
                ),
            }
        return {
            "schema_version": METRIC_SCHEMA_VERSION,
            "channel_order": list(CHANNELS),
            "reduction": (
                "For each channel, sum squared errors and squared reference values over all "
                "evaluated pairs/voxels, then take sqrt(sum_error_sq/sum_reference_sq)."
            ),
            "channel_average": "unweighted arithmetic mean of the eight per-channel relative L2 values",
            "update_count": self.update_count,
            "value_count_per_channel": {
                channel: int(self.value_counts[index])
                for index, channel in enumerate(CHANNELS)
            },
            "metrics": metrics,
            "squared_error_numerator_decomposition": decomposition,
        }


class ClampMaskAccumulator:
    """Keep target and model inverse-clamp masks separate, including sign."""

    def __init__(self, *, gamma: float, inverse_clamp_fraction: float) -> None:
        if gamma <= 0 or not 0 < inverse_clamp_fraction < 1:
            raise ValueError("Clamp diagnostics require gamma>0 and fraction in (0,1)")
        self.gamma = float(gamma)
        self.inverse_clamp_fraction = float(inverse_clamp_fraction)
        self.limit = self.gamma * self.inverse_clamp_fraction
        self.counts = {
            name: np.zeros(len(CHANNELS), dtype=np.int64)
            for name in (
                "total",
                "target",
                "target_positive",
                "target_negative",
                "model",
                "model_positive",
                "model_negative",
                "intersection",
                "model_only",
                "target_only",
                "neither",
                "same_sign_intersection",
                "opposite_sign_intersection",
            )
        }
        self.update_count = 0

    def update(
        self,
        *,
        normalized_target: np.ndarray | torch.Tensor,
        normalized_prediction: np.ndarray | torch.Tensor,
        channel_axis: int | None = None,
    ) -> None:
        normalized_target_rows = _as_channel_rows(normalized_target, channel_axis)
        normalized_prediction_rows = _as_channel_rows(normalized_prediction, channel_axis)
        if normalized_target_rows.shape != normalized_prediction_rows.shape:
            raise ValueError("Target and model normalized arrays must have equal shape")
        target_positive = normalized_target_rows > self.limit
        target_negative = normalized_target_rows < -self.limit
        model_positive = normalized_prediction_rows > self.limit
        model_negative = normalized_prediction_rows < -self.limit
        target_mask = target_positive | target_negative
        model_mask = model_positive | model_negative
        intersection = target_mask & model_mask
        same_sign_intersection = (target_positive & model_positive) | (
            target_negative & model_negative
        )
        masks = {
            "target": target_mask,
            "target_positive": target_positive,
            "target_negative": target_negative,
            "model": model_mask,
            "model_positive": model_positive,
            "model_negative": model_negative,
            "intersection": intersection,
            "model_only": model_mask & ~target_mask,
            "target_only": target_mask & ~model_mask,
            "neither": ~model_mask & ~target_mask,
            "same_sign_intersection": same_sign_intersection,
            "opposite_sign_intersection": intersection & ~same_sign_intersection,
        }
        self.counts["total"] += normalized_target_rows.shape[1]
        for name, mask in masks.items():
            self.counts[name] += np.count_nonzero(mask, axis=1)
        self.update_count += 1

    def finalize(self) -> dict[str, Any]:
        if self.update_count == 0:
            raise RuntimeError("Cannot finalize clamp diagnostics before an update")
        channels: dict[str, Any] = {}
        for index, channel in enumerate(CHANNELS):
            count = {name: int(values[index]) for name, values in self.counts.items()}
            union = count["intersection"] + count["model_only"] + count["target_only"]
            channels[channel] = {
                "value_count": count["total"],
                "target_clamp_count": count["target"],
                "target_clamp_fraction": count["target"] / count["total"],
                "target_positive_fraction": count["target_positive"] / count["total"],
                "target_negative_fraction": count["target_negative"] / count["total"],
                "model_clamp_count": count["model"],
                "model_clamp_fraction": count["model"] / count["total"],
                "model_positive_fraction": count["model_positive"] / count["total"],
                "model_negative_fraction": count["model_negative"] / count["total"],
                "intersection_fraction": count["intersection"] / count["total"],
                "model_only_fraction": count["model_only"] / count["total"],
                "target_only_fraction": count["target_only"] / count["total"],
                "neither_fraction": count["neither"] / count["total"],
                "jaccard": _safe_ratio(count["intersection"], union),
                "precision": _safe_ratio(count["intersection"], count["model"]),
                "recall": _safe_ratio(count["intersection"], count["target"]),
                "same_sign_intersection_fraction": (
                    count["same_sign_intersection"] / count["total"]
                ),
                "opposite_sign_intersection_fraction": (
                    count["opposite_sign_intersection"] / count["total"]
                ),
                "sign_agreement_within_intersection": _safe_ratio(
                    count["same_sign_intersection"], count["intersection"]
                ),
            }
        return {
            "schema_version": CLAMP_SCHEMA_VERSION,
            "channel_order": list(CHANNELS),
            "gamma": self.gamma,
            "inverse_clamp_fraction": self.inverse_clamp_fraction,
            "inverse_clamp_limit": self.limit,
            "mask_definition": "strictly greater than +limit or strictly less than -limit",
            "target_and_model_masks_kept_separate": True,
            "update_count": self.update_count,
            "channels": channels,
        }


def compute_paper_metrics(
    references: PaperReferenceStates,
    *,
    channel_axis: int | None = None,
) -> dict[str, Any]:
    accumulator = PaperMetricAccumulator()
    accumulator.update(references, channel_axis=channel_axis)
    return accumulator.finalize()


def compare_clamp_masks(
    *,
    normalized_target: np.ndarray | torch.Tensor,
    normalized_prediction: np.ndarray | torch.Tensor,
    gamma: float,
    inverse_clamp_fraction: float,
    channel_axis: int | None = None,
) -> dict[str, Any]:
    accumulator = ClampMaskAccumulator(
        gamma=gamma, inverse_clamp_fraction=inverse_clamp_fraction
    )
    accumulator.update(
        normalized_target=normalized_target,
        normalized_prediction=normalized_prediction,
        channel_axis=channel_axis,
    )
    return accumulator.finalize()
