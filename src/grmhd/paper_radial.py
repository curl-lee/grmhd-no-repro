"""Paper-adapted train-only radial baselines for ``rho`` and ``press``.

The literal candidate keeps Appendix C's no-intercept regression of
``log10(rho + press + epsilon_U)`` on physical spherical ``r``.  Turning that
scalar ``U`` curve into two model channels requires one explicit adaptation:
the fitted curve is partitioned using train-only global rho/press shares.  The
stronger spherical candidate instead fits each positive transformed channel
against ``log10(r)`` with an intercept.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import numpy as np
import torch

from . import CHANNELS
from .paper_preprocessing import PaperPreprocessor
from .paper_priors import (
    PAPER_PROTOCOL_NAME,
    PriorProvenance,
    RADIAL_MODES,
    read_prior_json,
    validate_radial_metadata,
    write_prior_json,
)


PAPER_REFERENCE_K = -1.78e-2


def _epsilon_from_min_positive(value: float, fallback: float = 1e-30) -> float:
    if not np.isfinite(value) or value <= 0:
        return float(fallback)
    epsilon = float(10.0 ** (np.floor(np.log10(value)) - 2.0))
    return epsilon if np.isfinite(epsilon) and epsilon > 0 else float(fallback)


def _positive_channel_normalized(
    physical: np.ndarray,
    channel: int,
    preprocessor: PaperPreprocessor,
) -> np.ndarray:
    transformed = np.log10(
        np.maximum(np.asarray(physical, dtype=np.float64), 0.0)
        + preprocessor.epsilon[channel]
    )
    z = (transformed - preprocessor.median[channel]) / preprocessor.scale[channel]
    return preprocessor.gamma * np.tanh(z / preprocessor.gamma)


@dataclass(frozen=True)
class PaperRadialBaseline:
    """A serialized radial-only rho/press baseline in physical and normalized space."""

    mode: str
    provenance: PriorProvenance
    r: tuple[float, ...]
    rho_physical: tuple[float, ...]
    press_physical: tuple[float, ...]
    rho_normalized: tuple[float, ...]
    press_normalized: tuple[float, ...]
    fit: Mapping[str, Any]
    exact_paper_formula: bool
    adaptation_level: str

    schema_version = "paper-radial-baseline-v1"

    def __post_init__(self) -> None:
        if self.mode not in RADIAL_MODES:
            raise ValueError("Unsupported paper radial mode")
        lengths = {
            len(self.r),
            len(self.rho_physical),
            len(self.press_physical),
            len(self.rho_normalized),
            len(self.press_normalized),
        }
        if lengths == {0} or len(lengths) != 1:
            raise ValueError("Radial baseline arrays must be non-empty and aligned")
        arrays = (
            self.r,
            self.rho_physical,
            self.press_physical,
            self.rho_normalized,
            self.press_normalized,
        )
        if any(not np.all(np.isfinite(values)) for values in arrays):
            raise ValueError("Radial baseline arrays must be finite")
        if np.any(np.asarray(self.r) <= 0) or np.any(np.diff(self.r) <= 0):
            raise ValueError("Radial coordinates must be positive and increasing")
        if min(self.rho_physical) <= 0 or min(self.press_physical) <= 0:
            raise ValueError("Physical rho/press baselines must remain positive")

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "protocol_name": self.provenance.protocol_name,
            "thermal_channel": self.provenance.thermal_channel,
            "baseline_space": "canonical_normalized_rho_press",
            "radial_coordinate": "physical_spherical_r",
            "stored_components_not_cartesian": True,
            "exact_paper_formula": self.exact_paper_formula,
            "adaptation_level": self.adaptation_level,
        }

    def state(
        self,
        spatial_shape: tuple[int, int, int],
        *,
        normalized: bool,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
        batch_size: int | None = None,
    ) -> torch.Tensor:
        nphi, ntheta, nr = (int(value) for value in spatial_shape)
        if nr != len(self.r):
            raise ValueError("Radial baseline and requested grid have different Nr")
        rho = self.rho_normalized if normalized else self.rho_physical
        press = self.press_normalized if normalized else self.press_physical
        output = torch.zeros((8, nphi, ntheta, nr), dtype=dtype, device=device)
        output[3] = torch.as_tensor(rho, dtype=dtype, device=device).view(1, 1, nr)
        output[4] = torch.as_tensor(press, dtype=dtype, device=device).view(1, 1, nr)
        if batch_size is not None:
            output = output.unsqueeze(0).expand(int(batch_size), -1, -1, -1, -1)
        return output

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "provenance": self.provenance.as_dict(),
            "metadata": self.metadata,
            "r": list(self.r),
            "physical_baseline": {
                "rho": list(self.rho_physical),
                "press": list(self.press_physical),
            },
            "normalized_baseline": {
                "rho": list(self.rho_normalized),
                "press": list(self.press_normalized),
            },
            "fit": dict(self.fit),
        }

    def save(self, path: str | Path) -> None:
        write_prior_json(path, self.as_dict())

    @classmethod
    def load(cls, path: str | Path, **expected: Any) -> "PaperRadialBaseline":
        payload = read_prior_json(path, expected_schema=cls.schema_version, **expected)
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "PaperRadialBaseline":
        if values.get("schema_version") != cls.schema_version:
            raise ValueError("Radial baseline schema mismatch")
        metadata = values["metadata"]
        validate_radial_metadata(metadata)
        physical = values["physical_baseline"]
        normalized = values["normalized_baseline"]
        return cls(
            mode=str(values["mode"]),
            provenance=PriorProvenance.from_dict(values["provenance"]),
            r=tuple(float(value) for value in values["r"]),
            rho_physical=tuple(float(value) for value in physical["rho"]),
            press_physical=tuple(float(value) for value in physical["press"]),
            rho_normalized=tuple(float(value) for value in normalized["rho"]),
            press_normalized=tuple(float(value) for value in normalized["press"]),
            fit=dict(values["fit"]),
            exact_paper_formula=bool(metadata["exact_paper_formula"]),
            adaptation_level=str(metadata["adaptation_level"]),
        )


def _validate_fit(
    h5_path: str,
    indices: tuple[int, ...],
    r: np.ndarray,
    preprocessor: PaperPreprocessor,
) -> tuple[int, int, int]:
    if not indices or tuple(sorted(set(indices))) != indices:
        raise ValueError("Radial fit indices must be non-empty, unique, and increasing")
    preprocessor.validate_compatibility(
        h5_path=h5_path,
        expected_training_indices=indices,
        expected_protocol_name=PAPER_PROTOCOL_NAME,
    )
    with h5py.File(h5_path, "r") as handle:
        snapshots = handle["snapshots"]
        if snapshots.shape[1] != 8 or snapshots.shape[-1] != len(r):
            raise ValueError("Radial fit HDF5 shape is incompatible")
        nphi, ntheta, nr = (int(value) for value in snapshots.shape[2:])
    r = np.asarray(r, dtype=np.float64)
    if r.shape != (nr,) or np.any(r <= 0) or np.any(np.diff(r) <= 0):
        raise ValueError("Physical r must be positive, increasing, and match HDF5 Nr")
    return nphi, ntheta, nr


def fit_appendix_literal_press_proxy(
    h5_path: str,
    *,
    training_indices: Iterable[int],
    r: np.ndarray,
    preprocessor: PaperPreprocessor,
    provenance: PriorProvenance,
) -> PaperRadialBaseline:
    """Fit ``log10(rho+press+epsilon_U) ~= k*r`` with no intercept."""

    indices = tuple(int(index) for index in training_indices)
    nphi, ntheta, _ = _validate_fit(h5_path, indices, r, preprocessor)
    provenance.validate(
        source_hdf5_checksum=preprocessor.source_hdf5_checksum,
        training_indices=indices,
        protocol_name=preprocessor.protocol_name,
        thermal_channel=preprocessor.thermal_channel,
    )
    minimum_positive = np.inf
    rho_sum = 0.0
    press_sum = 0.0
    with h5py.File(h5_path, "r") as handle:
        snapshots = handle["snapshots"]
        for index in indices:
            rho = np.asarray(snapshots[index, 3], dtype=np.float64)
            press = np.asarray(snapshots[index, 4], dtype=np.float64)
            total = rho + press
            positive = total[total > 0]
            if positive.size:
                minimum_positive = min(minimum_positive, float(np.min(positive)))
            rho_sum += float(np.sum(rho, dtype=np.float64))
            press_sum += float(np.sum(press, dtype=np.float64))
    epsilon_u = _epsilon_from_min_positive(minimum_positive)
    numerator = 0.0
    r_values = np.asarray(r, dtype=np.float64)
    denominator = len(indices) * nphi * ntheta * float(np.sum(np.square(r_values)))
    with h5py.File(h5_path, "r") as handle:
        snapshots = handle["snapshots"]
        for index in indices:
            total = (
                np.asarray(snapshots[index, 3], dtype=np.float64)
                + np.asarray(snapshots[index, 4], dtype=np.float64)
            )
            response = np.log10(total + epsilon_u)
            numerator += float(np.sum(response * r_values.reshape(1, 1, -1)))
    if denominator <= 0:
        raise FloatingPointError("Literal radial least-squares denominator is invalid")
    k = numerator / denominator
    total_sum = rho_sum + press_sum
    if total_sum <= 0 or not np.isfinite(k):
        raise FloatingPointError("Literal radial fit is not finite/positive")
    rho_share = rho_sum / total_sum
    press_share = press_sum / total_sum
    u_physical = np.maximum(10.0 ** (k * r_values) - epsilon_u, np.finfo(np.float64).tiny)
    rho_physical = np.maximum(rho_share * u_physical, np.finfo(np.float64).tiny)
    press_physical = np.maximum(press_share * u_physical, np.finfo(np.float64).tiny)
    rho_normalized = _positive_channel_normalized(rho_physical, 3, preprocessor)
    press_normalized = _positive_channel_normalized(press_physical, 4, preprocessor)
    return PaperRadialBaseline(
        mode="appendix_literal_press_proxy",
        provenance=provenance,
        r=tuple(float(value) for value in r_values),
        rho_physical=tuple(float(value) for value in rho_physical),
        press_physical=tuple(float(value) for value in press_physical),
        rho_normalized=tuple(float(value) for value in rho_normalized),
        press_normalized=tuple(float(value) for value in press_normalized),
        fit={
            "formula": "log10(rho + press + epsilon_U) ~= k*r",
            "k": float(k),
            "intercept": 0.0,
            "intercept_enabled": False,
            "epsilon_U": float(epsilon_u),
            "epsilon_rule": "10^(floor(log10(min_positive_train_U))-2)",
            "paper_reference_k": PAPER_REFERENCE_K,
            "paper_reference_for_comparison_only": True,
            "thermal_proxy": "press",
            "channel_partition": {
                "method": "train_global_fraction_of_rho_plus_press",
                "rho": float(rho_share),
                "press": float(press_share),
                "paper_adaptation": True,
            },
        },
        exact_paper_formula=True,
        adaptation_level="appendix_literal_with_press_proxy_and_channel_partition",
    )


def fit_spherical_logr_channelwise(
    h5_path: str,
    *,
    training_indices: Iterable[int],
    r: np.ndarray,
    preprocessor: PaperPreprocessor,
    provenance: PriorProvenance,
) -> PaperRadialBaseline:
    """Fit canonical positive pre-transforms against ``log10(r)`` by channel."""

    indices = tuple(int(index) for index in training_indices)
    nphi, ntheta, _ = _validate_fit(h5_path, indices, r, preprocessor)
    provenance.validate(
        source_hdf5_checksum=preprocessor.source_hdf5_checksum,
        training_indices=indices,
        protocol_name=preprocessor.protocol_name,
        thermal_channel=preprocessor.thermal_channel,
    )
    x = np.log10(np.asarray(r, dtype=np.float64))
    repeats = len(indices) * nphi * ntheta
    count = repeats * len(x)
    sum_x = repeats * float(np.sum(x))
    sum_x2 = repeats * float(np.sum(np.square(x)))
    coefficients: dict[str, dict[str, float]] = {}
    physical: dict[str, np.ndarray] = {}
    normalized: dict[str, np.ndarray] = {}
    with h5py.File(h5_path, "r") as handle:
        snapshots = handle["snapshots"]
        for channel in (3, 4):
            sum_y = 0.0
            sum_xy = 0.0
            for index in indices:
                values = np.asarray(snapshots[index, channel], dtype=np.float64)
                transformed = np.log10(values + preprocessor.epsilon[channel])
                sum_y += float(np.sum(transformed, dtype=np.float64))
                sum_xy += float(
                    np.sum(transformed * x.reshape(1, 1, -1), dtype=np.float64)
                )
            denominator = count * sum_x2 - sum_x * sum_x
            if denominator <= 0:
                raise FloatingPointError("Adapted radial least-squares denominator is invalid")
            k = (count * sum_xy - sum_x * sum_y) / denominator
            b = (sum_y - k * sum_x) / count
            transformed_baseline = k * x + b
            channel_physical = np.maximum(
                10.0 ** transformed_baseline - preprocessor.epsilon[channel],
                np.finfo(np.float64).tiny,
            )
            name = CHANNELS[channel]
            coefficients[name] = {"k": float(k), "b": float(b)}
            physical[name] = channel_physical
            normalized[name] = _positive_channel_normalized(
                channel_physical, channel, preprocessor
            )
    return PaperRadialBaseline(
        mode="spherical_logr_channelwise",
        provenance=provenance,
        r=tuple(float(value) for value in r),
        rho_physical=tuple(float(value) for value in physical["rho"]),
        press_physical=tuple(float(value) for value in physical["press"]),
        rho_normalized=tuple(float(value) for value in normalized["rho"]),
        press_normalized=tuple(float(value) for value in normalized["press"]),
        fit={
            "formula": "B_c(r) = k_c*log10(r) + b_c in positive transformed space",
            "channels": coefficients,
            "exact_paper_formula": False,
        },
        exact_paper_formula=False,
        adaptation_level="stronger_spherical_adaptation",
    )


def _sample_flat(values: np.ndarray, limit: int) -> np.ndarray:
    flat = np.asarray(values).reshape(-1)
    if flat.size <= limit:
        return flat.astype(np.float64, copy=True)
    indices = np.linspace(0, flat.size - 1, num=limit, dtype=np.int64)
    return flat[indices].astype(np.float64, copy=False)


def evaluate_radial_baseline(
    baseline: PaperRadialBaseline,
    h5_path: str,
    *,
    snapshot_indices: Iterable[int],
    preprocessor: PaperPreprocessor,
    shell_edges: Iterable[float],
    envelope_delta: float = 1.5,
    max_samples_per_snapshot_channel: int = 4096,
) -> dict[str, Any]:
    """Report fixed-candidate diagnostics without fitting on the supplied split."""

    indices = tuple(int(index) for index in snapshot_indices)
    if not indices:
        raise ValueError("Radial evaluation needs at least one snapshot")
    edges = np.asarray(tuple(shell_edges), dtype=np.float64)
    r = np.asarray(baseline.r, dtype=np.float64)
    shell_index = np.digitize(r, edges[1:-1], right=False)
    n_shells = len(edges) - 1
    channel_records: dict[str, Any] = {}
    with h5py.File(h5_path, "r") as handle:
        shape = tuple(int(value) for value in handle["snapshots"].shape[2:])
        normalized_baseline = baseline.state(shape, normalized=True).numpy()
        physical_baseline = baseline.state(shape, normalized=False).numpy()
        for channel, name in ((3, "rho"), (4, "press")):
            count = 0
            sum_value = 0.0
            sum_square = 0.0
            minimum = np.inf
            maximum = -np.inf
            violation_count = 0
            physical_error_square = 0.0
            physical_truth_square = 0.0
            shell_sum = np.zeros(n_shells, dtype=np.float64)
            shell_count = np.zeros(n_shells, dtype=np.int64)
            samples: list[np.ndarray] = []
            for index in indices:
                raw = np.asarray(handle["snapshots"][index], dtype=np.float32)
                normalized = preprocessor.encode(raw)
                residual = normalized[channel].astype(np.float64) - normalized_baseline[
                    channel
                ].astype(np.float64)
                count += residual.size
                sum_value += float(np.sum(residual, dtype=np.float64))
                sum_square += float(np.sum(np.square(residual), dtype=np.float64))
                minimum = min(minimum, float(np.min(residual)))
                maximum = max(maximum, float(np.max(residual)))
                violation_count += int(np.count_nonzero(np.abs(residual) > envelope_delta))
                samples.append(_sample_flat(residual, max_samples_per_snapshot_channel))
                physical_error = raw[channel].astype(np.float64) - physical_baseline[
                    channel
                ].astype(np.float64)
                physical_error_square += float(
                    np.sum(np.square(physical_error), dtype=np.float64)
                )
                physical_truth_square += float(
                    np.sum(np.square(raw[channel].astype(np.float64)), dtype=np.float64)
                )
                for shell in range(n_shells):
                    selection = shell_index == shell
                    shell_values = residual[..., selection]
                    shell_sum[shell] += float(np.sum(shell_values, dtype=np.float64))
                    shell_count[shell] += shell_values.size
            sampled = np.concatenate(samples)
            mean = sum_value / count
            std = np.sqrt(max(sum_square / count - mean * mean, 0.0))
            median = float(np.median(sampled))
            mad = float(np.median(np.abs(sampled - median)))
            qlow, qhigh = (float(value) for value in np.quantile(sampled, (0.001, 0.999)))
            shell_means = shell_sum / shell_count
            channel_records[name] = {
                "baseline_normalized_range": [
                    float(np.min(normalized_baseline[channel])),
                    float(np.max(normalized_baseline[channel])),
                ],
                "baseline_physical_range": [
                    float(np.min(physical_baseline[channel])),
                    float(np.max(physical_baseline[channel])),
                ],
                "residual_mean": float(mean),
                "residual_std": float(std),
                "residual_median": median,
                "residual_mad": mad,
                "residual_min": float(minimum),
                "residual_max": float(maximum),
                "residual_q0.001": qlow,
                "residual_q0.999": qhigh,
                "residual_q_span": float(qhigh - qlow),
                "shell_residual_mean": [float(value) for value in shell_means],
                "inner_two_shell_bias": float(np.mean(shell_means[:2])),
                "outer_two_shell_bias": float(np.mean(shell_means[-2:])),
                "residual_dynamic_range": float(maximum - minimum),
                "physical_reconstruction_relative_l2": float(
                    np.sqrt(physical_error_square / physical_truth_square)
                ),
                "envelope_delta": float(envelope_delta),
                "envelope_violation_fraction": float(violation_count / count),
                "finite": bool(
                    np.all(np.isfinite(sampled)) and np.all(np.isfinite(shell_means))
                ),
                "quantile_sampling": {
                    "method": "deterministic_even_stride_per_snapshot",
                    "sample_count": int(sampled.size),
                },
            }
    finite = all(record["finite"] for record in channel_records.values())
    return {
        "mode": baseline.mode,
        "snapshot_indices": list(indices),
        "snapshot_count": len(indices),
        "channels": channel_records,
        "finite": finite,
    }


def select_radial_mode(
    literal_audit: Mapping[str, Any], adapted_audit: Mapping[str, Any]
) -> tuple[str, str]:
    """Keep the literal path unless it has an explicit implementation/numeric failure."""

    if bool(literal_audit.get("finite")):
        return (
            "appendix_literal_press_proxy",
            "literal candidate is finite; lower adapted residual is not a selection criterion",
        )
    if bool(adapted_audit.get("finite")):
        return (
            "spherical_logr_channelwise",
            "literal candidate failed finite/implementation checks",
        )
    raise FloatingPointError("Neither radial candidate passed finite checks")
