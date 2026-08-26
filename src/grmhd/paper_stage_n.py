"""Isolated Stage N transform prototypes and read-only operator probes.

Nothing in this module mutates the canonical preprocessor or trains a model.
Prototype statistics and mappings are deliberately separate artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import torch

from . import CHANNELS
from .dataset import sha256_file
from .paper_preprocessing import (
    PAPER_GAMMA,
    PAPER_INVERSE_CLAMP_FRACTION,
    PAPER_TRANSFORMS,
    PaperPreprocessor,
)
from .paper_stage_l_attribution import (
    SPECTRAL_BANDS,
    basic_field_metrics,
    radial_profile,
    relative_l2,
    shell_metrics,
    spectrum_metrics,
)
from .paper_stage_m import forward_nonlinear, inverse_nonlinear


STAGE_N_GATE_VERSION = "stage_m_v1"
PROTOTYPE_METADATA = {
    "classification": "diagnostic_transform_prototype",
    "canonical_replacement": False,
    "authorized_for_training": False,
    "authorized_for_checkpoint_evaluation": False,
    "paper_faithful": False,
}
CANONICAL = "canonical"
NO_SOFTCLIP = "no_softclip"
MINIMAL_INVERSE_CLAMP = "minimal_inverse_clamp"
CORE_METRICS = (
    "global_variance_retention",
    "shell_radial_variance_retention",
    "high_k_retention",
    "dynamic_span_retention",
)


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def minimal_inverse_clamp_limit(
    gamma: float = PAPER_GAMMA, *, dtype: np.dtype[Any] = np.dtype(np.float32)
) -> float:
    """Largest representable value strictly inside the atanh singularity."""

    value = np.asarray(gamma, dtype=dtype)
    zero = np.asarray(0.0, dtype=dtype)
    limit = float(np.nextafter(value, zero))
    if not np.isfinite(limit) or not 0.0 < limit < float(gamma):
        raise FloatingPointError("Could not construct a finite inverse-softclip boundary")
    return limit


@dataclass(frozen=True)
class PrototypeSpec:
    key: str
    name: str
    channel_policies: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.channel_policies) != len(CHANNELS):
            raise ValueError("Stage N prototype needs exactly eight channel policies")
        allowed = {CANONICAL, NO_SOFTCLIP, MINIMAL_INVERSE_CLAMP}
        if not set(self.channel_policies) <= allowed:
            raise ValueError("Unknown Stage N transform policy")

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            **PROTOTYPE_METADATA,
            "channel_policies": dict(zip(CHANNELS, self.channel_policies, strict=True)),
        }


def prototype_specs(bcc2_policy: str | None = None) -> dict[str, PrototypeSpec]:
    canonical = [CANONICAL] * len(CHANNELS)
    p1 = canonical.copy()
    p1[2] = NO_SOFTCLIP
    p1[7] = NO_SOFTCLIP
    p2a = canonical.copy()
    p2a[1] = MINIMAL_INVERSE_CLAMP
    p2b = canonical.copy()
    p2b[1] = NO_SOFTCLIP
    result = {
        "P0": PrototypeSpec("P0", "CANONICAL_CONTROL", tuple(canonical)),
        "P1": PrototypeSpec("P1", "BCC3_VEL3_NO_SOFTCLIP", tuple(p1)),
        "P2A": PrototypeSpec("P2A", "BCC2_MINIMAL_INVERSE_CLAMP", tuple(p2a)),
        "P2B": PrototypeSpec("P2B", "BCC2_NO_SOFTCLIP", tuple(p2b)),
    }
    if bcc2_policy is not None:
        if bcc2_policy not in {MINIMAL_INVERSE_CLAMP, NO_SOFTCLIP}:
            raise ValueError("Combined prototype requires a frozen Bcc2 policy")
        p3 = p1.copy()
        p3[1] = bcc2_policy
        result["P3"] = PrototypeSpec("P3", "COMBINED_PROTOTYPE_V1", tuple(p3))
    return result


def assert_train_only_indices(
    indices: Iterable[int], *, train_range: tuple[int, int] = (11, 91)
) -> tuple[int, ...]:
    observed = tuple(int(value) for value in indices)
    expected = tuple(range(*train_range))
    if observed != expected:
        raise ValueError("Stage N prototype statistics require exactly train snapshots 11..90")
    return observed


class PrototypePreprocessor:
    """Prototype-only wrapper around separately fitted robust statistics."""

    def __init__(self, base: PaperPreprocessor, spec: PrototypeSpec) -> None:
        self.base = base
        self.spec = spec
        # ``PaperPreprocessor`` already requires non-empty, unique, strictly
        # increasing fit indices.  Keep the historical Stage-N 11..90 freeze
        # in ``fit_hdf5`` below, but do not bake that experiment-specific split
        # into the reusable transform wrapper/load path.  Later stages may
        # refit the same frozen channel mapping on a different train-only split.
        self.minimal_limit = minimal_inverse_clamp_limit(base.gamma)

    @property
    def gamma(self) -> float:
        return self.base.gamma

    @property
    def inverse_clamp_fraction(self) -> float:
        return self.base.inverse_clamp_fraction

    @property
    def epsilon(self) -> np.ndarray:
        return self.base.epsilon

    @property
    def median(self) -> np.ndarray:
        return self.base.median

    @property
    def scale(self) -> np.ndarray:
        return self.base.scale

    @property
    def training_indices(self) -> tuple[int, ...]:
        return self.base.training_indices

    @property
    def source_hdf5_checksum(self) -> str:
        return self.base.source_hdf5_checksum

    @property
    def protocol_name(self) -> str:
        return self.base.protocol_name

    @property
    def thermal_channel(self) -> str:
        return self.base.thermal_channel

    @property
    def paper_adaptation(self) -> bool:
        return self.base.paper_adaptation

    @property
    def eos_conversion(self) -> str:
        return self.base.eos_conversion

    @classmethod
    def fit_hdf5(
        cls,
        h5_path: str | Path,
        *,
        spec: PrototypeSpec,
        training_indices: Iterable[int],
        expected_source_hdf5_checksum: str,
        protocol_name: str,
    ) -> "PrototypePreprocessor":
        indices = assert_train_only_indices(training_indices)
        base = PaperPreprocessor.fit_hdf5(
            h5_path,
            training_indices=indices,
            protocol_name=f"{protocol_name}__stage_n__{spec.key.lower()}",
            expected_source_hdf5_checksum=expected_source_hdf5_checksum,
            gamma=PAPER_GAMMA,
            inverse_clamp_fraction=PAPER_INVERSE_CLAMP_FRACTION,
            thermal_channel="press",
            paper_adaptation=True,
            eos_conversion="disabled_unverified_gamma",
        )
        return cls(base, spec)

    def encode_numpy(self, values: np.ndarray, channel_axis: int | None = None) -> np.ndarray:
        array = np.asarray(values)
        axis = self.base._channel_axis(array.shape, channel_axis)
        output = np.empty_like(array, dtype=np.result_type(array.dtype, np.float32))
        for channel, policy in enumerate(self.spec.channel_policies):
            selection = [slice(None)] * array.ndim
            selection[axis] = channel
            key = tuple(selection)
            transformed = np.asarray(array[key], dtype=np.float64).copy()
            self.base._transform_values_inplace(
                transformed, channel, self.base.epsilon[channel]
            )
            z = (transformed - self.base.median[channel]) / self.base.scale[channel]
            output[key] = z if policy == NO_SOFTCLIP else self.base.gamma * np.tanh(
                z / self.base.gamma
            )
        if not np.isfinite(output).all():
            raise FloatingPointError("Stage N prototype encode produced NaN/Inf")
        return output

    def decode_numpy(self, values: np.ndarray, channel_axis: int | None = None) -> np.ndarray:
        array = np.asarray(values)
        axis = self.base._channel_axis(array.shape, channel_axis)
        output = np.empty_like(array, dtype=np.result_type(array.dtype, np.float32))
        for channel, policy in enumerate(self.spec.channel_policies):
            selection = [slice(None)] * array.ndim
            selection[axis] = channel
            key = tuple(selection)
            item = np.asarray(array[key], dtype=np.float64)
            if policy == NO_SOFTCLIP:
                z = item
            else:
                limit = (
                    self.minimal_limit
                    if policy == MINIMAL_INVERSE_CLAMP
                    else self.base.inverse_clamp_fraction * self.base.gamma
                )
                clipped = np.clip(item, -limit, limit)
                with np.errstate(divide="raise", invalid="raise", over="raise"):
                    z = self.base.gamma * np.arctanh(clipped / self.base.gamma)
            transformed = z * self.base.scale[channel] + self.base.median[channel]
            output[key] = inverse_nonlinear(
                transformed,
                kind=PAPER_TRANSFORMS[channel],
                epsilon=self.base.epsilon[channel],
                physical_limit=np.finfo(output.dtype).max,
            )
        if not np.isfinite(output).all():
            raise FloatingPointError("Stage N prototype decode produced NaN/Inf")
        return output

    def encode_tensor(
        self, values: torch.Tensor, channel_axis: int | None = None
    ) -> torch.Tensor:
        axis = self.base._channel_axis(tuple(values.shape), channel_axis)
        output = torch.empty_like(values)
        for channel, (kind, policy) in enumerate(
            zip(PAPER_TRANSFORMS, self.spec.channel_policies, strict=True)
        ):
            selection = [slice(None)] * values.ndim
            selection[axis] = channel
            key = tuple(selection)
            item = values[key]
            if not torch.isfinite(item).all():
                raise FloatingPointError(f"{CHANNELS[channel]} contains NaN/Inf")
            epsilon = torch.as_tensor(
                self.base.epsilon[channel], dtype=item.dtype, device=item.device
            )
            if kind == "positive_log":
                if torch.any(item < 0):
                    raise ValueError(f"{CHANNELS[channel]} contains negative values")
                transformed = torch.log10(item + epsilon)
            elif kind == "signed_log":
                transformed = torch.sign(item) * torch.log10(
                    1.0 + torch.abs(item) / epsilon
                )
            else:
                transformed = item
            median = torch.as_tensor(
                self.base.median[channel], dtype=item.dtype, device=item.device
            )
            scale = torch.as_tensor(
                self.base.scale[channel], dtype=item.dtype, device=item.device
            )
            z = (transformed - median) / scale
            encoded = (
                z
                if policy == NO_SOFTCLIP
                else self.gamma * torch.tanh(z / self.gamma)
            )
            if not torch.isfinite(encoded).all():
                work = item.to(dtype=torch.float64)
                epsilon64 = torch.as_tensor(
                    self.base.epsilon[channel],
                    dtype=torch.float64,
                    device=item.device,
                )
                if kind == "positive_log":
                    transformed64 = torch.log10(work + epsilon64)
                elif kind == "signed_log":
                    transformed64 = torch.sign(work) * torch.log10(
                        1.0 + torch.abs(work) / epsilon64
                    )
                else:
                    transformed64 = work
                median64 = torch.as_tensor(
                    self.base.median[channel],
                    dtype=torch.float64,
                    device=item.device,
                )
                scale64 = torch.as_tensor(
                    self.base.scale[channel],
                    dtype=torch.float64,
                    device=item.device,
                )
                z64 = (transformed64 - median64) / scale64
                encoded = (
                    z64
                    if policy == NO_SOFTCLIP
                    else self.gamma * torch.tanh(z64 / self.gamma)
                ).to(dtype=item.dtype)
            output[key] = encoded
        if not torch.isfinite(output).all():
            raise FloatingPointError("Stage N prototype encode produced NaN/Inf")
        return output

    def decode_tensor(
        self, values: torch.Tensor, channel_axis: int | None = None
    ) -> torch.Tensor:
        axis = self.base._channel_axis(tuple(values.shape), channel_axis)
        output = torch.empty_like(values)
        for channel, (kind, policy) in enumerate(
            zip(PAPER_TRANSFORMS, self.spec.channel_policies, strict=True)
        ):
            selection = [slice(None)] * values.ndim
            selection[axis] = channel
            key = tuple(selection)
            item = values[key]
            work_dtype = (
                torch.float64
                if item.dtype in (torch.float16, torch.bfloat16, torch.float32)
                else item.dtype
            )
            work = item.to(dtype=work_dtype)
            if policy == NO_SOFTCLIP:
                z = work
            else:
                limit = (
                    self.minimal_limit
                    if policy == MINIMAL_INVERSE_CLAMP
                    else self.inverse_clamp_fraction * self.gamma
                )
                z = self.gamma * torch.atanh(
                    torch.clamp(work, -limit, limit) / self.gamma
                )
            median = torch.as_tensor(
                self.base.median[channel], dtype=work_dtype, device=item.device
            )
            scale = torch.as_tensor(
                self.base.scale[channel], dtype=work_dtype, device=item.device
            )
            transformed = z * scale + median
            epsilon = torch.as_tensor(
                self.base.epsilon[channel], dtype=work_dtype, device=item.device
            )
            physical_limit = torch.finfo(item.dtype).max
            physical_limit_tensor = torch.as_tensor(
                physical_limit, dtype=work_dtype, device=item.device
            )
            if kind == "positive_log":
                exponent_limit = torch.log10(physical_limit_tensor + epsilon)
                decoded = (
                    torch.pow(10.0, torch.minimum(transformed, exponent_limit))
                    - epsilon
                )
                decoded = torch.clamp_min(decoded, torch.finfo(item.dtype).tiny)
            elif kind == "signed_log":
                exponent_limit = torch.log10(physical_limit_tensor) - torch.log10(
                    epsilon
                )
                decoded = torch.sign(transformed) * epsilon * (
                    torch.pow(
                        10.0, torch.minimum(torch.abs(transformed), exponent_limit)
                    )
                    - 1.0
                )
            else:
                decoded = torch.clamp(
                    transformed, -physical_limit, physical_limit
                )
            decoded = torch.clamp(decoded, -physical_limit_tensor, physical_limit_tensor)
            output[key] = decoded.to(dtype=item.dtype)
        if not torch.isfinite(output).all():
            raise FloatingPointError("Stage N prototype decode produced NaN/Inf")
        return output

    def encode(
        self, values: np.ndarray | torch.Tensor, channel_axis: int | None = None
    ):
        return (
            self.encode_tensor(values, channel_axis)
            if isinstance(values, torch.Tensor)
            else self.encode_numpy(values, channel_axis)
        )

    def decode(
        self, values: np.ndarray | torch.Tensor, channel_axis: int | None = None
    ):
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
        self.base.validate_compatibility(
            h5_path=h5_path,
            expected_source_hdf5_checksum=expected_source_hdf5_checksum,
            expected_training_indices=expected_training_indices,
            expected_protocol_name=expected_protocol_name,
        )

    def round_trip(self, values: np.ndarray, channel_axis: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        encoded = self.encode_numpy(values, channel_axis)
        return encoded, self.decode_numpy(encoded, channel_axis)

    def resolved_config(self) -> dict[str, Any]:
        return {
            "schema_version": "paper-stage-n-transform-prototype-v1",
            **self.spec.as_dict(),
            "canonical_gamma": self.base.gamma,
            "canonical_inverse_clamp": self.base.inverse_clamp_fraction * self.base.gamma,
            "minimal_inverse_clamp": self.minimal_limit,
            "minimal_inverse_clamp_derivation": "float32 nextafter(gamma, 0)",
            "training_indices": list(self.base.training_indices),
            "validation_indices_used_for_fit": [],
            "source_hdf5_checksum": self.base.source_hdf5_checksum,
        }

    def normalizer_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": "paper-stage-n-prototype-normalizer-v1",
            **PROTOTYPE_METADATA,
            "prototype": self.spec.key,
            "epsilon": self.base.epsilon.tolist(),
            "median": self.base.median.tolist(),
            "scale": self.base.scale.tolist(),
            "gamma": self.base.gamma,
            "training_indices": list(self.base.training_indices),
            "validation_indices_used_for_fit": [],
            "fit_scope": self.base.fit_scope,
            "sample_count_per_channel": self.base.sample_count_per_channel,
            "source_hdf5_checksum": self.base.source_hdf5_checksum,
            "protocol_name": self.base.protocol_name,
            "fit_warnings": list(self.base.fit_warnings),
        }

    @classmethod
    def load(cls, directory: str | Path, *, spec: PrototypeSpec) -> "PrototypePreprocessor":
        directory = Path(directory)
        metadata = json.loads((directory / "normalizer.json").read_text(encoding="utf-8"))
        if metadata.get("prototype") != spec.key or any(
            metadata.get(key) != value for key, value in PROTOTYPE_METADATA.items()
        ):
            raise ValueError("Stage N prototype normalizer identity mismatch")
        with np.load(directory / "normalizer.npz", allow_pickle=False) as values:
            base = PaperPreprocessor(
                values["epsilon"], values["median"], values["scale"],
                training_indices=tuple(int(value) for value in values["training_indices"]),
                source_hdf5_checksum=metadata["source_hdf5_checksum"],
                protocol_name=metadata["protocol_name"],
                thermal_channel="press", paper_adaptation=True,
                eos_conversion="disabled_unverified_gamma",
                fit_scope=metadata["fit_scope"],
                fit_warnings=metadata["fit_warnings"],
                sample_count_per_channel=metadata["sample_count_per_channel"],
            )
        return cls(base, spec)

    def save(self, directory: str | Path) -> dict[str, Any]:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        config = self.resolved_config()
        config_path = output / "resolved_config.json"
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
        metadata = self.normalizer_metadata()
        metadata_path = output / "normalizer.json"
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        artifact_path = output / "normalizer.npz"
        np.savez(
            artifact_path,
            epsilon=self.base.epsilon,
            median=self.base.median,
            scale=self.base.scale,
            training_indices=np.asarray(self.base.training_indices, dtype=np.int64),
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        )
        fit_log = {
            "status": "fit_complete",
            "fit_split": "train_only",
            "snapshots_read": list(self.base.training_indices),
            "validation_snapshots_read": [],
            "finite_statistics": bool(
                np.isfinite(self.base.epsilon).all()
                and np.isfinite(self.base.median).all()
                and np.isfinite(self.base.scale).all()
            ),
        }
        fit_path = output / "fit_log.json"
        fit_path.write_text(json.dumps(fit_log, indent=2, sort_keys=True) + "\n")
        return {
            "directory": str(output),
            "config_sha256": sha256_file(config_path),
            "normalizer_json_sha256": sha256_file(metadata_path),
            "normalizer_artifact_sha256": sha256_file(artifact_path),
            "fit_log_sha256": sha256_file(fit_path),
        }


def _ratio(numerator: float, denominator: float, epsilon: float = 1.0e-30) -> float | None:
    return None if abs(float(denominator)) <= epsilon else float(numerator) / float(denominator)


def roundtrip_metrics(
    raw: np.ndarray,
    decoded: np.ndarray,
    encoded: np.ndarray,
    *,
    shell_index: np.ndarray,
    softclip_applied: bool,
    gamma: float = PAPER_GAMMA,
) -> dict[str, Any]:
    raw = np.asarray(raw, dtype=np.float64)
    decoded = np.asarray(decoded, dtype=np.float64)
    encoded = np.asarray(encoded, dtype=np.float64)
    if raw.shape != decoded.shape or raw.shape != encoded.shape or raw.ndim != 3:
        raise ValueError("Stage N round-trip metrics require aligned 3-D fields")
    finite = np.isfinite(decoded)
    if not finite.all():
        return {"finite": False, "finite_count": int(finite.sum()), "value_count": int(raw.size)}
    raw_basic = basic_field_metrics(raw)
    decoded_basic = basic_field_metrics(decoded)
    raw_shells = shell_metrics(raw, shell_index)
    decoded_shells = shell_metrics(decoded, shell_index)
    shell_retentions = [
        _ratio(item["variance"], reference["variance"])
        for item, reference in zip(decoded_shells, raw_shells, strict=True)
    ]
    shell_defined = [value for value in shell_retentions if value is not None]
    raw_radial = radial_profile(raw)
    decoded_radial = radial_profile(decoded)
    raw_spectrum = spectrum_metrics(raw, axis="combined", demean=True)
    decoded_spectrum = spectrum_metrics(decoded, axis="combined", demean=True)
    spectral_retentions = {
        band: _ratio(decoded_spectrum[f"{band}_energy"], raw_spectrum[f"{band}_energy"])
        for band in SPECTRAL_BANDS
    }
    shell_median = None if not shell_defined else float(np.median(shell_defined))
    radial_retention = _ratio(decoded_radial["variance"], raw_radial["variance"])
    shell_radial_defined = [v for v in (shell_median, radial_retention) if v is not None]
    return {
        "finite": True,
        "finite_count": int(raw.size),
        "value_count": int(raw.size),
        "raw_to_oracle_relative_l2": relative_l2(decoded, raw, epsilon=1.0e-30),
        "global_variance_retention": _ratio(decoded_basic["variance"], raw_basic["variance"]),
        "standard_deviation_retention": _ratio(decoded_basic["std"], raw_basic["std"]),
        "dynamic_span_retention": _ratio(
            decoded_basic["dynamic_span_q99_q01"], raw_basic["dynamic_span_q99_q01"]
        ),
        "shell_variance_retention": shell_median,
        "radial_profile_variance_retention": radial_retention,
        "shell_radial_variance_retention": (
            None if not shell_radial_defined else float(min(shell_radial_defined))
        ),
        "total_variation_retention": _ratio(
            decoded_basic["total_variation"], raw_basic["total_variation"]
        ),
        "gradient_energy_retention": _ratio(
            decoded_basic["gradient_energy"], raw_basic["gradient_energy"]
        ),
        "low_k_retention": spectral_retentions["low_k"],
        "mid_k_retention": spectral_retentions["mid_k"],
        "high_k_retention": spectral_retentions["high_k"],
        "sign_agreement": float(np.mean(np.signbit(decoded) == np.signbit(raw))),
        "near_zero_occupancy": decoded_basic["near_zero_occupancy"],
        "exact_saturation_occupancy": (
            float(np.mean(np.abs(encoded) == float(gamma))) if softclip_applied else 0.0
        ),
        "inverse_clamp_occupancy": float(
            np.mean(np.abs(encoded) > PAPER_INVERSE_CLAMP_FRACTION * float(gamma))
        ),
        "decoded_minimum": decoded_basic["minimum"],
        "decoded_maximum": decoded_basic["maximum"],
    }


def aggregate_roundtrip(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot aggregate empty Stage N round-trip rows")
    metrics = sorted(
        key
        for key in rows[0]
        if key not in {"prototype", "snapshot", "split", "channel", "policy"}
        and isinstance(rows[0][key], (int, float, bool))
    )
    result: dict[str, Any] = {"snapshot_count": len(rows)}
    for metric in metrics:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        result[f"median_{metric}"] = None if not values else float(np.median(values))
        result[f"minimum_{metric}"] = None if not values else float(np.min(values))
    result["all_finite"] = all(bool(row.get("finite")) for row in rows)
    return result


def classify_readiness(
    *,
    target_summary: Mapping[str, Any],
    canonical_target_summary: Mapping[str, Any],
    control_summaries: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
    relative_l2_improvement_minimum: float = 0.50,
    core_retention_minimum: float = 0.80,
    core_categories_required: int = 3,
    control_degradation_maximum: float = 0.05,
) -> dict[str, Any]:
    target_l2 = target_summary["median_raw_to_oracle_relative_l2"]
    canonical_l2 = canonical_target_summary["median_raw_to_oracle_relative_l2"]
    improvement = 1.0 - float(target_l2) / max(float(canonical_l2), 1.0e-30)
    core = {
        name: target_summary[f"median_{name}"] for name in CORE_METRICS
    }
    core_passes = sum(
        value is not None and float(value) >= core_retention_minimum
        for value in core.values()
    )
    controls = []
    for candidate, canonical in control_summaries:
        current = float(candidate["median_raw_to_oracle_relative_l2"])
        reference = float(canonical["median_raw_to_oracle_relative_l2"])
        degradation = (current - reference) / max(reference, 1.0e-30)
        controls.append(degradation)
    max_control_degradation = max(controls, default=0.0)
    new_saturation = float(target_summary["median_exact_saturation_occupancy"]) > float(
        canonical_target_summary["median_exact_saturation_occupancy"]
    ) + 1.0e-15
    checks = {
        "A_all_train_finite": bool(target_summary["all_finite"]),
        "B_relative_l2_improvement": improvement >= relative_l2_improvement_minimum,
        "C_core_retention": core_passes >= core_categories_required,
        "D_control_channels": max_control_degradation <= control_degradation_maximum,
        "E_no_new_saturation_or_nonfinite": bool(target_summary["all_finite"]) and not new_saturation,
    }
    return {
        "ready": all(checks.values()),
        "checks": checks,
        "relative_l2_improvement": improvement,
        "core_retentions": core,
        "core_passes": core_passes,
        "maximum_control_relative_l2_degradation": max_control_degradation,
        "new_exact_saturation": new_saturation,
    }


def select_bcc2_candidate(p2a: Mapping[str, Any], p2b: Mapping[str, Any]) -> dict[str, Any]:
    a = bool(p2a["ready"])
    b = bool(p2b["ready"])
    if a and b:
        return {
            "selected": "P2A",
            "policy": MINIMAL_INVERSE_CLAMP,
            "reason": "both ready; preserve softclip, fewer branches, then smaller mapping deviation",
        }
    if a:
        return {"selected": "P2A", "policy": MINIMAL_INVERSE_CLAMP, "reason": "only P2A is train-ready"}
    if b:
        return {"selected": "P2B", "policy": NO_SOFTCLIP, "reason": "only P2B is train-ready"}
    return {"selected": None, "policy": None, "reason": "neither Bcc2 candidate is train-ready"}


def channel_prototype_decision(
    *, train_ready: bool, validation_summary: Mapping[str, Any], canonical_validation_l2: float
) -> str:
    if not bool(validation_summary.get("all_finite", False)):
        return "D. PROTOTYPE_UNSTABLE"
    l2 = float(validation_summary["median_raw_to_oracle_relative_l2"])
    core = sum(
        validation_summary[f"median_{metric}"] is not None
        and float(validation_summary[f"median_{metric}"]) >= 0.8
        for metric in CORE_METRICS
    )
    improved = l2 <= 0.5 * max(float(canonical_validation_l2), 1.0e-30)
    if train_ready and core >= 3 and improved:
        return "A. PROTOTYPE_READY"
    if train_ready:
        return "B. TRAIN_READY_VALIDATION_SHIFT"
    if l2 < float(canonical_validation_l2):
        return "C. PARTIAL_RECOVERY_ONLY"
    return "C. PARTIAL_RECOVERY_ONLY"


def combined_prototype_decision(channel_decisions: Mapping[str, str], *, p3_finite: bool) -> str:
    if not p3_finite:
        return "4. COMBINED_PROTOTYPE_ENGINEERING_FAILURE"
    choices = list(channel_decisions.values())
    if choices and all(value.startswith("A.") for value in choices):
        return "1. COMBINED_PROTOTYPE_READY_FOR_SHORT_PILOT"
    if any(value.startswith(("D.", "E.")) for value in choices):
        return "3. COMBINED_PROTOTYPE_NOT_READY"
    if any(value.startswith(("A.", "B.", "C.")) for value in choices):
        return "2. COMBINED_PROTOTYPE_PARTIALLY_READY"
    return "4. COMBINED_PROTOTYPE_ENGINEERING_FAILURE"


def normalized_operator_input(state: torch.Tensor, shells: torch.Tensor) -> torch.Tensor:
    if state.ndim != 5 or state.shape[1] != 8:
        raise ValueError("Stage N normalized physical state must be (B,8,phi,theta,r)")
    if shells.ndim == 4:
        shells = shells.unsqueeze(0)
    if shells.shape[1] != 8 or shells.shape[2:] != state.shape[2:]:
        raise ValueError("Stage N shell channels do not align with the state")
    return torch.cat((state, shells.expand(state.shape[0], -1, -1, -1, -1)), dim=1)


def apply_operator(model: torch.nn.Module, state: torch.Tensor, shells: torch.Tensor) -> torch.Tensor:
    output = model(x=normalized_operator_input(state, shells))
    if output.shape != state.shape or not torch.isfinite(output).all():
        raise FloatingPointError("Stage N operator response is invalid")
    return output


def constant_perturbation(state: torch.Tensor, channel: int, epsilon: float) -> torch.Tensor:
    result = torch.zeros_like(state)
    result[:, int(channel)] = float(epsilon)
    return result


def compact_impulse(
    state: torch.Tensor, channel: int, center: tuple[int, int, int], epsilon: float
) -> torch.Tensor:
    result = torch.zeros_like(state)
    slices = tuple(slice(max(0, value - 1), min(size, value + 2)) for value, size in zip(center, state.shape[2:], strict=True))
    result[(slice(None), int(channel), *slices)] = float(epsilon)
    return result


def shell_localized_perturbation(
    state: torch.Tensor, shell_mask: torch.Tensor, channel: int, epsilon: float
) -> torch.Tensor:
    mask = shell_mask.to(device=state.device, dtype=state.dtype)
    if mask.shape != state.shape[2:]:
        raise ValueError("Shell perturbation mask has the wrong shape")
    phi = torch.arange(state.shape[2], device=state.device, dtype=state.dtype)
    pattern = torch.sin(2.0 * torch.pi * phi / state.shape[2]).reshape(-1, 1, 1)
    pattern = pattern * mask
    selected = pattern[mask.bool()]
    if selected.numel() == 0:
        raise ValueError("Shell perturbation mask is empty")
    pattern = pattern - mask * selected.mean()
    norm = pattern.square().mean().sqrt().clamp_min(torch.finfo(state.dtype).eps)
    result = torch.zeros_like(state)
    result[:, int(channel)] = float(epsilon) * pattern / norm
    return result


def radial_mode(shape: tuple[int, int, int], name: str) -> np.ndarray:
    _, _, nr = shape
    u = np.linspace(-1.0, 1.0, nr, dtype=np.float64)
    modes = {
        "constant": np.ones(nr),
        "linear_log_r": u,
        "inner_localized": np.exp(-8.0 * np.square(u + 1.0)),
        "outer_localized": np.exp(-8.0 * np.square(u - 1.0)),
        "mid_frequency": np.sin(4.0 * np.pi * (u + 1.0) / 2.0),
    }
    if name not in modes:
        raise ValueError(f"Unknown Stage N radial mode {name}")
    values = modes[name].astype(np.float64)
    if name != "constant":
        values -= values.mean()
    values /= max(float(np.sqrt(np.mean(np.square(values)))), 1.0e-30)
    return np.broadcast_to(values.reshape(1, 1, nr), shape).copy()


def directional_mode(shape: tuple[int, int, int], axis: str, band: str) -> np.ndarray:
    axes = {"phi": 0, "theta": 1, "r": 2}
    if axis not in axes or band not in SPECTRAL_BANDS:
        raise ValueError("Stage N directional mode must reuse frozen axes/bands")
    size = shape[axes[axis]]
    cycles = {"low_k": max(1, size // 16), "mid_k": max(1, size // 5), "high_k": max(1, size * 3 // 8)}[band]
    coordinate = np.arange(size, dtype=np.float64)
    values = np.sin(2.0 * np.pi * cycles * coordinate / size)
    reshape = [1, 1, 1]
    reshape[axes[axis]] = size
    result = np.broadcast_to(values.reshape(reshape), shape).copy()
    result -= result.mean()
    result /= max(float(np.sqrt(np.mean(np.square(result)))), 1.0e-30)
    return result


def finite_difference_jvp(
    model: torch.nn.Module,
    state: torch.Tensor,
    shells: torch.Tensor,
    direction: torch.Tensor,
    epsilon: float,
    *,
    base_output: torch.Tensor | None = None,
) -> torch.Tensor:
    if direction.shape != state.shape:
        raise ValueError("Finite-difference direction must align with state")
    base = apply_operator(model, state, shells) if base_output is None else base_output
    return (apply_operator(model, state + float(epsilon) * direction, shells) - base) / float(epsilon)


def two_application_probe(
    model: torch.nn.Module, state: torch.Tensor, shells: torch.Tensor, *, applications: int = 2
) -> tuple[torch.Tensor, torch.Tensor]:
    if applications != 2:
        raise ValueError("Stage N fixed-point probe permits exactly two model applications")
    first = apply_operator(model, state, shells)
    second = apply_operator(model, first, shells)
    return first, second


def response_matrix(input_directions: Sequence[torch.Tensor], output_deltas: Sequence[torch.Tensor]) -> np.ndarray:
    if len(input_directions) != 8 or len(output_deltas) != 8:
        raise ValueError("Stage N channel response requires eight input directions")
    matrix = np.empty((8, 8), dtype=np.float64)
    for channel_in, (direction, delta) in enumerate(zip(input_directions, output_deltas, strict=True)):
        denominator = float(torch.linalg.vector_norm(direction[:, channel_in]).cpu())
        for channel_out in range(8):
            matrix[channel_out, channel_in] = float(
                torch.linalg.vector_norm(delta[:, channel_out]).cpu()
            ) / max(denominator, 1.0e-30)
    return matrix


def tensor_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def validate_manifest_dataset(manifest: Mapping[str, Any], h5_path: str | Path) -> None:
    path = Path(h5_path)
    if sha256_file(path) != manifest["dataset"]["sha256"]:
        raise RuntimeError("Stage N HDF5 checksum mismatch")
    with h5py.File(path, "r") as handle:
        if tuple(handle["snapshots"].shape) != (111, 8, 64, 64, 64):
            raise RuntimeError("Stage N HDF5 shape mismatch")
