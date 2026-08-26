"""Train-only prior metadata and the paper radial residual envelope.

This module is deliberately independent from :mod:`grmhd.priors`, whose
objects belong to the Round 1--3 extension path.  Stage D artifacts use the
frozen ``paper_reduced100`` protocol and carry enough provenance to reject
cross-dataset or cross-normalizer reuse.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch


PAPER_PROTOCOL_NAME = "paper_reduced100_press_spherical_ks"
PAPER_TRAIN_INDICES = tuple(range(11, 91))
PAPER_VALIDATION_INDICES = tuple(range(91, 111))
PAPER_THERMAL_CHANNEL = "press"
PAPER_COORDINATE_SYSTEM = "spherical Kerr-Schild: phi,theta,r"
RADIAL_MODES = ("appendix_literal_press_proxy", "spherical_logr_channelwise")


@dataclass(frozen=True)
class PriorProvenance:
    """Identity shared by every Stage D artifact."""

    source_hdf5_checksum: str
    preprocessing_stats_checksum: str
    training_indices: tuple[int, ...]
    protocol_name: str = PAPER_PROTOCOL_NAME
    thermal_channel: str = PAPER_THERMAL_CHANNEL
    paper_adaptation: bool = True
    eos_conversion: str = "disabled_unverified_gamma"
    coordinate_system: str = PAPER_COORDINATE_SYSTEM
    stored_components_not_cartesian: bool = True
    validation_not_used_for_fit: bool = True

    def __post_init__(self) -> None:
        if not self.source_hdf5_checksum or not self.preprocessing_stats_checksum:
            raise ValueError("Prior provenance requires HDF5 and preprocessing checksums")
        indices = tuple(int(index) for index in self.training_indices)
        if not indices or tuple(sorted(set(indices))) != indices:
            raise ValueError("Prior training indices must be non-empty, unique, and increasing")
        if self.protocol_name != PAPER_PROTOCOL_NAME:
            raise ValueError("Stage D prior protocol mismatch")
        if self.thermal_channel != PAPER_THERMAL_CHANNEL or not self.paper_adaptation:
            raise ValueError("Stage D requires the explicit press thermal adaptation")
        if self.eos_conversion != "disabled_unverified_gamma":
            raise ValueError("Stage D cannot enable an unverified EOS conversion")
        if self.coordinate_system != PAPER_COORDINATE_SYSTEM:
            raise ValueError("Stage D coordinate-system mismatch")
        if not self.stored_components_not_cartesian:
            raise ValueError("Stored vector components must remain marked non-Cartesian")
        if not self.validation_not_used_for_fit:
            raise ValueError("Validation data cannot be used to fit Stage D priors")

    def validate(
        self,
        *,
        source_hdf5_checksum: str | None = None,
        preprocessing_stats_checksum: str | None = None,
        training_indices: Iterable[int] | None = None,
        protocol_name: str | None = None,
        thermal_channel: str | None = None,
    ) -> None:
        if (
            source_hdf5_checksum is not None
            and source_hdf5_checksum != self.source_hdf5_checksum
        ):
            raise ValueError("Prior HDF5 checksum mismatch")
        if (
            preprocessing_stats_checksum is not None
            and preprocessing_stats_checksum != self.preprocessing_stats_checksum
        ):
            raise ValueError("Prior preprocessing checksum mismatch")
        if training_indices is not None and tuple(int(i) for i in training_indices) != self.training_indices:
            raise ValueError("Prior training indices mismatch")
        if protocol_name is not None and protocol_name != self.protocol_name:
            raise ValueError("Prior protocol mismatch")
        if thermal_channel is not None and thermal_channel != self.thermal_channel:
            raise ValueError("Prior thermal channel mismatch")

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_hdf5_checksum": self.source_hdf5_checksum,
            "preprocessing_stats_checksum": self.preprocessing_stats_checksum,
            "training_indices": list(self.training_indices),
            "protocol_name": self.protocol_name,
            "thermal_channel": self.thermal_channel,
            "paper_adaptation": self.paper_adaptation,
            "eos_conversion": self.eos_conversion,
            "coordinate_system": self.coordinate_system,
            "stored_components_not_cartesian": self.stored_components_not_cartesian,
            "validation_not_used_for_fit": self.validation_not_used_for_fit,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "PriorProvenance":
        return cls(
            source_hdf5_checksum=str(values["source_hdf5_checksum"]),
            preprocessing_stats_checksum=str(values["preprocessing_stats_checksum"]),
            training_indices=tuple(int(index) for index in values["training_indices"]),
            protocol_name=str(values.get("protocol_name", PAPER_PROTOCOL_NAME)),
            thermal_channel=str(values.get("thermal_channel", PAPER_THERMAL_CHANNEL)),
            paper_adaptation=bool(values.get("paper_adaptation", True)),
            eos_conversion=str(values.get("eos_conversion", "disabled_unverified_gamma")),
            coordinate_system=str(values.get("coordinate_system", PAPER_COORDINATE_SYSTEM)),
            stored_components_not_cartesian=bool(
                values.get("stored_components_not_cartesian", True)
            ),
            validation_not_used_for_fit=bool(
                values.get("validation_not_used_for_fit", True)
            ),
        )


def write_prior_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")


def read_prior_json(
    path: str | Path,
    *,
    expected_schema: str,
    source_hdf5_checksum: str,
    preprocessing_stats_checksum: str,
    training_indices: Iterable[int],
    protocol_name: str,
    thermal_channel: str,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != expected_schema:
        raise ValueError(f"Expected {expected_schema!r} artifact schema")
    provenance = PriorProvenance.from_dict(payload["provenance"])
    provenance.validate(
        source_hdf5_checksum=source_hdf5_checksum,
        preprocessing_stats_checksum=preprocessing_stats_checksum,
        training_indices=training_indices,
        protocol_name=protocol_name,
        thermal_channel=thermal_channel,
    )
    return payload


def validate_radial_metadata(metadata: Mapping[str, Any]) -> None:
    if metadata.get("mode") not in RADIAL_MODES:
        raise ValueError("Unsupported paper radial mode")
    if metadata.get("protocol_name") != PAPER_PROTOCOL_NAME:
        raise ValueError("Radial baseline protocol mismatch")
    if metadata.get("baseline_space") != "canonical_normalized_rho_press":
        raise ValueError("Radial baseline must be in canonical normalized rho/press space")
    if metadata.get("thermal_channel") != PAPER_THERMAL_CHANNEL:
        raise ValueError("Radial baseline thermal channel mismatch")


@dataclass(frozen=True)
class PaperResidualEnvelope:
    """Fixed Appendix-C envelope for normalized rho/press radial residuals."""

    provenance: PriorProvenance
    radial_mode: str
    delta_rho: float = 1.5
    delta_press: float = 1.5
    weight_rho: float = 0.05
    weight_press: float = 0.05
    space: str = "canonical_normalized_rho_press"

    schema_version = "paper-residual-envelope-v1"

    def __post_init__(self) -> None:
        if self.radial_mode not in RADIAL_MODES:
            raise ValueError("Unsupported paper radial mode")
        if self.space != "canonical_normalized_rho_press":
            raise ValueError("Residual envelope space changed")
        if min(self.delta_rho, self.delta_press) <= 0:
            raise ValueError("Residual envelope deltas must be positive")
        if min(self.weight_rho, self.weight_press) < 0:
            raise ValueError("Residual envelope weights cannot be negative")

    @staticmethod
    def _channel_axis(values: torch.Tensor, channel_axis: int | None) -> int:
        if channel_axis is None:
            if values.ndim >= 2 and values.shape[1] == 8:
                return 1
            if values.ndim >= 1 and values.shape[0] == 8:
                return 0
            raise ValueError(f"Cannot infer channel axis from {tuple(values.shape)}")
        axis = channel_axis % values.ndim
        if values.shape[axis] != 8:
            raise ValueError("Paper residual envelope requires an eight-channel axis")
        return axis

    def penalty_components(
        self,
        prediction: torch.Tensor,
        baseline: torch.Tensor,
        *,
        radial_metadata: Mapping[str, Any],
        channel_axis: int | None = None,
    ) -> dict[str, torch.Tensor]:
        validate_radial_metadata(radial_metadata)
        if radial_metadata["mode"] != self.radial_mode:
            raise ValueError("Residual envelope radial mode mismatch")
        if prediction.shape != baseline.shape:
            raise ValueError("Prediction and radial baseline shapes differ")
        axis = self._channel_axis(prediction, channel_axis)
        residual = prediction - baseline
        rho = residual.select(axis, 3)
        press = residual.select(axis, 4)
        rho_penalty = self.weight_rho * torch.mean(
            torch.relu(torch.abs(rho) - self.delta_rho).square()
        )
        press_penalty = self.weight_press * torch.mean(
            torch.relu(torch.abs(press) - self.delta_press).square()
        )
        return {
            "rho": rho_penalty,
            "press": press_penalty,
            "total": rho_penalty + press_penalty,
        }

    def penalty(
        self,
        prediction: torch.Tensor,
        baseline: torch.Tensor,
        *,
        radial_metadata: Mapping[str, Any],
        channel_axis: int | None = None,
    ) -> torch.Tensor:
        return self.penalty_components(
            prediction,
            baseline,
            radial_metadata=radial_metadata,
            channel_axis=channel_axis,
        )["total"]

    def violation_fractions(
        self,
        prediction: torch.Tensor,
        baseline: torch.Tensor,
        *,
        channel_axis: int | None = None,
    ) -> dict[str, float]:
        if prediction.shape != baseline.shape:
            raise ValueError("Prediction and radial baseline shapes differ")
        axis = self._channel_axis(prediction, channel_axis)
        residual = prediction - baseline
        return {
            "rho": float(torch.mean((torch.abs(residual.select(axis, 3)) > self.delta_rho).float())),
            "press": float(
                torch.mean((torch.abs(residual.select(axis, 4)) > self.delta_press).float())
            ),
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provenance": self.provenance.as_dict(),
            "radial_mode": self.radial_mode,
            "space": self.space,
            "channels": {
                "rho": {"delta": self.delta_rho, "weight": self.weight_rho},
                "press": {"delta": self.delta_press, "weight": self.weight_press},
            },
            "formula": "weight_c * mean(relu(abs(prediction_c-baseline_c)-Delta_c)^2)",
            "combined_into_total_paper_loss": False,
        }

    def save(self, path: str | Path) -> None:
        write_prior_json(path, self.as_dict())

    @classmethod
    def load(cls, path: str | Path, **expected: Any) -> "PaperResidualEnvelope":
        payload = read_prior_json(path, expected_schema=cls.schema_version, **expected)
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "PaperResidualEnvelope":
        if values.get("schema_version") != cls.schema_version:
            raise ValueError("Residual envelope schema mismatch")
        channels = values["channels"]
        return cls(
            provenance=PriorProvenance.from_dict(values["provenance"]),
            radial_mode=str(values["radial_mode"]),
            delta_rho=float(channels["rho"]["delta"]),
            delta_press=float(channels["press"]["delta"]),
            weight_rho=float(channels["rho"]["weight"]),
            weight_press=float(channels["press"]["weight"]),
            space=str(values["space"]),
        )
