"""Explicit reference semantics for the paper-adapted GRMHD protocol.

The paper preprocessing inverse is intentionally lossy above ``0.99 * gamma``.
Consequently, a raw HDF5 target, its canonical encode/decode image, and a model
prediction decoded through the same inverse are three distinct objects.  This
module gives each object a mandatory, unambiguous name.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch

from . import CHANNELS
from .paper_preprocessing import PaperPreprocessor


REFERENCE_SEMANTICS_VERSION = "paper-reference-semantics-v1"
COORDINATE_COMPONENTS = {
    "basis": "native spherical Kerr-Schild coordinate components",
    "magnetic": ["Bcc1", "Bcc2", "Bcc3"],
    "velocity": ["vel1", "vel2", "vel3"],
    "not_cartesian": True,
}

ArrayLike = np.ndarray | torch.Tensor


def paper_reference_metadata(preprocessor: PaperPreprocessor) -> dict[str, Any]:
    """Return the provenance that must accompany every reference bundle."""
    prototype = getattr(getattr(preprocessor, "spec", None), "key", None)
    transform_label = "stage_n_p3" if prototype == "P3" else "canonical_paper"
    return {
        "reference_semantics_version": REFERENCE_SEMANTICS_VERSION,
        "canonical_gamma": preprocessor.gamma,
        "inverse_clamp_fraction": preprocessor.inverse_clamp_fraction,
        "inverse_clamp_limit": preprocessor.gamma * preprocessor.inverse_clamp_fraction,
        "thermal_channel": preprocessor.thermal_channel,
        "paper_adaptation": preprocessor.paper_adaptation,
        "eos_conversion": preprocessor.eos_conversion,
        "coordinate_components": dict(COORDINATE_COMPONENTS),
        "channel_order": list(CHANNELS),
        "preprocessing_mode": transform_label,
        "canonical_replacement": False,
        "definitions": {
            "raw_physical_target": "HDF5 target without model preprocessing",
            "oracle_physical_target": (
                f"raw_physical_target after {transform_label} encode then decode"
            ),
            "normalized_target": f"{transform_label} encode(raw_physical_target)",
            "normalized_prediction": f"prediction in {transform_label} normalized space",
            "model_physical_prediction": (
                f"{transform_label} decode(normalized_prediction) under its frozen channel policies"
            ),
        },
    }


@dataclass(frozen=True)
class PaperReferenceStates:
    """The five explicitly named arrays needed by oracle-aware evaluation."""

    raw_physical_target: ArrayLike
    oracle_physical_target: ArrayLike
    normalized_target: ArrayLike
    normalized_prediction: ArrayLike
    model_physical_prediction: ArrayLike
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        shapes = {
            tuple(self.raw_physical_target.shape),
            tuple(self.oracle_physical_target.shape),
            tuple(self.normalized_target.shape),
            tuple(self.normalized_prediction.shape),
            tuple(self.model_physical_prediction.shape),
        }
        if len(shapes) != 1:
            raise ValueError(f"Paper reference arrays must have one common shape, found {shapes}")
        shape = next(iter(shapes))
        if not shape or len(CHANNELS) not in shape[:2]:
            raise ValueError(
                "Paper reference arrays require an eight-channel axis in position 0 or 1"
            )
        required_metadata = {
            "reference_semantics_version",
            "canonical_gamma",
            "inverse_clamp_fraction",
            "thermal_channel",
            "coordinate_components",
            "channel_order",
        }
        missing = required_metadata.difference(self.metadata)
        if missing:
            raise ValueError(f"Paper reference metadata is missing {sorted(missing)}")
        if tuple(self.metadata["channel_order"]) != CHANNELS:
            raise ValueError("Paper reference channel order changed")

    def as_dict(self) -> dict[str, Any]:
        """Expose only explicit scientific names; no ``y``/``truth`` aliases."""
        return {
            "raw_physical_target": self.raw_physical_target,
            "oracle_physical_target": self.oracle_physical_target,
            "normalized_target": self.normalized_target,
            "normalized_prediction": self.normalized_prediction,
            "model_physical_prediction": self.model_physical_prediction,
            "metadata": dict(self.metadata),
        }


def build_paper_reference_states(
    *,
    raw_physical_target: ArrayLike,
    normalized_prediction: ArrayLike,
    preprocessor: PaperPreprocessor,
    channel_axis: int | None = None,
) -> PaperReferenceStates:
    """Construct raw, canonical-oracle, and model-physical reference states."""
    if not isinstance(raw_physical_target, (np.ndarray, torch.Tensor)):
        raise TypeError("raw_physical_target must be a NumPy array or torch tensor")
    if not isinstance(normalized_prediction, (np.ndarray, torch.Tensor)):
        raise TypeError("normalized_prediction must be a NumPy array or torch tensor")
    if tuple(raw_physical_target.shape) != tuple(normalized_prediction.shape):
        raise ValueError(
            "raw_physical_target and normalized_prediction shapes differ: "
            f"{tuple(raw_physical_target.shape)} != {tuple(normalized_prediction.shape)}"
        )
    normalized_target = preprocessor.encode(raw_physical_target, channel_axis=channel_axis)
    oracle_physical_target = preprocessor.decode(
        normalized_target, channel_axis=channel_axis
    )
    model_physical_prediction = preprocessor.decode(
        normalized_prediction, channel_axis=channel_axis
    )
    return PaperReferenceStates(
        raw_physical_target=raw_physical_target,
        oracle_physical_target=oracle_physical_target,
        normalized_target=normalized_target,
        normalized_prediction=normalized_prediction,
        model_physical_prediction=model_physical_prediction,
        metadata=paper_reference_metadata(preprocessor),
    )
