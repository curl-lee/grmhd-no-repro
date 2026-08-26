"""Appendix-C normalized global-norm gate for the paper-adapted protocol.

This is a normalized array-state regularizer.  It is not a physical
dissipation rate, a covariant GRMHD norm, or the Round 1--3 gradient extension.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import numpy as np
import torch

from .paper_preprocessing import PaperPreprocessor
from .paper_priors import PriorProvenance, read_prior_json, write_prior_json


def global_state_norm(state: torch.Tensor) -> torch.Tensor:
    """L2 over channel and all three spatial axes, never across batch."""

    if state.ndim == 4 and state.shape[0] == 8:
        return torch.linalg.vector_norm(state)
    if state.ndim == 5 and state.shape[1] == 8:
        return torch.linalg.vector_norm(state.flatten(start_dim=1), dim=1)
    raise ValueError("Global paper state norm expects (8,...) or batch-first (B,8,...)")


@dataclass(frozen=True)
class DissipativeGateResult:
    y_target: torch.Tensor
    gate: torch.Tensor
    y_blend: torch.Tensor
    input_norm: torch.Tensor
    blend_norm: torch.Tensor
    penalty: torch.Tensor


@dataclass(frozen=True)
class PaperDissipativeReference:
    provenance: PriorProvenance
    rmax: float
    rin: float
    rout: float
    snapshot_norms: tuple[tuple[int, float], ...]
    beta: float = 10.0
    alpha: float = 5e-4

    schema_version = "paper-dissipative-reference-v1"

    def __post_init__(self) -> None:
        if not all(np.isfinite(value) and value > 0 for value in (self.rmax, self.rin, self.rout)):
            raise ValueError("Dissipative radii must be finite and positive")
        if not np.isclose(self.rin, 1.05 * self.rmax):
            raise ValueError("Dissipative Rin must equal 1.05*Rmax")
        if not np.isclose(self.rout, 1.5 * self.rin):
            raise ValueError("Dissipative Rout must equal 1.5*Rin")
        if self.beta != 10 or self.alpha != 5e-4:
            raise ValueError("Canonical dissipative beta/alpha changed")
        indices = tuple(index for index, _ in self.snapshot_norms)
        if indices != self.provenance.training_indices:
            raise ValueError("Dissipative snapshot norms must cover train indices exactly")
        norms = tuple(value for _, value in self.snapshot_norms)
        if not norms or not np.all(np.isfinite(norms)) or not np.isclose(max(norms), self.rmax):
            raise ValueError("Dissipative Rmax must be the maximum train snapshot norm")

    def apply(self, x: torch.Tensor, y_prediction: torch.Tensor) -> DissipativeGateResult:
        if x.shape != y_prediction.shape:
            raise ValueError("Dissipative input and prediction shapes differ")
        input_norm = global_state_norm(x)
        y_target = (self.rin / self.rout) * x
        gate = torch.sigmoid(self.beta * (x.new_tensor(self.rin) - input_norm))
        view_shape = () if x.ndim == 4 else (x.shape[0],) + (1,) * (x.ndim - 1)
        gate_view = gate.reshape(view_shape)
        y_blend = gate_view * y_prediction + (1.0 - gate_view) * y_target
        blend_norm = global_state_norm(y_blend)
        penalty = self.alpha * torch.mean(torch.relu(blend_norm - input_norm))
        return DissipativeGateResult(
            y_target=y_target,
            gate=gate,
            y_blend=y_blend,
            input_norm=input_norm,
            blend_norm=blend_norm,
            penalty=penalty,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provenance": self.provenance.as_dict(),
            "Rmax": self.rmax,
            "Rin": self.rin,
            "Rout": self.rout,
            "beta": self.beta,
            "alpha": self.alpha,
            "train_snapshot_norms": [
                {"snapshot_index": index, "global_l2_norm": value}
                for index, value in self.snapshot_norms
            ],
            "norm_axes": "channel + phi + theta + r; batch excluded",
            "formula": {
                "y_target": "(Rin/Rout)*x",
                "gate": "sigmoid(beta*(Rin-||x||_2))",
                "blend": "gate*y_prediction + (1-gate)*y_target",
                "penalty": "alpha*mean(relu(||blend||_2-||x||_2))",
            },
            "semantics": {
                "normalized_global_norm_regularizer": True,
                "physical_dissipation_rate": False,
                "covariant_grmhd_norm": False,
                "round2_3_gradient_extension": False,
                "combined_into_total_paper_loss": False,
            },
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "PaperDissipativeReference":
        if values.get("schema_version") != cls.schema_version:
            raise ValueError("Dissipative reference schema mismatch")
        return cls(
            provenance=PriorProvenance.from_dict(values["provenance"]),
            rmax=float(values["Rmax"]),
            rin=float(values["Rin"]),
            rout=float(values["Rout"]),
            beta=float(values["beta"]),
            alpha=float(values["alpha"]),
            snapshot_norms=tuple(
                (int(record["snapshot_index"]), float(record["global_l2_norm"]))
                for record in values["train_snapshot_norms"]
            ),
        )

    def save(self, path: str | Path) -> None:
        write_prior_json(path, self.as_dict())

    @classmethod
    def load(cls, path: str | Path, **expected: Any) -> "PaperDissipativeReference":
        payload = read_prior_json(path, expected_schema=cls.schema_version, **expected)
        return cls.from_dict(payload)


def fit_dissipative_reference(
    h5_path: str | Path,
    *,
    training_indices: Iterable[int],
    preprocessor: PaperPreprocessor,
    provenance: PriorProvenance,
) -> PaperDissipativeReference:
    indices = tuple(int(index) for index in training_indices)
    if not indices or tuple(sorted(set(indices))) != indices:
        raise ValueError("Dissipative fit indices must be non-empty, unique, and increasing")
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
    records: list[tuple[int, float]] = []
    with h5py.File(h5_path, "r") as handle:
        for index in indices:
            raw = np.asarray(handle["snapshots"][index], dtype=np.float32)
            normalized = preprocessor.encode(raw)
            norm = float(np.sqrt(np.sum(np.square(normalized.astype(np.float64)))))
            if not np.isfinite(norm):
                raise FloatingPointError(f"Nonfinite canonical state norm at snapshot {index}")
            records.append((index, norm))
    rmax = max(value for _, value in records)
    rin = 1.05 * rmax
    rout = 1.5 * rin
    return PaperDissipativeReference(
        provenance=provenance,
        rmax=rmax,
        rin=rin,
        rout=rout,
        snapshot_norms=tuple(records),
    )
