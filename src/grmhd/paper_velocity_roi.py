"""Canonical paper velocity ROI on stored spherical Kerr--Schild components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .paper_preprocessing import PaperPreprocessor
from .paper_priors import PriorProvenance, read_prior_json, write_prior_json


def _as_batch(values: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if values.ndim == 4 and values.shape[0] == 8:
        return values.unsqueeze(0), True
    if values.ndim == 5 and values.shape[1] == 8:
        return values, False
    raise ValueError(f"Expected (8,Nphi,Ntheta,Nr) or batch-first state, found {values.shape}")


def stored_component_speed_proxy(physical_state: torch.Tensor) -> torch.Tensor:
    """Euclidean proxy over stored components; not a metric-correct velocity norm."""

    batch, squeezed = _as_batch(physical_state)
    speed = torch.sqrt(torch.sum(batch[:, 5:8].square(), dim=1))
    return speed[0] if squeezed else speed


def top_fraction_mask(speed: torch.Tensor, top_fraction: float = 0.20) -> torch.Tensor:
    """Select an exact deterministic count per sample, with low flat index winning ties."""

    if not 0 < top_fraction <= 1:
        raise ValueError("top_fraction must lie in (0,1]")
    squeezed = speed.ndim == 3
    batch = speed.unsqueeze(0) if squeezed else speed
    if batch.ndim != 4:
        raise ValueError("Speed proxy must be spatial or batch-first spatial")
    flat = batch.flatten(start_dim=1)
    count = max(1, int(np.ceil(top_fraction * flat.shape[1])))
    order = torch.argsort(flat, dim=1, descending=True, stable=True)
    mask_flat = torch.zeros_like(flat, dtype=torch.bool)
    mask_flat.scatter_(1, order[:, :count], True)
    mask = mask_flat.reshape_as(batch)
    return mask[0] if squeezed else mask


def canonical_oracle_velocity_roi_mask(
    raw_physical_target: torch.Tensor,
    preprocessor: PaperPreprocessor,
    *,
    top_fraction: float = 0.20,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the main ROI and the lossy canonical decode used to construct it."""

    batch, squeezed = _as_batch(raw_physical_target)
    normalized = preprocessor.encode(batch, channel_axis=1)
    canonical = preprocessor.decode(normalized, channel_axis=1)
    mask = top_fraction_mask(stored_component_speed_proxy(canonical), top_fraction)
    return (mask[0], canonical[0]) if squeezed else (mask, canonical)


def raw_physical_velocity_roi_diagnostic_mask(
    raw_physical_target: torch.Tensor,
    *,
    top_fraction: float = 0.20,
) -> torch.Tensor:
    """Diagnostic extension only; this mask is not accepted by the main ROI loss API."""

    return top_fraction_mask(stored_component_speed_proxy(raw_physical_target), top_fraction)


def normalized_velocity_roi_relative_error(
    normalized_prediction: torch.Tensor,
    normalized_target: torch.Tensor,
    canonical_roi_mask: torch.Tensor,
    *,
    denominator_epsilon: float = 1e-12,
) -> torch.Tensor:
    """Batch-mean normalized velocity relative L2 on the canonical oracle ROI."""

    prediction, prediction_squeezed = _as_batch(normalized_prediction)
    target, target_squeezed = _as_batch(normalized_target)
    if prediction_squeezed != target_squeezed or prediction.shape != target.shape:
        raise ValueError("Normalized prediction and target shapes differ")
    mask = canonical_roi_mask.unsqueeze(0) if canonical_roi_mask.ndim == 3 else canonical_roi_mask
    if mask.shape != prediction.shape[:1] + prediction.shape[2:]:
        raise ValueError("Canonical ROI mask shape does not match normalized tensors")
    if denominator_epsilon <= 0:
        raise ValueError("ROI denominator epsilon must be positive")
    expanded = mask.unsqueeze(1)
    difference = (prediction[:, 5:8] - target[:, 5:8]) * expanded
    reference = target[:, 5:8] * expanded
    # ``vector_norm`` defines a finite zero subgradient when prediction equals
    # target.  The algebraic value is identical to sqrt(sum(square(...))).
    numerator = torch.linalg.vector_norm(difference.flatten(start_dim=1), dim=1)
    denominator = torch.linalg.vector_norm(
        reference.flatten(start_dim=1), dim=1
    ).clamp_min(denominator_epsilon)
    return torch.mean(numerator / denominator)


def paper_roi_ramp(epoch: int | float, ramp_epochs: int = 375) -> float:
    if epoch < 0 or ramp_epochs <= 0:
        raise ValueError("ROI epoch must be nonnegative and ramp_epochs positive")
    return float(min(1.0, float(epoch) / ramp_epochs))


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def roi_mask_diagnostics(
    canonical_mask: torch.Tensor,
    raw_mask: torch.Tensor,
    velocity_target_clamp_mask: torch.Tensor,
    *,
    shell_index: Sequence[int],
    snapshot_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Compare canonical/raw ROI masks and their overlap with velocity clamp hits."""

    canonical = canonical_mask.unsqueeze(0) if canonical_mask.ndim == 3 else canonical_mask
    raw = raw_mask.unsqueeze(0) if raw_mask.ndim == 3 else raw_mask
    clamp = (
        velocity_target_clamp_mask.unsqueeze(0)
        if velocity_target_clamp_mask.ndim == 3
        else velocity_target_clamp_mask
    )
    if canonical.shape != raw.shape or canonical.shape != clamp.shape or canonical.ndim != 4:
        raise ValueError("ROI diagnostic masks must be aligned batch-first spatial masks")
    shell_index_array = np.asarray(shell_index, dtype=np.int64)
    if shell_index_array.shape != (canonical.shape[-1],):
        raise ValueError("ROI shell index does not match Nr")
    canonical = canonical.bool()
    raw = raw.bool()
    clamp = clamp.bool()
    intersection = canonical & raw
    union = canonical | raw
    total = canonical.numel()
    canonical_count = int(canonical.sum())
    raw_count = int(raw.sum())
    intersection_count = int(intersection.sum())
    records: list[dict[str, Any]] = []
    indices = list(range(canonical.shape[0])) if snapshot_indices is None else list(snapshot_indices)
    if len(indices) != canonical.shape[0]:
        raise ValueError("snapshot_indices length does not match ROI batch")
    for slot, snapshot_index in enumerate(indices):
        c = canonical[slot]
        r = raw[slot]
        inter = c & r
        records.append(
            {
                "snapshot_index": int(snapshot_index),
                "canonical_fraction": float(c.float().mean()),
                "raw_fraction": float(r.float().mean()),
                "jaccard": _safe_ratio(int(inter.sum()), int((c | r).sum())),
                "disagreement_fraction": float((c ^ r).float().mean()),
                "canonical_clamp_overlap": _safe_ratio(int((c & clamp[slot]).sum()), int(c.sum())),
                "raw_clamp_overlap": _safe_ratio(int((r & clamp[slot]).sum()), int(r.sum())),
            }
        )
    shells: list[dict[str, Any]] = []
    for shell in range(int(shell_index_array.max()) + 1):
        radial = torch.as_tensor(
            shell_index_array == shell, dtype=torch.bool, device=canonical.device
        ).view(1, 1, 1, -1)
        radial = radial.expand_as(canonical)
        c = canonical & radial
        r = raw & radial
        inter = c & r
        shell_total = int(radial.sum())
        shells.append(
            {
                "shell": shell,
                "voxel_count": shell_total,
                "canonical_fraction": float(c.sum() / shell_total),
                "raw_fraction": float(r.sum() / shell_total),
                "jaccard": _safe_ratio(int(inter.sum()), int((c | r).sum())),
                "canonical_clamp_overlap": _safe_ratio(int((c & clamp).sum()), int(c.sum())),
                "raw_clamp_overlap": _safe_ratio(int((r & clamp).sum()), int(r.sum())),
            }
        )
    inner_radial = torch.as_tensor(
        shell_index_array < 2, dtype=torch.bool, device=canonical.device
    ).view(1, 1, 1, -1).expand_as(canonical)
    inner_c = canonical & inner_radial
    inner_r = raw & inner_radial
    return {
        "mask_semantics": {
            "main": "canonical paper-decoded stored-component speed proxy top 20%",
            "diagnostic": "raw physical stored-component speed proxy top 20%",
            "not_metric_correct_relativistic_speed": True,
        },
        "canonical_fraction": float(canonical.float().mean()),
        "raw_fraction": float(raw.float().mean()),
        "intersection_fraction": float(intersection.float().mean()),
        "jaccard": _safe_ratio(intersection_count, int(union.sum())),
        "precision_canonical_against_raw": _safe_ratio(intersection_count, canonical_count),
        "recall_canonical_against_raw": _safe_ratio(intersection_count, raw_count),
        "disagreement_fraction": float((canonical ^ raw).float().mean()),
        "canonical_clamp_overlap": _safe_ratio(int((canonical & clamp).sum()), canonical_count),
        "raw_clamp_overlap": _safe_ratio(int((raw & clamp).sum()), raw_count),
        "shells": shells,
        "inner_two_shells": {
            "canonical_fraction_of_inner": float(inner_c.sum() / inner_radial.sum()),
            "raw_fraction_of_inner": float(inner_r.sum() / inner_radial.sum()),
            "jaccard": _safe_ratio(int((inner_c & inner_r).sum()), int((inner_c | inner_r).sum())),
            "canonical_clamp_overlap": _safe_ratio(
                int((inner_c & clamp).sum()), int(inner_c.sum())
            ),
            "raw_clamp_overlap": _safe_ratio(int((inner_r & clamp).sum()), int(inner_r.sum())),
        },
        "snapshot_records": records,
        "voxel_count": total,
    }


@dataclass(frozen=True)
class PaperVelocityROI:
    provenance: PriorProvenance
    top_fraction: float = 0.20
    kappa: float = 8.0
    ramp_epochs: int = 375
    denominator_epsilon: float = 1e-12

    schema_version = "paper-velocity-roi-v1"

    def __post_init__(self) -> None:
        if self.top_fraction != 0.20 or self.kappa != 8 or self.ramp_epochs != 375:
            raise ValueError("Canonical paper ROI parameters changed")
        if self.denominator_epsilon <= 0:
            raise ValueError("ROI denominator epsilon must be positive")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "provenance": self.provenance.as_dict(),
            "top_fraction": self.top_fraction,
            "kappa": self.kappa,
            "ramp": {"formula": "min(1, epoch/375)", "epochs": self.ramp_epochs},
            "denominator_epsilon": self.denominator_epsilon,
            "main_mask": "canonical_oracle_physical stored-component speed proxy",
            "raw_mask_role": "diagnostic_extension_only",
            "stored_components_not_cartesian": True,
            "not_metric_correct_relativistic_speed": True,
            "combined_into_total_paper_loss": False,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "PaperVelocityROI":
        if values.get("schema_version") != cls.schema_version:
            raise ValueError("Velocity ROI schema mismatch")
        return cls(
            provenance=PriorProvenance.from_dict(values["provenance"]),
            top_fraction=float(values["top_fraction"]),
            kappa=float(values["kappa"]),
            ramp_epochs=int(values["ramp"]["epochs"]),
            denominator_epsilon=float(values["denominator_epsilon"]),
        )

    def save(self, path: str | Path) -> None:
        write_prior_json(path, self.as_dict())

    @classmethod
    def load(cls, path: str | Path, **expected: Any) -> "PaperVelocityROI":
        payload = read_prior_json(path, expected_schema=cls.schema_version, **expected)
        return cls.from_dict(payload)
