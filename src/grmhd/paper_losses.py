"""Stage E losses for the paper-adapted reduced protocol.

Every training term operates in the canonical normalized state space. Physical
targets and the optional raw ROI mask are carried only for diagnostics; they
cannot affect the total loss.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from neuralop import H1Loss, LpLoss
from torch import nn

from . import CHANNELS
from .paper_bounds import PaperPhysicalBounds
from .paper_dissipation import PaperDissipativeReference, global_state_norm
from .paper_priors import (
    PAPER_PROTOCOL_NAME,
    PAPER_THERMAL_CHANNEL,
    PaperResidualEnvelope,
    validate_radial_metadata,
)
from .paper_velocity_roi import (
    PaperVelocityROI,
    normalized_velocity_roi_relative_error,
    paper_roi_ramp,
)


EXPECTED_RADIAL_MODE = "appendix_literal_press_proxy"
EXPECTED_COORDINATE_SYSTEM = "spherical Kerr-Schild: phi,theta,r"


def _validate_state(name: str, value: torch.Tensor) -> None:
    if value.ndim != 5 or value.shape[1] != 8:
        raise ValueError(f"{name} must have shape (B,8,Nphi,Ntheta,Nr)")
    if not value.is_floating_point():
        raise TypeError(f"{name} must be floating point")


def _safe_mask_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    zero = numerator.new_zeros((), dtype=torch.float32)
    return torch.where(denominator > 0, numerator.float() / denominator.float(), zero)


@dataclass(frozen=True)
class PaperLossContext:
    """Unambiguous state and prior references for one Stage E loss call."""

    normalized_input: torch.Tensor
    normalized_target: torch.Tensor
    raw_physical_target: torch.Tensor
    oracle_physical_target: torch.Tensor
    canonical_roi_mask: torch.Tensor
    radial_baseline_normalized: torch.Tensor
    normalized_bounds: Mapping[str, tuple[float, float]]
    epoch: int | float
    snapshot_indices: Sequence[int]
    protocol_metadata: Mapping[str, Any]
    raw_roi_diagnostic_mask: torch.Tensor | None = None

    def validate(self, normalized_prediction: torch.Tensor) -> None:
        _validate_state("normalized_prediction", normalized_prediction)
        for name, value in (
            ("normalized_input", self.normalized_input),
            ("normalized_target", self.normalized_target),
            ("raw_physical_target", self.raw_physical_target),
            ("oracle_physical_target", self.oracle_physical_target),
        ):
            _validate_state(name, value)
            if value.shape != normalized_prediction.shape:
                raise ValueError(f"{name} shape differs from normalized_prediction")
            if value.device != normalized_prediction.device:
                raise ValueError(f"{name} device differs from normalized_prediction")
            if value.dtype != normalized_prediction.dtype:
                raise ValueError(f"{name} dtype differs from normalized_prediction")
        expected_mask_shape = (
            normalized_prediction.shape[0],
            *normalized_prediction.shape[2:],
        )
        if self.canonical_roi_mask.shape != expected_mask_shape:
            raise ValueError("canonical_roi_mask shape differs from normalized states")
        if self.canonical_roi_mask.dtype != torch.bool:
            raise TypeError("canonical_roi_mask must be boolean")
        if self.canonical_roi_mask.device != normalized_prediction.device:
            raise ValueError("canonical_roi_mask device differs from normalized_prediction")
        if self.raw_roi_diagnostic_mask is not None:
            if (
                self.raw_roi_diagnostic_mask.shape != expected_mask_shape
                or self.raw_roi_diagnostic_mask.dtype != torch.bool
                or self.raw_roi_diagnostic_mask.device != normalized_prediction.device
            ):
                raise ValueError("raw_roi_diagnostic_mask must match canonical_roi_mask")
        baseline = self.radial_baseline_normalized
        allowed_baselines = {
            tuple(normalized_prediction.shape),
            (1, *normalized_prediction.shape[1:]),
            tuple(normalized_prediction.shape[1:]),
        }
        if tuple(baseline.shape) not in allowed_baselines:
            raise ValueError("radial_baseline_normalized is not broadcast-compatible")
        if baseline.device != normalized_prediction.device or baseline.dtype != normalized_prediction.dtype:
            raise ValueError("radial_baseline_normalized dtype/device mismatch")
        if set(self.normalized_bounds) != {"rho", "press"}:
            raise ValueError("normalized_bounds must contain exactly rho and press")
        for lower, upper in self.normalized_bounds.values():
            if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                raise ValueError("normalized_bounds must be finite and ordered")
        if float(self.epoch) < 0:
            raise ValueError("epoch must be nonnegative")
        if len(tuple(self.snapshot_indices)) != normalized_prediction.shape[0]:
            raise ValueError("snapshot_indices length must equal batch size")
        required_metadata = {
            "protocol_name": PAPER_PROTOCOL_NAME,
            "thermal_channel": PAPER_THERMAL_CHANNEL,
            "paper_adaptation": True,
            "validation_not_used_for_fit": True,
            "selected_radial_mode": EXPECTED_RADIAL_MODE,
            "coordinate_system": EXPECTED_COORDINATE_SYSTEM,
            "canonical_roi_source": "oracle_physical_target",
        }
        for key, expected in required_metadata.items():
            if self.protocol_metadata.get(key) != expected:
                raise ValueError(f"Paper loss context metadata mismatch for {key}")

    def expanded_radial_baseline(self, normalized_prediction: torch.Tensor) -> torch.Tensor:
        baseline = self.radial_baseline_normalized
        if baseline.ndim == 4:
            baseline = baseline.unsqueeze(0)
        return baseline.expand_as(normalized_prediction)


@dataclass(frozen=True)
class PaperLossResult:
    total: torch.Tensor
    base_fidelity_raw: torch.Tensor
    base_fidelity_weighted: torch.Tensor
    h1_raw: torch.Tensor
    h1_weighted: torch.Tensor
    roi_raw: torch.Tensor
    roi_ramp: torch.Tensor
    roi_weighted: torch.Tensor
    bounds_rho_raw: torch.Tensor
    bounds_press_raw: torch.Tensor
    bounds_weighted: torch.Tensor
    envelope_rho_raw: torch.Tensor
    envelope_press_raw: torch.Tensor
    envelope_weighted: torch.Tensor
    dissipation_raw: torch.Tensor
    dissipation_weighted: torch.Tensor
    diagnostics: Mapping[str, Any]
    metadata: Mapping[str, Any]

    def component_tensors(self) -> dict[str, torch.Tensor]:
        return {
            name: getattr(self, name)
            for name in (
                "total",
                "base_fidelity_raw",
                "base_fidelity_weighted",
                "h1_raw",
                "h1_weighted",
                "roi_raw",
                "roi_ramp",
                "roi_weighted",
                "bounds_rho_raw",
                "bounds_press_raw",
                "bounds_weighted",
                "envelope_rho_raw",
                "envelope_press_raw",
                "envelope_weighted",
                "dissipation_raw",
                "dissipation_weighted",
            )
        }

    def detached_log(self) -> dict[str, Any]:
        def detach(value: Any) -> Any:
            if torch.is_tensor(value):
                detached = value.detach().cpu()
                return float(detached) if detached.numel() == 1 else detached.tolist()
            if isinstance(value, Mapping):
                return {str(key): detach(item) for key, item in value.items()}
            if isinstance(value, (tuple, list)):
                return [detach(item) for item in value]
            return value

        return {
            "components": detach(self.component_tensors()),
            "diagnostics": detach(self.diagnostics),
            "metadata": detach(self.metadata),
        }


class PaperComponentFidelityLoss(nn.Module):
    """Spatial per-channel MSE, channel sum, and batch mean."""

    def __init__(
        self,
        *,
        magnetic_weight: float = 1.2,
        velocity_weight: float = 1.0,
    ) -> None:
        super().__init__()
        if magnetic_weight < 0 or velocity_weight < 0:
            raise ValueError("Paper fidelity weights must be nonnegative")
        self.magnetic_weight = float(magnetic_weight)
        self.velocity_weight = float(velocity_weight)

    @property
    def channel_weights(self) -> tuple[float, ...]:
        return (
            self.magnetic_weight,
            self.magnetic_weight,
            self.magnetic_weight,
            1.0,
            1.0,
            self.velocity_weight,
            self.velocity_weight,
            self.velocity_weight,
        )

    def components(
        self,
        normalized_prediction: torch.Tensor,
        normalized_target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        _validate_state("normalized_prediction", normalized_prediction)
        _validate_state("normalized_target", normalized_target)
        if normalized_prediction.shape != normalized_target.shape:
            raise ValueError("Normalized prediction and target shapes differ")
        channel_raw = (normalized_prediction - normalized_target).square().mean(
            dim=(0, 2, 3, 4)
        )
        weights = normalized_prediction.new_tensor(self.channel_weights)
        return {
            "channel_raw": channel_raw,
            "raw": torch.sum(channel_raw),
            "weighted": torch.sum(channel_raw * weights),
        }

    def forward(
        self,
        normalized_prediction: torch.Tensor,
        normalized_target: torch.Tensor,
    ) -> torch.Tensor:
        return self.components(normalized_prediction, normalized_target)["weighted"]


class PaperH1GradientLoss(nn.Module):
    """Paper squared gradient seminorm using pinned upstream finite differences."""

    def __init__(self) -> None:
        super().__init__()
        self.upstream_h1 = H1Loss(d=3, reduction="mean")
        self.upstream_l2 = LpLoss(d=3, p=2, reduction="mean")

    def forward(
        self,
        normalized_prediction: torch.Tensor,
        normalized_target: torch.Tensor,
    ) -> torch.Tensor:
        _validate_state("normalized_prediction", normalized_prediction)
        _validate_state("normalized_target", normalized_target)
        if normalized_prediction.shape != normalized_target.shape:
            raise ValueError("Normalized prediction and target shapes differ")
        full_h1_squared = self.upstream_h1.abs(
            normalized_prediction, normalized_target, take_root=False
        )
        l2_squared = self.upstream_l2.abs(
            normalized_prediction, normalized_target, take_root=False
        )
        return normalized_prediction.shape[1] * (full_h1_squared - l2_squared)


class PlainL2Loss(nn.Module):
    """Strict Appendix-D unit-weight normalized per-voxel squared L2."""

    prior_flag_names = (
        "h1",
        "roi",
        "bounds",
        "envelope",
        "dissipation",
        "component_weights",
    )

    def __init__(self) -> None:
        super().__init__()
        self.fidelity = PaperComponentFidelityLoss(
            magnetic_weight=1.0, velocity_weight=1.0
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": "paper-plain-l2-v1",
            "space": "canonical_normalized_state",
            "channel_weights": [1.0] * 8,
            "reduction": "spatial_mean_per_channel_then_channel_sum_then_batch_mean",
            "enabled_priors": [],
        }

    def forward(
        self,
        normalized_prediction: torch.Tensor,
        normalized_target: torch.Tensor,
        *,
        enabled_prior_flags: Mapping[str, bool] | None = None,
    ) -> torch.Tensor:
        if enabled_prior_flags is not None:
            unknown = set(enabled_prior_flags) - set(self.prior_flag_names)
            if unknown:
                raise ValueError(f"Unknown Plain L2 prior flags: {sorted(unknown)}")
            enabled = sorted(name for name, value in enabled_prior_flags.items() if value)
            if enabled:
                raise ValueError(f"PlainL2Loss rejects enabled prior flags: {enabled}")
        return self.fidelity(normalized_prediction, normalized_target)


class PaperCompositeLoss(nn.Module):
    """Full paper loss composed only from the Stage E contract components."""

    def __init__(
        self,
        *,
        bounds: PaperPhysicalBounds,
        envelope: PaperResidualEnvelope,
        roi: PaperVelocityROI,
        dissipation: PaperDissipativeReference,
        radial_metadata: Mapping[str, Any],
        magnetic_weight: float = 1.2,
        velocity_weight: float = 1.0,
        h1_weight: float = 0.05,
        roi_kappa: float = 8.0,
        bounds_rho_low_weight: float = 0.05,
        bounds_press_low_weight: float = 0.05,
        bounds_rho_high_weight: float = 0.0,
        bounds_press_high_weight: float = 0.0,
        envelope_rho_weight: float = 0.05,
        envelope_press_weight: float = 0.05,
        dissipation_alpha: float = 5e-4,
        gamma: float = 6.0,
        inverse_clamp_fraction: float = 0.99,
    ) -> None:
        super().__init__()
        validate_radial_metadata(radial_metadata)
        if radial_metadata["mode"] != EXPECTED_RADIAL_MODE:
            raise ValueError("Stage E requires the selected appendix literal radial mode")
        if envelope.radial_mode != EXPECTED_RADIAL_MODE:
            raise ValueError("Residual envelope does not use the selected literal mode")
        provenances = (
            bounds.provenance,
            envelope.provenance,
            roi.provenance,
            dissipation.provenance,
        )
        if any(provenance != provenances[0] for provenance in provenances[1:]):
            raise ValueError("Stage D loss artifacts have different provenance")
        coefficients = (
            magnetic_weight,
            velocity_weight,
            h1_weight,
            roi_kappa,
            bounds_rho_low_weight,
            bounds_press_low_weight,
            bounds_rho_high_weight,
            bounds_press_high_weight,
            envelope_rho_weight,
            envelope_press_weight,
            dissipation_alpha,
        )
        if any(not np.isfinite(value) or value < 0 for value in coefficients):
            raise ValueError("Paper loss coefficients must be finite and nonnegative")
        if gamma != 6.0 or inverse_clamp_fraction != 0.99:
            raise ValueError("Canonical paper gamma/inverse clamp changed")
        self.bounds = bounds
        self.envelope = envelope
        self.roi = roi
        self.dissipation = dissipation
        self.radial_metadata = dict(radial_metadata)
        self.fidelity = PaperComponentFidelityLoss(
            magnetic_weight=magnetic_weight,
            velocity_weight=velocity_weight,
        )
        self.h1 = PaperH1GradientLoss()
        self.h1_weight = float(h1_weight)
        self.roi_kappa = float(roi_kappa)
        self.bounds_weights = {
            "rho_low": float(bounds_rho_low_weight),
            "press_low": float(bounds_press_low_weight),
            "rho_high": float(bounds_rho_high_weight),
            "press_high": float(bounds_press_high_weight),
        }
        self.envelope_weights = {
            "rho": float(envelope_rho_weight),
            "press": float(envelope_press_weight),
        }
        self.dissipation_alpha = float(dissipation_alpha)
        self.gamma = float(gamma)
        self.inverse_clamp_fraction = float(inverse_clamp_fraction)
        self._unit_bounds = replace(
            bounds,
            lambda_low_rho=1.0,
            lambda_low_press=1.0,
            lambda_high_rho=1.0,
            lambda_high_press=1.0,
        )
        self._unit_envelope = replace(
            envelope,
            weight_rho=1.0,
            weight_press=1.0,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": "paper-composite-loss-v1",
            "protocol_name": self.bounds.provenance.protocol_name,
            "space": "canonical_normalized_state",
            "radial_mode": self.envelope.radial_mode,
            "component_weights": {
                "Bcc1": self.fidelity.magnetic_weight,
                "Bcc2": self.fidelity.magnetic_weight,
                "Bcc3": self.fidelity.magnetic_weight,
                "rho": 1.0,
                "press": 1.0,
                "vel1": self.fidelity.velocity_weight,
                "vel2": self.fidelity.velocity_weight,
                "vel3": self.fidelity.velocity_weight,
            },
            "h1_weight": self.h1_weight,
            "h1": {
                "computational_grid_H1_adaptation": True,
                "spherical_metric_H1": False,
                "upstream_absolute_squared_minus_l2": True,
            },
            "roi_kappa": self.roi_kappa,
            "roi_ramp_epochs": self.roi.ramp_epochs,
            "bounds_weights": dict(self.bounds_weights),
            "envelope_weights": dict(self.envelope_weights),
            "envelope_delta": {
                "rho": self.envelope.delta_rho,
                "press": self.envelope.delta_press,
            },
            "dissipation_alpha": self.dissipation_alpha,
            "gamma": self.gamma,
            "inverse_clamp_fraction": self.inverse_clamp_fraction,
            "forbidden_extensions": [
                "bounded_residual",
                "hybrid_target",
                "rollout_aware_loss",
                "range_loss",
                "fold_b",
                "recency_weighting",
                "physical_evaluation_error",
            ],
        }

    def _validate_context_bounds(self, context: PaperLossContext) -> None:
        for name in ("rho", "press"):
            actual = tuple(float(value) for value in context.normalized_bounds[name])
            expected = tuple(float(value) for value in self.bounds.normalized_bounds[name])
            if actual != expected:
                raise ValueError(f"Context normalized {name} bounds differ from artifact")

    def components(
        self,
        normalized_prediction: torch.Tensor,
        *,
        context: PaperLossContext,
    ) -> PaperLossResult:
        context.validate(normalized_prediction)
        self._validate_context_bounds(context)
        baseline = context.expanded_radial_baseline(normalized_prediction)
        base = self.fidelity.components(
            normalized_prediction, context.normalized_target
        )
        h1_raw = self.h1(normalized_prediction, context.normalized_target)
        h1_weighted = self.h1_weight * h1_raw
        roi_raw = normalized_velocity_roi_relative_error(
            normalized_prediction,
            context.normalized_target,
            context.canonical_roi_mask,
            denominator_epsilon=self.roi.denominator_epsilon,
        )
        roi_ramp = normalized_prediction.new_tensor(
            paper_roi_ramp(context.epoch, self.roi.ramp_epochs)
        )
        roi_weighted = self.roi_kappa * roi_ramp * roi_raw
        bounds_raw = self._unit_bounds.penalty_components(normalized_prediction)
        bounds_rho_raw = bounds_raw["rho_low"]
        bounds_press_raw = bounds_raw["press_low"]
        bounds_weighted = (
            self.bounds_weights["rho_low"] * bounds_raw["rho_low"]
            + self.bounds_weights["press_low"] * bounds_raw["press_low"]
            + self.bounds_weights["rho_high"] * bounds_raw["rho_high"]
            + self.bounds_weights["press_high"] * bounds_raw["press_high"]
        )
        envelope_raw = self._unit_envelope.penalty_components(
            normalized_prediction,
            baseline,
            radial_metadata=self.radial_metadata,
        )
        envelope_rho_raw = envelope_raw["rho"]
        envelope_press_raw = envelope_raw["press"]
        envelope_weighted = (
            self.envelope_weights["rho"] * envelope_rho_raw
            + self.envelope_weights["press"] * envelope_press_raw
        )
        dissipative = self.dissipation.apply(
            context.normalized_input, normalized_prediction
        )
        dissipation_raw = dissipative.penalty / self.dissipation.alpha
        dissipation_weighted = self.dissipation_alpha * dissipation_raw
        total = (
            base["weighted"]
            + h1_weighted
            + roi_weighted
            + bounds_weighted
            + envelope_weighted
            + dissipation_weighted
        )
        clamp_limit = self.gamma * self.inverse_clamp_fraction
        target_clamp = torch.abs(context.normalized_target) > clamp_limit
        velocity_target_clamp = torch.any(target_clamp[:, 5:8], dim=1)
        canonical_count = context.canonical_roi_mask.sum()
        canonical_raw_jaccard = normalized_prediction.new_tensor(float("nan"))
        if context.raw_roi_diagnostic_mask is not None:
            intersection = (
                context.canonical_roi_mask & context.raw_roi_diagnostic_mask
            ).sum()
            union = (
                context.canonical_roi_mask | context.raw_roi_diagnostic_mask
            ).sum()
            canonical_raw_jaccard = _safe_mask_ratio(intersection, union).to(
                normalized_prediction
            )
        prediction_norm = global_state_norm(normalized_prediction)
        below_rho = normalized_prediction[:, 3] < self.bounds.normalized_bounds["rho"][0]
        below_press = normalized_prediction[:, 4] < self.bounds.normalized_bounds["press"][0]
        above_rho = normalized_prediction[:, 3] > self.bounds.normalized_bounds["rho"][1]
        above_press = normalized_prediction[:, 4] > self.bounds.normalized_bounds["press"][1]
        envelope_fractions = self.envelope.violation_fractions(
            normalized_prediction, baseline
        )
        diagnostics: dict[str, Any] = {
            "target_clamp_fraction_by_channel": {
                name: target_clamp[:, channel].float().mean()
                for channel, name in enumerate(CHANNELS)
            },
            "roi_voxel_fraction": context.canonical_roi_mask.float().mean(),
            "roi_clamp_overlap": _safe_mask_ratio(
                (context.canonical_roi_mask & velocity_target_clamp).sum(),
                canonical_count,
            ).to(normalized_prediction),
            "canonical_raw_roi_jaccard": canonical_raw_jaccard,
            "bounds_violation_fraction": {
                "rho_below": below_rho.float().mean(),
                "press_below": below_press.float().mean(),
                "rho_above": above_rho.float().mean(),
                "press_above": above_press.float().mean(),
            },
            "envelope_violation_fraction": {
                name: normalized_prediction.new_tensor(value)
                for name, value in envelope_fractions.items()
            },
            "dissipation_gate_mean": dissipative.gate.mean(),
            "dissipation_gate_min": dissipative.gate.min(),
            "dissipation_gate_max": dissipative.gate.max(),
            "input_norm_mean": dissipative.input_norm.mean(),
            "prediction_norm_mean": prediction_norm.mean(),
            "fraction_above_Rin": (
                dissipative.input_norm > self.dissipation.rin
            ).float().mean(),
            "fraction_above_Rout": (
                dissipative.input_norm > self.dissipation.rout
            ).float().mean(),
            "base_channel_raw": {
                name: base["channel_raw"][channel]
                for channel, name in enumerate(CHANNELS)
            },
        }
        result = PaperLossResult(
            total=total,
            base_fidelity_raw=base["raw"],
            base_fidelity_weighted=base["weighted"],
            h1_raw=h1_raw,
            h1_weighted=h1_weighted,
            roi_raw=roi_raw,
            roi_ramp=roi_ramp,
            roi_weighted=roi_weighted,
            bounds_rho_raw=bounds_rho_raw,
            bounds_press_raw=bounds_press_raw,
            bounds_weighted=bounds_weighted,
            envelope_rho_raw=envelope_rho_raw,
            envelope_press_raw=envelope_press_raw,
            envelope_weighted=envelope_weighted,
            dissipation_raw=dissipation_raw,
            dissipation_weighted=dissipation_weighted,
            diagnostics=diagnostics,
            metadata=self.metadata,
        )
        for name, value in result.component_tensors().items():
            if value.ndim != 0 or value.dtype != normalized_prediction.dtype or value.device != normalized_prediction.device:
                raise RuntimeError(f"Paper loss component {name} is not a matching scalar")
        return result

    def forward(
        self,
        normalized_prediction: torch.Tensor,
        *,
        context: PaperLossContext,
    ) -> torch.Tensor:
        return self.components(normalized_prediction, context=context).total
