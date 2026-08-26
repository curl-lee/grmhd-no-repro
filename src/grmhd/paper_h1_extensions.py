"""Controlled Stage I H1 diagnostic extensions.

The frozen Stage E :class:`~grmhd.paper_losses.PaperCompositeLoss` remains the
paper-adapted reference implementation.  This module subclasses that loss and
replaces only its H1 module.  It does not define an optimizer, scheduler, data
split, or training policy.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Mapping

import torch
from torch import nn

from .paper_h1_diagnostics import H1Diagnostic, h1_diagnostic
from .paper_losses import (
    PaperCompositeLoss,
    PaperH1GradientLoss,
    PaperLossContext,
    PaperLossResult,
)


PAPER_REFERENCE_H1_WEIGHT = 0.05
STAGE_I_EXTENSION_REASON = "spherical_grid_H1_adaptation_diagnosis"
STAGE_I_UNIT_INDEX_EXTENSION_REASON = (
    "isolate_unit_cube_spacing_amplification"
)
STAGE_I_STORED_COORDINATE_EXTENSION_REASON = (
    "stored_spherical_coordinate_H1_adaptation_diagnosis"
)


class DiagnosticH1Mode(str, Enum):
    """The three authorized Stage I H1 diagnostic modes."""

    NO_H1 = "no_h1"
    UNIT_INDEX = "unit_index"
    STORED_COORDINATE_VOLUME_PROXY = "stored_coordinate_volume_proxy"


@dataclass(frozen=True)
class DiagnosticH1Result:
    """Differentiable selected-H1 value and additive decompositions."""

    mode: str
    raw_h1: torch.Tensor
    weighted_h1: torch.Tensor
    per_channel: torch.Tensor
    per_direction: Mapping[str, torch.Tensor]
    per_shell: torch.Tensor
    metadata: Mapping[str, Any]


def _validate_extension_state(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if prediction.ndim != 5 or prediction.shape[1] != 8:
        raise ValueError(
            "Stage I H1 extensions require shape (B,8,Nphi,Ntheta,Nr); "
            "extra shell channels are not loss targets"
        )
    if target.shape != prediction.shape:
        raise ValueError("Stage I prediction and target shapes differ")
    if not prediction.is_floating_point() or not target.is_floating_point():
        raise TypeError("Stage I prediction and target must be floating point")
    if prediction.device != target.device or prediction.dtype != target.dtype:
        raise ValueError("Stage I prediction and target dtype/device differ")


def _per_shell(
    diagnostic: H1Diagnostic,
    shell_masks: torch.Tensor,
) -> torch.Tensor:
    density = sum(diagnostic.direction_density.values())
    weights = diagnostic.spatial_weights.to(
        device=density.device,
        dtype=density.dtype,
    )
    masks = shell_masks.to(device=density.device, dtype=density.dtype)
    values = []
    for shell in masks:
        sample_values = torch.sum(
            density * weights * shell,
            dim=(1, 2, 3, 4),
        )
        values.append(sample_values.mean())
    output = torch.stack(values)
    if not torch.allclose(
        output.sum(),
        diagnostic.total,
        rtol=2.0e-5,
        atol=1.0e-6,
    ):
        raise RuntimeError("Stage I shell H1 decomposition is not additive")
    return output


class DiagnosticH1Extension(nn.Module):
    """Selected Stage I H1 term, independent of all other Full components."""

    def __init__(
        self,
        *,
        mode: DiagnosticH1Mode | str,
        coordinates: Mapping[str, torch.Tensor],
        shell_masks: torch.Tensor,
        paper_reference_weight: float = PAPER_REFERENCE_H1_WEIGHT,
    ) -> None:
        super().__init__()
        self.mode = DiagnosticH1Mode(mode)
        if float(paper_reference_weight) != PAPER_REFERENCE_H1_WEIGHT:
            raise ValueError("Stage I must preserve the paper reference H1 weight 0.05")
        if set(coordinates) != {"phi", "theta", "r"}:
            raise ValueError("Stage I coordinates must contain exactly phi, theta, and r")
        for name in ("phi", "theta", "r"):
            coordinate = coordinates[name]
            if coordinate.ndim != 1 or coordinate.numel() < 4:
                raise ValueError(f"Stage I coordinate {name} is invalid")
            self.register_buffer(
                f"coordinate_{name}",
                coordinate.detach().clone().to(dtype=torch.float64),
            )
        if shell_masks.ndim != 4 or shell_masks.shape[0] != 8:
            raise ValueError("Stage I shell masks must have shape (8,Nphi,Ntheta,Nr)")
        boolean_shells = shell_masks.detach().clone().to(dtype=torch.bool)
        if not torch.all(boolean_shells.sum(dim=0) == 1):
            raise ValueError("Every Stage I voxel must belong to exactly one shell")
        expected_shape = tuple(
            int(coordinates[name].numel()) for name in ("phi", "theta", "r")
        )
        if tuple(boolean_shells.shape[1:]) != expected_shape:
            raise ValueError("Stage I shell masks and coordinates have different shapes")
        self.register_buffer("shell_masks", boolean_shells)
        self.paper_reference_weight = float(paper_reference_weight)
        self.last_result: DiagnosticH1Result | None = None

    @property
    def coordinates(self) -> dict[str, torch.Tensor]:
        return {
            name: getattr(self, f"coordinate_{name}")
            for name in ("phi", "theta", "r")
        }

    def _zero_result(self, error: torch.Tensor) -> DiagnosticH1Result:
        raw = error.sum() * 0.0
        zero_channels = error.new_zeros((8,))
        zero_directions = {
            name: error.new_zeros(()) for name in ("phi", "theta", "r")
        }
        return DiagnosticH1Result(
            mode=self.mode.value,
            raw_h1=raw,
            weighted_h1=self.paper_reference_weight * raw,
            per_channel=zero_channels,
            per_direction=zero_directions,
            per_shell=error.new_zeros((8,)),
            metadata={
                "mode": self.mode.value,
                "enabled": False,
                "paper_reference_weight": self.paper_reference_weight,
                "weighted_contribution": 0.0,
                "diagnostic_proxy_only": True,
                "extension_reason": STAGE_I_EXTENSION_REASON,
            },
        )

    def components(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> DiagnosticH1Result:
        _validate_extension_state(prediction, target)
        error = prediction - target
        spatial_shape = tuple(int(value) for value in error.shape[-3:])
        expected_shape = tuple(int(value) for value in self.shell_masks.shape[1:])
        if spatial_shape != expected_shape:
            raise ValueError(
                f"Stage I H1 geometry is {expected_shape}, received {spatial_shape}"
            )
        if self.mode is DiagnosticH1Mode.NO_H1:
            result = self._zero_result(error)
            self.last_result = result
            return result

        if self.mode is DiagnosticH1Mode.UNIT_INDEX:
            diagnostic = h1_diagnostic(
                error,
                variant="H1_unit_index",
                reduction="uniform",
            )
            extension_metadata: dict[str, Any] = {
                "mode": self.mode.value,
                "enabled": True,
                "spacings": [1.0, 1.0, 1.0],
                "boundary": "periodic_wrap_all_three_axes",
                "reduction": "uniform_voxel_mean",
                "diagnostic_proxy_only": True,
            }
        else:
            diagnostic = h1_diagnostic(
                error,
                variant="H3_stored_r",
                coordinates=self.coordinates,
                reduction="volume_proxy",
            )
            extension_metadata = {
                "mode": self.mode.value,
                "enabled": True,
                "coordinate_source": "frozen_hdf5_centers",
                "phi_boundary": "periodic_centered",
                "theta_boundary": "open_three_point_lagrange",
                "r_boundary": "open_nonuniform_three_point_lagrange",
                "reduction": "normalized_r2_sin_theta_coordinate_volume_proxy",
                "covariant_GRMHD_H1": False,
                "proper_Kerr_Schild_volume": "unverified",
                "stored_components_covariant_derivative": False,
                "stored_vector_covariant_derivative": False,
                "diagnostic_proxy_only": True,
                "radial_coordinate": "physical_r",
                "pole_sin_floor": 0.0,
                "pole_handling": "theta_center_sin_clamp_min_0",
                "volume_weight_normalization": "explicit_sum_to_one",
            }
        extension_reason = (
            STAGE_I_UNIT_INDEX_EXTENSION_REASON
            if self.mode is DiagnosticH1Mode.UNIT_INDEX
            else STAGE_I_STORED_COORDINATE_EXTENSION_REASON
        )
        metadata = {
            **diagnostic.metadata,
            **extension_metadata,
            "paper_reference_weight": self.paper_reference_weight,
            "extension_reason": extension_reason,
            "shell_attribution": "derivative_density_assigned_to_evaluation_cell",
        }
        result = DiagnosticH1Result(
            mode=self.mode.value,
            raw_h1=diagnostic.total,
            weighted_h1=self.paper_reference_weight * diagnostic.total,
            per_channel=diagnostic.per_channel,
            per_direction=diagnostic.per_direction,
            per_shell=_per_shell(diagnostic, self.shell_masks),
            metadata=metadata,
        )
        self.last_result = result
        return result

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        return self.components(prediction, target).raw_h1


class DiagnosticPaperCompositeLoss(PaperCompositeLoss):
    """Full Stage E loss with exactly one controlled H1 substitution."""

    def __init__(
        self,
        *,
        diagnostic_h1_mode: DiagnosticH1Mode | str,
        coordinates: Mapping[str, torch.Tensor],
        shell_masks: torch.Tensor,
        **paper_loss_kwargs: Any,
    ) -> None:
        configured_weight = float(
            paper_loss_kwargs.get("h1_weight", PAPER_REFERENCE_H1_WEIGHT)
        )
        if configured_weight != PAPER_REFERENCE_H1_WEIGHT:
            raise ValueError("Stage I cannot modify the paper H1 coefficient")
        super().__init__(**paper_loss_kwargs)
        self.diagnostic_h1_mode = DiagnosticH1Mode(diagnostic_h1_mode)
        self.current_upstream_h1_diagnostic = PaperH1GradientLoss()
        self.h1 = DiagnosticH1Extension(
            mode=self.diagnostic_h1_mode,
            coordinates=coordinates,
            shell_masks=shell_masks,
            paper_reference_weight=configured_weight,
        )

    @property
    def metadata(self) -> dict[str, Any]:
        extension_reason = {
            DiagnosticH1Mode.NO_H1: STAGE_I_EXTENSION_REASON,
            DiagnosticH1Mode.UNIT_INDEX: (
                STAGE_I_UNIT_INDEX_EXTENSION_REASON
            ),
            DiagnosticH1Mode.STORED_COORDINATE_VOLUME_PROXY: (
                STAGE_I_STORED_COORDINATE_EXTENSION_REASON
            ),
        }[self.diagnostic_h1_mode]
        return {
            **super().metadata,
            "schema_version": "paper-h1-diagnostic-extension-loss-v1",
            "reproduction_level": "diagnostic_extension",
            "paper_faithful_full": False,
            "extension_reason": extension_reason,
            "diagnostic_h1": {
                "mode": self.diagnostic_h1_mode.value,
                "paper_reference_weight": self.h1_weight,
                "current_upstream_h1_logged_detached": True,
                "unit_index_h1_logged_detached": True,
            },
        }

    def components(
        self,
        normalized_prediction: torch.Tensor,
        *,
        context: PaperLossContext,
    ) -> PaperLossResult:
        result = super().components(
            normalized_prediction,
            context=context,
        )
        selected = self.h1.last_result
        if selected is None:
            raise RuntimeError("Stage I selected H1 components were not recorded")
        with torch.no_grad():
            current_upstream = self.current_upstream_h1_diagnostic(
                normalized_prediction.detach(),
                context.normalized_target.detach(),
            )
            unit_index = h1_diagnostic(
                normalized_prediction.detach()
                - context.normalized_target.detach(),
                variant="H1_unit_index",
                reduction="uniform",
            ).total
        diagnostics = dict(result.diagnostics)
        diagnostics.update(
            {
                "diagnostic_current_upstream_h1_raw": current_upstream,
                "diagnostic_unit_index_h1_raw": unit_index,
                "selected_h1_per_channel": selected.per_channel,
                "selected_h1_per_direction": dict(selected.per_direction),
                "selected_h1_per_shell": selected.per_shell,
                "selected_h1_metadata": dict(selected.metadata),
            }
        )
        return replace(
            result,
            diagnostics=diagnostics,
            metadata=self.metadata,
        )


__all__ = [
    "DiagnosticH1Extension",
    "DiagnosticH1Mode",
    "DiagnosticH1Result",
    "DiagnosticPaperCompositeLoss",
    "PAPER_REFERENCE_H1_WEIGHT",
    "STAGE_I_EXTENSION_REASON",
    "STAGE_I_STORED_COORDINATE_EXTENSION_REASON",
    "STAGE_I_UNIT_INDEX_EXTENSION_REASON",
]
