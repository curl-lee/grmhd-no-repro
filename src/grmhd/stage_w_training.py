"""Frozen model construction and decision contracts for Stage W."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import torch

from grmhd.models import trainable_parameter_count
from grmhd.paper_stage_g import tensor_state_sha256
from grmhd.stage_w_mixed_basis import (
    AxisBoundaryFiniteDifferenceConvolution3d,
    MixedBasisSpectralConv3d,
)


VARIANTS = ("mixed_basis", "mixed_basis_boundary")
BASELINE_PARAMETER_COUNT = 358_296


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _model_kwargs(stage_s: Mapping[str, Any]) -> dict[str, Any]:
    spec = stage_s["model"]
    return {
        "in_channels": int(spec["in_channels"]),
        "out_channels": int(spec["out_channels"]),
        "n_modes": tuple(int(value) for value in spec["n_modes"]),
        "hidden_channels": int(spec["hidden_channels"]),
        "n_layers": int(spec["n_layers"]),
        "default_in_shape": tuple(int(value) for value in spec["default_in_shape"]),
        "positional_embedding": spec["positional_embedding"],
        "fin_diff_kernel_size": int(spec["fin_diff_kernel_size"]),
        "mix_derivatives": bool(spec["mix_derivatives"]),
        "conv_padding_mode": str(spec["conv_padding_mode"]),
        "use_channel_mlp": bool(spec["use_channel_mlp"]),
        "local_no_skip": spec["local_no_skip"],
        "norm": spec["normalization"],
        "enforce_hermitian_symmetry": bool(spec["enforce_hermitian_symmetry"]),
    }


def _load_permuted_stage_t_initialization(
    model: torch.nn.Module, initial_state: Mapping[str, torch.Tensor]
) -> dict[str, Any]:
    target = model.state_dict()
    mapped: dict[str, torch.Tensor] = {}
    spectral_pairs: dict[str, str] = {}
    for key, value in target.items():
        # neuraloperator's BaseModel appends a descriptive, non-tensor
        # ``_metadata`` entry.  It is not loadable model state.
        if not torch.is_tensor(value):
            continue
        if key.startswith("local_no_blocks.convs.") and key.endswith(".weight"):
            source_key = key + ".tensor"
            source = initial_state[source_key]
            # Stage T stores (phi FFT, theta FFT, r rFFT) = (8,8,5).
            # Stage W stores (phi rFFT, theta DCT, r DCT) = (5,8,8).
            projected = source.permute(0, 1, 4, 3, 2).contiguous()
            if projected.shape != value.shape:
                raise ValueError(
                    f"Spectral permutation {source_key} -> {key} has shape "
                    f"{tuple(projected.shape)}, expected {tuple(value.shape)}"
                )
            mapped[key] = projected
            spectral_pairs[key] = source_key
        else:
            if key not in initial_state:
                raise KeyError(f"No frozen Stage-T initialization tensor for {key}")
            if initial_state[key].shape != value.shape:
                raise ValueError(f"Frozen tensor shape changed for {key}")
            mapped[key] = initial_state[key]
    incompatible = model.load_state_dict(mapped, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError(f"Mixed-basis initialization was not strict: {incompatible}")
    non_spectral = {
        key: value for key, value in model.state_dict().items()
        if not key.startswith("local_no_blocks.convs.")
    }
    return {
        "method": "strict common-tensor load plus reversible spectral-axis permutation",
        "spectral_source_to_target": spectral_pairs,
        "spectral_permutation": [0, 1, 4, 3, 2],
        "source_modal_shape": [8, 8, 5],
        "target_modal_shape": [5, 8, 8],
        "non_spectral_tensor_state_sha256": tensor_state_sha256(non_spectral),
    }


def _replace_fd_boundaries(model: torch.nn.Module) -> None:
    replacements = []
    for upstream in model.local_no_blocks.differential:
        replacement = AxisBoundaryFiniteDifferenceConvolution3d(
            upstream.in_channels,
            int(upstream.weight.shape[0]),
            kernel_size=int(upstream.kernel_size),
            groups=int(upstream.groups),
            phi_padding="circular",
            theta_padding="replicate",
            radial_padding="replicate",
        )
        replacement.load_state_dict(upstream.state_dict(), strict=True)
        replacements.append(replacement)
    model.local_no_blocks.differential = torch.nn.ModuleList(replacements)


def build_stage_w_model(
    stage_s: Mapping[str, Any],
    variant: str,
    *,
    initial_state: Mapping[str, torch.Tensor],
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Construct W1/W2 with identical initial tensors and only frozen changes."""

    if variant not in VARIANTS:
        raise ValueError(f"Unknown Stage W variant {variant!r}")
    from neuralop.models.local_no import LocalNO

    model = LocalNO(
        **_model_kwargs(stage_s),
        disco_layers=False,
        diff_layers=True,
        conv_module=MixedBasisSpectralConv3d,
    )
    initialization = _load_permuted_stage_t_initialization(model, initial_state)
    pre_boundary_hash = tensor_state_sha256(model.state_dict())
    if variant == "mixed_basis_boundary":
        _replace_fd_boundaries(model)
    full_hash = tensor_state_sha256(model.state_dict())
    if full_hash != pre_boundary_hash:
        raise ValueError("W2 boundary replacement changed the common initial tensors")
    count = trainable_parameter_count(model)
    difference = count - BASELINE_PARAMETER_COUNT
    initialization.update({
        "variant": variant,
        "full_initial_tensor_state_sha256": full_hash,
        "parameter_count": count,
        "parameter_count_difference": difference,
        "parameter_count_relative_difference": difference / BASELINE_PARAMETER_COUNT,
        "spectral_basis": ["rFFT_phi", "DCT-II_theta", "DCT-II_log_r"],
        "differential_boundary": (
            {"phi": "circular", "theta": "replicate", "r": "replicate"}
            if variant == "mixed_basis_boundary"
            else {"phi": "circular", "theta": "circular", "r": "circular"}
        ),
        "MIXED_BASIS_IS_NOT_SPHERICAL_HARMONICS": True,
        "MIXED_BASIS_IS_NOT_KERR_SCHILD_COVARIANT": True,
    })
    return model, initialization


def stage_w_decision(
    w1: Mapping[str, object],
    w2: Mapping[str, object],
    *,
    boundary_benefit: bool,
    baseline_shell: float = -1.67947,
    baseline_radial: float = -0.348089,
) -> str:
    """Apply the frozen Stage-W A/B/C/D scientific decision labels."""

    def retained(row: Mapping[str, object]) -> bool:
        return (
            float(row["state_l2"]) <= 0.280047
            and float(row["residual_l2"]) < 1.0
            and float(row["cosine"]) > 0.7
        )

    def improvement(row: Mapping[str, object]) -> tuple[float, float]:
        return (
            float(row["shell_skill"]) - baseline_shell,
            float(row["radial_skill"]) - baseline_radial,
        )

    w1_delta = improvement(w1)
    w2_delta = improvement(w2)
    w1_gain = max(w1_delta)
    w2_gain = max(w2_delta)
    w1_rollout = w1.get("first10x")
    w2_rollout = w2.get("first10x")
    rollout_gain = (
        w1_rollout is None or int(w1_rollout) > 1
        or w2_rollout is None or int(w2_rollout) > 1
    )
    if boundary_benefit and w1_gain < 0.10 and w2_gain >= 0.10 and retained(w2):
        return "C"
    supported = [row for row in (w1, w2) if retained(row)]
    if supported:
        strong = any(
            float(row["shell_skill"]) > 0 or float(row["radial_skill"]) > 0
            for row in supported
        ) or rollout_gain
        clear = any(max(improvement(row)) >= 0.10 for row in supported)
        if clear and strong:
            return "A"
        if clear:
            return "B"
    return "D"
