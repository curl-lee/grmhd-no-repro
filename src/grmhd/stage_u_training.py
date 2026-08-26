"""Frozen construction and training contracts for Stage U variants."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from grmhd.models import build_model, trainable_parameter_count
from grmhd.paper_stage_g import tensor_state_sha256
from grmhd.stage_u_geometry import (
    CoordinateFDPolicy,
    StoredCoordinateFiniteDifferenceConvolution3D,
    project_upstream_weight,
)


VARIANTS = ("spectral_only", "coordinate_fd", "spherical_proxy_fd")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_stage_u_model(
    stage_s: Mapping[str, Any],
    variant: str,
    *,
    coords: Mapping[str, Sequence[float]],
    initial_state: Mapping[str, torch.Tensor],
    padding: Mapping[str, str],
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Build one geometry-only ablation with frozen common initialization."""

    if variant not in VARIANTS:
        raise ValueError(f"Unknown Stage U variant {variant!r}")
    spec = stage_s["model"]
    common = dict(
        in_channels=int(spec["in_channels"]),
        out_channels=int(spec["out_channels"]),
        n_modes=tuple(int(value) for value in spec["n_modes"]),
        hidden_channels=int(spec["hidden_channels"]),
        n_layers=int(spec["n_layers"]),
        default_in_shape=tuple(int(value) for value in spec["default_in_shape"]),
        positional_embedding=spec["positional_embedding"],
        fin_diff_kernel_size=int(spec["fin_diff_kernel_size"]),
        mix_derivatives=bool(spec["mix_derivatives"]),
        conv_padding_mode=str(spec["conv_padding_mode"]),
        use_channel_mlp=bool(spec["use_channel_mlp"]),
        local_no_skip=spec["local_no_skip"],
        norm=spec["normalization"],
        enforce_hermitian_symmetry=bool(spec["enforce_hermitian_symmetry"]),
    )
    if variant == "spectral_only":
        # Instantiate through the pinned public model with the actual requested
        # diff_layers=False rather than retaining unused differential parameters.
        from neuralop.models.local_no import LocalNO

        local_kwargs = dict(common)
        default_shape = local_kwargs.pop("default_in_shape")
        model = LocalNO(
            **local_kwargs,
            default_in_shape=default_shape,
            disco_layers=False,
            diff_layers=False,
        )
        expected = model.state_dict()
        filtered = {key: value for key, value in initial_state.items() if key in expected}
        incompatible = model.load_state_dict(filtered, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise ValueError(f"Spectral-only shared initialization failed: {incompatible}")
        initialization = {
            "method": "strict subset of Stage K shared state after diff_layers=False construction",
            "common_tensor_count": len(filtered),
            "common_tensor_state_sha256": tensor_state_sha256(filtered),
            "geometry_parameter_initialization": "none",
        }
    else:
        model = build_model("localno_differential_3d", **common)
        model.load_state_dict(initial_state, strict=True)
        policy = CoordinateFDPolicy(
            phi_padding=str(padding["phi"]),
            theta_padding=str(padding["theta"]),
            r_padding=str(padding["r"]),
            spherical_proxy=variant == "spherical_proxy_fd",
        )
        replacements = []
        for upstream in model.local_no_blocks.differential:
            replacement = StoredCoordinateFiniteDifferenceConvolution3D(
                upstream.in_channels,
                int(upstream.weight.shape[0]),
                phi=coords["phi"], theta=coords["theta"], r=coords["r"],
                policy=policy, groups=upstream.groups,
            )
            project_upstream_weight(upstream.weight.detach(), replacement)
            replacements.append(replacement)
        model.local_no_blocks.differential = torch.nn.ModuleList(replacements)
        geometry = {
            key: value for key, value in model.state_dict().items()
            if ".differential." in key
        }
        common_state = {
            key: value for key, value in model.state_dict().items()
            if ".differential." not in key
        }
        initialization = {
            "method": "strict Stage K shared load then deterministic axial marginal projection",
            "common_tensor_count": len(common_state),
            "common_tensor_state_sha256": tensor_state_sha256(common_state),
            "geometry_tensor_state_sha256": tensor_state_sha256(geometry),
            "geometry_parameter_initialization": "sum frozen 3x3x3 kernel over orthogonal axes and divide each of three additive paths by 3",
            "spherical_proxy": policy.spherical_proxy,
            "padding": {"phi": policy.phi_padding, "theta": policy.theta_padding, "r": policy.r_padding},
        }
    initialization.update({
        "variant": variant,
        "parameter_count": trainable_parameter_count(model),
        "full_initial_tensor_state_sha256": tensor_state_sha256(model.state_dict()),
    })
    return model, initialization


def load_coords(dataset: Path) -> dict[str, np.ndarray]:
    import h5py

    with h5py.File(dataset, "r") as handle:
        return {
            name: np.asarray(handle[f"coords/{name}"], dtype=np.float64)
            for name in ("phi", "theta", "r")
        }


def variant_parameter_difference(parameter_count: int, baseline_count: int = 358_296) -> dict[str, float | int]:
    difference = int(parameter_count) - int(baseline_count)
    return {
        "baseline": int(baseline_count),
        "variant": int(parameter_count),
        "absolute_difference": difference,
        "relative_difference": difference / int(baseline_count),
    }
