"""Strict Stage F metadata around pinned neuraloperator training-state bundles."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import torch
from torch import nn

from .paper_config import ResolvedPaperConfig
from .paper_h1_diagnostics import spherical_proxy_volume_weights
from .paper_stage_g import tensor_state_sha256
from .paper_velocity_roi import paper_roi_ramp
from .upstream_adapters import (
    load_upstream_training_bundle,
    save_upstream_training_bundle,
)


CHECKPOINT_SCHEMA_VERSION = "paper-stage-f-checkpoint-v1"
UPSTREAM_SAVE_NAME = "paper"


def stage_i_selected_h1_definition(
    config: ResolvedPaperConfig,
) -> dict[str, Any] | None:
    """Return the strict selected-H1 identity for trained Stage I variants."""

    if config.diagnostic_h1_mode == "unit_index":
        return {
            "mode": "unit_index",
            "weight": 0.05,
            "spacings": [1.0, 1.0, 1.0],
            "stencil": "second_order_centered",
            "boundary": "periodic_wrap_all_three_axes",
            "reduction": "uniform_voxel_mean",
            "spherical_metric_factors": False,
            "physical_r_spacing": False,
            "volume_weighting": False,
            "shell_weighting": False,
        }
    if config.diagnostic_h1_mode != "stored_coordinate_volume_proxy":
        return None

    dataset = config.resolve_path(config.values["protocol"]["dataset"])
    with h5py.File(dataset, "r") as handle:
        coordinates = {
            name: torch.as_tensor(
                handle[f"coords/{name}"][...],
                dtype=torch.float64,
            )
            for name in ("phi", "theta", "r")
        }
    coordinate_hashes = {
        name: tensor_state_sha256({name: coordinate})
        for name, coordinate in coordinates.items()
    }
    volume_weights = spherical_proxy_volume_weights(coordinates)
    return {
        "mode": "stored_coordinate_volume_proxy",
        "weight": 0.05,
        "coordinate_source": "frozen_hdf5_centers",
        "coordinate_sha256": coordinate_hashes,
        "coordinate_combined_sha256": tensor_state_sha256(coordinates),
        "radial_coordinate": "physical_r",
        "stencil": "centered_or_three_point_coordinate_lagrange",
        "phi_boundary": "periodic_centered",
        "theta_boundary": "open_three_point_lagrange",
        "r_boundary": "open_nonuniform_three_point_lagrange",
        "reduction": "normalized_r2_sin_theta_coordinate_volume_proxy",
        "volume_proxy": {
            "formula": "r2_sin_theta_dphi_dtheta_dr",
            "weight_sha256": tensor_state_sha256(
                {"spherical_coordinate_volume_proxy": volume_weights}
            ),
            "normalization": "explicit_sum_to_one",
            "normalization_sum": float(volume_weights.sum()),
            "finite": bool(torch.isfinite(volume_weights).all()),
            "nonnegative": bool(torch.all(volume_weights >= 0)),
            "pole_sin_floor": 0.0,
            "pole_handling": "theta_center_sin_clamp_min_0",
        },
        "diagnostic_proxy_only": True,
        "covariant_GRMHD_H1": False,
        "proper_Kerr_Schild_volume": "unverified",
        "stored_vector_covariant_derivative": False,
    }


def build_paper_checkpoint_metadata(
    config: ResolvedPaperConfig,
    *,
    project_commit: str,
    config_checksum: str,
    epoch: int,
    experiment_name: str,
    gradient_accumulation: int,
    validation_probe_sha256: str | None = None,
    stage_g: Mapping[str, Any] | None = None,
    stage_i: Mapping[str, Any] | None = None,
    stage_k: Mapping[str, Any] | None = None,
    stage_o: Mapping[str, Any] | None = None,
    stage_r: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build all identity fields required for strict resume/evaluation."""

    epoch = int(epoch)
    if epoch < 0:
        raise ValueError("Checkpoint epoch must be nonnegative")
    values = config.values
    provenance = values["provenance"]
    metadata = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "experiment_name": str(experiment_name),
        "mode": config.mode,
        "model_config": dict(values["model"]),
        "loss_contract": {
            "contract_version": values["loss"]["contract_version"],
            "implementation_version": values["loss"]["implementation_version"],
            "name": values["loss"]["name"],
        },
        "project_commit": str(project_commit),
        "upstream_commit": provenance["upstream"]["commit"],
        "config_checksum": str(config_checksum),
        "hdf5_checksum": provenance["dataset"]["sha256"],
        "preprocessing_checksum": provenance["preprocessing"]["sha256"],
        "stage_d_artifact_checksums": {
            name: record["sha256"] for name, record in provenance["artifacts"].items()
        },
        "stage_e_loss_contract_checksum": provenance["stage_e_loss_contract"]["sha256"],
        "oracle_baseline_checksum": provenance["oracle_baseline"]["sha256"],
        "oracle_reference_semantics_version": provenance[
            "oracle_reference_semantics_version"
        ],
        "train_indices": list(range(*values["protocol"]["train_snapshots"])),
        "validation_indices": list(range(*values["protocol"]["validation_snapshots"])),
        "dropped_transition": list(values["protocol"]["dropped_transition"]),
        "radial_mode": values["representation"]["radial"]["selected_mode"],
        "radial_runtime_semantics": "envelope_reference_only_no_state_subtraction",
        "epoch": epoch,
        "roi_ramp": paper_roi_ramp(epoch, 375),
        "optimizer": dict(values["optimizer"]),
        "scheduler": dict(values["scheduler"]),
        "gradient_accumulation": int(gradient_accumulation),
        "seed": int(values["runtime"]["seed"]),
        "thermal_adaptation": dict(values["thermal"]),
        "coordinate_adaptation": dict(values["coordinates"]),
        "representation": dict(values["representation"]),
        "validation_not_used_for_fit": provenance["validation_not_used_for_fit"],
        "validation_probe_sha256": validation_probe_sha256,
        "scientific_checkpoint": False,
    }
    if config.stage_r:
        metadata["prediction_contract"] = dict(values["prediction"])
    if stage_g is not None:
        required = {
            "shared_initial_state_sha256",
            "pair_order_sha256",
            "optimizer_step",
            "h1_weight",
            "gradient_clip_norm",
            "resource_scaled_training",
        }
        if set(stage_g) != required:
            raise ValueError(
                "Stage G checkpoint metadata fields changed: "
                f"missing={sorted(required - set(stage_g))}, "
                f"extra={sorted(set(stage_g) - required)}"
            )
        if not isinstance(stage_g["optimizer_step"], int) or stage_g["optimizer_step"] < 0:
            raise ValueError("Stage G optimizer step must be nonnegative")
        metadata["stage_g"] = dict(stage_g)
    if stage_i is not None:
        required = {
            "diagnostic_h1_mode",
            "reproduction_level",
            "paper_faithful_full",
            "extension_reason",
            "comparison_parent",
            "stage_g_parent_config",
            "selected_h1_training_coefficient",
            "diagnostic_current_upstream_h1_trained",
            "non_h1_loss_contract",
        }
        if stage_i_selected_h1_definition(config) is not None:
            required.add("selected_h1_definition")
        if set(stage_i) != required:
            raise ValueError(
                "Stage I checkpoint metadata fields changed: "
                f"missing={sorted(required - set(stage_i))}, "
                f"extra={sorted(set(stage_i) - required)}"
            )
        metadata["stage_i"] = dict(stage_i)
    if stage_k is not None:
        required = {
            "classification",
            "architecture",
            "n_dim",
            "differential_enabled",
            "disco_enabled",
            "parameter_count",
            "input_channels",
            "output_channels",
            "prediction_mode",
            "plain_loss_contract",
            "initial_state_sha256",
            "pair_order_sha256",
            "optimizer_step",
            "gradient_clip_norm",
            "resource_scaled_training",
            "run_kind",
        }
        if set(stage_k) != required:
            raise ValueError(
                "Stage K checkpoint metadata fields changed: "
                f"missing={sorted(required - set(stage_k))}, "
                f"extra={sorted(set(stage_k) - required)}"
            )
        metadata["stage_k"] = dict(stage_k)
    if stage_o is not None:
        required = {
            "classification",
            "p3_config_sha256",
            "p3_statistics_sha256",
            "p3_train_indices",
            "architecture_config_sha256",
            "stage_k_initial_state_file_sha256",
            "initial_state_sha256",
            "pair_order_sha256",
            "optimizer_step",
            "run_kind",
            "oracle_conditioned_gate_version",
            "canonical_replacement",
        }
        if set(stage_o) != required:
            raise ValueError(
                "Stage O checkpoint metadata fields changed: "
                f"missing={sorted(required - set(stage_o))}, "
                f"extra={sorted(set(stage_o) - required)}"
            )
        metadata["stage_o"] = dict(stage_o)
    if stage_r is not None:
        required = {
            "classification",
            "prediction_mode",
            "state_skip",
            "residual_scale",
            "learnable_scale",
            "clipping",
            "loss_target",
            "loss_equivalence_contract",
            "p3_config_sha256",
            "p3_statistics_sha256",
            "architecture_config_sha256",
            "initial_state_sha256",
            "pair_order_sha256",
            "optimizer_step",
            "run_kind",
            "oracle_conditioned_gate_version",
        }
        if set(stage_r) != required:
            raise ValueError(
                "Stage R checkpoint metadata fields changed: "
                f"missing={sorted(required - set(stage_r))}, "
                f"extra={sorted(set(stage_r) - required)}"
            )
        metadata["stage_r"] = dict(stage_r)
    return metadata


def _expected_identity(config: ResolvedPaperConfig) -> dict[str, Any]:
    values = config.values
    provenance = values["provenance"]
    identity = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "mode": config.mode,
        "model_config": dict(values["model"]),
        "loss_contract": {
            "contract_version": values["loss"]["contract_version"],
            "implementation_version": values["loss"]["implementation_version"],
            "name": values["loss"]["name"],
        },
        "upstream_commit": provenance["upstream"]["commit"],
        "hdf5_checksum": provenance["dataset"]["sha256"],
        "preprocessing_checksum": provenance["preprocessing"]["sha256"],
        "stage_d_artifact_checksums": {
            name: record["sha256"] for name, record in provenance["artifacts"].items()
        },
        "stage_e_loss_contract_checksum": provenance["stage_e_loss_contract"]["sha256"],
        "oracle_baseline_checksum": provenance["oracle_baseline"]["sha256"],
        "oracle_reference_semantics_version": provenance[
            "oracle_reference_semantics_version"
        ],
        "train_indices": list(range(*values["protocol"]["train_snapshots"])),
        "validation_indices": list(range(*values["protocol"]["validation_snapshots"])),
        "dropped_transition": list(values["protocol"]["dropped_transition"]),
        "radial_mode": values["representation"]["radial"]["selected_mode"],
        "radial_runtime_semantics": "envelope_reference_only_no_state_subtraction",
        "optimizer": dict(values["optimizer"]),
        "scheduler": dict(values["scheduler"]),
        "seed": int(values["runtime"]["seed"]),
        "thermal_adaptation": dict(values["thermal"]),
        "coordinate_adaptation": dict(values["coordinates"]),
        "representation": dict(values["representation"]),
        "validation_not_used_for_fit": True,
        "scientific_checkpoint": False,
    }
    if config.stage_r:
        identity["prediction_contract"] = dict(values["prediction"])
    return identity


def validate_paper_checkpoint_metadata(
    metadata: Mapping[str, Any],
    config: ResolvedPaperConfig,
    *,
    expected_config_checksum: str | None = None,
) -> None:
    for key, expected in _expected_identity(config).items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Checkpoint metadata mismatch for {key}: "
                f"expected={expected!r}, actual={metadata.get(key)!r}"
            )
    if expected_config_checksum is not None and metadata.get("config_checksum") != expected_config_checksum:
        raise ValueError("Checkpoint config checksum mismatch")
    epoch = metadata.get("epoch")
    if not isinstance(epoch, int) or epoch < 0:
        raise ValueError("Checkpoint epoch is invalid")
    expected_ramp = paper_roi_ramp(epoch, 375)
    if metadata.get("roi_ramp") != expected_ramp:
        raise ValueError("Checkpoint epoch/ROI ramp mismatch")
    if not isinstance(metadata.get("gradient_accumulation"), int) or metadata[
        "gradient_accumulation"
    ] <= 0:
        raise ValueError("Checkpoint gradient accumulation is invalid")
    stage_g = metadata.get("stage_g")
    if stage_g is not None:
        expected_keys = {
            "shared_initial_state_sha256",
            "pair_order_sha256",
            "optimizer_step",
            "h1_weight",
            "gradient_clip_norm",
            "resource_scaled_training",
        }
        if not isinstance(stage_g, Mapping) or set(stage_g) != expected_keys:
            raise ValueError("Checkpoint Stage G metadata is incomplete")
        if not isinstance(stage_g["optimizer_step"], int) or stage_g["optimizer_step"] < 0:
            raise ValueError("Checkpoint Stage G optimizer step is invalid")
        if stage_g["gradient_clip_norm"] != 1.0:
            raise ValueError("Checkpoint Stage G gradient clip changed")
        if stage_g["resource_scaled_training"] is not True:
            raise ValueError("Checkpoint Stage G resource-scaled flag changed")
        expected_h1 = (
            0.0
            if config.diagnostic_h1_mode == "no_h1"
            else 0.05
            if config.mode == "full"
            else None
        )
        if stage_g["h1_weight"] != expected_h1:
            raise ValueError("Checkpoint Stage G H1 weight/mode mismatch")
        for key in ("shared_initial_state_sha256", "pair_order_sha256"):
            value = stage_g[key]
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"Checkpoint Stage G {key} is invalid")
    stage_i = metadata.get("stage_i")
    if config.diagnostic_h1_mode is None:
        if stage_i is not None:
            raise ValueError("Non-extension checkpoint unexpectedly has Stage I metadata")
    else:
        expected_keys = {
            "diagnostic_h1_mode",
            "reproduction_level",
            "paper_faithful_full",
            "extension_reason",
            "comparison_parent",
            "stage_g_parent_config",
            "selected_h1_training_coefficient",
            "diagnostic_current_upstream_h1_trained",
            "non_h1_loss_contract",
        }
        if stage_i_selected_h1_definition(config) is not None:
            expected_keys.add("selected_h1_definition")
        if not isinstance(stage_i, Mapping) or set(stage_i) != expected_keys:
            raise ValueError("Checkpoint Stage I diagnostic metadata is incomplete")
        if stage_i["diagnostic_h1_mode"] != config.diagnostic_h1_mode:
            raise ValueError("Checkpoint Stage I diagnostic H1 mode changed")
        if stage_i["reproduction_level"] != "diagnostic_extension":
            raise ValueError("Checkpoint Stage I reproduction level changed")
        if stage_i["paper_faithful_full"] is not False:
            raise ValueError("Checkpoint Stage I paper-faithful flag changed")
        expected_reason = (
            "isolate_H1_contribution_under_spherical_grid_adaptation"
            if config.diagnostic_h1_mode == "no_h1"
            else config.values["reproduction_metadata"]["extension_reason"]
        )
        if stage_i["extension_reason"] != expected_reason:
            raise ValueError("Checkpoint Stage I extension reason changed")
        if stage_i["comparison_parent"] != "stage_g_paper_adapted_full":
            raise ValueError("Checkpoint Stage I comparison parent changed")
        if stage_i["stage_g_parent_config"] != (
            "configs/paper_reduced100/full_fno_proxy.yaml"
        ):
            raise ValueError("Checkpoint Stage I parent config changed")
        expected_coefficient = (
            0.0 if config.diagnostic_h1_mode == "no_h1" else 0.05
        )
        if stage_i["selected_h1_training_coefficient"] != expected_coefficient:
            raise ValueError("Checkpoint Stage I selected H1 coefficient changed")
        if stage_i["diagnostic_current_upstream_h1_trained"] is not False:
            raise ValueError("Checkpoint Stage I diagnostic H1 entered training")
        expected_non_h1 = {
            name: config.values["loss"][name]
            for name in ("base", "roi", "bounds", "envelope", "dissipation")
        }
        if stage_i["non_h1_loss_contract"] != expected_non_h1:
            raise ValueError("Checkpoint Stage I non-H1 loss contract changed")
        expected_definition = stage_i_selected_h1_definition(config)
        if expected_definition is not None and (
            stage_i["selected_h1_definition"] != expected_definition
        ):
            raise ValueError("Checkpoint Stage I selected H1 definition changed")
    stage_k = metadata.get("stage_k")
    if not config.stage_k:
        if stage_k is not None:
            raise ValueError("Non-Stage-K checkpoint unexpectedly has Stage K metadata")
        return
    expected_keys = {
        "classification",
        "architecture",
        "n_dim",
        "differential_enabled",
        "disco_enabled",
        "parameter_count",
        "input_channels",
        "output_channels",
        "prediction_mode",
        "plain_loss_contract",
        "initial_state_sha256",
        "pair_order_sha256",
        "optimizer_step",
        "gradient_clip_norm",
        "resource_scaled_training",
        "run_kind",
    }
    if not isinstance(stage_k, Mapping) or set(stage_k) != expected_keys:
        raise ValueError("Checkpoint Stage K metadata is incomplete")
    model = config.values["model"]
    expected_values = {
        "classification": config.values["reproduction_metadata"]["classification"],
        "architecture": "localno_differential_3d",
        "n_dim": 3,
        "differential_enabled": True,
        "disco_enabled": False,
        "parameter_count": 358296,
        "input_channels": int(model["in_channels"]),
        "output_channels": int(model["out_channels"]),
        "prediction_mode": config.prediction_mode,
        "plain_loss_contract": dict(config.values["loss"]),
        "gradient_clip_norm": 1.0,
        "resource_scaled_training": True,
    }
    for key, expected in expected_values.items():
        if stage_k[key] != expected:
            raise ValueError(f"Checkpoint Stage K {key} changed")
    if stage_k["run_kind"] not in {"engineering_smoke", "pilot30"}:
        raise ValueError("Checkpoint Stage K run kind is invalid")
    if not isinstance(stage_k["optimizer_step"], int) or stage_k["optimizer_step"] < 0:
        raise ValueError("Checkpoint Stage K optimizer step is invalid")
    for key in ("initial_state_sha256", "pair_order_sha256"):
        value = stage_k[key]
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Checkpoint Stage K {key} is invalid")
    stage_r = metadata.get("stage_r")
    if config.stage_r:
        required_r = {
            "classification",
            "prediction_mode",
            "state_skip",
            "residual_scale",
            "learnable_scale",
            "clipping",
            "loss_target",
            "loss_equivalence_contract",
            "p3_config_sha256",
            "p3_statistics_sha256",
            "architecture_config_sha256",
            "initial_state_sha256",
            "pair_order_sha256",
            "optimizer_step",
            "run_kind",
            "oracle_conditioned_gate_version",
        }
        if not isinstance(stage_r, Mapping) or set(stage_r) != required_r:
            raise ValueError("Checkpoint Stage R metadata is incomplete")
        frozen = config.values["provenance"]["stage_o_frozen"]
        expected_r = {
            "classification": "adapted_residual_contract_model_pilot",
            "prediction_mode": "normalized_residual",
            "state_skip": "identity",
            "residual_scale": 1.0,
            "learnable_scale": False,
            "clipping": False,
            "loss_target": "normalized_residual",
            "loss_equivalence_contract": (
                "PlainL2(r_theta,delta_z_true)=="
                "PlainL2(z_t+r_theta,z_t1)"
            ),
            "p3_config_sha256": config.values["preprocessing"][
                "prototype_config_checksum"
            ],
            "p3_statistics_sha256": config.values["preprocessing"][
                "stats_checksum"
            ],
            "architecture_config_sha256": frozen["stage_k_config"]["sha256"],
            "initial_state_sha256": stage_k["initial_state_sha256"],
            "pair_order_sha256": stage_k["pair_order_sha256"],
            "oracle_conditioned_gate_version": "stage_m_v1",
        }
        for key, expected in expected_r.items():
            if stage_r[key] != expected:
                raise ValueError(f"Checkpoint Stage R {key} changed")
        if stage_r["run_kind"] not in {"engineering_smoke", "pilot30"}:
            raise ValueError("Checkpoint Stage R run kind is invalid")
        if not isinstance(stage_r["optimizer_step"], int) or stage_r["optimizer_step"] < 0:
            raise ValueError("Checkpoint Stage R optimizer step is invalid")
    elif stage_r is not None:
        raise ValueError("Non-Stage-R checkpoint unexpectedly has Stage R metadata")
    stage_o = metadata.get("stage_o")
    if not config.stage_o:
        if stage_o is not None:
            raise ValueError("Non-Stage-O checkpoint unexpectedly has Stage O metadata")
        return
    required_o = {
        "classification",
        "p3_config_sha256",
        "p3_statistics_sha256",
        "p3_train_indices",
        "architecture_config_sha256",
        "stage_k_initial_state_file_sha256",
        "initial_state_sha256",
        "pair_order_sha256",
        "optimizer_step",
        "run_kind",
        "oracle_conditioned_gate_version",
        "canonical_replacement",
    }
    if not isinstance(stage_o, Mapping) or set(stage_o) != required_o:
        raise ValueError("Checkpoint Stage O metadata is incomplete")
    frozen = config.values["provenance"]["stage_o_frozen"]
    expected_o = {
        "classification": "adapted_transform_model_pilot",
        "p3_config_sha256": config.values["preprocessing"][
            "prototype_config_checksum"
        ],
        "p3_statistics_sha256": config.values["preprocessing"]["stats_checksum"],
        "p3_train_indices": list(range(11, 91)),
        "architecture_config_sha256": frozen["stage_k_config"]["sha256"],
        "stage_k_initial_state_file_sha256": frozen[
            "stage_k_initial_state_file"
        ]["sha256"],
        "initial_state_sha256": stage_k["initial_state_sha256"],
        "pair_order_sha256": stage_k["pair_order_sha256"],
        "oracle_conditioned_gate_version": "stage_m_v1",
        "canonical_replacement": False,
    }
    for key, expected in expected_o.items():
        if stage_o[key] != expected:
            raise ValueError(f"Checkpoint Stage O {key} changed")
    if stage_o["run_kind"] not in {"engineering_smoke", "pilot30"}:
        raise ValueError("Checkpoint Stage O run kind is invalid")
    if not isinstance(stage_o["optimizer_step"], int) or stage_o["optimizer_step"] < 0:
        raise ValueError("Checkpoint Stage O optimizer step is invalid")


def save_paper_checkpoint(
    checkpoint_dir: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    metadata: Mapping[str, Any],
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    epoch = int(metadata["epoch"])
    save_upstream_training_bundle(
        checkpoint_dir,
        UPSTREAM_SAVE_NAME,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=epoch,
        metadata=dict(metadata),
    )


@dataclass(frozen=True)
class LoadedPaperCheckpoint:
    model: nn.Module
    optimizer: torch.optim.Optimizer | None
    scheduler: Any | None
    epoch: int
    metadata: Mapping[str, Any]


def load_paper_checkpoint(
    checkpoint_dir: str | Path,
    *,
    config: ResolvedPaperConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    expected_config_checksum: str | None = None,
) -> LoadedPaperCheckpoint:
    """Validate identity before strict upstream state restoration."""

    checkpoint_dir = Path(checkpoint_dir)
    sidecar = checkpoint_dir / f"{UPSTREAM_SAVE_NAME}_grmhd_metadata.json"
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    validate_paper_checkpoint_metadata(
        metadata, config, expected_config_checksum=expected_config_checksum
    )
    model, optimizer, scheduler, epoch, loaded = load_upstream_training_bundle(
        checkpoint_dir,
        UPSTREAM_SAVE_NAME,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    if epoch is None or int(epoch) != int(metadata["epoch"]):
        raise ValueError("Upstream manifest epoch differs from paper metadata")
    if loaded != metadata:
        raise ValueError("Upstream-loaded metadata differs from prevalidated sidecar")
    return LoadedPaperCheckpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=int(epoch),
        metadata=metadata,
    )
