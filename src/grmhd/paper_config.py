"""Strict paired configuration handling for the Stage F reproduction.

The Full and Plain experiments deliberately share one resolver.  Resolution
validates every frozen checksum but does not fit, rewrite, or otherwise mutate
any Stage C--E artifact.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import h5py
import torch
import yaml

from .dataset import sha256_file
from .paper_bounds import PaperPhysicalBounds
from .paper_dissipation import PaperDissipativeReference
from .paper_h1_extensions import (
    DiagnosticH1Mode,
    DiagnosticPaperCompositeLoss,
    STAGE_I_EXTENSION_REASON,
    STAGE_I_STORED_COORDINATE_EXTENSION_REASON,
    STAGE_I_UNIT_INDEX_EXTENSION_REASON,
)
from .paper_losses import PaperCompositeLoss, PlainL2Loss
from .paper_priors import PaperResidualEnvelope
from .paper_protocol import EXPECTED_UPSTREAM_COMMIT, PaperReduced100Protocol
from .paper_radial import PaperRadialBaseline
from .paper_velocity_roi import PaperVelocityROI
from .models import build_model
from .shells import radial_shells_tensor
from .upstream_adapters import UpstreamFNOConfig, build_upstream_fno


PAIR_ALLOWED_DIFFERENCES = frozenset(
    {"experiment_name", "loss.*", "output_dir", "checkpoint_prefix"}
)
STAGE_I_PAIR_ALLOWED_DIFFERENCES = frozenset(
    {
        "experiment_name",
        "output_dir",
        "checkpoint_prefix",
        "loss.h1.*",
        "reproduction_metadata.reproduction_level",
        "reproduction_metadata.paper_faithful_full",
        "reproduction_metadata.extension_reason",
    }
)
EXPECTED_PROTOCOL_NAME = "paper_reduced100_press_spherical_ks"
EXPECTED_HDF5_SHA256 = (
    "cd3966bed1f617f49d775151cff87ae92eb2432a560bb82ae9355605e7aea70a"
)
EXPECTED_STATS_SHA256 = (
    "1645714424d95b02de2283fc16a8ef05f3864d15f11025580af13f3e67964001"
)
EXPECTED_STAGE_O_P3_STATS_SHA256 = (
    "aea6d3f91e54865deeb9b2b1142c49a633ae8b67223075273e6c0b88bf541948"
)
EXPECTED_STAGE_O_P3_CONFIG_SHA256 = (
    "41409e803cc0f54bede7cc73a84d9986e081bb07265c84dcac2fd64a44778809"
)
EXPECTED_STAGE_O_P3_METADATA_SHA256 = (
    "71919bffd5a13d96a500b76fa06ea8308dc578c524893984fd4f168cd9eba0e2"
)
EXPECTED_STAGE_O_PAIR_ORDER_SHA256 = (
    "5e1e361ea75e01ffd667822d8bfe870d70c5bc69570d52cdf806e637b2df2b52"
)
EXPECTED_ORACLE_SEMANTICS = "paper-reference-semantics-v1"
EXPECTED_LOSS_CONTRACT = "paper-stage-e-loss-contract-v1"
EXPECTED_RADIAL_MODE = "appendix_literal_press_proxy"


COMMON_CONTRACT: dict[str, Any] = {
    "protocol": {
        "name": EXPECTED_PROTOCOL_NAME,
        "dataset": "data_proc/grmhd_regrid_inner_r200_64.h5",
        "source_snapshots": [11, 111],
        "train_snapshots": [11, 91],
        "validation_snapshots": [91, 111],
        "dropped_transition": [90, 91],
        "test_split": None,
    },
    "thermal": {
        "channel": "press",
        "paper_adaptation": True,
        "eos_conversion": "disabled_unverified_gamma",
    },
    "coordinates": {
        "system": "spherical_kerr_schild",
        "tensor_axes": ["phi", "theta", "r"],
        "stored_components_not_cartesian": True,
    },
    "preprocessing": {
        "mode": "canonical_paper",
        "gamma": 6,
        "inverse_clamp_fraction": 0.99,
        "stats_path": "outputs/paper_reduced100/stats/normalizer.npz",
        "stats_checksum": EXPECTED_STATS_SHA256,
    },
    "representation": {
        "shells": {
            "enabled": True,
            "count": 8,
            "mode": "spherical_r_adaptation",
        },
        "radial": {
            "selected_mode": EXPECTED_RADIAL_MODE,
            "artifact": "outputs/paper_reduced100/priors/radial_selected.json",
            "exact_cartesian_paper_radius": False,
        },
    },
    "evaluation": {
        "oracle_aware": True,
        "rho_press_eval_clamp": True,
        "report_normalized": True,
        "report_model_to_oracle": True,
        "report_model_to_raw": True,
        "report_oracle_floor": True,
    },
    "model": {
        "source": "neuralop.models.FNO",
        "in_channels": 16,
        "out_channels": 8,
        "n_modes": [8, 8, 8],
        "hidden_channels": 16,
        "n_layers": 4,
        "positional_embedding": None,
        "mixed_precision": False,
    },
    "optimizer": {
        "name": "Adam",
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-4,
    },
    "scheduler": {
        "name": "warmup_cosine",
        "paper_epochs": 1200,
        "actual_default_epochs": 30,
        "paper_warmup_epochs": 75,
        "actual_default_warmup_epochs": 2,
        "min_learning_rate": 1.0e-6,
        "resource_scaled_training": True,
    },
    "runtime": {
        "batch_size": 1,
        "gradient_accumulation": 4,
        "gradient_clip_norm": 1.0,
        "seed": 42,
        "num_workers": 0,
        "pin_memory": True,
        "wandb": False,
    },
}


STAGE_O_P3_PREPROCESSING_CONTRACT: dict[str, Any] = {
    "mode": "stage_n_p3",
    "artifact": "outputs/paper_reduced100/stage_n/prototypes/p3_combined_v1",
    "stats_path": (
        "outputs/paper_reduced100/stage_n/prototypes/p3_combined_v1/normalizer.npz"
    ),
    "stats_checksum": EXPECTED_STAGE_O_P3_STATS_SHA256,
    "prototype_config": (
        "outputs/paper_reduced100/stage_n/prototypes/p3_combined_v1/resolved_config.json"
    ),
    "prototype_config_checksum": EXPECTED_STAGE_O_P3_CONFIG_SHA256,
    "prototype_metadata": (
        "outputs/paper_reduced100/stage_n/prototypes/p3_combined_v1/normalizer.json"
    ),
    "prototype_metadata_checksum": EXPECTED_STAGE_O_P3_METADATA_SHA256,
    "prototype": "P3",
    "prototype_name": "COMBINED_PROTOTYPE_V1",
    "gamma": 6,
    "inverse_clamp_fraction": 0.99,
    "canonical_replacement": False,
    "validation_used_for_fit": False,
}


STAGE_K_MODEL_CONTRACT: dict[str, Any] = {
    "source": "neuralop.models.LocalNO",
    "architecture": "localno_differential_3d",
    "n_dim": 3,
    "in_channels": 16,
    "out_channels": 8,
    "n_modes": [8, 8, 8],
    "hidden_channels": 16,
    "n_layers": 4,
    "default_in_shape": [64, 64, 64],
    "positional_embedding": None,
    "mixed_precision": False,
    "use_differential": True,
    "use_disco": False,
    "use_local_integral": False,
    "spectral_enabled": True,
    "fin_diff_kernel_size": 3,
    "mix_derivatives": True,
    "conv_padding_mode": "periodic",
    "use_channel_mlp": False,
    "local_no_skip": "linear",
    "normalization": None,
    "enforce_hermitian_symmetry": True,
    "prediction_mode": "direct",
}


STAGE_R_MODEL_CONTRACT: dict[str, Any] = {
    **STAGE_K_MODEL_CONTRACT,
    "prediction_mode": "normalized_residual",
}


STAGE_R_PREDICTION_CONTRACT: dict[str, Any] = {
    "mode": "normalized_residual",
    "state_skip": "identity",
    "residual_scale": 1.0,
    "learnable_scale": False,
    "clipping": False,
}


STAGE_K_REPRODUCTION_METADATA: dict[str, Any] = {
    "classification": "adapted_method_reproduction",
    "reproduction_level": "adapted_method_reproduction",
    "exact": [
        "ordered_reduced100_split",
        "canonical_paper_preprocessing",
        "plain_l2_loss_contract",
        "pinned_upstream_differential_localno_implementation",
    ],
    "adapted": [
        "differential_localno_without_disco_integral",
        "press_thermal_proxy",
        "spherical_kerr_schild_grid_and_stored_components",
        "spherical_r_shell_inputs",
        "resource_scaled_30_epochs",
    ],
    "blocked": [
        "exact_3d_disco_backbone",
        "unverified_eint_eos_conversion",
        "cartesian_kerr_schild_conversion",
        "paper_300_snapshot_dataset",
        "paper_1200_epoch_budget",
        "coarse_fine_coupled_simulation",
    ],
}


STAGE_O_REPRODUCTION_METADATA: dict[str, Any] = {
    "classification": "adapted_transform_model_pilot",
    "reproduction_level": "adapted_transform_model_pilot",
    "exact": [
        "ordered_reduced100_split",
        "frozen_stage_n_p3_transform_and_train_only_statistics",
        "stage_k_localno_architecture_and_initial_tensor_state",
        "plain_l2_loss_contract",
        "stage_g_stage_k_pair_order",
    ],
    "adapted": [
        "diagnostic_transform_prototype_used_for_authorized_stage_o_training",
        "differential_localno_without_disco_integral",
        "press_thermal_proxy",
        "spherical_kerr_schild_grid_and_stored_components",
        "spherical_r_shell_inputs",
        "resource_scaled_30_epochs",
    ],
    "blocked": list(STAGE_K_REPRODUCTION_METADATA["blocked"]),
    "canonical_replacement": False,
    "comparison_parent": "stage_k_canonical_localno",
}


STAGE_R_REPRODUCTION_METADATA: dict[str, Any] = {
    "classification": "adapted_residual_contract_model_pilot",
    "reproduction_level": "adapted_residual_contract_model_pilot",
    "exact": [
        "ordered_reduced100_split",
        "frozen_stage_n_p3_transform_and_train_only_statistics",
        "stage_k_localno_architecture_and_initial_tensor_state",
        "plain_l2_loss_contract",
        "stage_g_stage_k_pair_order",
        "stage_o_optimizer_scheduler_and_epoch_budget",
    ],
    "adapted": [
        "normalized_residual_target_with_identity_state_skip",
        "diagnostic_transform_prototype_used_for_authorized_stage_r_training",
        "differential_localno_without_disco_integral",
        "press_thermal_proxy",
        "spherical_kerr_schild_grid_and_stored_components",
        "spherical_r_shell_inputs",
        "resource_scaled_30_epochs",
    ],
    "blocked": list(STAGE_K_REPRODUCTION_METADATA["blocked"]),
    "canonical_replacement": False,
    "comparison_parent": "stage_o_p3_direct_localno",
    "one_factor_change": "prediction_and_target_contract",
}


FULL_LOSS_CONTRACT: dict[str, Any] = {
    "name": "PaperCompositeLoss",
    "contract_version": EXPECTED_LOSS_CONTRACT,
    "implementation_version": "paper-composite-loss-v1",
    "base": {
        "magnetic_weight": 1.2,
        "density_weight": 1.0,
        "pressure_weight": 1.0,
        "velocity_weight": 1.0,
    },
    "h1": {
        "enabled": True,
        "weight": 0.05,
        "implementation": "pinned_upstream_H1Loss_d3",
        "metric_adaptation": "computational_grid",
    },
    "roi": {
        "enabled": True,
        "top_fraction": 0.20,
        "kappa": 8,
        "ramp_epochs": 375,
        "mask_source": "canonical_oracle_physical_target",
    },
    "bounds": {
        "enabled": True,
        "artifact": "outputs/paper_reduced100/priors/physical_bounds.json",
        "rho_lower_weight": 0.05,
        "press_lower_weight": 0.05,
        "upper_weights": 0,
    },
    "envelope": {
        "enabled": True,
        "artifact": "outputs/paper_reduced100/priors/residual_envelope.json",
        "delta_rho": 1.5,
        "delta_press": 1.5,
        "weight_rho": 0.05,
        "weight_press": 0.05,
    },
    "dissipation": {
        "enabled": True,
        "artifact": "outputs/paper_reduced100/priors/dissipative_reference.json",
        "alpha": 5.0e-4,
    },
}


PLAIN_LOSS_CONTRACT: dict[str, Any] = {
    "name": "PlainL2Loss",
    "contract_version": EXPECTED_LOSS_CONTRACT,
    "implementation_version": "paper-plain-l2-v1",
    "unit_channel_weights": True,
    "h1": {"enabled": False},
    "roi": {"enabled": False},
    "bounds": {"training_penalty_enabled": False},
    "envelope": {"enabled": False},
    "dissipation": {"enabled": False},
    "diagnostic_extensions_enabled": False,
}


STAGE_R_LOSS_CONTRACT: dict[str, Any] = {
    **PLAIN_LOSS_CONTRACT,
    "mode": "strict_plain_l2",
    "target": "normalized_residual",
}


STAGE_I_EXTENSION_METADATA: dict[str, Any] = {
    "reproduction_level": "diagnostic_extension",
    "paper_faithful_full": False,
    "extension_reason": STAGE_I_EXTENSION_REASON,
}


STAGE_I_H1_CONTRACTS: dict[str, dict[str, Any]] = {
    DiagnosticH1Mode.NO_H1.value: {
        "enabled": False,
        "weight": 0.05,
        "implementation": "diagnostic_no_h1",
        "metric_adaptation": "selected_h1_disabled",
        "mode": DiagnosticH1Mode.NO_H1.value,
        "paper_reference_weight": 0.05,
        "weighted_contribution": 0.0,
        "diagnostic_current_upstream_h1": True,
    },
    DiagnosticH1Mode.UNIT_INDEX.value: {
        "enabled": True,
        "weight": 0.05,
        "implementation": "stage_h_H1_unit_index",
        "metric_adaptation": "unit_index_spacing",
        "mode": DiagnosticH1Mode.UNIT_INDEX.value,
        "paper_reference_weight": 0.05,
        "weighted_contribution": "paper_reference_weight_times_raw_h1",
        "diagnostic_current_upstream_h1": True,
        "spacings": [1.0, 1.0, 1.0],
        "boundary": "periodic_wrap_all_three_axes",
        "reduction": "uniform_voxel_mean",
    },
    DiagnosticH1Mode.STORED_COORDINATE_VOLUME_PROXY.value: {
        "enabled": True,
        "weight": 0.05,
        "implementation": "stage_h_H3_stored_r_volume_proxy",
        "metric_adaptation": "stored_coordinate_volume_proxy",
        "mode": DiagnosticH1Mode.STORED_COORDINATE_VOLUME_PROXY.value,
        "paper_reference_weight": 0.05,
        "weighted_contribution": "paper_reference_weight_times_raw_h1",
        "diagnostic_current_upstream_h1": True,
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
    },
}


def _deep_merge(
    base: Mapping[str, Any],
    override: Mapping[str, Any],
) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _infer_project_root(config_path: Path) -> Path:
    for parent in config_path.parents:
        if parent.name == "configs":
            return parent.parent.resolve()
    raise ValueError(f"Cannot infer project root from config path {config_path}")


def _load_config_values(config_path: Path, root: Path) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Paper experiment config must contain a YAML mapping")
    if "extends" not in raw:
        return raw

    if raw.get("preprocessing", {}).get("mode") == "stage_n_p3":
        stage_r = raw.get("prediction", {}).get("mode") == "normalized_residual"
        allowed = {
            "extends",
            "experiment_name",
            "output_dir",
            "checkpoint_prefix",
            "preprocessing",
            "evaluation",
            "provenance",
            "reproduction_metadata",
            *({"model", "prediction", "loss"} if stage_r else set()),
        }
        if set(raw) != allowed:
            raise ValueError(
                "Stage O/Stage R config must contain exactly the frozen P3 overrides"
            )
        base_path = (config_path.parent / str(raw["extends"])).resolve()
        expected_base = (
            root / "configs/paper_reduced100/stage_k_localno_differential_plain.yaml"
        ).resolve()
        if base_path != expected_base:
            raise ValueError("Stage O/Stage R must inherit the frozen Stage K LocalNO config")
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        if not isinstance(base, dict) or "extends" in base:
            raise ValueError("Frozen Stage K config cannot itself inherit another config")
        overrides = {key: value for key, value in raw.items() if key != "extends"}
        return _deep_merge(base, overrides)

    allowed_override_keys = {
        "extends",
        "experiment_name",
        "output_dir",
        "checkpoint_prefix",
        "loss",
        "reproduction_metadata",
    }
    if not set(raw).issubset(allowed_override_keys):
        raise ValueError("Stage I extension config contains a non-paired override")
    if set(raw.get("loss", {})) != {"h1"}:
        raise ValueError("Stage I extension config may override only loss.h1")
    raw_h1_mode = raw.get("loss", {}).get("h1", {}).get("mode")
    extension_reason = {
        DiagnosticH1Mode.NO_H1.value: STAGE_I_EXTENSION_REASON,
        DiagnosticH1Mode.UNIT_INDEX.value: STAGE_I_UNIT_INDEX_EXTENSION_REASON,
        DiagnosticH1Mode.STORED_COORDINATE_VOLUME_PROXY.value: (
            STAGE_I_STORED_COORDINATE_EXTENSION_REASON
        ),
    }.get(raw_h1_mode)
    if extension_reason is None:
        raise ValueError("Stage I extension H1 mode is invalid")
    expected_extension_metadata = {
        **STAGE_I_EXTENSION_METADATA,
        "extension_reason": extension_reason,
    }
    if raw.get("reproduction_metadata") != expected_extension_metadata:
        raise ValueError("Stage I extension metadata changed")
    required_overrides = {
        "experiment_name",
        "output_dir",
        "checkpoint_prefix",
        "loss",
        "reproduction_metadata",
    }
    if not required_overrides.issubset(raw):
        raise ValueError("Stage I extension config is missing a paired override")

    base_path = (config_path.parent / str(raw["extends"])).resolve()
    expected_base = (root / "configs/paper_reduced100/full_fno_proxy.yaml").resolve()
    if base_path != expected_base:
        raise ValueError("Stage I extensions must inherit the frozen Full config")
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(base, dict) or "extends" in base:
        raise ValueError("Frozen Full config cannot itself inherit another config")
    overrides = {key: value for key, value in raw.items() if key != "extends"}
    return _deep_merge(base, overrides)


def _require_subset(actual: Mapping[str, Any], expected: Mapping[str, Any], prefix: str = "") -> None:
    for key, expected_value in expected.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if key not in actual:
            raise ValueError(f"Resolved paper config is missing {path}")
        actual_value = actual[key]
        if isinstance(expected_value, Mapping):
            if not isinstance(actual_value, Mapping):
                raise ValueError(f"Resolved paper config {path} must be a mapping")
            _require_subset(actual_value, expected_value, path)
        elif actual_value != expected_value:
            raise ValueError(
                f"Resolved paper config {path} changed: "
                f"expected={expected_value!r}, actual={actual_value!r}"
            )


def _resolved_path(root: Path, configured: str) -> Path:
    path = Path(configured)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _validate_checksum(root: Path, record: Mapping[str, Any], name: str) -> None:
    if set(record).issuperset({"path", "sha256"}) is False:
        raise ValueError(f"Provenance record {name} requires path and sha256")
    path = _resolved_path(root, str(record["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"Frozen {name} artifact is missing: {path}")
    actual = sha256_file(path)
    if actual != str(record["sha256"]):
        raise ValueError(
            f"Stale {name} checksum: configured={record['sha256']}, actual={actual}"
        )


def _validate_upstream(root: Path, provenance: Mapping[str, Any]) -> None:
    upstream = provenance.get("upstream")
    if not isinstance(upstream, Mapping):
        raise ValueError("provenance.upstream must be a mapping")
    if upstream.get("commit") != EXPECTED_UPSTREAM_COMMIT:
        raise ValueError("Configured neuraloperator commit changed")
    path = _resolved_path(root, str(upstream.get("path", "")))
    head = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != EXPECTED_UPSTREAM_COMMIT:
        raise ValueError(f"Pinned neuraloperator checkout changed: {head}")
    status = subprocess.run(
        ["git", "-C", str(path), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status:
        raise ValueError("Pinned neuraloperator worktree is dirty")


def _validate_frozen_provenance(root: Path, values: Mapping[str, Any]) -> None:
    provenance = values.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Paper config requires provenance metadata")
    _validate_upstream(root, provenance)
    uses_p3 = values.get("preprocessing", {}).get("mode") == "stage_n_p3"
    for name in ("dataset", "manifest", "preprocessing"):
        record = provenance.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"provenance.{name} must be a mapping")
        _validate_checksum(root, record, name)
    artifacts = provenance.get("artifacts")
    required_artifacts = {"radial", "bounds", "roi", "envelope", "dissipation", "shells"}
    if not isinstance(artifacts, Mapping) or set(artifacts) != required_artifacts:
        raise ValueError("Provenance must contain exactly the six Stage D runtime artifacts")
    for name in sorted(required_artifacts):
        record = artifacts[name]
        if not isinstance(record, Mapping):
            raise ValueError(f"provenance.artifacts.{name} must be a mapping")
        _validate_checksum(root, record, f"Stage D {name}")
    for name in ("stage_e_loss_contract", "oracle_baseline", "loss_contract_document"):
        record = provenance.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"provenance.{name} must be a mapping")
        _validate_checksum(root, record, name)
    if provenance["dataset"]["sha256"] != EXPECTED_HDF5_SHA256:
        raise ValueError("Canonical HDF5 checksum changed")
    expected_stats = (
        EXPECTED_STAGE_O_P3_STATS_SHA256 if uses_p3 else EXPECTED_STATS_SHA256
    )
    if provenance["preprocessing"]["sha256"] != expected_stats:
        raise ValueError("Configured canonical/P3 preprocessing checksum changed")
    if values["preprocessing"]["stats_checksum"] != provenance["preprocessing"]["sha256"]:
        raise ValueError("Preprocessing checksum is inconsistent across config sections")
    if values["protocol"]["dataset"] != provenance["dataset"]["path"]:
        raise ValueError("Dataset path is inconsistent across config sections")
    if values["representation"]["radial"]["artifact"] != artifacts["radial"]["path"]:
        raise ValueError("Radial artifact path is inconsistent across config sections")
    if provenance["stage_e_loss_contract"].get("schema_version") != EXPECTED_LOSS_CONTRACT:
        raise ValueError("Stage E loss contract version changed")
    if provenance.get("oracle_reference_semantics_version") != EXPECTED_ORACLE_SEMANTICS:
        raise ValueError("Oracle reference semantics version changed")
    if provenance["oracle_baseline"].get("semantics_version") != EXPECTED_ORACLE_SEMANTICS:
        raise ValueError("Oracle baseline semantics version changed")
    if provenance.get("validation_not_used_for_fit") is not True:
        raise ValueError("Validation must remain excluded from every fit")

    if uses_p3:
        canonical = provenance.get("canonical_preprocessing")
        if not isinstance(canonical, Mapping):
            raise ValueError("Stage O requires the frozen canonical preprocessing record")
        _validate_checksum(root, canonical, "Stage O canonical preprocessing")
        if canonical["sha256"] != EXPECTED_STATS_SHA256:
            raise ValueError("Stage O canonical preprocessing checksum changed")
        frozen = provenance.get("stage_o_frozen")
        expected_hashes = {
            "prototype_config": EXPECTED_STAGE_O_P3_CONFIG_SHA256,
            "prototype_metadata": EXPECTED_STAGE_O_P3_METADATA_SHA256,
            "prototype_manifest": "935ebe068b2afad03bb44dd3f5aab895fef3e91e241bfa560c462deadb447c4d",
            "stage_n_config": "e366d5a94b14ea70777b42ed29bdd3b8f6477b2360e70bf9ed6ea874a1e9f1ff",
            "stage_n_decision": "9b4a8492a0a8729fa75956a1c1aaf8d0bf6540d33f1b75e9751a7742eb4b5dc5",
            "stage_k_config": "8b229c5c2528d66bd4e11d0b0c4af16d00e906129386f8119af1670721ba30af",
            "stage_k_initial_state_file": "07db393f5c3e831f44bcc2d47b8132c3f9a0e72e2a258618080db487cc13cbb0",
            "pair_order": EXPECTED_STAGE_O_PAIR_ORDER_SHA256,
            "stage_m_gate": "a829a3760a71a006f54ae18b399c569a903898d27f3ef3a72236982fd033e5f7",
        }
        if not isinstance(frozen, Mapping) or set(frozen) != set(expected_hashes):
            raise ValueError("Stage O frozen provenance record set changed")
        for name, expected_hash in expected_hashes.items():
            record = frozen[name]
            if not isinstance(record, Mapping):
                raise ValueError(f"Stage O frozen provenance {name} must be a mapping")
            _validate_checksum(root, record, f"Stage O {name}")
            if record["sha256"] != expected_hash:
                raise ValueError(f"Stage O frozen provenance hash changed: {name}")
        prototype_config = json.loads(
            _resolved_path(
                root, str(frozen["prototype_config"]["path"])
            ).read_text(encoding="utf-8")
        )
        if (
            prototype_config.get("key") != "P3"
            or prototype_config.get("name") != "COMBINED_PROTOTYPE_V1"
            or prototype_config.get("training_indices") != list(range(11, 91))
            or prototype_config.get("validation_indices_used_for_fit") != []
            or prototype_config.get("canonical_replacement") is not False
        ):
            raise ValueError("Stage O P3 prototype identity or train-only fit changed")
        policies = prototype_config.get("channel_policies", {})
        expected_policies = {
            "Bcc1": "canonical",
            "Bcc2": "no_softclip",
            "Bcc3": "no_softclip",
            "rho": "canonical",
            "press": "canonical",
            "vel1": "canonical",
            "vel2": "canonical",
            "vel3": "no_softclip",
        }
        if policies != expected_policies:
            raise ValueError("Stage O P3 channel policy changed")

        if values.get("prediction") == STAGE_R_PREDICTION_CONTRACT:
            stage_r_frozen = provenance.get("stage_r_frozen")
            expected_stage_r = {
                "stage_o_best_checkpoint": "756ee7f31a9780942aa0f35dce4e5584cbeaf04ed8e9777721e7765962f04e12",
                "stage_o_training_summary": "a0171a8c8c41dda043df8e63c9fd2a48b1e1fe27711bc6ce86d910c45d7d8cf7",
                "stage_o_validation_metrics": "b97f9735f0d405837ee3326c8548cd2bb25b848e4217bfaffd49e080affe3051",
                "stage_o_gt_rollout": "e929d4bc74f69ba027edf21aff88be1210a817716af8a9e65f104ca66d47772c",
                "stage_o_no_gt_rollout": "3b432059439ede11b505b3f4377655f9e8d1dff4f3dac0eb44000824e57d16a3",
                "stage_o_decision": "79fd718b7a0216f49d5a1df313ad19086e92c14ad800dfbe2215ec091930fa7f",
                "stage_q_decision": "3851687d6757c946cd64782fe5e83373d883066ed58493ca3bdaf44c1e548587",
            }
            if not isinstance(stage_r_frozen, Mapping) or set(stage_r_frozen) != set(
                expected_stage_r
            ):
                raise ValueError("Stage R frozen Stage O/Q provenance record set changed")
            for name, expected_hash in expected_stage_r.items():
                record = stage_r_frozen[name]
                if not isinstance(record, Mapping):
                    raise ValueError(f"Stage R frozen provenance {name} must be a mapping")
                _validate_checksum(root, record, f"Stage R {name}")
                if record["sha256"] != expected_hash:
                    raise ValueError(f"Stage R frozen provenance hash changed: {name}")

    manifest = json.loads(
        _resolved_path(root, str(provenance["manifest"]["path"])).read_text(encoding="utf-8")
    )
    if manifest.get("protocol_name") != EXPECTED_PROTOCOL_NAME:
        raise ValueError("Frozen manifest protocol changed")
    split = manifest.get("splits", {})
    if split.get("train", {}).get("snapshot_range") != [11, 91]:
        raise ValueError("Frozen manifest train indices changed")
    if split.get("validation", {}).get("snapshot_range") != [91, 111]:
        raise ValueError("Frozen manifest validation indices changed")

    contract = json.loads(
        _resolved_path(root, str(provenance["stage_e_loss_contract"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    if contract.get("schema_version") != EXPECTED_LOSS_CONTRACT or contract.get("status") != "passed":
        raise ValueError("Stage E loss contract is not the frozen passed contract")
    oracle = json.loads(
        _resolved_path(root, str(provenance["oracle_baseline"]["path"])).read_text(
            encoding="utf-8"
        )
    )
    semantics = oracle.get("reference_semantics", {}).get("reference_semantics_version")
    if semantics != EXPECTED_ORACLE_SEMANTICS:
        raise ValueError("Frozen oracle baseline semantics changed")


def _validate_protocol(root: Path, values: Mapping[str, Any]) -> None:
    protocol = PaperReduced100Protocol.from_yaml(
        root / "configs/data/paper_reduced100.yaml", project_root=root
    )
    protocol.validate_hdf5()
    if tuple(values["protocol"]["train_snapshots"]) != (
        protocol.train_snapshot_start,
        protocol.train_snapshot_end,
    ):
        raise ValueError("Experiment and frozen data protocol train ranges differ")
    if tuple(values["protocol"]["validation_snapshots"]) != (
        protocol.validation_snapshot_start,
        protocol.validation_snapshot_end,
    ):
        raise ValueError("Experiment and frozen data protocol validation ranges differ")


@dataclass(frozen=True)
class ResolvedPaperConfig:
    """Validated, serializable Stage F configuration."""

    values: Mapping[str, Any]
    config_path: Path
    project_root: Path

    @property
    def loss_name(self) -> str:
        return str(self.values["loss"]["name"])

    @property
    def mode(self) -> str:
        return "full" if self.loss_name == "PaperCompositeLoss" else "plain"

    @property
    def diagnostic_h1_mode(self) -> str | None:
        mode = self.values["loss"]["h1"].get("mode")
        return None if mode is None else str(mode)

    @property
    def architecture(self) -> str:
        return str(self.values["model"].get("architecture", "fno"))

    @property
    def stage_k(self) -> bool:
        return self.architecture == "localno_differential_3d"

    @property
    def uses_p3(self) -> bool:
        return self.values["preprocessing"].get("mode") == "stage_n_p3"

    @property
    def stage_r(self) -> bool:
        return self.values.get("prediction") == STAGE_R_PREDICTION_CONTRACT

    @property
    def stage_o(self) -> bool:
        return self.uses_p3 and not self.stage_r

    @property
    def prediction_mode(self) -> str:
        return str(self.values["model"].get("prediction_mode", "direct"))

    @property
    def prior_preprocessing_checksum(self) -> str:
        provenance = self.values["provenance"]
        record = (
        provenance["canonical_preprocessing"]
            if self.uses_p3
            else provenance["preprocessing"]
        )
        return str(record["sha256"])

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(dict(self.values))

    def resolve_path(self, configured: str) -> Path:
        return _resolved_path(self.project_root, configured)


def load_paper_experiment_config(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> ResolvedPaperConfig:
    """Load one Full/Plain config and reject any stale frozen dependency."""

    config_path = Path(path).resolve()
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else _infer_project_root(config_path)
    )
    values = _load_config_values(config_path, root)
    stage_r = values.get("prediction") == STAGE_R_PREDICTION_CONTRACT
    required_top = {
        "experiment_name",
        "output_dir",
        "checkpoint_prefix",
        *COMMON_CONTRACT,
        "loss",
        "provenance",
        "reproduction_metadata",
        *({"prediction"} if stage_r else set()),
    }
    if set(values) != required_top:
        missing = sorted(required_top - set(values))
        extra = sorted(set(values) - required_top)
        raise ValueError(f"Paper config top-level schema mismatch: missing={missing}, extra={extra}")
    model_values = values.get("model")
    if not isinstance(model_values, Mapping):
        raise ValueError("Resolved paper config model must be a mapping")
    if (
        int(model_values.get("n_dim", len(model_values.get("n_modes", [])))) == 3
        and bool(model_values.get("use_disco", False))
    ):
        raise ValueError("volumetric 3D DISCO is unavailable in pinned upstream")
    stage_k = model_values.get("architecture") == "localno_differential_3d"
    uses_p3 = values.get("preprocessing", {}).get("mode") == "stage_n_p3"
    stage_o = uses_p3 and not stage_r
    if uses_p3 and not stage_k:
        raise ValueError("Stage O P3 is authorized only with the Stage K LocalNO architecture")
    if stage_r and values.get("prediction") != STAGE_R_PREDICTION_CONTRACT:
        raise ValueError("Stage R normalized-residual prediction contract changed")
    expected_common = copy.deepcopy(COMMON_CONTRACT)
    if stage_k:
        expected_common["model"] = (
            STAGE_R_MODEL_CONTRACT if stage_r else STAGE_K_MODEL_CONTRACT
        )
    if uses_p3:
        expected_common["preprocessing"] = STAGE_O_P3_PREPROCESSING_CONTRACT
    if stage_r:
        expected_common["evaluation"] = {
            **expected_common["evaluation"],
            "rho_press_eval_clamp": False,
        }
    _require_subset(values, expected_common)
    expected_model = STAGE_R_MODEL_CONTRACT if stage_r else STAGE_K_MODEL_CONTRACT
    if stage_k and dict(model_values) != expected_model:
        raise ValueError("Stage K LocalNO model contract changed")
    if stage_r and values["loss"] != STAGE_R_LOSS_CONTRACT:
        raise ValueError("Stage R requires the frozen strict Plain L2 contract")
    loss_name = values.get("loss", {}).get("name")
    if loss_name == "PaperCompositeLoss":
        h1_mode = values["loss"]["h1"].get("mode")
        if h1_mode is None:
            if values["loss"] != FULL_LOSS_CONTRACT:
                raise ValueError(
                    "Full loss configuration changed from the Stage E contract"
                )
        else:
            expected_h1 = STAGE_I_H1_CONTRACTS.get(str(h1_mode))
            if expected_h1 is None:
                raise ValueError(f"Unsupported Stage I H1 mode {h1_mode!r}")
            expected_loss = copy.deepcopy(FULL_LOSS_CONTRACT)
            expected_loss["h1"] = expected_h1
            if values["loss"] != expected_loss:
                raise ValueError("Stage I H1 extension contract changed")
            full_values = yaml.safe_load(
                (root / "configs/paper_reduced100/full_fno_proxy.yaml").read_text(
                    encoding="utf-8"
                )
            )
            extension_reason = {
                DiagnosticH1Mode.NO_H1.value: STAGE_I_EXTENSION_REASON,
                DiagnosticH1Mode.UNIT_INDEX.value: (
                    STAGE_I_UNIT_INDEX_EXTENSION_REASON
                ),
                DiagnosticH1Mode.STORED_COORDINATE_VOLUME_PROXY.value: (
                    STAGE_I_STORED_COORDINATE_EXTENSION_REASON
                ),
            }[h1_mode]
            expected_reproduction = {
                **full_values["reproduction_metadata"],
                **STAGE_I_EXTENSION_METADATA,
                "extension_reason": extension_reason,
            }
            if values["reproduction_metadata"] != expected_reproduction:
                raise ValueError("Stage I reproduction metadata changed")
    elif loss_name == "PlainL2Loss":
        expected_plain = STAGE_R_LOSS_CONTRACT if stage_r else PLAIN_LOSS_CONTRACT
        if values["loss"] != expected_plain:
            raise ValueError("Plain loss configuration changed from the Stage E contract")
    else:
        raise ValueError(f"Unsupported paper loss {loss_name!r}")
    expected_reproduction = (
        STAGE_R_REPRODUCTION_METADATA
        if stage_r
        else STAGE_O_REPRODUCTION_METADATA
        if stage_o
        else STAGE_K_REPRODUCTION_METADATA
    )
    if stage_k and values["reproduction_metadata"] != expected_reproduction:
        raise ValueError(
            "Stage K/Stage O/Stage R reproduction classification or adaptation metadata changed"
        )
    if uses_p3:
        evaluation = values["evaluation"]
        required_evaluation = {
            "legacy_detector": True,
            "oracle_conditioned_gate": True,
            "oracle_conditioned_gate_version": "stage_m_v1",
        }
        _require_subset(evaluation, required_evaluation, "evaluation")
    _validate_frozen_provenance(root, values)
    _validate_protocol(root, values)
    return ResolvedPaperConfig(copy.deepcopy(values), config_path, root)


def build_paper_model(config: ResolvedPaperConfig) -> torch.nn.Module:
    model = config.values["model"]
    if config.stage_k:
        return build_model(
            "localno_differential_3d",
            in_channels=int(model["in_channels"]),
            out_channels=int(model["out_channels"]),
            n_modes=tuple(int(value) for value in model["n_modes"]),
            hidden_channels=int(model["hidden_channels"]),
            n_layers=int(model["n_layers"]),
            default_in_shape=tuple(int(value) for value in model["default_in_shape"]),
            positional_embedding=model["positional_embedding"],
            fin_diff_kernel_size=int(model["fin_diff_kernel_size"]),
            mix_derivatives=bool(model["mix_derivatives"]),
            conv_padding_mode=str(model["conv_padding_mode"]),
            use_channel_mlp=bool(model["use_channel_mlp"]),
            local_no_skip=model["local_no_skip"],
            norm=model["normalization"],
            enforce_hermitian_symmetry=bool(model["enforce_hermitian_symmetry"]),
        )
    return build_upstream_fno(
        UpstreamFNOConfig(
            in_channels=int(model["in_channels"]),
            out_channels=int(model["out_channels"]),
            n_modes=tuple(int(value) for value in model["n_modes"]),
            hidden_channels=int(model["hidden_channels"]),
            n_layers=int(model["n_layers"]),
            positional_embedding=model["positional_embedding"],
        )
    )


def _prior_expectations(config: ResolvedPaperConfig) -> dict[str, Any]:
    values = config.values
    return {
        "source_hdf5_checksum": values["provenance"]["dataset"]["sha256"],
        "preprocessing_stats_checksum": config.prior_preprocessing_checksum,
        "training_indices": tuple(range(*values["protocol"]["train_snapshots"])),
        "protocol_name": values["protocol"]["name"],
        "thermal_channel": values["thermal"]["channel"],
    }


def _load_stage_i_h1_geometry(
    config: ResolvedPaperConfig,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    dataset_path = config.resolve_path(str(config.values["protocol"]["dataset"]))
    with h5py.File(dataset_path, "r") as handle:
        coordinates = {
            name: torch.as_tensor(
                handle[f"coords/{name}"][...],
                dtype=torch.float64,
            )
            for name in ("phi", "theta", "r")
        }
    shell_record = config.values["provenance"]["artifacts"]["shells"]
    shell_payload = json.loads(
        config.resolve_path(str(shell_record["path"])).read_text(encoding="utf-8")
    )
    frozen_shells = shell_payload.get("shells")
    if not isinstance(frozen_shells, Mapping):
        raise ValueError("Frozen Stage I shell metadata is missing")
    shell_masks, generated = radial_shells_tensor(
        coordinates["r"].cpu().numpy(),
        int(coordinates["phi"].numel()),
        int(coordinates["theta"].numel()),
        n_shells=int(config.values["representation"]["shells"]["count"]),
    )
    if generated.as_dict() != dict(frozen_shells):
        raise ValueError("Stage I generated shells differ from the frozen artifact")
    return coordinates, shell_masks.to(dtype=torch.bool)


def build_paper_training_loss(
    config: ResolvedPaperConfig,
) -> PaperCompositeLoss | PlainL2Loss:
    """Instantiate only the loss objects enabled by the selected mode."""

    loss_config = config.values["loss"]
    if config.mode == "plain":
        flags = {
            "h1": bool(loss_config["h1"]["enabled"]),
            "roi": bool(loss_config["roi"]["enabled"]),
            "bounds": bool(loss_config["bounds"]["training_penalty_enabled"]),
            "envelope": bool(loss_config["envelope"]["enabled"]),
            "dissipation": bool(loss_config["dissipation"]["enabled"]),
        }
        # This validates the strict Plain contract without loading a Full prior.
        PlainL2Loss()(torch.zeros(1, 8, 1, 1, 1), torch.zeros(1, 8, 1, 1, 1), enabled_prior_flags=flags)
        return PlainL2Loss()

    artifacts = config.values["provenance"]["artifacts"]
    expected = _prior_expectations(config)
    radial = PaperRadialBaseline.load(
        config.resolve_path(str(artifacts["radial"]["path"])), **expected
    )
    bounds = PaperPhysicalBounds.load(
        config.resolve_path(str(artifacts["bounds"]["path"])), **expected
    )
    envelope = PaperResidualEnvelope.load(
        config.resolve_path(str(artifacts["envelope"]["path"])), **expected
    )
    roi = PaperVelocityROI.load(
        config.resolve_path(str(artifacts["roi"]["path"])), **expected
    )
    dissipation = PaperDissipativeReference.load(
        config.resolve_path(str(artifacts["dissipation"]["path"])), **expected
    )
    paper_loss_kwargs = {
        "bounds": bounds,
        "envelope": envelope,
        "roi": roi,
        "dissipation": dissipation,
        "radial_metadata": radial.metadata,
        "magnetic_weight": float(loss_config["base"]["magnetic_weight"]),
        "velocity_weight": float(loss_config["base"]["velocity_weight"]),
        "h1_weight": float(loss_config["h1"]["weight"]),
        "roi_kappa": float(loss_config["roi"]["kappa"]),
        "bounds_rho_low_weight": float(loss_config["bounds"]["rho_lower_weight"]),
        "bounds_press_low_weight": float(loss_config["bounds"]["press_lower_weight"]),
        "bounds_rho_high_weight": float(loss_config["bounds"]["upper_weights"]),
        "bounds_press_high_weight": float(loss_config["bounds"]["upper_weights"]),
        "envelope_rho_weight": float(loss_config["envelope"]["weight_rho"]),
        "envelope_press_weight": float(loss_config["envelope"]["weight_press"]),
        "dissipation_alpha": float(loss_config["dissipation"]["alpha"]),
        "gamma": float(config.values["preprocessing"]["gamma"]),
        "inverse_clamp_fraction": float(
            config.values["preprocessing"]["inverse_clamp_fraction"]
        ),
    }
    if config.diagnostic_h1_mode is not None:
        coordinates, shell_masks = _load_stage_i_h1_geometry(config)
        return DiagnosticPaperCompositeLoss(
            diagnostic_h1_mode=config.diagnostic_h1_mode,
            coordinates=coordinates,
            shell_masks=shell_masks,
            **paper_loss_kwargs,
        )
    return PaperCompositeLoss(
        **paper_loss_kwargs,
    )


def recursive_config_differences(
    full: Mapping[str, Any], plain: Mapping[str, Any], prefix: str = ""
) -> list[dict[str, Any]]:
    """Return deterministic leaf differences between two resolved mappings."""

    differences: list[dict[str, Any]] = []
    keys = sorted(set(full) | set(plain))
    missing = object()
    for key in keys:
        path = f"{prefix}.{key}" if prefix else str(key)
        full_value = full.get(key, missing)
        plain_value = plain.get(key, missing)
        if isinstance(full_value, Mapping) and isinstance(plain_value, Mapping):
            differences.extend(recursive_config_differences(full_value, plain_value, path))
        elif full_value != plain_value:
            differences.append(
                {
                    "path": path,
                    "full": "<missing>" if full_value is missing else full_value,
                    "plain": "<missing>" if plain_value is missing else plain_value,
                }
            )
    return differences


def difference_is_allowed(path: str) -> bool:
    return path in {"experiment_name", "output_dir", "checkpoint_prefix"} or path.startswith(
        "loss."
    )


def stage_i_difference_is_allowed(path: str) -> bool:
    """Return whether a resolved Full/Stage-I leaf difference is authorized."""

    return (
        path in {
            "experiment_name",
            "output_dir",
            "checkpoint_prefix",
            "reproduction_metadata.reproduction_level",
            "reproduction_metadata.paper_faithful_full",
            "reproduction_metadata.extension_reason",
        }
        or path.startswith("loss.h1.")
    )


def model_tensor_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def paired_config_audit(
    full_config: ResolvedPaperConfig, plain_config: ResolvedPaperConfig
) -> dict[str, Any]:
    differences = recursive_config_differences(
        full_config.as_dict(), plain_config.as_dict()
    )
    allowed = [item for item in differences if difference_is_allowed(item["path"])]
    unexpected = [item for item in differences if not difference_is_allowed(item["path"])]
    seed = int(full_config.values["runtime"]["seed"])
    torch.manual_seed(seed)
    full_model = build_paper_model(full_config)
    torch.manual_seed(seed)
    plain_model = build_paper_model(plain_config)
    full_hash = model_tensor_state_sha256(full_model)
    plain_hash = model_tensor_state_sha256(plain_model)
    full_count = sum(parameter.numel() for parameter in full_model.parameters())
    plain_count = sum(parameter.numel() for parameter in plain_model.parameters())
    return {
        "schema_version": "paper-stage-f-paired-config-audit-v1",
        "status": "passed" if not unexpected and full_hash == plain_hash else "failed",
        "allowed_difference_patterns": sorted(PAIR_ALLOWED_DIFFERENCES),
        "allowed_differences": allowed,
        "unexpected_differences": unexpected,
        "model_initialization": {
            "seed": seed,
            "full_parameter_count": full_count,
            "plain_parameter_count": plain_count,
            "counts_equal": full_count == plain_count,
            "full_tensor_state_sha256": full_hash,
            "plain_tensor_state_sha256": plain_hash,
            "state_identical": full_hash == plain_hash,
        },
        "checksums": copy.deepcopy(full_config.values["provenance"]),
    }


def write_paired_config_audit(payload: Mapping[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "paired_config_diff.json").write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    model = payload["model_initialization"]
    lines = [
        "# Stage F paired configuration audit",
        "",
        f"- Status: `{payload['status']}`",
        f"- Allowed leaf differences: `{len(payload['allowed_differences'])}`",
        f"- Unexpected differences: `{len(payload['unexpected_differences'])}`",
        f"- Parameter count: `{model['full_parameter_count']}`",
        f"- Same-seed state identical: `{str(model['state_identical']).lower()}`",
        f"- Initial tensor-state SHA256: `{model['full_tensor_state_sha256']}`",
        "",
        "## Allowed differences",
        "",
    ]
    lines.extend(f"- `{item['path']}`" for item in payload["allowed_differences"])
    lines.extend(["", "## Unexpected differences", ""])
    if payload["unexpected_differences"]:
        lines.extend(f"- `{item['path']}`" for item in payload["unexpected_differences"])
    else:
        lines.append("None.")
    (output / "paired_config_diff.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-config",
        type=Path,
        default=Path("configs/paper_reduced100/full_fno_proxy.yaml"),
    )
    parser.add_argument(
        "--plain-config",
        type=Path,
        default=Path("configs/paper_reduced100/plain_l2_fno.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper_reduced100/stage_f"),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    full = load_paper_experiment_config(args.full_config, project_root=root)
    plain = load_paper_experiment_config(args.plain_config, project_root=root)
    payload = paired_config_audit(full, plain)
    write_paired_config_audit(payload, args.output_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
