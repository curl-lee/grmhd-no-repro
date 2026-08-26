#!/usr/bin/env python
"""Run the authorized no-training Stage Q persistence-anchor audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import h5py
import numpy as np
import torch
import yaml
from neuralop.layers.differential_conv import FiniteDifferenceConvolution
from neuralop.layers.discrete_continuous_convolution import (
    EquidistantDiscreteContinuousConv2d,
)

from grmhd import CHANNELS
from grmhd.dataset import sha256_file
from grmhd.paper_config import build_paper_model, load_paper_experiment_config
from grmhd.paper_data_processor import PaperDataProcessor
from grmhd.paper_dissipation import PaperDissipativeReference, global_state_norm
from grmhd.paper_stage_g_evaluation import artifact_diagnostics, radial_shell_indices
from grmhd.paper_stage_l_attribution import (
    basic_field_metrics,
    radial_profile,
    spectrum_metrics,
)
from grmhd.paper_stage_m import radial_profile_vector, transport_metrics, variance_vector
from grmhd.paper_stage_n import directional_mode, radial_mode
from grmhd.paper_stage_o import load_frozen_p3
from grmhd.paper_stage_p import (
    TARGET_CHANNELS,
    decoder_derivative_channel,
    load_stage_o_model_only,
    ood_diagnostics,
    relative_l2,
)
from grmhd.paper_stage_q import (
    ALPHA_GRID,
    FROZEN_HISTORY,
    LOCAL_GAIN_EPSILONS,
    STAGE_Q_CLASSIFICATION,
    anchored_output,
    expected_transform_counts,
    model_state_sha256,
    readiness_predicates,
    residual_diagnostics,
    select_candidate_alpha,
    selection_artifact_sha256,
)


EXPECTED = {
    "upstream": "86a8bc7812a31b42c4f7895693cf4ac11521c066",
    "dataset": "cd3966bed1f617f49d775151cff87ae92eb2432a560bb82ae9355605e7aea70a",
    "p3_config": "41409e803cc0f54bede7cc73a84d9986e081bb07265c84dcac2fd64a44778809",
    "p3_statistics": "aea6d3f91e54865deeb9b2b1142c49a633ae8b67223075273e6c0b88bf541948",
    "stage_o_config": "d34b15cecf21be49faf9d98a5be55e63a7651224b9b625a5d6b153a45bbddcce",
    "stage_o_best": "756ee7f31a9780942aa0f35dce4e5584cbeaf04ed8e9777721e7765962f04e12",
    "stage_k_config": "8b229c5c2528d66bd4e11d0b0c4af16d00e906129386f8119af1670721ba30af",
    "stage_k_initial": "07db393f5c3e831f44bcc2d47b8132c3f9a0e72e2a258618080db487cc13cbb0",
    "pair_order": "5e1e361ea75e01ffd667822d8bfe870d70c5bc69570d52cdf806e637b2df2b52",
    "stage_m_gate": "a829a3760a71a006f54ae18b399c569a903898d27f3ef3a72236982fd033e5f7",
    "stage_p_run_manifest": "492cab5a097ee9def7d93ced99a78ae02fbeea0468bea5edbe4552f8e7aa4249",
    "stage_p_mechanism": "c190b8d59a32ec6d14dd3f02749cc8a4d64efbc294bbe7ab50a77c1334661047",
    "stage_p_decision": "6d4085cd5e2fa156f6dd54c3e0c623c5ab68302770ab530003f95581de07f2f8",
    "stage_p_projection": "f9da941bae7898ead47980951cc62a4e405d30bf006e752115d0ef47ab4c6179",
    "stage_p_reset": "5d02d11272f36701fb68465e183f40f81e7f66a8a441bafb4b65288415888540",
    "stage_p_driver": "120b75f13b4a2fae024665c0dfbf3ab898c31af1907c250d035654358cc9265e",
}


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if torch.is_tensor(value):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    if not fields:
        fields = ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(json_safe(value), sort_keys=True)
                        if isinstance(value, (dict, list, tuple))
                        else json_safe(value)
                    )
                    for key, value in row.items()
                }
            )


def git(*args: str, root: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def provenance_gate(
    root: Path,
    stage_o_config,
    qconfig: Mapping[str, Any],
    qconfig_path: Path,
) -> tuple[dict[str, Any], torch.nn.Module]:
    paths = {
        "dataset": root / "data_proc/grmhd_regrid_inner_r200_64.h5",
        "p3_config": root / "outputs/paper_reduced100/stage_n/prototypes/p3_combined_v1/resolved_config.json",
        "p3_statistics": root / "outputs/paper_reduced100/stage_n/prototypes/p3_combined_v1/normalizer.npz",
        "stage_o_config": root / "configs/paper_reduced100/stage_o_localno_p3_plain.yaml",
        "stage_o_best": root / "outputs/paper_reduced100/stage_o/localno_p3_plain/best_validation_l2/paper_state_dict.pt",
        "stage_k_config": root / "configs/paper_reduced100/stage_k_localno_differential_plain.yaml",
        "stage_k_initial": root / "outputs/paper_reduced100/stage_k/shared_localno_initial_state.pt",
        "pair_order": root / "outputs/paper_reduced100/stage_g/epoch_pair_order.json",
        "stage_m_gate": root / "outputs/paper_reduced100/stage_m/candidate_gate_config.json",
        "stage_p_run_manifest": root / "outputs/paper_reduced100/stage_p/run_manifest.json",
        "stage_p_mechanism": root / "outputs/paper_reduced100/stage_p/mechanism_evidence.json",
        "stage_p_decision": root / "outputs/paper_reduced100/stage_p/stage_p_decision.json",
        "stage_p_projection": root / "outputs/paper_reduced100/stage_p/envelope_projection.json",
        "stage_p_reset": root / "outputs/paper_reduced100/stage_p/channel_reset.json",
        "stage_p_driver": root / "outputs/paper_reduced100/stage_p/channel_driver_matrix.json",
    }
    observed = {name: sha256_file(path) for name, path in paths.items()}
    mismatches = {
        name: {"expected": EXPECTED[name], "observed": value}
        for name, value in observed.items()
        if value != EXPECTED[name]
    }
    upstream = git("-C", "external/neuraloperator", "rev-parse", "HEAD", root=root)
    upstream_status = git("-C", "external/neuraloperator", "status", "--short", root=root)
    branch = git("branch", "--show-current", root=root)
    if upstream != EXPECTED["upstream"] or upstream_status:
        mismatches["upstream"] = {
            "expected": EXPECTED["upstream"],
            "observed": upstream,
            "status": upstream_status,
        }
    if branch != "main":
        mismatches["branch"] = {
            "expected": "main",
            "observed": branch,
        }
    frozen = qconfig["frozen_provenance"]
    config_checks = {
        "dataset": frozen["dataset_sha256"],
        "p3_config": frozen["p3_config_sha256"],
        "p3_statistics": frozen["p3_statistics_sha256"],
        "stage_o_best": frozen["stage_o_best_checkpoint_sha256"],
        "stage_k_config": frozen["stage_k_architecture_sha256"],
        "stage_k_initial": frozen["stage_k_initial_state_sha256"],
        "pair_order": frozen["pair_order_sha256"],
        "stage_m_gate": frozen["stage_m_gate_sha256"],
        "stage_p_run_manifest": frozen["stage_p_run_manifest_sha256"],
        "stage_p_mechanism": frozen["stage_p_mechanism_sha256"],
        "stage_p_decision": frozen["stage_p_decision_sha256"],
    }
    for name, expected in config_checks.items():
        if expected != observed[name]:
            mismatches[f"qconfig:{name}"] = {
                "expected": expected,
                "observed": observed[name],
            }
    stage_p_mechanism = json.loads(paths["stage_p_mechanism"].read_text())
    stage_p_decision = json.loads(paths["stage_p_decision"].read_text())
    expected_mechanisms = {
        "M1": "supported",
        "M2": "supported",
        "M3": "weakly_supported",
        "M4": "weakly_supported",
        "M5": "supported",
    }
    actual_mechanisms = {
        key: payload["status"]
        for key, payload in stage_p_mechanism["mechanisms"].items()
    }
    if actual_mechanisms != expected_mechanisms:
        mismatches["stage_p_mechanisms"] = {
            "expected": expected_mechanisms,
            "observed": actual_mechanisms,
        }
    if stage_p_decision["decision"] != "6. MIXED_CLOSED_LOOP_FAILURE":
        mismatches["stage_p_final"] = stage_p_decision["decision"]
    if mismatches:
        raise RuntimeError(f"Stage Q provenance gate failed: {mismatches}")
    model, epoch, checkpoint_metadata = load_stage_o_model_only(
        paths["stage_o_best"].parent,
        config=stage_o_config,
        model=build_paper_model(stage_o_config),
        expected_config_checksum=observed["stage_o_config"],
    )
    differential_count = sum(
        isinstance(module, FiniteDifferenceConvolution) for module in model.modules()
    )
    disco_count = sum(
        isinstance(module, EquidistantDiscreteContinuousConv2d)
        for module in model.modules()
    )
    if epoch != 23 or differential_count <= 0 or disco_count != 0:
        raise RuntimeError("Stage Q frozen model architecture/reload gate failed")
    return {
        "schema_version": "paper-stage-q-run-manifest-v1",
        "classification": STAGE_Q_CLASSIFICATION,
        "project": {
            "branch": branch,
            "commit": git("rev-parse", "HEAD", root=root),
            "stage_p_base": "56d3959",
        },
        "upstream_commit": upstream,
        "provenance_passed": True,
        "checksums": observed,
        "stage_q_config_sha256": sha256_file(qconfig_path),
        "stage_p_mechanisms": actual_mechanisms,
        "stage_p_decision": stage_p_decision["decision"],
        "stage_p_summary": stage_p_mechanism["summary_values"],
        "checkpoint_reload": {
            "model_only": True,
            "strict": True,
            "epoch": epoch,
            "optimizer": False,
            "scheduler": False,
            "metadata": checkpoint_metadata,
        },
        "architecture": {
            "name": stage_o_config.architecture,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "differential_modules": differential_count,
            "disco_modules": disco_count,
            "input_channels": 16,
            "output_channels": 8,
        },
        "frozen_history": FROZEN_HISTORY,
    }, model


def prepare_structure(
    input_field: np.ndarray, target_field: np.ndarray, shell_index: np.ndarray
) -> dict[str, Any]:
    target_basic = basic_field_metrics(target_field)
    target_shell = variance_vector(target_field, shell_index)
    target_radial_profile = radial_profile(target_field)
    return {
        "input_shell": variance_vector(input_field, shell_index),
        "input_radial": radial_profile_vector(input_field),
        "target_basic": target_basic,
        "target_shell": target_shell,
        "target_radial_vector": radial_profile_vector(target_field),
        "target_radial_variance": target_radial_profile["variance"],
        "target_high_k": spectrum_metrics(
            target_field, axis="combined", demean=True
        )["high_k_energy"],
    }


def evaluate_structure(
    prepared: Mapping[str, Any], prediction: np.ndarray, shell_index: np.ndarray
) -> dict[str, Any]:
    model_basic = basic_field_metrics(prediction)
    model_shell = variance_vector(prediction, shell_index)
    model_radial = radial_profile(prediction)["variance"]
    model_high = spectrum_metrics(prediction, axis="combined", demean=True)[
        "high_k_energy"
    ]
    target_basic = prepared["target_basic"]
    target_shell = prepared["target_shell"]
    ratios = {
        "global_variance": model_basic["variance"]
        / max(target_basic["variance"], 1.0e-30),
        "shell_radial_variance": min(
            float(np.median(model_shell / np.maximum(target_shell, 1.0e-30))),
            model_radial / max(prepared["target_radial_variance"], 1.0e-30),
        ),
        "dynamic_span": model_basic["dynamic_span_q99_q01"]
        / max(target_basic["dynamic_span_q99_q01"], 1.0e-30),
        "high_k_energy": model_high / max(prepared["target_high_k"], 1.0e-30),
    }
    severe = {name: value < 0.5 for name, value in ratios.items()}
    shell = transport_metrics(
        prepared["input_shell"],
        target_shell,
        model_shell,
        epsilon=1.0e-30,
        sign_zero_tolerance=1.0e-12,
    )
    radial = transport_metrics(
        prepared["input_radial"],
        prepared["target_radial_vector"],
        radial_profile_vector(prediction),
        epsilon=1.0e-30,
        sign_zero_tolerance=1.0e-12,
    )
    skills = (shell["persistence_relative_skill"], radial["persistence_relative_skill"])
    signs = (shell["signed_transport_agreement"], radial["signed_transport_agreement"])
    defined = all(value is not None and np.isfinite(value) for value in (*skills, *signs))
    gate3 = bool(
        defined
        and all(float(value) >= 0.0 for value in skills)
        and any(float(value) > 0.0 for value in skills)
        and all(float(value) >= 0.5 for value in signs)
    )
    return {
        "retention": ratios,
        "severe": severe,
        "gate_2_failed": sum(severe.values()) >= 2,
        "gate_3_passed": gate3,
        "shell_skill": shell["persistence_relative_skill"],
        "shell_sign_agreement": shell["signed_transport_agreement"],
        "radial_skill": radial["persistence_relative_skill"],
        "radial_sign_agreement": radial["signed_transport_agreement"],
    }


def contract_definition_output(
    *,
    qconfig: Mapping[str, Any],
    config_sha256: str,
    output: Path,
) -> dict[str, Any]:
    payload = {
        "schema_version": "paper-stage-q-anchor-contract-v1",
        "classification": STAGE_Q_CLASSIFICATION,
        "direct_output": "y_direct = F_theta(concat(z_t, fixed_shells))",
        "residual": "r_theta(z_t) = y_direct - z_t",
        "anchored_output": "y_alpha = z_t + alpha * r_theta(z_t)",
        "affine_equivalent": "y_alpha = (1-alpha)*z_t + alpha*y_direct",
        "alpha_grid": list(ALPHA_GRID),
        "candidate_alphas": qconfig["contract"]["candidate_alphas"],
        "endpoint_semantics": {
            "0.0": "exact_persistence_control",
            "1.0": "exact_stage_o_direct_control",
        },
        "scalar_alpha_only": True,
        "fixed_shells": True,
        "no_clipping_or_transform_in_contract": True,
        "config_sha256": config_sha256,
        "rules_frozen_before_alpha_results": True,
    }
    write_json(output / "contract_definition.json", payload)
    (output / "contract_definition.md").write_text(
        "# Stage Q persistence-anchored output contract\n\n"
        "`y_direct = F_theta(concat(z_t, fixed_shells))`\n\n"
        "`r_theta = y_direct - z_t`\n\n"
        "`y_alpha = z_t + alpha * r_theta = (1-alpha)z_t + alpha*y_direct`\n\n"
        "The scalar grid is `0/0.125/0.25/0.5/1`. Endpoints are controls; only "
        "intermediate values may become future-smoke candidates. The contract contains no "
        "decode, encode, clamp, clipping, repair, or shell mutation.\n",
        encoding="utf-8",
    )
    return payload


def contract_identity_output(
    *,
    model: torch.nn.Module,
    p3,
    processor: PaperDataProcessor,
    h5_path: Path,
    qconfig: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    with h5py.File(h5_path, "r") as handle:
        raw = torch.from_numpy(np.asarray(handle["snapshots"][11], dtype=np.float32))
    z = p3.encode_tensor(raw.unsqueeze(0).to(processor.device), channel_axis=1)
    shells = processor.shells.expand_as(z)
    before = model_state_sha256(model)
    shell_before = hashlib.sha256(shells.detach().cpu().numpy().tobytes()).hexdigest()
    with torch.no_grad():
        direct = model(x=torch.cat((z, shells), dim=1))
    endpoint0 = anchored_output(z, direct, 0.0)
    endpoint1 = anchored_output(z, direct, 1.0)
    affine_errors = {}
    affine_checks = {}
    residual_errors = {}
    residual_checks = {}
    residual_norms = {}
    tolerance = qconfig["contract"]["identity_tolerance"]
    ulp_factor = float(tolerance["ulp_factor"])
    unit_floor = float(tolerance["unit_floor"])
    dtype_epsilon = float(torch.finfo(z.dtype).eps)

    def within_dtype_roundoff(
        left: torch.Tensor, right: torch.Tensor, *operation_scale: torch.Tensor
    ) -> bool:
        references = operation_scale or (left, right)
        scale = torch.full_like(left, unit_floor)
        for reference in references:
            scale = scale + torch.abs(reference)
        allowed = ulp_factor * dtype_epsilon * scale
        return bool(torch.all(torch.abs(left - right) <= allowed))
    for alpha in ALPHA_GRID:
        anchored = anchored_output(z, direct, alpha)
        affine = (1.0 - alpha) * z + alpha * direct
        affine_errors[str(alpha)] = float(torch.max(torch.abs(anchored - affine)).cpu())
        affine_checks[str(alpha)] = within_dtype_roundoff(anchored, affine)
        residual_expected = alpha * (direct - z)
        residual_errors[str(alpha)] = float(
            torch.max(torch.abs((anchored - z) - residual_expected)).cpu()
        )
        residual_checks[str(alpha)] = within_dtype_roundoff(
            anchored - z, residual_expected, anchored, z, residual_expected
        )
        residual_norms[str(alpha)] = float(
            torch.linalg.vector_norm((anchored - z).to(torch.float64)).cpu()
        )
    fixed = torch.randn_like(z)
    fixed_errors = {
        str(alpha): float(torch.max(torch.abs(anchored_output(fixed, fixed, alpha) - fixed)).cpu())
        for alpha in ALPHA_GRID
    }
    shell_after = hashlib.sha256(shells.detach().cpu().numpy().tobytes()).hexdigest()
    source = inspect.getsource(anchored_output)
    forbidden = ("clip(", "clamp(", "nan_to_num", "decoder", "encoder", "bounds")
    results = {
        "Q1_endpoint_identity": bool(torch.equal(endpoint0, z) and torch.equal(endpoint1, direct)),
        "Q2_affine_identity": all(affine_checks.values()),
        "Q3_fixed_point_preservation": max(fixed_errors.values()) == 0.0,
        "Q4_residual_scaling": all(residual_checks.values()),
        "Q5_pairwise_alpha_ordering": all(
            left <= right
            for left, right in zip(
                [residual_norms[str(alpha)] for alpha in ALPHA_GRID],
                [residual_norms[str(alpha)] for alpha in ALPHA_GRID][1:],
            )
        ),
        "Q6_shell_invariance": shell_before == shell_after,
        "Q7_no_hidden_clamp_or_transform": not any(token in source for token in forbidden),
        "Q8_no_parameter_or_buffer_mutation": before == model_state_sha256(model),
    }
    payload = {
        "schema_version": "paper-stage-q-contract-identity-tests-v1",
        "results": results,
        "passed": all(results.values()),
        "affine_max_abs_errors": affine_errors,
        "residual_scaling_max_abs_errors": residual_errors,
        "fixed_point_max_abs_errors": fixed_errors,
        "frozen_float_tolerance": {
            "mode": tolerance["mode"],
            "ulp_factor": ulp_factor,
            "unit_floor": unit_floor,
            "dtype": str(z.dtype),
            "dtype_epsilon": dtype_epsilon,
        },
        "residual_norms": residual_norms,
        "parameter_buffer_sha256_before": before,
        "parameter_buffer_sha256_after": model_state_sha256(model),
        "shell_sha256_before": shell_before,
        "shell_sha256_after": shell_after,
    }
    write_json(output / "contract_identity_tests.json", payload)
    if not payload["passed"]:
        raise RuntimeError(f"Stage Q contract identity gate failed: {results}")
    return payload


def split_teacher_forced_audit(
    *,
    split: str,
    source_indices: Sequence[int],
    alphas: Sequence[float],
    config,
    model: torch.nn.Module,
    processor: PaperDataProcessor,
    p3,
    envelope: Mapping[str, Any],
    shell_index: np.ndarray,
    dissipation: PaperDissipativeReference,
) -> dict[str, Any]:
    if split not in {"train", "validation"}:
        raise ValueError("Stage Q split must be train or validation")
    if split == "train" and tuple(source_indices) != tuple(range(11, 90)):
        raise ValueError("Stage Q calibration must use exactly train transitions 11->12..89->90")
    h5_path = config.resolve_path(config.values["protocol"]["dataset"])
    shells = processor.shells.expand(1, -1, -1, -1, -1).to(processor.device)
    rows: list[dict[str, Any]] = []
    accumulators = {
        alpha: {
            "channel_errors": {name: [] for name in CHANNELS},
            "channel_numerators": {name: 0.0 for name in CHANNELS},
            "channel_denominators": {name: 0.0 for name in CHANNELS},
            "residual_cosines": {name: [] for name in CHANNELS},
            "residual_ratios": [],
            "residual_norms": [],
            "true_residual_norms": [],
            "q_ood": [],
            "range_ood": [],
            "rout": 0,
            "gate2": 0,
            "gate3_pass": 0,
            "shell_skill": [],
            "radial_skill": [],
            "legacy_ripple": 0,
            "legacy_stripe": 0,
            "legacy_collapse": 0,
            "all_finite": True,
            "positive": True,
            "nonfinite_derivatives": 0,
            "max_persistence_difference": 0.0,
            "max_persistence_scale": 0.0,
            "global_numerator": 0.0,
            "global_denominator": 0.0,
            "prediction_decodes": 0,
        }
        for alpha in alphas
    }
    with h5py.File(h5_path, "r") as handle, torch.no_grad():
        snapshots = handle["snapshots"]
        for source_index in source_indices:
            target_index = int(source_index) + 1
            raw_input = torch.from_numpy(
                np.asarray(snapshots[source_index], dtype=np.float32)
            ).unsqueeze(0).to(processor.device)
            raw_target = torch.from_numpy(
                np.asarray(snapshots[target_index], dtype=np.float32)
            ).unsqueeze(0).to(processor.device)
            z_input = p3.encode_tensor(raw_input, channel_axis=1)
            z_target = p3.encode_tensor(raw_target, channel_axis=1)
            oracle_target = p3.decode_tensor(z_target, channel_axis=1)
            direct = model(x=torch.cat((z_input, shells), dim=1))
            true_global = residual_diagnostics(z_input, z_target, z_target)
            prepared = {
                name: prepare_structure(
                    raw_input[0, CHANNELS.index(name)].cpu().numpy(),
                    oracle_target[0, CHANNELS.index(name)].cpu().numpy(),
                    shell_index,
                )
                for name in TARGET_CHANNELS
            }
            for alpha in alphas:
                anchored = anchored_output(z_input, direct, alpha)
                decoded = processor.decode_prediction(
                    anchored, apply_evaluation_clamp=True
                )
                physical = decoded.physical_prediction
                acc = accumulators[alpha]
                acc["prediction_decodes"] += 1
                finite = bool(
                    torch.isfinite(anchored).all()
                    and torch.isfinite(physical).all()
                )
                positive = bool(
                    torch.all(physical[:, CHANNELS.index("rho")] > 0)
                    and torch.all(physical[:, CHANNELS.index("press")] > 0)
                )
                acc["all_finite"] = acc["all_finite"] and finite
                acc["positive"] = acc["positive"] and positive
                state_norm = float(global_state_norm(anchored)[0].cpu())
                rout_failure = state_norm > dissipation.rout
                acc["rout"] += int(rout_failure)
                global_residual = residual_diagnostics(z_input, anchored, z_target)
                acc["residual_ratios"].append(
                    global_residual["residual_over_true_residual"]
                )
                acc["residual_norms"].append(global_residual["residual_norm"])
                acc["true_residual_norms"].append(
                    true_global["true_residual_norm"]
                )
                acc["max_persistence_difference"] = max(
                    acc["max_persistence_difference"],
                    float(torch.max(torch.abs(anchored - z_input)).cpu()),
                )
                acc["max_persistence_scale"] = max(
                    acc["max_persistence_scale"],
                    float(torch.max(torch.abs(z_input)).cpu()),
                )
                delta = (anchored - z_target).to(torch.float64)
                acc["global_numerator"] += float(torch.sum(delta.square()).cpu())
                acc["global_denominator"] += float(
                    torch.sum(z_target.to(torch.float64).square()).cpu()
                )
                artifacts = artifact_diagnostics(
                    physical,
                    raw_target,
                    raw_target,
                    reference_kind="ground_truth",
                )
                acc["legacy_ripple"] += sum(
                    "possible_high_frequency_ripple" in flag
                    for flag in artifacts["flags"]
                )
                acc["legacy_stripe"] += sum(
                    "possible_stripe_anisotropy" in flag for flag in artifacts["flags"]
                )
                acc["legacy_collapse"] += sum(
                    "possible_field_collapse" in flag for flag in artifacts["flags"]
                )
                for channel, name in enumerate(CHANNELS):
                    y_np = anchored[0, channel].cpu().numpy()
                    target_np = z_target[0, channel].cpu().numpy()
                    physical_np = physical[0, channel].cpu().numpy()
                    ood = ood_diagnostics(
                        y_np, envelope["channels"][name]["normalized"]
                    )
                    derivative = decoder_derivative_channel(
                        p3, y_np, channel=channel, source_dtype=y_np.dtype
                    )
                    derivative_finite = bool(np.isfinite(derivative).all())
                    acc["nonfinite_derivatives"] += int(not derivative_finite)
                    error = relative_l2(y_np, target_np)
                    channel_delta = y_np.astype(np.float64) - target_np.astype(
                        np.float64
                    )
                    acc["channel_numerators"][name] += float(
                        np.sum(np.square(channel_delta))
                    )
                    acc["channel_denominators"][name] += float(
                        np.sum(np.square(target_np.astype(np.float64)))
                    )
                    residual = residual_diagnostics(
                        z_input[:, channel : channel + 1],
                        anchored[:, channel : channel + 1],
                        z_target[:, channel : channel + 1],
                    )
                    acc["channel_errors"][name].append(error)
                    acc["residual_cosines"][name].append(
                        residual["residual_cosine"]
                    )
                    acc["q_ood"].append(
                        ood["fraction_outside_train_q001_q999"]
                    )
                    acc["range_ood"].append(
                        ood["fraction_outside_train_min_max"]
                    )
                    structure = None
                    if name in TARGET_CHANNELS:
                        structure = evaluate_structure(
                            prepared[name], physical_np, shell_index
                        )
                        acc["gate2"] += int(structure["gate_2_failed"])
                        acc["gate3_pass"] += int(structure["gate_3_passed"])
                        if structure["shell_skill"] is not None:
                            acc["shell_skill"].append(structure["shell_skill"])
                        if structure["radial_skill"] is not None:
                            acc["radial_skill"].append(structure["radial_skill"])
                    inverse_policy = p3.spec.channel_policies[channel]
                    inverse_limit = (
                        None
                        if inverse_policy == "no_softclip"
                        else (
                            p3.minimal_limit
                            if inverse_policy == "minimal_inverse_clamp"
                            else p3.inverse_clamp_fraction * p3.gamma
                        )
                    )
                    inverse_occupancy = (
                        0.0
                        if inverse_limit is None
                        else float(np.mean(np.abs(y_np) > inverse_limit))
                    )
                    row = {
                        "split": split,
                        "source_index": int(source_index),
                        "target_index": target_index,
                        "alpha": alpha,
                        "channel": name,
                        "normalized_relative_l2": error,
                        **residual,
                        **ood,
                        "normalized_state_norm": state_norm,
                        "rout_failure": rout_failure,
                        "physical_finite": finite,
                        "rho_press_positive": positive,
                        "physical_minimum": float(np.min(physical_np)),
                        "physical_maximum": float(np.max(physical_np)),
                        "physical_relative_l2_to_raw": relative_l2(
                            physical_np, raw_target[0, channel].cpu().numpy()
                        ),
                        "physical_relative_l2_to_p3_oracle": relative_l2(
                            physical_np, oracle_target[0, channel].cpu().numpy()
                        ),
                        "decoder_derivative_finite": derivative_finite,
                        "decoder_derivative_median": float(np.median(derivative)),
                        "decoder_derivative_maximum": float(np.max(derivative)),
                        "derivative_over_train_q999": float(
                            np.max(derivative)
                            / max(
                                envelope["channels"][name]["decoder_derivative"]["q999"],
                                1.0e-30,
                            )
                        ),
                        "inverse_clamp_policy": inverse_policy,
                        "inverse_clamp_occupancy": inverse_occupancy,
                        "evaluation_clamp_occupancy": float(
                            decoded.clamp_mask[:, channel].float().mean().cpu()
                        ),
                        "gate_0_passed": finite and positive and not rout_failure,
                        "gate_1_floor_reference": "frozen_stage_p_p3_floor",
                        "gate_2_failed": None
                        if structure is None
                        else structure["gate_2_failed"],
                        "gate_3_passed": None
                        if structure is None
                        else structure["gate_3_passed"],
                        "global_variance_retention": None
                        if structure is None
                        else structure["retention"]["global_variance"],
                        "shell_radial_variance_retention": None
                        if structure is None
                        else structure["retention"]["shell_radial_variance"],
                        "high_k_retention": None
                        if structure is None
                        else structure["retention"]["high_k_energy"],
                        "shell_transport_skill": None
                        if structure is None
                        else structure["shell_skill"],
                        "radial_transport_skill": None
                        if structure is None
                        else structure["radial_skill"],
                        "stage_p_primary_driver": name in TARGET_CHANNELS,
                        "legacy_flags": ";".join(
                            flag
                            for flag in artifacts["flags"]
                            if flag.startswith(f"{name}:")
                        ),
                    }
                    rows.append(row)
            del raw_input, raw_target, z_input, z_target, oracle_target, direct
    summaries = {}
    for alpha in alphas:
        acc = accumulators[alpha]
        per_channel = {
            name: math.sqrt(
                acc["channel_numerators"][name]
                / max(acc["channel_denominators"][name], 1.0e-30)
            )
            for name in CHANNELS
        }
        atol = 1.0e-7
        rtol = 1.0e-6
        persistence_equivalent = acc["max_persistence_difference"] <= (
            atol + rtol * acc["max_persistence_scale"]
        )
        summaries[alpha] = {
            "alpha": alpha,
            "pairs": len(source_indices),
            "source_indices": list(source_indices),
            "target_indices": [int(index) + 1 for index in source_indices],
            "per_channel_relative_l2": per_channel,
            "normalized_average": float(np.mean(list(per_channel.values()))),
            "normalized_global": math.sqrt(
                acc["global_numerator"] / max(acc["global_denominator"], 1.0e-30)
            ),
            "normalized_q_ood_fraction_median": float(np.median(acc["q_ood"])),
            "normalized_minmax_exceedance_median": float(
                np.median(acc["range_ood"])
            ),
            "rout_failure_count": acc["rout"],
            "residual_norm_median": float(np.median(acc["residual_norms"])),
            "true_residual_norm_median": float(
                np.median(acc["true_residual_norms"])
            ),
            "residual_over_true_residual_median": float(
                np.median(acc["residual_norms"])
                / max(np.median(acc["true_residual_norms"]), 1.0e-30)
            ),
            "residual_cosine_channel_medians": {
                name: float(np.median(values))
                for name, values in acc["residual_cosines"].items()
            },
            "persistence_max_abs_difference": acc["max_persistence_difference"],
            "persistence_equivalent": persistence_equivalent,
            "gate_2_severe_count": acc["gate2"],
            "gate_3_pass_count": acc["gate3_pass"],
            "shell_transport_skill_median": float(np.median(acc["shell_skill"])),
            "radial_transport_skill_median": float(np.median(acc["radial_skill"])),
            "legacy_ripple_count": acc["legacy_ripple"],
            "legacy_stripe_count": acc["legacy_stripe"],
            "legacy_collapse_count": acc["legacy_collapse"],
            "engineering": {
                "all_finite": acc["all_finite"],
                "rho_press_positive": acc["positive"],
                "transform_counter_errors": int(
                    acc["prediction_decodes"] != len(source_indices)
                ),
                "new_nonfinite_decoder_derivatives": acc["nonfinite_derivatives"],
                "prediction_decodes": acc["prediction_decodes"],
            },
        }
    return {
        "schema_version": f"paper-stage-q-{split}-alpha-audit-v1",
        "split": split,
        "train_only_selection_source": split == "train",
        "validation_confirmatory_only": split == "validation",
        "alphas": list(alphas),
        "rows": rows,
        "summaries": {str(alpha): value for alpha, value in summaries.items()},
        "_summaries_by_alpha": summaries,
    }


def write_train_and_selection(
    *,
    train: Mapping[str, Any],
    qconfig: Mapping[str, Any],
    qconfig_sha256: str,
    pair_order_sha256: str,
    data_sha256: str,
    p3_statistics_sha256: str,
    output: Path,
) -> tuple[dict[str, Any], dict[float, Any]]:
    summaries = train["_summaries_by_alpha"]
    persistence = summaries[0.0]
    direct = summaries[1.0]
    readiness = {
        alpha: readiness_predicates(
            summaries[alpha],
            persistence=persistence,
            direct=direct,
            config=qconfig,
        )
        for alpha in (0.125, 0.25, 0.5)
    }
    selection = select_candidate_alpha(
        summaries, readiness, config=qconfig
    )
    selection.update(
        {
            "schema_version": "paper-stage-q-alpha-selection-v1",
            "selection_split": "train_only_snapshots_11_90",
            "train_pair_count": 79,
            "train_source_indices": list(range(11, 90)),
            "validation_indices_read_before_selection": [],
            "alpha_grid": list(ALPHA_GRID),
            "readiness": {str(alpha): value for alpha, value in readiness.items()},
            "stage_q_config_sha256": qconfig_sha256,
            "selection_rule_sha256": qconfig_sha256,
            "pair_order_sha256": pair_order_sha256,
            "data_sha256": data_sha256,
            "p3_statistics_sha256": p3_statistics_sha256,
        }
    )
    selection["selection_artifact_sha256"] = selection_artifact_sha256(selection)
    train_payload = {key: value for key, value in train.items() if not key.startswith("_")}
    train_payload["readiness"] = {str(alpha): value for alpha, value in readiness.items()}
    train_payload["candidate_alpha"] = selection["candidate_alpha"]
    write_json(output / "train_alpha_sweep.json", train_payload)
    write_csv(output / "train_alpha_sweep.csv", train["rows"])
    lines = [
        "# Stage Q train-only alpha sweep",
        "",
        "Selection used exactly train transitions `11->12` through `89->90`; validation was not read.",
        "",
        "| alpha | normalized average | / persistence | q-tail OOD median | Rout failures | Gate 2 count | Gate 3 passes | readiness |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for alpha in ALPHA_GRID:
        summary = summaries[alpha]
        status = "control" if alpha in (0.0, 1.0) else str(readiness[alpha]["passed"])
        lines.append(
            f"| {alpha} | {summary['normalized_average']:.8g} | "
            f"{summary['normalized_average']/max(persistence['normalized_average'],1e-30):.8g} | "
            f"{summary['normalized_q_ood_fraction_median']:.8g} | "
            f"{summary['rout_failure_count']} | {summary['gate_2_severe_count']} | "
            f"{summary['gate_3_pass_count']} | `{status}` |"
        )
    lines.extend(
        [
            "",
            f"Frozen candidate: `{selection['candidate_alpha']}`.",
            f"Reason: `{selection['reason']}`.",
            "",
        ]
    )
    (output / "train_alpha_sweep.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(output / "alpha_selection.json", selection)
    (output / "alpha_selection.md").write_text(
        "# Stage Q frozen alpha selection\n\n"
        f"- Candidate alpha: `{selection['candidate_alpha']}`.\n"
        f"- Reason: `{selection['reason']}`.\n"
        "- Selection source: train snapshots `11..90` only.\n"
        "- Validation used for selection: `false`.\n"
        f"- Selection artifact SHA256: `{selection['selection_artifact_sha256']}`.\n",
        encoding="utf-8",
    )
    return selection, readiness


def write_validation(
    validation: Mapping[str, Any], *, candidate_alpha: float | None, output: Path
) -> None:
    payload = {key: value for key, value in validation.items() if not key.startswith("_")}
    payload.update(
        {
            "candidate_alpha_frozen_before_validation": candidate_alpha,
            "validation_is_confirmatory": True,
            "validation_used_for_alpha_selection": False,
            "candidate_metrics_present": candidate_alpha is not None,
        }
    )
    write_json(output / "validation_one_step.json", payload)
    write_csv(output / "validation_one_step.csv", validation["rows"])


def overall_ood(
    normalized: torch.Tensor, envelope: Mapping[str, Any]
) -> tuple[float, float]:
    q_values = []
    range_values = []
    for channel, name in enumerate(CHANNELS):
        metrics = ood_diagnostics(
            normalized[0, channel].detach().cpu().numpy(),
            envelope["channels"][name]["normalized"],
        )
        q_values.append(metrics["fraction_outside_train_q001_q999"])
        range_values.append(metrics["fraction_outside_train_min_max"])
    return float(np.mean(q_values)), float(np.mean(range_values))


def collect_fixed_states(
    *,
    config,
    model: torch.nn.Module,
    processor: PaperDataProcessor,
    p3,
) -> dict[str, dict[str, Any]]:
    h5_path = config.resolve_path(config.values["protocol"]["dataset"])
    device = processor.device
    shells = processor.shells.expand(1, -1, -1, -1, -1).to(device)
    states: dict[str, dict[str, Any]] = {}
    with h5py.File(h5_path, "r") as handle:
        snapshots = handle["snapshots"]
        for label, index, target_index in (
            ("train_early", 11, 12),
            ("train_middle", 50, 51),
            ("train_late", 90, None),
            ("validation_snapshot91", 91, 92),
        ):
            raw = torch.from_numpy(np.asarray(snapshots[index], dtype=np.float32)).unsqueeze(0)
            target = (
                None
                if target_index is None
                else torch.from_numpy(
                    np.asarray(snapshots[target_index], dtype=np.float32)
                ).unsqueeze(0)
            )
            states[label] = {
                "normalized": p3.encode_tensor(raw.to(device), channel_axis=1).detach().cpu(),
                "physical_input": raw,
                "physical_target": target,
                "source_index": index,
                "target_index": target_index,
                "kind": "oracle_state",
            }
        initial = torch.from_numpy(
            np.asarray(snapshots[91], dtype=np.float32)
        ).unsqueeze(0).to(device)
        current = p3.encode_tensor(initial, channel_axis=1)
        with torch.no_grad():
            for step in range(1, 20):
                if step in (1, 5, 10, 19):
                    target_index = 91 + step
                    states[f"stage_o_actual_step{step}_input"] = {
                        "normalized": current.detach().cpu(),
                        "physical_input": p3.decode_tensor(
                            current, channel_axis=1
                        ).detach().cpu(),
                        "physical_target": torch.from_numpy(
                            np.asarray(snapshots[target_index], dtype=np.float32)
                        ).unsqueeze(0),
                        "source_index": None,
                        "target_index": target_index,
                        "kind": "replayed_stage_o_actual_input",
                    }
                direct = model(x=torch.cat((current, shells), dim=1))
                decoded = processor.decode_prediction(
                    direct, apply_evaluation_clamp=True
                ).physical_prediction
                current = p3.encode_tensor(decoded, channel_axis=1)
    return states


def fixed_state_response(
    *,
    states: Mapping[str, Mapping[str, Any]],
    alphas: Sequence[float],
    model: torch.nn.Module,
    processor: PaperDataProcessor,
    p3,
    envelope: Mapping[str, Any],
    shell_index: np.ndarray,
    dissipation: PaperDissipativeReference,
    output: Path,
) -> dict[str, Any]:
    shells = processor.shells.expand(1, -1, -1, -1, -1).to(processor.device)
    rows = []
    with torch.no_grad():
        for state_name, payload in states.items():
            state = payload["normalized"].to(processor.device)
            direct = model(x=torch.cat((state, shells), dim=1))
            physical_input = payload["physical_input"][0].numpy()
            target_raw = payload["physical_target"]
            target_oracle = (
                None
                if target_raw is None
                else p3.decode_tensor(
                    p3.encode_tensor(target_raw.to(processor.device), channel_axis=1),
                    channel_axis=1,
                )
            )
            prepared = (
                {}
                if target_oracle is None
                else {
                    name: prepare_structure(
                        physical_input[CHANNELS.index(name)],
                        target_oracle[0, CHANNELS.index(name)].cpu().numpy(),
                        shell_index,
                    )
                    for name in TARGET_CHANNELS
                }
            )
            for alpha in alphas:
                anchored = anchored_output(state, direct, alpha)
                decoded = processor.decode_prediction(
                    anchored, apply_evaluation_clamp=True
                )
                physical = decoded.physical_prediction
                state_norm = float(global_state_norm(anchored)[0].cpu())
                for channel, name in enumerate(CHANNELS):
                    y = anchored[0, channel].cpu().numpy()
                    x = physical[0, channel].cpu().numpy()
                    derivative = decoder_derivative_channel(
                        p3, y, channel=channel, source_dtype=y.dtype
                    )
                    ood = ood_diagnostics(
                        y, envelope["channels"][name]["normalized"]
                    )
                    structure = (
                        None
                        if name not in prepared
                        else evaluate_structure(prepared[name], x, shell_index)
                    )
                    rows.append(
                        {
                            "state": state_name,
                            "state_kind": payload["kind"],
                            "source_index": payload["source_index"],
                            "target_index": payload["target_index"],
                            "alpha": alpha,
                            "channel": name,
                            "output_state_delta": relative_l2(
                                y, state[0, channel].cpu().numpy()
                            ),
                            **ood,
                            "normalized_minimum": float(np.min(y)),
                            "normalized_maximum": float(np.max(y)),
                            "decoded_minimum": float(np.min(x)),
                            "decoded_maximum": float(np.max(x)),
                            "decoded_absolute_maximum": float(np.max(np.abs(x))),
                            "decoder_derivative_maximum": float(np.max(derivative)),
                            "derivative_over_train_q999": float(
                                np.max(derivative)
                                / max(
                                    envelope["channels"][name]["decoder_derivative"]["q999"],
                                    1.0e-30,
                                )
                            ),
                            "normalized_state_norm": state_norm,
                            "rout_failure": state_norm > dissipation.rout,
                            "variance": float(np.var(x, dtype=np.float64)),
                            "shell_variance": variance_vector(x, shell_index).tolist(),
                            "radial_profile": radial_profile_vector(x).tolist(),
                            "high_k_energy": spectrum_metrics(
                                x, axis="combined", demean=True
                            )["high_k_energy"],
                            "gate_2_failed": None
                            if structure is None
                            else structure["gate_2_failed"],
                            "gate_3_passed": None
                            if structure is None
                            else structure["gate_3_passed"],
                            "shell_transport_skill": None
                            if structure is None
                            else structure["shell_skill"],
                            "radial_transport_skill": None
                            if structure is None
                            else structure["radial_skill"],
                        }
                    )
    payload = {
        "schema_version": "paper-stage-q-fixed-state-response-v1",
        "states": list(states),
        "alphas": list(alphas),
        "rows": rows,
    }
    write_json(output / "fixed_state_response.json", payload)
    write_csv(output / "fixed_state_response.csv", rows)
    return payload


def gain_directions(
    state: torch.Tensor, processor: PaperDataProcessor
) -> dict[str, torch.Tensor]:
    directions = {}
    for name in TARGET_CHANNELS:
        value = torch.zeros_like(state)
        value[:, CHANNELS.index(name)] = 1.0
        directions[f"channel_{name}"] = value
    all_direction = state.clone()
    norm = all_direction.square().mean().sqrt()
    directions["all_channel"] = (
        torch.ones_like(all_direction) if float(norm) == 0.0 else all_direction / norm
    )
    inner = (processor.shells[:, 0] + processor.shells[:, 1]).clamp_max(1.0)
    outer = (processor.shells[:, 6] + processor.shells[:, 7]).clamp_max(1.0)
    directions["inner_shell"] = inner.expand_as(state)
    directions["outer_shell"] = outer.expand_as(state)
    shape = tuple(state.shape[2:])
    for label, values in (
        ("radial_low", radial_mode(shape, "linear_log_r")),
        ("radial_mid", radial_mode(shape, "mid_frequency")),
        ("low_k", directional_mode(shape, "phi", "low_k")),
        ("high_k", directional_mode(shape, "phi", "high_k")),
    ):
        tensor = torch.from_numpy(values).to(device=state.device, dtype=state.dtype)
        directions[label] = tensor.reshape(1, 1, *shape).expand_as(state).clone()
    return directions


def local_gain_audit(
    *,
    states: Mapping[str, Mapping[str, Any]],
    alphas: Sequence[float],
    model: torch.nn.Module,
    processor: PaperDataProcessor,
    output: Path,
) -> dict[str, Any]:
    shells = processor.shells.expand(1, -1, -1, -1, -1).to(processor.device)
    rows = []
    with torch.no_grad():
        for state_name, payload in states.items():
            state = payload["normalized"].to(processor.device)
            direct_base = model(x=torch.cat((state, shells), dim=1))
            for basis, direction in gain_directions(state, processor).items():
                direction_norm = float(
                    torch.linalg.vector_norm(direction.to(torch.float64)).cpu()
                )
                for epsilon in LOCAL_GAIN_EPSILONS:
                    direct_perturbed = model(
                        x=torch.cat((state + epsilon * direction, shells), dim=1)
                    )
                    jf = (direct_perturbed - direct_base) / epsilon
                    for alpha in alphas:
                        anchored_base = anchored_output(state, direct_base, alpha)
                        anchored_perturbed = anchored_output(
                            state + epsilon * direction,
                            direct_perturbed,
                            alpha,
                        )
                        numeric = (anchored_perturbed - anchored_base) / epsilon
                        analytic = (1.0 - alpha) * direction + alpha * jf
                        analytic_difference = float(
                            (
                                torch.linalg.vector_norm(
                                    (numeric - analytic).to(torch.float64)
                                )
                                / torch.linalg.vector_norm(
                                    analytic.to(torch.float64)
                                ).clamp_min(1.0e-30)
                            ).cpu()
                        )
                        gain = float(
                            torch.linalg.vector_norm(numeric.to(torch.float64)).cpu()
                            / max(direction_norm, 1.0e-30)
                        )
                        inner = (
                            processor.shells[:, 0:1] + processor.shells[:, 1:2]
                        ).clamp_max(1.0)
                        outer = (
                            processor.shells[:, 6:7] + processor.shells[:, 7:8]
                        ).clamp_max(1.0)
                        response_profile = numeric.to(torch.float64).mean(dim=(2, 3))
                        direction_profile = direction.to(torch.float64).mean(dim=(2, 3))
                        row = {
                            "state": state_name,
                            "basis": basis,
                            "epsilon": epsilon,
                            "alpha": alpha,
                            "J_A_gain": gain,
                            "analytic_numeric_relative_difference": analytic_difference,
                            "J_A_inner_shell_gain": float(
                                torch.linalg.vector_norm(
                                    (numeric * inner).to(torch.float64)
                                ).cpu()
                                / max(direction_norm, 1.0e-30)
                            ),
                            "J_A_outer_shell_gain": float(
                                torch.linalg.vector_norm(
                                    (numeric * outer).to(torch.float64)
                                ).cpu()
                                / max(direction_norm, 1.0e-30)
                            ),
                            "J_A_radial_profile_gain": float(
                                torch.linalg.vector_norm(response_profile).cpu()
                                / max(
                                    float(
                                        torch.linalg.vector_norm(
                                            direction_profile
                                        ).cpu()
                                    ),
                                    1.0e-30,
                                )
                            ),
                            "used_backward": False,
                        }
                        for channel, name in enumerate(CHANNELS):
                            row[f"J_A_cross_{name}"] = float(
                                torch.linalg.vector_norm(
                                    numeric[:, channel].to(torch.float64)
                                ).cpu()
                                / max(direction_norm, 1.0e-30)
                            )
                        rows.append(row)
    for row in rows:
        companion = next(
            other
            for other in rows
            if other["state"] == row["state"]
            and other["basis"] == row["basis"]
            and other["alpha"] == row["alpha"]
            and other["epsilon"] != row["epsilon"]
        )
        row["epsilon_relative_difference"] = abs(
            row["J_A_gain"] - companion["J_A_gain"]
        ) / max(abs(row["J_A_gain"]), abs(companion["J_A_gain"]), 1.0e-30)
        row["epsilon_consistent_20pct"] = (
            row["epsilon_relative_difference"] <= 0.2
        )
        direct_companion = next(
            other
            for other in rows
            if other["state"] == row["state"]
            and other["basis"] == row["basis"]
            and other["epsilon"] == row["epsilon"]
            and other["alpha"] == 1.0
        )
        row["gain_reduction_vs_alpha1"] = (
            direct_companion["J_A_gain"] - row["J_A_gain"]
        ) / max(abs(direct_companion["J_A_gain"]), 1.0e-30)
    payload = {
        "schema_version": "paper-stage-q-local-gain-v1",
        "states": list(states),
        "alphas": list(alphas),
        "epsilons": list(LOCAL_GAIN_EPSILONS),
        "parameter_backward": False,
        "not_full_jacobian_spectral_radius": True,
        "rows": rows,
        "summary": {
            "row_count": len(rows),
            "finite_rows": sum(
                math.isfinite(row["J_A_gain"])
                and math.isfinite(row["analytic_numeric_relative_difference"])
                for row in rows
            ),
            "epsilon_consistent_rows": sum(
                row["epsilon_consistent_20pct"] for row in rows
            ),
            "analytic_numeric_max_relative_difference": max(
                row["analytic_numeric_relative_difference"] for row in rows
            ),
        },
    }
    write_json(output / "local_gain.json", payload)
    write_csv(output / "local_gain.csv", rows)
    return payload


def two_application_audit(
    *,
    states: Mapping[str, Mapping[str, Any]],
    alphas: Sequence[float],
    model: torch.nn.Module,
    processor: PaperDataProcessor,
    p3,
    envelope: Mapping[str, Any],
    shell_index: np.ndarray,
    dissipation: PaperDissipativeReference,
    output: Path,
) -> dict[str, Any]:
    shells = processor.shells.expand(1, -1, -1, -1, -1).to(processor.device)
    rows = []
    with torch.no_grad():
        for state_name, payload in states.items():
            initial = payload["normalized"].to(processor.device)
            target_raw = payload["physical_target"]
            target_oracle = (
                None
                if target_raw is None
                else p3.decode_tensor(
                    p3.encode_tensor(target_raw.to(processor.device), channel_axis=1),
                    channel_axis=1,
                )
            )
            physical_input = payload["physical_input"][0].numpy()
            prepared = (
                {}
                if target_oracle is None
                else {
                    name: prepare_structure(
                        physical_input[CHANNELS.index(name)],
                        target_oracle[0, CHANNELS.index(name)].cpu().numpy(),
                        shell_index,
                    )
                    for name in TARGET_CHANNELS
                }
            )
            for alpha in alphas:
                current = initial
                first_delta = None
                for application in (1, 2):
                    direct = model(x=torch.cat((current, shells), dim=1))
                    anchored = anchored_output(current, direct, alpha)
                    delta = relative_l2(
                        anchored[0].cpu().numpy(), current[0].cpu().numpy()
                    )
                    if application == 1:
                        first_delta = delta
                    decoded = processor.decode_prediction(
                        anchored, apply_evaluation_clamp=True
                    ).physical_prediction
                    state_norm = float(global_state_norm(anchored)[0].cpu())
                    for channel, name in enumerate(CHANNELS):
                        y = anchored[0, channel].cpu().numpy()
                        x = decoded[0, channel].cpu().numpy()
                        ood = ood_diagnostics(
                            y, envelope["channels"][name]["normalized"]
                        )
                        structure = (
                            None
                            if name not in prepared
                            else evaluate_structure(prepared[name], x, shell_index)
                        )
                        rows.append(
                            {
                                "state": state_name,
                                "alpha": alpha,
                                "application": application,
                                "channel": name,
                                "normalized_state_norm": state_norm,
                                "output_delta": delta,
                                "second_over_first_delta": None
                                if application == 1
                                else delta / max(float(first_delta), 1.0e-30),
                                **ood,
                                "decoded_minimum": float(np.min(x)),
                                "decoded_maximum": float(np.max(x)),
                                "decoded_variance": float(np.var(x, dtype=np.float64)),
                                "shell_variance": variance_vector(x, shell_index).tolist(),
                                "radial_variance": radial_profile(x)["variance"],
                                "rout_failure": state_norm > dissipation.rout,
                                "gate_2_failed": None
                                if structure is None
                                else structure["gate_2_failed"],
                                "fixed_point_tendency": None
                                if application == 1
                                else delta < float(first_delta),
                            }
                        )
                    current = anchored
    payload = {
        "schema_version": "paper-stage-q-two-application-v1",
        "maximum_applications": 2,
        "not_production_rollout": True,
        "states": list(states),
        "alphas": list(alphas),
        "rows": rows,
    }
    write_json(output / "two_application.json", payload)
    write_csv(output / "two_application.csv", rows)
    return payload


def rollout_audit(
    *,
    candidate_alpha: float | None,
    config,
    model: torch.nn.Module,
    processor: PaperDataProcessor,
    p3,
    envelope: Mapping[str, Any],
    shell_index: np.ndarray,
    dissipation: PaperDissipativeReference,
    output: Path,
) -> dict[str, Any]:
    if candidate_alpha is None:
        payload = {
            "schema_version": "paper-stage-q-rollout-v1",
            "status": "not_run_no_candidate",
            "candidate_alpha": None,
            "steps": 0,
            "rows": [],
            "transform_counts": {},
        }
        write_json(output / "rollout_gt.json", payload)
        write_csv(output / "rollout_gt.csv", [{"status": "not_run_no_candidate"}])
        return payload
    h5_path = config.resolve_path(config.values["protocol"]["dataset"])
    alphas = (0.0, 1.0, float(candidate_alpha))
    shells = processor.shells.expand(1, -1, -1, -1, -1).to(processor.device)
    rows = []
    counts = {}
    with h5py.File(h5_path, "r") as handle, torch.no_grad():
        snapshots = handle["snapshots"]
        initial = torch.from_numpy(
            np.asarray(snapshots[91], dtype=np.float32)
        ).unsqueeze(0).to(processor.device)
        for alpha in alphas:
            current = p3.encode_tensor(initial, channel_axis=1)
            alpha_counts = {"prediction_decode": 0, "next_input_encode": 0}
            for step in range(1, 20):
                raw_target = torch.from_numpy(
                    np.asarray(snapshots[91 + step], dtype=np.float32)
                ).unsqueeze(0).to(processor.device)
                target = p3.encode_tensor(raw_target, channel_axis=1)
                oracle = p3.decode_tensor(target, channel_axis=1)
                input_physical = p3.decode_tensor(current, channel_axis=1)
                direct = model(x=torch.cat((current, shells), dim=1))
                anchored = anchored_output(current, direct, alpha)
                decoded_result = processor.decode_prediction(
                    anchored, apply_evaluation_clamp=True
                )
                physical = decoded_result.physical_prediction
                alpha_counts["prediction_decode"] += 1
                feedback = p3.encode_tensor(physical, channel_axis=1)
                alpha_counts["next_input_encode"] += 1
                state_norm = float(global_state_norm(anchored)[0].cpu())
                artifacts = artifact_diagnostics(
                    physical,
                    raw_target,
                    raw_target,
                    reference_kind="ground_truth",
                )
                for channel, name in enumerate(CHANNELS):
                    y = anchored[0, channel].cpu().numpy()
                    x = physical[0, channel].cpu().numpy()
                    derivative = decoder_derivative_channel(
                        p3, y, channel=channel, source_dtype=y.dtype
                    )
                    ood = ood_diagnostics(
                        y, envelope["channels"][name]["normalized"]
                    )
                    structure = None
                    if name in TARGET_CHANNELS:
                        prepared = prepare_structure(
                            input_physical[0, channel].cpu().numpy(),
                            oracle[0, channel].cpu().numpy(),
                            shell_index,
                        )
                        structure = evaluate_structure(prepared, x, shell_index)
                    rows.append(
                        {
                            "alpha": alpha,
                            "step": step,
                            "selected_report_step": step in (1, 3, 5, 10, 19),
                            "channel": name,
                            "normalized_relative_l2": relative_l2(
                                y, target[0, channel].cpu().numpy()
                            ),
                            "model_to_p3_oracle": relative_l2(
                                x, oracle[0, channel].cpu().numpy()
                            ),
                            "model_to_raw": relative_l2(
                                x, raw_target[0, channel].cpu().numpy()
                            ),
                            **ood,
                            "normalized_state_norm": state_norm,
                            "persistence_ratio": None,
                            "decoded_minimum": float(np.min(x)),
                            "decoded_maximum": float(np.max(x)),
                            "rho_press_positive": bool(
                                torch.all(physical[:, CHANNELS.index("rho")] > 0)
                                and torch.all(physical[:, CHANNELS.index("press")] > 0)
                            ),
                            "decoder_derivative_maximum": float(np.max(derivative)),
                            "derivative_over_train_q999": float(
                                np.max(derivative)
                                / max(
                                    envelope["channels"][name]["decoder_derivative"]["q999"],
                                    1.0e-30,
                                )
                            ),
                            "evaluation_clamp": float(
                                decoded_result.clamp_mask[:, channel]
                                .float()
                                .mean()
                                .cpu()
                            ),
                            "rout_failure": state_norm > dissipation.rout,
                            "variance": float(np.var(x, dtype=np.float64)),
                            "shell_variance": variance_vector(x, shell_index).tolist(),
                            "radial_profile": radial_profile_vector(x).tolist(),
                            "high_k_energy": spectrum_metrics(
                                x, axis="combined", demean=True
                            )["high_k_energy"],
                            "gate_2_failed": None
                            if structure is None
                            else structure["gate_2_failed"],
                            "gate_3_passed": None
                            if structure is None
                            else structure["gate_3_passed"],
                            "shell_transport_skill": None
                            if structure is None
                            else structure["shell_skill"],
                            "radial_transport_skill": None
                            if structure is None
                            else structure["radial_skill"],
                            "legacy_flags": ";".join(
                                flag
                                for flag in artifacts["flags"]
                                if flag.startswith(f"{name}:")
                            ),
                        }
                    )
                current = feedback
            counts[str(alpha)] = alpha_counts
            if alpha_counts != expected_transform_counts(19):
                raise RuntimeError("Stage Q rollout transform counters changed")
    # Add the same-step persistence ratio without using it for alpha selection.
    lookup = {
        (row["alpha"], row["step"], row["channel"]): row for row in rows
    }
    for row in rows:
        persistence_error = lookup[(0.0, row["step"], row["channel"])][
            "normalized_relative_l2"
        ]
        row["persistence_ratio"] = row["normalized_relative_l2"] / max(
            persistence_error, 1.0e-30
        )
    payload = {
        "schema_version": "paper-stage-q-rollout-v1",
        "status": "completed_candidate_counterfactual",
        "candidate_alpha": candidate_alpha,
        "steps": 19,
        "alphas": list(alphas),
        "rows": rows,
        "transform_counts": counts,
    }
    write_json(output / "rollout_gt.json", payload)
    write_csv(output / "rollout_gt.csv", rows)
    return payload


def final_decision_output(
    *,
    selection: Mapping[str, Any],
    train: Mapping[str, Any],
    validation: Mapping[str, Any],
    rollout: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    candidate = selection["candidate_alpha"]
    train_summaries = train["_summaries_by_alpha"]
    validation_summaries = validation["_summaries_by_alpha"]
    evidence: dict[str, Any] = {
        "candidate_exists": candidate is not None,
        "selection_used_validation": False,
        "contract_tests_passed": True,
    }
    if candidate is None:
        decision = "D. NO_VALID_ANCHOR_CANDIDATE"
        evidence["reason"] = "no_intermediate_alpha_passed_all_train_only_readiness_groups"
    else:
        candidate = float(candidate)
        train_candidate = train_summaries[candidate]
        val_candidate = validation_summaries[candidate]
        val_direct = validation_summaries[1.0]
        val_persistence = validation_summaries[0.0]
        finite_positive = bool(
            val_candidate["engineering"]["all_finite"]
            and val_candidate["engineering"]["rho_press_positive"]
        )
        rout_improved = val_candidate["rout_failure_count"] <= 0.25 * max(
            val_direct["rout_failure_count"], 1
        )
        gate2_improved = val_candidate["gate_2_severe_count"] <= 0.5 * max(
            val_direct["gate_2_severe_count"], 1
        )
        dynamics = {
            "normalized_average_better_than_persistence": val_candidate[
                "normalized_average"
            ]
            < val_persistence["normalized_average"],
            "four_positive_residual_cosines": sum(
                value > 0
                for value in val_candidate[
                    "residual_cosine_channel_medians"
                ].values()
            )
            >= 4,
            "positive_shell_skill": val_candidate["shell_transport_skill_median"] > 0,
            "positive_radial_skill": val_candidate["radial_transport_skill_median"] > 0,
        }
        rollout_completed = rollout["status"] == "completed_candidate_counterfactual"
        candidate_rows = [
            row
            for row in rollout.get("rows", [])
            if row["alpha"] == candidate
        ]
        no_persistent_explosion = bool(
            rollout_completed
            and sum(row["rout_failure"] for row in candidate_rows) < len(candidate_rows)
        )
        overshoot_improved = (
            val_candidate["normalized_q_ood_fraction_median"]
            <= 0.25 * val_direct["normalized_q_ood_fraction_median"]
        )
        evidence.update(
            {
                "validation_finite_positive": finite_positive,
                "validation_rout_improved": rout_improved,
                "validation_gate2_improved": gate2_improved,
                "validation_overshoot_improved": overshoot_improved,
                "dynamical_evidence": dynamics,
                "rollout_no_persistent_explosion": no_persistent_explosion,
                "train_persistence_equivalent": train_candidate[
                    "persistence_equivalent"
                ],
            }
        )
        if not finite_positive:
            decision = "E. INCONCLUSIVE_OR_ENGINEERING_FAILURE"
        elif train_candidate["persistence_equivalent"]:
            decision = "C. ANCHOR_COLLAPSES_TO_PERSISTENCE"
        elif (
            rout_improved
            and gate2_improved
            and no_persistent_explosion
            and any(dynamics.values())
        ):
            decision = "A. ANCHOR_CONTRACT_READY_FOR_PAIRED_SMOKE"
        elif overshoot_improved and rout_improved and gate2_improved:
            decision = "B. OVERSHOOT_REDUCED_BUT_NO_DYNAMICAL_SKILL"
        else:
            decision = "D. NO_VALID_ANCHOR_CANDIDATE"
    payload = {
        "schema_version": "paper-stage-q-decision-v1",
        "classification": STAGE_Q_CLASSIFICATION,
        "decision": decision,
        "candidate_alpha": selection["candidate_alpha"],
        "evidence": evidence,
        "frozen_history": FROZEN_HISTORY,
        "historical_status_changed": False,
        "no_training": True,
        "next_action_proposal_only": True,
    }
    write_json(output / "stage_q_decision.json", payload)
    (output / "stage_q_decision.md").write_text(
        "# Stage Q decision\n\n"
        f"## {decision}\n\n"
        f"Frozen train-only candidate alpha: `{selection['candidate_alpha']}`.\n\n"
        "This no-training output-contract audit does not change Stage O C or Stage P 6 "
        "and does not authorize a smoke, pilot, clipping change, or new model.\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = root / "outputs/paper_reduced100/stage_q"
    output.mkdir(parents=True, exist_ok=True)
    qconfig_path = root / "configs/paper_reduced100/stage_q_residual_anchor_audit.yaml"
    qconfig = yaml.safe_load(qconfig_path.read_text(encoding="utf-8"))
    stage_o_config_path = root / "configs/paper_reduced100/stage_o_localno_p3_plain.yaml"
    stage_o_config = load_paper_experiment_config(
        stage_o_config_path, project_root=root
    )
    manifest, model = provenance_gate(root, stage_o_config, qconfig, qconfig_path)
    if not torch.cuda.is_available():
        raise RuntimeError("Stage Q frozen LocalNO forward audit requires verified WSL CUDA")
    device = torch.device("cuda:0")
    model = model.to(device).eval()
    parameter_hash_before = model_state_sha256(model)
    processor = PaperDataProcessor.from_config(stage_o_config).to(device)
    processor.eval()
    p3 = load_frozen_p3(stage_o_config)
    values = stage_o_config.values
    expected_artifacts = {
        "source_hdf5_checksum": values["provenance"]["dataset"]["sha256"],
        "preprocessing_stats_checksum": stage_o_config.prior_preprocessing_checksum,
        "training_indices": tuple(range(*values["protocol"]["train_snapshots"])),
        "protocol_name": values["protocol"]["name"],
        "thermal_channel": values["thermal"]["channel"],
    }
    dissipation = PaperDissipativeReference.load(
        stage_o_config.resolve_path(
            values["provenance"]["artifacts"]["dissipation"]["path"]
        ),
        **expected_artifacts,
    )
    envelope = json.loads(
        (root / "outputs/paper_reduced100/stage_p/train_envelope.json").read_text()
    )
    if envelope["indices"] != list(range(11, 91)) or envelope[
        "validation_indices_used"
    ]:
        raise RuntimeError("Stage Q train envelope has validation leakage")
    with h5py.File(
        stage_o_config.resolve_path(values["protocol"]["dataset"]), "r"
    ) as handle:
        r = np.asarray(handle["coords/r"][...], dtype=np.float64)
    _, shell_index = radial_shell_indices(r, 8)

    contract = contract_definition_output(
        qconfig=qconfig,
        config_sha256=manifest["stage_q_config_sha256"],
        output=output,
    )
    identities = contract_identity_output(
        model=model,
        p3=p3,
        processor=processor,
        h5_path=stage_o_config.resolve_path(values["protocol"]["dataset"]),
        qconfig=qconfig,
        output=output,
    )

    # Calibration is completed and selection is serialized before validation is read.
    train = split_teacher_forced_audit(
        split="train",
        source_indices=tuple(range(11, 90)),
        alphas=ALPHA_GRID,
        config=stage_o_config,
        model=model,
        processor=processor,
        p3=p3,
        envelope=envelope,
        shell_index=shell_index,
        dissipation=dissipation,
    )
    selection, readiness = write_train_and_selection(
        train=train,
        qconfig=qconfig,
        qconfig_sha256=manifest["stage_q_config_sha256"],
        pair_order_sha256=manifest["checksums"]["pair_order"],
        data_sha256=manifest["checksums"]["dataset"],
        p3_statistics_sha256=manifest["checksums"]["p3_statistics"],
        output=output,
    )
    candidate = selection["candidate_alpha"]
    validation_alphas = [0.0, 1.0]
    if candidate is not None:
        validation_alphas.insert(1, float(candidate))
    validation = split_teacher_forced_audit(
        split="validation",
        source_indices=tuple(range(91, 110)),
        alphas=tuple(validation_alphas),
        config=stage_o_config,
        model=model,
        processor=processor,
        p3=p3,
        envelope=envelope,
        shell_index=shell_index,
        dissipation=dissipation,
    )
    write_validation(validation, candidate_alpha=candidate, output=output)

    states = collect_fixed_states(
        config=stage_o_config,
        model=model,
        processor=processor,
        p3=p3,
    )
    comparison_alphas = [0.0, 1.0]
    if candidate is not None:
        comparison_alphas.insert(1, float(candidate))
    fixed = fixed_state_response(
        states=states,
        alphas=tuple(comparison_alphas),
        model=model,
        processor=processor,
        p3=p3,
        envelope=envelope,
        shell_index=shell_index,
        dissipation=dissipation,
        output=output,
    )
    local_gain = local_gain_audit(
        states=states,
        alphas=tuple(comparison_alphas),
        model=model,
        processor=processor,
        output=output,
    )
    two_application = two_application_audit(
        states=states,
        alphas=tuple(comparison_alphas),
        model=model,
        processor=processor,
        p3=p3,
        envelope=envelope,
        shell_index=shell_index,
        dissipation=dissipation,
        output=output,
    )
    rollout = rollout_audit(
        candidate_alpha=candidate,
        config=stage_o_config,
        model=model,
        processor=processor,
        p3=p3,
        envelope=envelope,
        shell_index=shell_index,
        dissipation=dissipation,
        output=output,
    )
    decision = final_decision_output(
        selection=selection,
        train=train,
        validation=validation,
        rollout=rollout,
        output=output,
    )
    parameter_hash_after = model_state_sha256(model)
    if parameter_hash_after != parameter_hash_before:
        raise RuntimeError("Stage Q mutated frozen model parameters or buffers")
    manifest.update(
        {
            "device": {
                "name": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
            },
            "contract": contract,
            "contract_identity_tests_passed": identities["passed"],
            "parameter_buffer_sha256_before": parameter_hash_before,
            "parameter_buffer_sha256_after": parameter_hash_after,
            "parameter_mutated": False,
            "train_indices": list(range(11, 91)),
            "train_pairs": 79,
            "validation_indices_used_for_selection": [],
            "candidate_alpha": candidate,
            "selection_artifact_sha256": selection[
                "selection_artifact_sha256"
            ],
            "diagnostic_contract": {
                "no_training": True,
                "no_backward": True,
                "no_optimizer": True,
                "no_scheduler": True,
                "no_checkpoint_write": True,
                "no_clipping_in_output_contract": True,
                "validation_confirmatory_only": True,
                "rollout_max_steps": 19,
                "two_application_maximum": 2,
            },
            "outputs": {
                "train_rows": len(train["rows"]),
                "validation_rows": len(validation["rows"]),
                "fixed_state_rows": len(fixed["rows"]),
                "local_gain_rows": len(local_gain["rows"]),
                "two_application_rows": len(two_application["rows"]),
                "rollout_rows": len(rollout["rows"]),
            },
            "stage_q_decision": decision["decision"],
            "historical_status_changed": False,
        }
    )
    write_json(output / "run_manifest.json", manifest)
    (output / "run_manifest.md").write_text(
        "# Stage Q run manifest\n\n"
        "- Provenance: `passed`.\n"
        "- Frozen Stage O best checkpoint: strict model-only epoch `23`.\n"
        f"- Candidate alpha: `{candidate}`.\n"
        "- Validation used for selection: `false`.\n"
        "- No training, backward, optimizer, scheduler, checkpoint write, or contract clipping.\n"
        f"- Decision: `{decision['decision']}`.\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"candidate_alpha": candidate, "decision": decision["decision"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
