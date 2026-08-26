#!/usr/bin/env python
"""Run the authorized no-training Stage P extreme-range attribution audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import torch

from grmhd import CHANNELS
from grmhd.dataset import sha256_file
from grmhd.paper_config import build_paper_model, load_paper_experiment_config
from grmhd.paper_data_processor import PaperDataProcessor
from grmhd.paper_dissipation import PaperDissipativeReference, global_state_norm
from grmhd.paper_stage_g_evaluation import artifact_diagnostics, radial_shell_indices
from grmhd.paper_stage_l_attribution import basic_field_metrics, radial_profile, spectrum_metrics
from grmhd.paper_stage_m import radial_profile_vector, transport_metrics, variance_vector
from grmhd.paper_stage_n import directional_mode, radial_mode
from grmhd.paper_stage_o import load_frozen_p3
from grmhd.paper_stage_p import (
    DECODER_EPSILONS,
    FROZEN_HISTORY,
    LOCAL_GAIN_EPSILONS,
    SELECTED_GT_STEPS,
    SELECTED_NO_GT_STEPS,
    STAGE_P_CLASSIFICATION,
    TARGET_CHANNELS,
    decoder_derivative_channel,
    directional_gain,
    distribution_summary,
    driver_support,
    final_mechanism_decision,
    load_stage_o_model_only,
    ood_diagnostics,
    project_train_envelope,
    relative_l2,
    reset_channels,
    roundtrip_only,
)
from grmhd.paper_protocol import PaperReduced100Protocol


EXPECTED = {
    "upstream": "86a8bc7812a31b42c4f7895693cf4ac11521c066",
    "dataset": "cd3966bed1f617f49d775151cff87ae92eb2432a560bb82ae9355605e7aea70a",
    "stage_o_config": "d34b15cecf21be49faf9d98a5be55e63a7651224b9b625a5d6b153a45bbddcce",
    "p3_statistics": "aea6d3f91e54865deeb9b2b1142c49a633ae8b67223075273e6c0b88bf541948",
    "stage_k_config": "8b229c5c2528d66bd4e11d0b0c4af16d00e906129386f8119af1670721ba30af",
    "stage_k_initial": "07db393f5c3e831f44bcc2d47b8132c3f9a0e72e2a258618080db487cc13cbb0",
    "best_checkpoint": "756ee7f31a9780942aa0f35dce4e5584cbeaf04ed8e9777721e7765962f04e12",
    "last_checkpoint": "ec9c4f1f81f15de03952986d454acf150f3613fa12b78465426a3ceef1e3dfa7",
    "selected_states": "f42fc77c1d50089296516245e637dce9ae3927b5fdfee45cbfc481ec4990a700",
    "pair_order": "5e1e361ea75e01ffd667822d8bfe870d70c5bc69570d52cdf806e637b2df2b52",
    "stage_m_gate": "a829a3760a71a006f54ae18b399c569a903898d27f3ef3a72236982fd033e5f7",
    "stage_o_decision": "79fd718b7a0216f49d5a1df313ad19086e92c14ad800dfbe2215ec091930fa7f",
    "stage_o_gt": "bddbc5d33ddcb2844f12735c1f9dc1c4be4bfaf8c5dc6c4c91269b5d883d2cf7",
    "stage_o_no_gt": "c1e95637943f0e2d7ea811ff6bb5567d8e73606e0626f9410cb4986278ab5740",
}


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if torch.is_tensor(value):
        return json_safe(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([json_safe(row) for row in rows])


def git(*args: str, root: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def provenance_gate(root: Path, config, output: Path) -> tuple[dict[str, Any], torch.nn.Module]:
    paths = {
        "dataset": root / "data_proc/grmhd_regrid_inner_r200_64.h5",
        "stage_o_config": root / "configs/paper_reduced100/stage_o_localno_p3_plain.yaml",
        "p3_statistics": root / "outputs/paper_reduced100/stage_n/prototypes/p3_combined_v1/normalizer.npz",
        "stage_k_config": root / "configs/paper_reduced100/stage_k_localno_differential_plain.yaml",
        "stage_k_initial": root / "outputs/paper_reduced100/stage_k/shared_localno_initial_state.pt",
        "best_checkpoint": root / "outputs/paper_reduced100/stage_o/localno_p3_plain/best_validation_l2/paper_state_dict.pt",
        "last_checkpoint": root / "outputs/paper_reduced100/stage_o/localno_p3_plain/last/paper_state_dict.pt",
        "selected_states": root / "outputs/paper_reduced100/stage_o/localno_p3_plain/selected_states.pt",
        "pair_order": root / "outputs/paper_reduced100/stage_g/epoch_pair_order.json",
        "stage_m_gate": root / "outputs/paper_reduced100/stage_m/candidate_gate_config.json",
        "stage_o_decision": root / "outputs/paper_reduced100/stage_o/stage_o_decision.json",
        "stage_o_gt": root / "outputs/paper_reduced100/stage_o/localno_p3_plain/gt_rollout.json",
        "stage_o_no_gt": root / "outputs/paper_reduced100/stage_o/localno_p3_plain/no_gt_rollout.json",
    }
    observed = {name: sha256_file(path) for name, path in paths.items()}
    mismatches = {
        name: {"expected": EXPECTED[name], "observed": observed[name]}
        for name in EXPECTED
        if name != "upstream" and observed.get(name) != EXPECTED[name]
    }
    upstream = git("-C", "external/neuraloperator", "rev-parse", "HEAD", root=root)
    upstream_status = git("-C", "external/neuraloperator", "status", "--short", root=root)
    if upstream != EXPECTED["upstream"] or upstream_status:
        mismatches["upstream"] = {
            "expected": EXPECTED["upstream"],
            "observed": upstream,
            "status": upstream_status,
        }
    if mismatches:
        raise RuntimeError(f"Stage P provenance gate failed: {mismatches}")

    config_checksum = observed["stage_o_config"]
    best_model, best_epoch, best_metadata = load_stage_o_model_only(
        paths["best_checkpoint"].parent,
        config=config,
        model=build_paper_model(config),
        expected_config_checksum=config_checksum,
    )
    _, last_epoch, last_metadata = load_stage_o_model_only(
        paths["last_checkpoint"].parent,
        config=config,
        model=build_paper_model(config),
        expected_config_checksum=config_checksum,
    )
    if (best_epoch, last_epoch) != (23, 30):
        raise RuntimeError("Stage O best/last checkpoint epochs changed")
    selected = torch.load(paths["selected_states"], map_location="cpu", weights_only=True)
    rollout_hashes = {
        str(step): tensor_sha256(state)
        for step, state in selected["selected_steps"].items()
    }
    training = json.loads(
        (root / "outputs/paper_reduced100/stage_o/training_summary.json").read_text()
    )
    validation = json.loads(
        (root / "outputs/paper_reduced100/stage_o/validation_metrics.json").read_text()
    )
    no_gt = json.loads(paths["stage_o_no_gt"].read_text())
    gates = json.loads(
        (root / "outputs/paper_reduced100/stage_o/stage_m_v1_gate.json").read_text()
    )
    legacy = json.loads(
        (root / "outputs/paper_reduced100/stage_o/legacy_detector.json").read_text()
    )
    gt_payload = json.loads(paths["stage_o_gt"].read_text())
    facts = {
        "epochs": training["epochs"],
        "microbatches": training["microbatches"],
        "updates": training["optimizer_updates"],
        "best_epoch": best_epoch,
        "last_epoch": last_epoch,
        "strict_model_only_reload": True,
        "optimizer_constructed": False,
        "scheduler_constructed": False,
        "training_nonfinite": training["runtime"]["nonfinite_count"],
        "gt_finite": gt_payload["finite"],
        "gt_positive": gt_payload["rho_press_positive"],
        "no_gt_finite": no_gt["finite"],
        "no_gt_positive": no_gt["rho_press_positive"],
        "transform_counts": no_gt["transform_count_delta"],
        "p3_floor": {
            name: validation["p3_model"]["metrics"]["E_oracle_raw"]["per_channel"][name]
            for name in TARGET_CHANNELS
        },
        "model_over_persistence": (
            validation["p3_model"]["metrics"]["E_norm"]["arithmetic_average"]
            / validation["p3_persistence"]["metrics"]["E_norm"]["arithmetic_average"]
        ),
        "gate_2_failed": {
            name: gates["channels"][name]["gate"]["gate_2_model_added_degradation"]["failed"]
            for name in TARGET_CHANNELS
        },
        "gate_3_failed": {
            name: not gates["channels"][name]["gate"]["gate_3_transport_skill"]["passed"]
            for name in TARGET_CHANNELS
        },
        "validation_above_rout": validation["constraints"]["prediction_above_rout_fraction"],
        "gt_step19_average": gt_payload["records"][18]["oracle_aware"]["metrics"]["E_norm"]["arithmetic_average"],
        "no_gt_step100_clamp": no_gt["records"][-1]["evaluation_bound_clamp_fraction"],
        "legacy_target_collapse_count": sum(
            flag.startswith(tuple(f"{name}:" for name in TARGET_CHANNELS))
            and "possible_field_collapse" in flag
            for row in legacy["rows"]
            for flag in row["flags"].split(";")
        ),
        "legacy_all_channel_collapse_count": sum(
            row["collapse_count"] for row in legacy["rows"]
        ),
    }
    if facts["epochs"] != 30 or facts["microbatches"] != 2370 or facts["updates"] != 600:
        raise RuntimeError("Frozen Stage O training counts changed")
    if facts["transform_counts"] != {
        "input_encode": 100,
        "target_encode": 19,
        "oracle_decode": 19,
        "prediction_decode": 100,
    }:
        raise RuntimeError("Frozen Stage O transform counters changed")
    manifest = {
        "schema_version": "paper-stage-p-run-manifest-v1",
        "classification": STAGE_P_CLASSIFICATION,
        "project": {
            "branch": git("branch", "--show-current", root=root),
            "commit": git("rev-parse", "HEAD", root=root),
            "stage_o_base": "7fa0840252538bc05d50844cb64dbfbf7d51fb56",
        },
        "upstream_commit": upstream,
        "checksums": observed,
        "rollout_state_tensor_sha256": rollout_hashes,
        "checkpoint_reload": {
            "best": {"epoch": best_epoch, "metadata": best_metadata["stage_o"]},
            "last": {"epoch": last_epoch, "metadata": last_metadata["stage_o"]},
            "model_only": True,
            "optimizer": False,
            "scheduler": False,
        },
        "stage_o_facts_recomputed": facts,
        "stage_m_v1": True,
        "frozen_history": FROZEN_HISTORY,
        "provenance_passed": True,
    }
    write_json(output / "run_manifest.json", manifest)
    (output / "run_manifest.md").write_text(
        "# Stage P run manifest\n\n"
        "- Provenance: `passed`.\n"
        "- Frozen Stage O best/last model-only strict reload: `23/30`.\n"
        "- No optimizer, scheduler, backward, checkpoint write, or training.\n"
        "- Stage K--O historical decisions remain frozen.\n",
        encoding="utf-8",
    )
    return manifest, best_model


def build_train_envelope(root: Path, config, p3, output: Path) -> tuple[dict[str, Any], np.ndarray]:
    h5_path = config.resolve_path(config.values["protocol"]["dataset"])
    with h5py.File(h5_path, "r") as handle:
        raw = np.asarray(handle["snapshots"][11:91], dtype=np.float32)
        r = np.asarray(handle["coords/r"][...], dtype=np.float64)
    normalized = p3.encode_numpy(raw, channel_axis=1)
    _, shell_index = radial_shell_indices(r, 8)
    channels = {}
    for channel, name in enumerate(CHANNELS):
        raw_values = raw[:, channel]
        norm_values = normalized[:, channel]
        derivative = decoder_derivative_channel(
            p3, norm_values, channel=channel, source_dtype=normalized.dtype
        )
        shell_ranges = []
        for shell in range(8):
            selected = raw_values[:, shell_index == shell]
            shell_ranges.append(
                {"shell": shell, "minimum": float(selected.min()), "maximum": float(selected.max())}
            )
        profiles = raw_values.mean(axis=(1, 2), dtype=np.float64)
        channels[name] = {
            "normalized": distribution_summary(norm_values),
            "physical": {
                **distribution_summary(raw_values),
                "shell_ranges": shell_ranges,
                "radial_profile_minimum": np.min(profiles, axis=0).tolist(),
                "radial_profile_maximum": np.max(profiles, axis=0).tolist(),
            },
            "decoder_derivative": distribution_summary(derivative),
        }
    payload = {
        "schema_version": "paper-stage-p-train-envelope-v1",
        "indices": list(range(11, 91)),
        "validation_indices_used": [],
        "train_only": True,
        "quantiles": [0.001, 0.01, 0.5, 0.99, 0.999],
        "decoder_derivative": "analytic_frozen_P3",
        "finite_difference_validation_epsilons": list(DECODER_EPSILONS),
        "channels": channels,
    }
    write_json(output / "train_envelope.json", payload)
    (output / "train_envelope.md").write_text(
        "# Stage P train-only envelope\n\n"
        "- Source: snapshots `11..90` only; validation leakage: `false`.\n"
        "- Normalized and physical quantiles: `0.001/0.01/0.5/0.99/0.999`.\n"
        "- Decoder sensitivity: analytic frozen-P3 derivative, checked at fixed `1e-5/1e-4`.\n"
        "- All OOD and projection bounds in Stage P use this artifact.\n",
        encoding="utf-8",
    )
    del raw, normalized
    return payload, shell_index


def channel_relative_average(prediction: torch.Tensor, reference: torch.Tensor) -> float:
    values = []
    for channel in range(8):
        left = prediction[:, channel].detach().cpu().numpy()
        right = reference[:, channel].detach().cpu().numpy()
        values.append(relative_l2(left, right))
    return float(np.mean(values))


def shell_range(values: np.ndarray, shell_index: np.ndarray) -> list[dict[str, float | int]]:
    return [
        {
            "shell": shell,
            "minimum": float(values[shell_index == shell].min()),
            "maximum": float(values[shell_index == shell].max()),
            "variance": float(np.var(values[shell_index == shell], dtype=np.float64)),
        }
        for shell in range(8)
    ]


def structural_step(
    input_field: np.ndarray,
    target_field: np.ndarray,
    model_field: np.ndarray,
    shell_index: np.ndarray,
) -> dict[str, Any]:
    target_basic = basic_field_metrics(target_field)
    model_basic = basic_field_metrics(model_field)
    target_shell = variance_vector(target_field, shell_index)
    model_shell = variance_vector(model_field, shell_index)
    target_radial = radial_profile(target_field)["variance"]
    model_radial = radial_profile(model_field)["variance"]
    target_high = spectrum_metrics(target_field, axis="combined", demean=True)["high_k_energy"]
    model_high = spectrum_metrics(model_field, axis="combined", demean=True)["high_k_energy"]
    ratios = {
        "global_variance": model_basic["variance"] / max(target_basic["variance"], 1e-30),
        "shell_radial_variance": min(
            float(np.median(model_shell / np.maximum(target_shell, 1e-30))),
            model_radial / max(target_radial, 1e-30),
        ),
        "dynamic_span": model_basic["dynamic_span_q99_q01"]
        / max(target_basic["dynamic_span_q99_q01"], 1e-30),
        "high_k_energy": model_high / max(target_high, 1e-30),
    }
    severe = {name: value < 0.5 for name, value in ratios.items()}
    shell = transport_metrics(
        variance_vector(input_field, shell_index),
        target_shell,
        model_shell,
        epsilon=1e-30,
        sign_zero_tolerance=1e-12,
    )
    radial = transport_metrics(
        radial_profile_vector(input_field),
        radial_profile_vector(target_field),
        radial_profile_vector(model_field),
        epsilon=1e-30,
        sign_zero_tolerance=1e-12,
    )
    skills = (shell["persistence_relative_skill"], radial["persistence_relative_skill"])
    signs = (shell["signed_transport_agreement"], radial["signed_transport_agreement"])
    defined = all(value is not None and np.isfinite(value) for value in (*skills, *signs))
    gate3 = bool(
        defined
        and all(float(value) >= 0 for value in skills)
        and any(float(value) > 0 for value in skills)
        and all(float(value) >= 0.5 for value in signs)
    )
    return {
        "retention_ratios": ratios,
        "severe": severe,
        "gate_2_step_failed": sum(severe.values()) >= 2,
        "shell_transport": shell,
        "radial_transport": radial,
        "gate_3_step_passed": gate3,
    }


def failure_metrics(
    physical: torch.Tensor,
    *,
    structures: Mapping[str, Mapping[str, Any]],
    artifacts: Mapping[str, Any],
) -> dict[str, float]:
    array = physical[0].detach().cpu().numpy()
    flags = list(artifacts["flags"])
    shell_error = []
    radial_error = []
    for payload in structures.values():
        shell_error.append(max(0.0, -float(payload["shell_transport"]["persistence_relative_skill"])))
        radial_error.append(max(0.0, -float(payload["radial_transport"]["persistence_relative_skill"])))
    return {
        "Bcc2_range": float(np.max(np.abs(array[1]))),
        "Bcc3_range": float(np.max(np.abs(array[2]))),
        "vel3_range": float(np.max(np.abs(array[7]))),
        "total_decoded_range": float(np.max(np.abs(array))),
        "gate_2_count": float(sum(item["gate_2_step_failed"] for item in structures.values())),
        "gate_3_count": float(sum(not item["gate_3_step_passed"] for item in structures.values())),
        "shell_transport_error": float(np.mean(shell_error)),
        "radial_transport_error": float(np.mean(radial_error)),
        "ripple_count": float(sum("possible_high_frequency_ripple" in flag for flag in flags)),
        "stripe_count": float(sum("possible_stripe_anisotropy" in flag for flag in flags)),
    }


def trace_channel(
    *,
    step: int,
    channel: int,
    normalized_output: torch.Tensor,
    physical: torch.Tensor,
    feedback: torch.Tensor,
    normalized_input: torch.Tensor,
    envelope: Mapping[str, Any],
    shell_index: np.ndarray,
    p3,
    evaluation_clamp: float,
    structures: Mapping[str, Any] | None,
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    name = CHANNELS[channel]
    y = normalized_output[0, channel].detach().cpu().numpy()
    x = physical[0, channel].detach().cpu().numpy()
    z = feedback[0, channel].detach().cpu().numpy()
    zin = normalized_input[0, channel].detach().cpu().numpy()
    derivative = decoder_derivative_channel(p3, y, channel=channel, source_dtype=y.dtype)
    normalized_stats = distribution_summary(y)
    physical_stats = distribution_summary(x)
    feedback_stats = distribution_summary(z)
    ood = ood_diagnostics(y, envelope[name]["normalized"])
    train_derivative = envelope[name]["decoder_derivative"]
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
    inverse_clamp_occupancy = (
        0.0 if inverse_limit is None else float(np.mean(np.abs(y) > inverse_limit))
    )
    row: dict[str, Any] = {
        "step": step,
        "channel": name,
        "normalized_output": normalized_stats,
        "normalized_ood": ood,
        "physical_decode": {
            **physical_stats,
            "dynamic_span": physical_stats["maximum"] - physical_stats["minimum"],
            "shell_ranges": shell_range(x, shell_index),
            "radial_profile": np.mean(x, axis=(0, 1), dtype=np.float64).tolist(),
            "decoder_derivative_median": float(np.median(derivative)),
            "decoder_derivative_maximum": float(np.max(derivative)),
            "derivative_over_train_q999": float(
                np.max(derivative) / max(float(train_derivative["q999"]), 1e-30)
            ),
            "derivative_fraction_above_train_q999": float(
                np.mean(derivative > float(train_derivative["q999"]))
            ),
            "inverse_clamp_policy": inverse_policy,
            "inverse_clamp_limit": inverse_limit,
            "inverse_clamp_occupancy": inverse_clamp_occupancy,
            "evaluation_bounds_clamp_combined": evaluation_clamp,
        },
        "feedback": {
            "normalized": feedback_stats,
            "output_feedback_relative_l2": relative_l2(z, y),
            "maximum_discrepancy": float(np.max(np.abs(z - y))),
            "shell_discrepancy": shell_range(z - y, shell_index),
        },
        "next_model": {
            "input_norm": float(np.linalg.norm(zin.astype(np.float64))),
            "output_norm": float(np.linalg.norm(y.astype(np.float64))),
            "incremental_gain": float(
                np.linalg.norm(y.astype(np.float64))
                / max(np.linalg.norm(zin.astype(np.float64)), 1e-30)
            ),
            "legacy_flags": [flag for flag in artifacts["flags"] if flag.startswith(f"{name}:")],
        },
    }
    if structures is not None and name in structures:
        row["stage_m_v1_step"] = structures[name]
    return row


def actual_trajectory(
    *,
    root: Path,
    config,
    model: torch.nn.Module,
    processor: PaperDataProcessor,
    p3,
    envelope: Mapping[str, Any],
    shell_index: np.ndarray,
    dissipation: PaperDissipativeReference,
    output: Path,
) -> dict[str, Any]:
    h5_path = config.resolve_path(config.values["protocol"]["dataset"])
    with h5py.File(h5_path, "r") as handle:
        snapshots = np.asarray(handle["snapshots"][91:111], dtype=np.float32)
    device = processor.device
    initial = torch.from_numpy(snapshots[0]).unsqueeze(0).to(device)
    current_normalized = p3.encode_tensor(initial, channel_axis=1)
    initial_reference = initial
    shells = processor.shells.expand(1, -1, -1, -1, -1).to(device)
    selected_frozen = torch.load(
        root / "outputs/paper_reduced100/stage_o/localno_p3_plain/selected_states.pt",
        map_location="cpu",
        weights_only=True,
    )["selected_steps"]
    selected = set(SELECTED_GT_STEPS) | set(SELECTED_NO_GT_STEPS)
    trace_rows = []
    timeline_rows = []
    outputs: dict[int, torch.Tensor] = {}
    inputs: dict[int, torch.Tensor] = {}
    physical_states: dict[int, torch.Tensor] = {}
    step_metrics: dict[int, dict[str, float]] = {}
    replay_hashes = {}
    with torch.no_grad():
        for step in range(1, 101):
            if step in {1, 5, 10, 19, 25}:
                inputs[step] = current_normalized.detach().cpu()
            y = model(x=torch.cat((current_normalized, shells), dim=1))
            decoded = processor.decode_prediction(y, apply_evaluation_clamp=True)
            physical = decoded.physical_prediction
            feedback = p3.encode_tensor(physical, channel_axis=1)
            has_gt = step <= 19
            target = torch.from_numpy(snapshots[step]).unsqueeze(0).to(device) if has_gt else None
            oracle = p3.decode_tensor(p3.encode_tensor(target, channel_axis=1), channel_axis=1) if has_gt else None
            reference = target if has_gt else initial_reference
            artifacts = artifact_diagnostics(
                physical,
                reference,
                target if has_gt else initial_reference,
                reference_kind="ground_truth" if has_gt else "initial_snapshot_91",
            )
            structures = {}
            if has_gt:
                input_physical = p3.decode_tensor(current_normalized, channel_axis=1)
                for name in TARGET_CHANNELS:
                    channel = CHANNELS.index(name)
                    structures[name] = structural_step(
                        input_physical[0, channel].cpu().numpy(),
                        oracle[0, channel].cpu().numpy(),
                        physical[0, channel].cpu().numpy(),
                        shell_index,
                    )
                step_metrics[step] = failure_metrics(
                    physical, structures=structures, artifacts=artifacts
                )
            norm = float(global_state_norm(y)[0].cpu())
            for channel, name in enumerate(CHANNELS):
                y_np = y[0, channel].cpu().numpy()
                derivative = decoder_derivative_channel(
                    p3, y_np, channel=channel, source_dtype=y_np.dtype
                )
                ood = ood_diagnostics(y_np, envelope["channels"][name]["normalized"])
                flags = [flag for flag in artifacts["flags"] if flag.startswith(f"{name}:")]
                inverse_clamp = (
                    0.0
                    if p3.spec.channel_policies[channel] == "no_softclip"
                    else float(np.mean(np.abs(y_np) > p3.inverse_clamp_fraction * p3.gamma))
                )
                evaluation_channel_clamp = float(
                    decoded.clamp_mask[:, channel].float().mean().cpu()
                )
                clamp_occupancy = max(inverse_clamp, evaluation_channel_clamp)
                timeline_rows.append(
                    {
                        "step": step,
                        "channel": name,
                        "normalized_outside_q001_q999": ood["fraction_outside_train_q001_q999"] > 0,
                        "normalized_outside_min_max": ood["fraction_outside_train_min_max"] > 0,
                        "decode_sensitivity_above_train_q999": bool(
                            np.any(derivative > envelope["channels"][name]["decoder_derivative"]["q999"])
                        ),
                        "decoded_above_rout": norm > dissipation.rout,
                        "clamp_occupancy_positive": clamp_occupancy > 0,
                        "clamp_occupancy_gt_0_1": clamp_occupancy > 0.1,
                        "clamp_occupancy_gt_0_5": clamp_occupancy > 0.5,
                        "gate_2_failed": bool(structures.get(name, {}).get("gate_2_step_failed", False)),
                        "gate_3_failed": bool(
                            structures.get(name) is not None
                            and not structures[name]["gate_3_step_passed"]
                        ),
                        "ripple": any("possible_high_frequency_ripple" in flag for flag in flags),
                        "stripe": any("possible_stripe_anisotropy" in flag for flag in flags),
                        "field_collapse": any("possible_field_collapse" in flag for flag in flags),
                        "shell_radial_transport_severe": bool(
                            structures.get(name, {}).get("retention_ratios", {}).get(
                                "shell_radial_variance", 1.0
                            )
                            < 0.5
                        ),
                        "normalized_output_norm": norm,
                        "evaluation_clamp": decoded.clamp_fraction["combined"],
                        "channel_clamp_occupancy": clamp_occupancy,
                    }
                )
            if step in selected:
                outputs[step] = y.detach().cpu()
                physical_states[step] = physical.detach().cpu()
                for channel in range(8):
                    trace_rows.append(
                        trace_channel(
                            step=step,
                            channel=channel,
                            normalized_output=y,
                            physical=physical,
                            feedback=feedback,
                            normalized_input=current_normalized,
                            envelope=envelope["channels"],
                            shell_index=shell_index,
                            p3=p3,
                            evaluation_clamp=decoded.clamp_fraction["combined"],
                            structures=structures if has_gt else None,
                            artifacts=artifacts,
                        )
                    )
                if step in selected_frozen:
                    frozen = selected_frozen[step]
                    replay_hashes[str(step)] = {
                        "frozen": tensor_sha256(frozen),
                        "replayed": tensor_sha256(physical.detach().cpu()),
                        "bitwise_equal": bool(torch.equal(frozen, physical.detach().cpu())),
                    }
                    if not replay_hashes[str(step)]["bitwise_equal"]:
                        raise RuntimeError(f"Stage O trajectory replay changed at step {step}")
            current_normalized = feedback.detach()
    write_json(
        output / "trajectory_trace.json",
        {
            "schema_version": "paper-stage-p-trajectory-trace-v1",
            "contract": "G(z)=E_P3(D_P3(F_theta(concat(z,s)))) with frozen rho/press evaluation bounds",
            "selected_steps": sorted(selected),
            "rows": trace_rows,
            "replay_hashes": replay_hashes,
        },
    )
    flat_rows = []
    for row in trace_rows:
        flat_rows.append(
            {
                "step": row["step"],
                "channel": row["channel"],
                "normalized_min": row["normalized_output"]["minimum"],
                "normalized_max": row["normalized_output"]["maximum"],
                "normalized_ood_q_fraction": row["normalized_ood"]["fraction_outside_train_q001_q999"],
                "normalized_ood_range_fraction": row["normalized_ood"]["fraction_outside_train_min_max"],
                "physical_min": row["physical_decode"]["minimum"],
                "physical_max": row["physical_decode"]["maximum"],
                "derivative_median": row["physical_decode"]["decoder_derivative_median"],
                "derivative_max": row["physical_decode"]["decoder_derivative_maximum"],
                "derivative_over_train_q999": row["physical_decode"]["derivative_over_train_q999"],
                "feedback_relative_l2": row["feedback"]["output_feedback_relative_l2"],
                "feedback_max_discrepancy": row["feedback"]["maximum_discrepancy"],
                "incremental_gain": row["next_model"]["incremental_gain"],
            }
        )
    write_csv(output / "trajectory_trace.csv", flat_rows)
    return {
        "timeline_rows": timeline_rows,
        "outputs": outputs,
        "inputs": inputs,
        "physical_states": physical_states,
        "step_metrics": step_metrics,
        "trace_rows": trace_rows,
    }


TIMELINE_EVENTS = (
    "normalized_outside_q001_q999",
    "normalized_outside_min_max",
    "decode_sensitivity_above_train_q999",
    "decoded_above_rout",
    "clamp_occupancy_positive",
    "clamp_occupancy_gt_0_1",
    "clamp_occupancy_gt_0_5",
    "gate_2_failed",
    "gate_3_failed",
    "ripple",
    "stripe",
    "field_collapse",
    "shell_radial_transport_severe",
)


def timeline_output(rows: Sequence[Mapping[str, Any]], output: Path) -> dict[str, Any]:
    def relation(first: int | None, second: int | None) -> str:
        if first is None and second is None:
            return "both_absent"
        if first is None:
            return "first_absent"
        if second is None:
            return "second_absent"
        if first < second:
            return "precedes"
        if first == second:
            return "coincident"
        return "follows"

    result_rows = []
    by_channel = {name: [row for row in rows if row["channel"] == name] for name in CHANNELS}
    for name in CHANNELS:
        for event in TIMELINE_EVENTS:
            first = next((int(row["step"]) for row in by_channel[name] if row[event]), None)
            result_rows.append({"channel": name, "event": event, "first_step": first})
    target_first = {
        name: {
            event: next(
                (row["first_step"] for row in result_rows if row["channel"] == name and row["event"] == event),
                None,
            )
            for event in TIMELINE_EVENTS
        }
        for name in TARGET_CHANNELS
    }
    q_first = min(
        value["normalized_outside_q001_q999"]
        for value in target_first.values()
        if value["normalized_outside_q001_q999"] is not None
    )
    range_first = min(
        value["decoded_above_rout"]
        for value in target_first.values()
        if value["decoded_above_rout"] is not None
    )
    sensitivity_first = min(
        value["decode_sensitivity_above_train_q999"]
        for value in target_first.values()
        if value["decode_sensitivity_above_train_q999"] is not None
    )
    all_first = {
        event: min(
            (row["first_step"] for row in result_rows if row["event"] == event and row["first_step"] is not None),
            default=None,
        )
        for event in TIMELINE_EVENTS
    }
    clamp_first = all_first["clamp_occupancy_positive"]
    target_clamp_first = min(
        (
            value["clamp_occupancy_positive"]
            for value in target_first.values()
            if value["clamp_occupancy_positive"] is not None
        ),
        default=None,
    )
    earliest_step = min(
        value["normalized_outside_min_max"]
        for value in target_first.values()
        if value["normalized_outside_min_max"] is not None
    )
    earliest_channels = [
        name
        for name in TARGET_CHANNELS
        if target_first[name]["normalized_outside_min_max"] == earliest_step
    ]
    answers = {
        "failure_present_on_first_prediction": q_first == 1 or range_first == 1,
        "normalized_ood_to_physical_rout_relation": relation(q_first, range_first),
        "normalized_ood_precedes_or_equals_physical_rout": q_first <= range_first,
        "target_decode_sensitivity_to_target_clamp_relation": relation(
            sensitivity_first, target_clamp_first
        ),
        "target_decode_sensitivity_to_any_channel_clamp_relation": relation(
            sensitivity_first, clamp_first
        ),
        "decode_sensitivity_precedes_or_equals_any_channel_clamp": (
            clamp_first is not None and sensitivity_first <= clamp_first
        ),
        "earliest_target_channels": earliest_channels,
        "Bcc2_vs_Bcc3": "tie" if {"Bcc2", "Bcc3"} <= set(earliest_channels) else earliest_channels[0],
        "vel3_driver_status_deferred_to_driver_matrix": True,
    }
    payload = {
        "schema_version": "paper-stage-p-first-failure-timeline-v1",
        "events": result_rows,
        "target_channel_first_steps": target_first,
        "all_channel_first_steps": all_first,
        "answers": answers,
    }
    write_json(output / "first_failure_timeline.json", payload)
    write_csv(output / "first_failure_timeline.csv", result_rows)
    (output / "first_failure_timeline.md").write_text(
        "# Stage P first-failure timeline\n\n"
        f"- Failure on first prediction: `{answers['failure_present_on_first_prediction']}`.\n"
        f"- Normalized OOD versus Rout: `{answers['normalized_ood_to_physical_rout_relation']}`.\n"
        "- Target decode sensitivity versus target-channel clamp: "
        f"`{answers['target_decode_sensitivity_to_target_clamp_relation']}`.\n"
        "- Target decode sensitivity versus any-channel clamp: "
        f"`{answers['target_decode_sensitivity_to_any_channel_clamp_relation']}`.\n"
        f"- Earliest target channels by train min/max exit: `{','.join(earliest_channels)}`.\n"
        "- vel3 driver/passenger status is resolved only after reset and finite-difference evidence.\n",
        encoding="utf-8",
    )
    return payload


def teacher_forced_audit(
    *,
    config,
    model: torch.nn.Module,
    processor: PaperDataProcessor,
    p3,
    envelope: Mapping[str, Any],
    shell_index: np.ndarray,
    dissipation: PaperDissipativeReference,
    output: Path,
) -> dict[str, Any]:
    with h5py.File(config.resolve_path(config.values["protocol"]["dataset"]), "r") as handle:
        snapshots = np.asarray(handle["snapshots"][91:111], dtype=np.float32)
    device = processor.device
    shells = processor.shells.expand(1, -1, -1, -1, -1).to(device)
    rows = []
    outputs = {}
    with torch.no_grad():
        for step in range(1, 20):
            raw_input = torch.from_numpy(snapshots[step - 1]).unsqueeze(0).to(device)
            raw_target = torch.from_numpy(snapshots[step]).unsqueeze(0).to(device)
            z_input = p3.encode_tensor(raw_input, channel_axis=1)
            z_target = p3.encode_tensor(raw_target, channel_axis=1)
            oracle_input = p3.decode_tensor(z_input, channel_axis=1)
            oracle_target = p3.decode_tensor(z_target, channel_axis=1)
            y = model(x=torch.cat((z_input, shells), dim=1))
            decoded = processor.decode_prediction(y, apply_evaluation_clamp=True)
            physical = decoded.physical_prediction
            outputs[step] = y.detach().cpu()
            norm = float(global_state_norm(y)[0].cpu())
            artifacts = artifact_diagnostics(
                physical, raw_target, raw_target, reference_kind="ground_truth"
            )
            for channel, name in enumerate(CHANNELS):
                y_np = y[0, channel].cpu().numpy()
                derivative = decoder_derivative_channel(
                    p3, y_np, channel=channel, source_dtype=y_np.dtype
                )
                structure = None
                if name in TARGET_CHANNELS:
                    structure = structural_step(
                        oracle_input[0, channel].cpu().numpy(),
                        oracle_target[0, channel].cpu().numpy(),
                        physical[0, channel].cpu().numpy(),
                        shell_index,
                    )
                rows.append(
                    {
                        "step": step,
                        "channel": name,
                        **ood_diagnostics(y_np, envelope["channels"][name]["normalized"]),
                        "decoder_derivative_median": float(np.median(derivative)),
                        "decoder_derivative_maximum": float(np.max(derivative)),
                        "derivative_over_train_q999": float(
                            np.max(derivative)
                            / max(envelope["channels"][name]["decoder_derivative"]["q999"], 1e-30)
                        ),
                        "physical_minimum": float(physical[:, channel].min().cpu()),
                        "physical_maximum": float(physical[:, channel].max().cpu()),
                        "above_rout": norm > dissipation.rout,
                        "evaluation_clamp": decoded.clamp_fraction["combined"],
                        "normalized_relative_l2": relative_l2(
                            y[0, channel].cpu().numpy(), z_target[0, channel].cpu().numpy()
                        ),
                        "raw_relative_l2": relative_l2(
                            physical[0, channel].cpu().numpy(), raw_target[0, channel].cpu().numpy()
                        ),
                        "gate_2_failed": None if structure is None else structure["gate_2_step_failed"],
                        "gate_3_failed": None if structure is None else not structure["gate_3_step_passed"],
                        "shell_skill": None if structure is None else structure["shell_transport"]["persistence_relative_skill"],
                        "radial_skill": None if structure is None else structure["radial_transport"]["persistence_relative_skill"],
                        "legacy_flags": ";".join(
                            flag for flag in artifacts["flags"] if flag.startswith(f"{name}:")
                        ),
                    }
                )
    payload = {
        "schema_version": "paper-stage-p-teacher-forced-v1",
        "transitions": 19,
        "prediction_feedback_used": False,
        "rows": rows,
        "fraction_predictions_above_rout": float(
            np.mean([row["above_rout"] for row in rows if row["channel"] == "Bcc2"])
        ),
    }
    write_json(output / "teacher_forced.json", payload)
    write_csv(output / "teacher_forced.csv", [{k: v for k, v in row.items() if k != "legacy_flags"} | {"legacy_flags": row["legacy_flags"]} for row in rows])
    return {"payload": payload, "outputs": outputs}


def normalized_direct_audit(
    *,
    config,
    model: torch.nn.Module,
    processor: PaperDataProcessor,
    p3,
    envelope: Mapping[str, Any],
    shell_index: np.ndarray,
    actual_outputs: Mapping[int, torch.Tensor],
    output: Path,
) -> dict[str, Any]:
    with h5py.File(config.resolve_path(config.values["protocol"]["dataset"]), "r") as handle:
        initial = np.asarray(handle["snapshots"][91], dtype=np.float32)
    device = processor.device
    z = p3.encode_tensor(torch.from_numpy(initial).unsqueeze(0).to(device), channel_axis=1)
    shells = processor.shells.expand(1, -1, -1, -1, -1).to(device)
    rows = []
    outputs = {}
    with torch.no_grad():
        for step in range(1, 20):
            y = model(x=torch.cat((z, shells), dim=1))
            outputs[step] = y.detach().cpu()
            state_norm = float(global_state_norm(y)[0].cpu())
            for channel, name in enumerate(CHANNELS):
                values = y[0, channel].cpu().numpy()
                actual = actual_outputs.get(step)
                rows.append(
                    {
                        "step": step,
                        "channel": name,
                        "state_norm": state_norm,
                        "minimum": float(values.min()),
                        "maximum": float(values.max()),
                        "std": float(values.std(dtype=np.float64)),
                        **ood_diagnostics(values, envelope["channels"][name]["normalized"]),
                        "shell_variance": [
                            float(np.var(values[shell_index == shell], dtype=np.float64))
                            for shell in range(8)
                        ],
                        "radial_profile_variance": float(
                            np.var(np.mean(values, axis=(0, 1), dtype=np.float64))
                        ),
                        "relative_l2_to_physical_loop_output": None
                        if actual is None
                        else relative_l2(values, actual[0, channel].numpy()),
                    }
                )
            z = y.detach()
    payload = {
        "schema_version": "paper-stage-p-normalized-direct-v1",
        "steps": 19,
        "decoder_calls": 0,
        "encoder_calls": 0,
        "deployable": False,
        "rows": rows,
    }
    write_json(output / "normalized_direct_feedback.json", payload)
    write_csv(
        output / "normalized_direct_feedback.csv",
        [
            {k: v for k, v in row.items() if k not in {"shell_variance"}}
            | {"shell_variance": ";".join(str(value) for value in row["shell_variance"])}
            for row in rows
        ],
    )
    return {"payload": payload, "outputs": outputs}


def roundtrip_only_audit(
    *,
    config,
    p3,
    envelope: Mapping[str, Any],
    teacher_outputs: Mapping[int, torch.Tensor],
    actual_outputs: Mapping[int, torch.Tensor],
    output: Path,
) -> dict[str, Any]:
    sources: list[tuple[str, int, np.ndarray]] = []
    with h5py.File(config.resolve_path(config.values["protocol"]["dataset"]), "r") as handle:
        for index in range(11, 91):
            raw = np.asarray(handle["snapshots"][index], dtype=np.float32)
            sources.append(("train_oracle", index, p3.encode_numpy(raw, channel_axis=0)))
        for index in range(91, 111):
            raw = np.asarray(handle["snapshots"][index], dtype=np.float32)
            sources.append(("validation_oracle", index, p3.encode_numpy(raw, channel_axis=0)))
    sources.extend(
        ("teacher_one_step_output", step, value[0].numpy())
        for step, value in teacher_outputs.items()
    )
    sources.extend(
        ("actual_rollout_output", step, value[0].numpy())
        for step, value in actual_outputs.items()
    )
    rows = []
    for source, index, state in sources:
        h1 = roundtrip_only(p3, state, applications=1, channel_axis=0)
        h2 = roundtrip_only(p3, state, applications=2, channel_axis=0)
        physical = p3.decode_numpy(state, channel_axis=0)
        for channel, name in enumerate(CHANNELS):
            original = state[channel]
            first = h1[channel]
            second = h2[channel]
            derivative = decoder_derivative_channel(
                p3, original, channel=channel, source_dtype=state.dtype
            )
            delta1 = relative_l2(first, original)
            delta2 = relative_l2(second, first)
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
            rows.append(
                {
                    "source": source,
                    "index_or_step": index,
                    "channel": name,
                    "roundtrip_1_relative_l2": delta1,
                    "roundtrip_2_increment_relative_l2": delta2,
                    "fixed_point_delta_ratio": delta2 / max(delta1, 1e-30),
                    "std_ratio_h1_over_input": float(
                        np.std(first, dtype=np.float64)
                        / max(np.std(original, dtype=np.float64), 1e-30)
                    ),
                    "std_ratio_h2_over_h1": float(
                        np.std(second, dtype=np.float64)
                        / max(np.std(first, dtype=np.float64), 1e-30)
                    ),
                    "input_minimum": float(original.min()),
                    "input_maximum": float(original.max()),
                    "h1_minimum": float(first.min()),
                    "h1_maximum": float(first.max()),
                    "h2_minimum": float(second.min()),
                    "h2_maximum": float(second.max()),
                    "physical_absolute_maximum": float(np.max(np.abs(physical[channel]))),
                    "inverse_clamp_policy": inverse_policy,
                    "inverse_clamp_limit": inverse_limit,
                    "inverse_clamp_occupancy": (
                        0.0
                        if inverse_limit is None
                        else float(np.mean(np.abs(original) > inverse_limit))
                    ),
                    "decoder_derivative_median": float(np.median(derivative)),
                    "decoder_derivative_maximum": float(np.max(derivative)),
                    "derivative_over_train_q999": float(
                        np.max(derivative)
                        / max(envelope["channels"][name]["decoder_derivative"]["q999"], 1e-30)
                    ),
                }
            )
    payload = {
        "schema_version": "paper-stage-p-roundtrip-only-v1",
        "model_calls": 0,
        "maximum_applications": 2,
        "rows": rows,
    }
    write_json(output / "roundtrip_only.json", payload)
    write_csv(output / "roundtrip_only.csv", rows)
    return payload


def counterfactual_rollouts(
    *,
    kind: str,
    variants: Mapping[str, tuple[str, ...]],
    config,
    model: torch.nn.Module,
    processor: PaperDataProcessor,
    p3,
    shell_index: np.ndarray,
    dissipation: PaperDissipativeReference,
    actual_metrics: Mapping[int, Mapping[str, float]],
    envelope: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    if kind not in {"projection", "reset"}:
        raise ValueError("Unknown Stage P counterfactual")
    with h5py.File(config.resolve_path(config.values["protocol"]["dataset"]), "r") as handle:
        snapshots = np.asarray(handle["snapshots"][91:111], dtype=np.float32)
    device = processor.device
    raw_initial = torch.from_numpy(snapshots[0]).unsqueeze(0).to(device)
    shells = processor.shells.expand(1, -1, -1, -1, -1).to(device)
    rows = []
    metric_by_variant: dict[str, dict[int, dict[str, float]]] = {}
    with torch.no_grad():
        for variant, channels in variants.items():
            z = p3.encode_tensor(raw_initial, channel_axis=1)
            metric_by_variant[variant] = {}
            for step in range(1, 20):
                raw_target = torch.from_numpy(snapshots[step]).unsqueeze(0).to(device)
                z_target = p3.encode_tensor(raw_target, channel_axis=1)
                oracle_target = p3.decode_tensor(z_target, channel_axis=1)
                input_physical = p3.decode_tensor(z, channel_axis=1)
                y = model(x=torch.cat((z, shells), dim=1))
                if kind == "projection":
                    used = project_train_envelope(y, {name: envelope["channels"][name]["normalized"] for name in CHANNELS}, channels)
                else:
                    used = reset_channels(y, z_target, channels)
                decoded = processor.decode_prediction(used, apply_evaluation_clamp=True)
                physical = decoded.physical_prediction
                feedback = p3.encode_tensor(physical, channel_axis=1)
                structures = {}
                for name in TARGET_CHANNELS:
                    channel = CHANNELS.index(name)
                    structures[name] = structural_step(
                        input_physical[0, channel].cpu().numpy(),
                        oracle_target[0, channel].cpu().numpy(),
                        physical[0, channel].cpu().numpy(),
                        shell_index,
                    )
                artifacts = artifact_diagnostics(
                    physical, raw_target, raw_target, reference_kind="ground_truth"
                )
                metrics = failure_metrics(
                    physical, structures=structures, artifacts=artifacts
                )
                metric_by_variant[variant][step] = metrics
                baseline = actual_metrics[step]
                reductions = {
                    name: (baseline[name] - value) / max(abs(baseline[name]), 1e-30)
                    for name, value in metrics.items()
                }
                row = {
                    "counterfactual": kind,
                    "variant": variant,
                    "channels": ";".join(channels),
                    "step": step,
                    "diagnostic_teacher_reset": kind == "reset",
                    "normalized_average_relative_l2": channel_relative_average(used, z_target),
                    "raw_average_relative_l2": channel_relative_average(physical, raw_target),
                    "above_rout": float(global_state_norm(used)[0].cpu()) > dissipation.rout,
                    "evaluation_clamp": decoded.clamp_fraction["combined"],
                    "gate_2_count": metrics["gate_2_count"],
                    "gate_3_count": metrics["gate_3_count"],
                    "ripple_count": metrics["ripple_count"],
                    "stripe_count": metrics["stripe_count"],
                    **metrics,
                    **{f"reduction_{name}": value for name, value in reductions.items()},
                }
                for name in TARGET_CHANNELS:
                    structure = structures[name]
                    row[f"{name}_shell_skill"] = structure["shell_transport"]["persistence_relative_skill"]
                    row[f"{name}_radial_skill"] = structure["radial_transport"]["persistence_relative_skill"]
                rows.append(row)
                z = feedback.detach()
    filename = "envelope_projection" if kind == "projection" else "channel_reset"
    payload = {
        "schema_version": f"paper-stage-p-{kind}-v1",
        "steps": 19,
        "counterfactual_only": True,
        "train_only_projection": kind == "projection",
        "diagnostic_teacher_reset": kind == "reset",
        "rows": rows,
    }
    write_json(output / f"{filename}.json", payload)
    write_csv(output / f"{filename}.csv", rows)
    return {"payload": payload, "metrics": metric_by_variant}


def gain_directions(state: torch.Tensor, processor: PaperDataProcessor) -> dict[str, torch.Tensor]:
    directions = {}
    for name in TARGET_CHANNELS:
        value = torch.zeros_like(state)
        value[:, CHANNELS.index(name)] = 1.0
        directions[f"channel_{name}"] = value
    all_direction = state.clone()
    norm = all_direction.square().mean().sqrt()
    if float(norm) == 0:
        all_direction.fill_(1.0)
    else:
        all_direction /= norm
    directions["all_channel_normalized_direction"] = all_direction
    inner_mask = (processor.shells[:, 0] + processor.shells[:, 1]).clamp_max(1.0)
    outer_mask = (processor.shells[:, 6] + processor.shells[:, 7]).clamp_max(1.0)
    directions["inner_shell_mode"] = inner_mask.expand_as(state)
    directions["outer_shell_mode"] = outer_mask.expand_as(state)
    shape = tuple(state.shape[2:])
    for label, values in (
        ("radial_low_mode", radial_mode(shape, "linear_log_r")),
        ("radial_mid_mode", radial_mode(shape, "mid_frequency")),
        ("low_k", directional_mode(shape, "phi", "low_k")),
        ("high_k", directional_mode(shape, "phi", "high_k")),
    ):
        tensor = torch.from_numpy(values).to(device=state.device, dtype=state.dtype)
        directions[label] = tensor.reshape(1, 1, *shape).expand_as(state).clone()
    return directions


def local_gain_audit(
    *,
    model: torch.nn.Module,
    processor: PaperDataProcessor,
    p3,
    states: Mapping[str, torch.Tensor],
    output: Path,
) -> dict[str, Any]:
    shells = processor.shells.expand(1, -1, -1, -1, -1).to(processor.device)

    def f_map(value: torch.Tensor) -> torch.Tensor:
        return model(x=torch.cat((value, shells), dim=1))

    def g_map(value: torch.Tensor) -> torch.Tensor:
        prediction = f_map(value)
        physical = processor.decode_prediction(
            prediction, apply_evaluation_clamp=True
        ).physical_prediction
        return p3.encode_tensor(physical, channel_axis=1)

    rows = []
    for state_name, cpu_state in states.items():
        state = cpu_state.to(processor.device)
        for basis, direction in gain_directions(state, processor).items():
            direction_norm = float(torch.linalg.vector_norm(direction.to(torch.float64)).cpu())
            for epsilon in LOCAL_GAIN_EPSILONS:
                f_result = directional_gain(f_map, state, direction, epsilon=epsilon)
                g_result = directional_gain(g_map, state, direction, epsilon=epsilon)
                row = {
                    "state": state_name,
                    "basis": basis,
                    "epsilon": epsilon,
                    "J_F_gain": f_result["gain"],
                    "J_G_gain": g_result["gain"],
                    "decoder_feedback_contribution": g_result["gain"] / max(f_result["gain"], 1e-30),
                    "used_backward": False,
                }
                inner = (processor.shells[:, 0:1] + processor.shells[:, 1:2]).clamp_max(1.0)
                outer = (processor.shells[:, 6:7] + processor.shells[:, 7:8]).clamp_max(1.0)
                for prefix, response in (
                    ("J_F", f_result["response"]),
                    ("J_G", g_result["response"]),
                ):
                    row[f"{prefix}_inner_shell_gain"] = float(
                        torch.linalg.vector_norm((response * inner).to(torch.float64)).cpu()
                        / max(direction_norm, 1e-30)
                    )
                    row[f"{prefix}_outer_shell_gain"] = float(
                        torch.linalg.vector_norm((response * outer).to(torch.float64)).cpu()
                        / max(direction_norm, 1e-30)
                    )
                    response_profile = response.to(torch.float64).mean(dim=(2, 3))
                    direction_profile = direction.to(torch.float64).mean(dim=(2, 3))
                    row[f"{prefix}_radial_profile_gain"] = float(
                        torch.linalg.vector_norm(response_profile).cpu()
                        / max(float(torch.linalg.vector_norm(direction_profile).cpu()), 1e-30)
                    )
                for channel, name in enumerate(CHANNELS):
                    row[f"J_F_cross_{name}"] = float(
                        torch.linalg.vector_norm(
                            f_result["response"][:, channel].to(torch.float64)
                        ).cpu()
                        / max(direction_norm, 1e-30)
                    )
                    row[f"J_G_cross_{name}"] = float(
                        torch.linalg.vector_norm(
                            g_result["response"][:, channel].to(torch.float64)
                        ).cpu()
                        / max(direction_norm, 1e-30)
                    )
                rows.append(row)
    for row in rows:
        companion = next(
            item
            for item in rows
            if item["state"] == row["state"]
            and item["basis"] == row["basis"]
            and item["epsilon"] != row["epsilon"]
        )
        row["J_F_epsilon_relative_difference"] = abs(row["J_F_gain"] - companion["J_F_gain"]) / max(abs(row["J_F_gain"]), abs(companion["J_F_gain"]), 1e-30)
        row["J_G_epsilon_relative_difference"] = abs(row["J_G_gain"] - companion["J_G_gain"]) / max(abs(row["J_G_gain"]), abs(companion["J_G_gain"]), 1e-30)
        row["epsilon_consistent_20pct"] = row["J_F_epsilon_relative_difference"] <= 0.2 and row["J_G_epsilon_relative_difference"] <= 0.2
    payload = {
        "schema_version": "paper-stage-p-local-gain-v1",
        "not_full_jacobian_spectral_radius": True,
        "parameter_backward": False,
        "epsilons": list(LOCAL_GAIN_EPSILONS),
        "rows": rows,
    }
    write_json(output / "local_gain.json", payload)
    write_csv(output / "local_gain.csv", rows)
    return payload


FAILURE_METRICS = (
    "Bcc2_range",
    "Bcc3_range",
    "vel3_range",
    "total_decoded_range",
    "gate_2_count",
    "gate_3_count",
    "shell_transport_error",
    "radial_transport_error",
    "ripple_count",
    "stripe_count",
)


def channel_driver_output(
    *,
    projection: Mapping[str, Any],
    reset: Mapping[str, Any],
    actual_metrics: Mapping[int, Mapping[str, float]],
    local_gain: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    selected_steps = (1, 3, 5, 10, 19)
    rows = []
    channel_summary = {}
    for channel in TARGET_CHANNELS:
        p_variant = f"D_{channel}"
        r_variant = f"E_{channel}"
        primary_metrics = 0
        weak_metrics = 0
        for metric in FAILURE_METRICS:
            reductions = []
            for step in selected_steps:
                baseline = float(actual_metrics[step][metric])
                projected = float(projection[p_variant][step][metric])
                reset_value = float(reset[r_variant][step][metric])
                best = min(projected, reset_value)
                reduction = (baseline - best) / max(abs(baseline), 1e-30)
                reductions.append(reduction)
                rows.append(
                    {
                        "channel_in": channel,
                        "failure_metric_out": metric,
                        "step": step,
                        "actual": baseline,
                        "projection": projected,
                        "teacher_reset": reset_value,
                        "best_reduction": reduction,
                    }
                )
            primary = sum(value >= 0.5 for value in reductions) >= 2
            weak = sum(value >= 0.2 for value in reductions) >= 2
            primary_metrics += int(primary)
            weak_metrics += int(weak)
            for row in rows[-len(selected_steps):]:
                row["metric_primary_support"] = primary
                row["metric_weak_support"] = weak
        if primary_metrics >= 2:
            support = "supported_primary_driver"
        elif weak_metrics >= 2:
            support = "weak_driver"
        else:
            support = "not_supported"
        basis = f"channel_{channel}"
        gains = [
            row
            for row in local_gain["rows"]
            if row["basis"] == basis and row["epsilon_consistent_20pct"]
        ]
        cross = {
            target: None
            if not gains
            else float(np.median([row[f"J_G_cross_{target}"] for row in gains]))
            for target in TARGET_CHANNELS
        }
        channel_summary[channel] = {
            "support": support,
            "primary_metric_count": primary_metrics,
            "weak_metric_count": weak_metrics,
            "finite_difference_median_J_G_cross_channel_gain": cross,
            "frozen_rule": "at least two selected steps and two failure metrics at >=50%; 20-50% is weak",
        }
    for row in rows:
        row["channel_support"] = channel_summary[row["channel_in"]]["support"]
    payload = {
        "schema_version": "paper-stage-p-channel-driver-matrix-v1",
        "selected_steps": list(selected_steps),
        "thresholds": {"primary": 0.5, "weak": 0.2, "steps": 2, "metric_classes": 2},
        "channels": channel_summary,
        "rows": rows,
    }
    write_json(output / "channel_driver_matrix.json", payload)
    write_csv(output / "channel_driver_matrix.csv", rows)
    return payload


def _evidence_status(predicates: Mapping[str, bool], *, required: int = 3) -> str:
    count = sum(bool(value) for value in predicates.values())
    if count >= required:
        return "supported"
    if count == required - 1:
        return "weakly_supported"
    return "not_supported"


def mechanism_outputs(
    *,
    timeline: Mapping[str, Any],
    teacher: Mapping[str, Any],
    direct: Mapping[str, Any],
    roundtrip: Mapping[str, Any],
    projection: Mapping[str, Any],
    driver: Mapping[str, Any],
    local_gain: Mapping[str, Any],
    actual: Mapping[str, Any],
    output: Path,
) -> dict[str, Any]:
    teacher_target = [
        row for row in teacher["rows"] if row["channel"] in TARGET_CHANNELS
    ]
    teacher_ood_majority = float(
        np.mean([row["fraction_outside_train_min_max"] > 0 for row in teacher_target])
    ) > 0.5
    direct_norm = {
        step: next(
            row["state_norm"]
            for row in direct["rows"]
            if row["step"] == step and row["channel"] == "Bcc2"
        )
        for step in (1, 19)
    }
    direct_growth = direct_norm[19] / max(direct_norm[1], 1e-30)
    actual_norm_19 = float(global_state_norm(actual["outputs"][19])[0])
    all_projection = [
        row for row in projection["rows"] if row["variant"] == "D_all"
    ]
    projection_range_reduction = float(
        np.median(
            [
                row["reduction_total_decoded_range"]
                for row in all_projection
                if row["step"] in (1, 3, 5, 10, 19)
            ]
        )
    )
    target_trace = [
        row for row in actual["trace_rows"] if row["channel"] in TARGET_CHANNELS
    ]
    derivative_extreme = float(
        np.mean([row["physical_decode"]["derivative_over_train_q999"] > 10 for row in target_trace])
    ) > 0.25
    nonlinear_physical_amplification = any(
        row["physical_decode"]["absolute_maximum"]
        > 1.0e6 * max(row["normalized_output"]["absolute_maximum"], 1.0)
        for row in target_trace
    )
    actual_feedback = [
        row["feedback"]["output_feedback_relative_l2"] for row in actual["trace_rows"]
    ]
    roundtrip_actual = [
        row for row in roundtrip["rows"] if row["source"] == "actual_rollout_output"
    ]
    repeated_accumulation = float(
        np.mean(
            [
                row["roundtrip_2_increment_relative_l2"]
                > 0.5 * row["roundtrip_1_relative_l2"]
                and row["roundtrip_2_increment_relative_l2"] > 1e-3
                for row in roundtrip_actual
            ]
        )
    ) > 0.25
    consistent_gains = [row for row in local_gain["rows"] if row["epsilon_consistent_20pct"]]
    median_f_gain = (
        None
        if not consistent_gains
        else float(np.median([row["J_F_gain"] for row in consistent_gains]))
    )
    median_g_over_f = (
        None
        if not consistent_gains
        else float(np.median([row["decoder_feedback_contribution"] for row in consistent_gains]))
    )
    multiple_f_gain = sum(row["J_F_gain"] > 1 for row in consistent_gains) >= 4
    primary_drivers = [
        name
        for name, payload in driver["channels"].items()
        if payload["support"] == "supported_primary_driver"
    ]
    weak_drivers = [
        name
        for name, payload in driver["channels"].items()
        if payload["support"] == "weak_driver"
    ]
    mechanisms = {
        "M1": {
            "name": "NORMALIZED_MODEL_OVERSHOOT",
            "predicates": {
                "teacher_forced_majority_outside_train_minmax": teacher_ood_majority,
                "normalized_direct_growth_gt_2": direct_growth > 2,
                "normalized_ood_precedes_or_equals_Rout": timeline["answers"]["normalized_ood_precedes_or_equals_physical_rout"],
                "train_projection_median_range_reduction_gt_50pct": projection_range_reduction >= 0.5,
            },
        },
        "M2": {
            "name": "P3_DECODE_TAIL_AMPLIFICATION",
            "predicates": {
                "decoder_derivative_gt_10x_train_q999": derivative_extreme,
                "physical_to_normalized_tail_amplification_gt_1e6": nonlinear_physical_amplification,
                "normalized_direct_step19_norm_lt_physical_loop": direct_norm[19] < actual_norm_19,
                "train_projection_median_range_reduction_gt_50pct": projection_range_reduction >= 0.5,
            },
        },
        "M3": {
            "name": "ROUNDTRIP_CLAMP_FEEDBACK",
            "predicates": {
                "median_actual_output_feedback_relative_l2_gt_0_1": float(np.median(actual_feedback)) > 0.1,
                "repeated_H_accumulates": repeated_accumulation,
                "median_J_G_over_J_F_gt_2": median_g_over_f is not None and median_g_over_f > 2,
                "sensitivity_precedes_or_coincides_with_any_clamp": timeline["answers"]["decode_sensitivity_precedes_or_equals_any_channel_clamp"],
            },
        },
        "M4": {
            "name": "RECURRENT_OPERATOR_GAIN",
            "predicates": {
                "normalized_direct_growth_gt_2": direct_growth > 2,
                "median_J_F_gain_gt_1": median_f_gain is not None and median_f_gain > 1,
                "multiple_consistent_J_F_directions_gt_1": multiple_f_gain,
                "growth_without_decoder": direct_norm[19] > direct_norm[1],
            },
        },
        "M5": {
            "name": "CROSS_CHANNEL_FEEDBACK",
            "predicates": {
                "primary_driver_identified": bool(primary_drivers),
                "at_least_weak_driver_identified": bool(primary_drivers or weak_drivers),
                "stable_non_diagonal_gain": any(
                    value is not None and value > 1
                    for input_name, channel in driver["channels"].items()
                    for target, value in channel["finite_difference_median_J_G_cross_channel_gain"].items()
                    if target != input_name
                ),
            },
        },
    }
    for key, payload in mechanisms.items():
        payload["status"] = _evidence_status(
            payload["predicates"], required=2 if key == "M5" else 3
        )
    statuses = {key: payload["status"] for key, payload in mechanisms.items()}
    decision = final_mechanism_decision(statuses)
    evidence = {
        "schema_version": "paper-stage-p-mechanism-evidence-v1",
        "frozen_thresholds_selected_before_audit": True,
        "summary_values": {
            "teacher_ood_majority": teacher_ood_majority,
            "normalized_direct_growth_step19_over_step1": direct_growth,
            "physical_loop_output_norm_step19": actual_norm_19,
            "normalized_direct_norm_step19": direct_norm[19],
            "projection_median_total_range_reduction": projection_range_reduction,
            "local_gain_rows": len(local_gain["rows"]),
            "local_gain_epsilon_consistent_rows": len(consistent_gains),
            "local_gain_inference": (
                "inconclusive_under_frozen_20pct_epsilon_consistency"
                if not consistent_gains
                else "eligible_for_mechanism_inference"
            ),
            "median_J_F_gain": median_f_gain,
            "median_J_G_over_J_F": median_g_over_f,
            "primary_drivers": primary_drivers,
            "weak_drivers": weak_drivers,
        },
        "mechanisms": mechanisms,
        "decision": decision,
    }
    write_json(output / "mechanism_evidence.json", evidence)
    lines = ["# Stage P mechanism evidence", ""]
    for key, payload in mechanisms.items():
        lines.append(f"- {key} `{payload['name']}`: `{payload['status']}`.")
    lines.extend(["", f"Frozen-rule decision: **{decision}**.", ""])
    (output / "mechanism_evidence.md").write_text("\n".join(lines), encoding="utf-8")
    decision_payload = {
        "schema_version": "paper-stage-p-decision-v1",
        "classification": STAGE_P_CLASSIFICATION,
        "decision": decision,
        "mechanism_status": statuses,
        "engineering": {
            "no_training": True,
            "no_backward": True,
            "no_optimizer": True,
            "no_scheduler": True,
            "no_checkpoint_write": True,
            "strict_stage_o_model_reload": True,
        },
        "frozen_history": FROZEN_HISTORY,
        "stage_o_decision_changed": False,
        "next_authorized_action": "proposal_only; follow the decision-specific Stage P rule and change one factor at a time",
    }
    write_json(output / "stage_p_decision.json", decision_payload)
    (output / "stage_p_decision.md").write_text(
        "# Stage P decision\n\n"
        f"## {decision}\n\n"
        "This is a frozen-checkpoint post-hoc attribution result. It does not modify the model, "
        "P3, canonical preprocessing, or Stage O decision C.\n\n"
        "No training or new model is authorized by this result.\n",
        encoding="utf-8",
    )
    return {"evidence": evidence, "decision": decision_payload}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = root / "outputs/paper_reduced100/stage_p"
    output.mkdir(parents=True, exist_ok=True)
    config_path = root / "configs/paper_reduced100/stage_o_localno_p3_plain.yaml"
    config = load_paper_experiment_config(config_path, project_root=root)
    p3 = load_frozen_p3(config)

    manifest, model = provenance_gate(root, config, output)
    envelope, shell_index = build_train_envelope(root, config, p3, output)
    if not torch.cuda.is_available():
        raise RuntimeError("Stage P finite forward audit requires the verified WSL CUDA path")
    device = torch.device("cuda:0")
    model = model.to(device).eval()
    processor = PaperDataProcessor.from_config(config).to(device)
    processor.eval()
    values = config.values
    expected_artifacts = {
        "source_hdf5_checksum": values["provenance"]["dataset"]["sha256"],
        "preprocessing_stats_checksum": config.prior_preprocessing_checksum,
        "training_indices": tuple(range(*values["protocol"]["train_snapshots"])),
        "protocol_name": values["protocol"]["name"],
        "thermal_channel": values["thermal"]["channel"],
    }
    dissipation = PaperDissipativeReference.load(
        config.resolve_path(values["provenance"]["artifacts"]["dissipation"]["path"]),
        **expected_artifacts,
    )

    actual = actual_trajectory(
        root=root,
        config=config,
        model=model,
        processor=processor,
        p3=p3,
        envelope=envelope,
        shell_index=shell_index,
        dissipation=dissipation,
        output=output,
    )
    timeline = timeline_output(actual["timeline_rows"], output)
    teacher = teacher_forced_audit(
        config=config,
        model=model,
        processor=processor,
        p3=p3,
        envelope=envelope,
        shell_index=shell_index,
        dissipation=dissipation,
        output=output,
    )
    direct = normalized_direct_audit(
        config=config,
        model=model,
        processor=processor,
        p3=p3,
        envelope=envelope,
        shell_index=shell_index,
        actual_outputs=actual["outputs"],
        output=output,
    )
    roundtrip = roundtrip_only_audit(
        config=config,
        p3=p3,
        envelope=envelope,
        teacher_outputs=teacher["outputs"],
        actual_outputs=actual["outputs"],
        output=output,
    )
    projection = counterfactual_rollouts(
        kind="projection",
        variants={
            "D_all": tuple(CHANNELS),
            "D_Bcc2": ("Bcc2",),
            "D_Bcc3": ("Bcc3",),
            "D_vel3": ("vel3",),
            "D_Bcc2_Bcc3": ("Bcc2", "Bcc3"),
            "D_targets": tuple(TARGET_CHANNELS),
        },
        config=config,
        model=model,
        processor=processor,
        p3=p3,
        shell_index=shell_index,
        dissipation=dissipation,
        actual_metrics=actual["step_metrics"],
        envelope=envelope,
        output=output,
    )
    reset = counterfactual_rollouts(
        kind="reset",
        variants={
            "E_Bcc2": ("Bcc2",),
            "E_Bcc3": ("Bcc3",),
            "E_vel3": ("vel3",),
            "E_Bcc2_Bcc3": ("Bcc2", "Bcc3"),
            "E_targets": tuple(TARGET_CHANNELS),
        },
        config=config,
        model=model,
        processor=processor,
        p3=p3,
        shell_index=shell_index,
        dissipation=dissipation,
        actual_metrics=actual["step_metrics"],
        envelope=envelope,
        output=output,
    )
    gain_states = {
        "validation_initial_state": actual["inputs"][1],
        "GT_step_1_input": actual["inputs"][1].clone(),
        "GT_step_5_input": actual["inputs"][5],
        "actual_rollout_step_10_input": actual["inputs"][10],
        "actual_rollout_step_19_input": actual["inputs"][19],
        "no_GT_step_25_input": actual["inputs"][25],
    }
    local_gain = local_gain_audit(
        model=model,
        processor=processor,
        p3=p3,
        states=gain_states,
        output=output,
    )
    driver = channel_driver_output(
        projection=projection["metrics"],
        reset=reset["metrics"],
        actual_metrics=actual["step_metrics"],
        local_gain=local_gain,
        output=output,
    )
    decision = mechanism_outputs(
        timeline=timeline,
        teacher=teacher["payload"],
        direct=direct["payload"],
        roundtrip=roundtrip,
        projection=projection["payload"],
        driver=driver,
        local_gain=local_gain,
        actual=actual,
        output=output,
    )
    manifest["device"] = {
        "name": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    manifest["diagnostic_contract"] = {
        "no_training": True,
        "no_backward": True,
        "no_optimizer": True,
        "no_scheduler": True,
        "no_checkpoint_write": True,
        "physical_steps": 100,
        "teacher_forced_steps": 19,
        "normalized_direct_steps": 19,
        "counterfactual_steps": 19,
        "roundtrip_maximum_applications": 2,
    }
    manifest["stage_p_decision"] = decision["decision"]["decision"]
    write_json(output / "run_manifest.json", manifest)
    print(json.dumps({"decision": decision["decision"]["decision"]}, indent=2))


if __name__ == "__main__":
    main()
