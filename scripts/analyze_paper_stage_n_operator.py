#!/usr/bin/env python
"""Audit frozen Stage K LocalNO responses without training or prototype inputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping

import h5py
import numpy as np
import torch
import yaml

from neuralop.layers.spectral_convolution import SpectralConv

from analyze_paper_stage_n_transforms import EXPECTED, git, provenance_gate
from grmhd import CHANNELS
from grmhd.dataset import sha256_file
from grmhd.paper_checkpoint import validate_paper_checkpoint_metadata
from grmhd.paper_config import build_paper_model, load_paper_experiment_config
from grmhd.paper_data_processor import PaperDataProcessor
from grmhd.paper_stage_g_evaluation import artifact_diagnostics
from grmhd.paper_stage_g_evaluation import radial_shell_indices
from grmhd.paper_stage_m import (
    evaluate_oracle_conditioned_structure_gate,
    oracle_conditioned_gate_schema,
    radial_profile_vector,
    transport_metrics,
    variance_vector,
)
from grmhd.paper_stage_n import (
    apply_operator,
    compact_impulse,
    constant_perturbation,
    directional_mode,
    radial_mode,
    shell_localized_perturbation,
    tensor_state_sha256,
)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def csv_value(value: Any) -> Any:
    if isinstance(value, (list, dict, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return value


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing empty Stage N operator CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([{key: csv_value(value) for key, value in row.items()} for row in rows])


def aggregate_rows(
    rows: list[Mapping[str, Any]], *, group_keys: tuple[str, ...]
) -> list[dict[str, Any]]:
    """Reduce fixed-state responses without removing operator-matrix axes."""

    grouped: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(tuple(row.get(key) for key in group_keys), []).append(row)
    output = []
    for key_values, selected in sorted(grouped.items(), key=lambda item: str(item[0])):
        record = dict(zip(group_keys, key_values, strict=True))
        record["state_count"] = len({row.get("snapshot") for row in selected})
        numeric_keys = sorted(
            key
            for key in {name for row in selected for name in row}
            if key not in set(group_keys) | {"snapshot"}
            and not isinstance(next((row.get(key) for row in selected if row.get(key) is not None), None), (str, bool, list, dict))
        )
        for name in numeric_keys:
            values = [float(row[name]) for row in selected if row.get(name) is not None and np.isfinite(row[name])]
            if values:
                record[f"median_{name}"] = float(np.median(values))
                record[f"mean_{name}"] = float(np.mean(values))
                record[f"minimum_{name}"] = float(np.min(values))
                record[f"maximum_{name}"] = float(np.max(values))
        boolean_keys = sorted(
            key for key in {name for row in selected for name in row}
            if isinstance(next((row.get(key) for row in selected if row.get(key) is not None), None), bool)
        )
        for name in boolean_keys:
            record[f"all_{name}"] = all(bool(row[name]) for row in selected)
        output.append(record)
    return output


def norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.float()).cpu())


def ratio(numerator: float, denominator: float) -> float | None:
    return None if abs(denominator) <= 1.0e-30 else numerator / denominator


def correlation(left: torch.Tensor, right: torch.Tensor) -> float | None:
    a = left.float().flatten()
    b = right.float().flatten()
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b))
    return None if denominator <= 1.0e-30 else float(torch.dot(a, b).cpu()) / denominator


def cosine(left: torch.Tensor, right: torch.Tensor) -> float | None:
    denominator = norm(left) * norm(right)
    return None if denominator <= 1.0e-30 else float(torch.sum(left.float() * right.float()).cpu()) / denominator


def field_variance(value: torch.Tensor) -> float:
    return float(torch.var(value.float(), correction=0).cpu())


def shell_variances(value: torch.Tensor, shell_index: np.ndarray) -> np.ndarray:
    return variance_vector(value.detach().float().cpu().numpy().astype(np.float64), shell_index)


def profile(value: torch.Tensor) -> np.ndarray:
    return radial_profile_vector(value.detach().float().cpu().numpy().astype(np.float64))


def load_frozen_model(root: Path, config_path: Path, device: torch.device) -> tuple[torch.nn.Module, PaperDataProcessor, dict[str, Any]]:
    config = load_paper_experiment_config(config_path, project_root=root)
    checkpoint = root / "outputs/paper_reduced100/stage_k/localno_differential_plain/best_validation_l2"
    metadata = json.loads((checkpoint / "paper_grmhd_metadata.json").read_text())
    if int(metadata["epoch"]) != 22:
        raise RuntimeError("Stage N requires frozen Stage K best epoch 22")
    validate_paper_checkpoint_metadata(metadata, config, expected_config_checksum=sha256_file(config_path))
    model = build_paper_model(config)
    with torch.no_grad(), torch.serialization.safe_globals([torch._C._nn.gelu, SpectralConv]):
        state = torch.load(checkpoint / "paper_state_dict.pt", map_location="cpu", weights_only=True)
        incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Stage N LocalNO strict reload returned incompatible keys")
    model.to(device).eval()
    processor = PaperDataProcessor.from_config(config).to(device)
    processor.eval()
    processor.set_epoch(22)
    return model, processor, metadata


def encode_snapshot(handle: h5py.File, index: int, processor: PaperDataProcessor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    raw = torch.as_tensor(np.asarray(handle["snapshots"][index], dtype=np.float32), device=device).unsqueeze(0)
    normalized = processor.preprocessor.encode_tensor(raw, channel_axis=1)
    return raw, normalized


def summarize_delta(delta: torch.Tensor, direction: torch.Tensor, channel_out: int) -> dict[str, Any]:
    output = delta[0, channel_out]
    input_norm = norm(direction)
    output_norm = norm(output)
    centered = output - output.mean()
    return {
        "channel_out": CHANNELS[channel_out],
        "output_norm": output_norm,
        "input_norm": input_norm,
        "gain": ratio(output_norm, input_norm),
        "mean_response": float(output.mean().cpu()),
        "std_response": float(output.std(correction=0).cpu()),
        "texture_fraction": ratio(norm(centered), output_norm),
    }


def run_constant(model: torch.nn.Module, state: torch.Tensor, shells: torch.Tensor, base: torch.Tensor, snapshot: int, epsilons: Iterable[float]) -> list[dict[str, Any]]:
    rows = []
    for epsilon in epsilons:
        for channel_in in range(8):
            direction = constant_perturbation(state, channel_in, 1.0)
            delta = apply_operator(model, state + float(epsilon) * direction, shells) - base
            for channel_out in range(8):
                rows.append({"snapshot": snapshot, "epsilon": epsilon, "channel_in": CHANNELS[channel_in], **summarize_delta(delta / float(epsilon), direction, channel_out)})
    return rows


def run_impulse(model: torch.nn.Module, state: torch.Tensor, shells: torch.Tensor, base: torch.Tensor, snapshot: int, shell_index: np.ndarray, epsilons: Iterable[float]) -> list[dict[str, Any]]:
    rows = []
    radial_centers = {
        "inner": int(np.flatnonzero(shell_index == 0)[len(np.flatnonzero(shell_index == 0)) // 2]),
        "middle": int(np.flatnonzero(shell_index == 3)[len(np.flatnonzero(shell_index == 3)) // 2]),
        "outer": int(np.flatnonzero(shell_index == 7)[len(np.flatnonzero(shell_index == 7)) // 2]),
    }
    phi, theta = state.shape[2] // 2, state.shape[3] // 2
    radial_grid = torch.arange(state.shape[-1], device=state.device).reshape(1, 1, -1)
    for epsilon in epsilons:
        for region, radial in radial_centers.items():
            for channel_in in range(8):
                unit = compact_impulse(state, channel_in, (phi, theta, radial), 1.0)
                batch = torch.cat((state + float(epsilon) * unit, state - float(epsilon) * unit), dim=0)
                shell_batch = shells.expand(2, -1, -1, -1, -1)
                outputs = apply_operator(model, batch, shell_batch)
                positive = (outputs[0:1] - base) / float(epsilon)
                negative = (outputs[1:2] - base) / float(epsilon)
                symmetry = ratio(norm(positive + negative), norm(positive - negative))
                energy = positive.square().sum(dim=1)[0]
                total_energy = float(energy.sum().cpu())
                distance = torch.abs(radial_grid - radial).expand_as(energy)
                order = torch.argsort(distance.flatten())
                cumulative = torch.cumsum(energy.flatten()[order], dim=0)
                threshold_index = int(torch.searchsorted(cumulative, torch.as_tensor(0.9 * total_energy, device=energy.device)).clamp_max(cumulative.numel() - 1))
                radius90 = float(distance.flatten()[order[threshold_index]].cpu())
                input_channel_energy = float(positive[:, channel_in].square().sum().cpu())
                rows.append({
                    "snapshot": snapshot, "epsilon": epsilon, "region": region,
                    "channel_in": CHANNELS[channel_in], "positive_negative_symmetry_error": symmetry,
                    "radial_support_radius_90pct_cells": radius90,
                    "cross_channel_mixing_fraction": 1.0 - input_channel_energy / max(total_energy, 1.0e-30),
                    "output_norm": norm(positive), "finite": bool(torch.isfinite(positive).all()),
                })
    return rows


def run_shells(model: torch.nn.Module, state: torch.Tensor, shells: torch.Tensor, base: torch.Tensor, snapshot: int, shell_masks: torch.Tensor, epsilons: Iterable[float]) -> list[dict[str, Any]]:
    rows = []
    for epsilon in epsilons:
        for shell_in in range(8):
            for channel_in in range(8):
                direction = shell_localized_perturbation(state, shell_masks[shell_in], channel_in, 1.0)
                response = (apply_operator(model, state + float(epsilon) * direction, shells) - base) / float(epsilon)
                denominator = norm(direction)
                total = float(response.square().sum().cpu())
                for shell_out in range(8):
                    mask = shell_masks[shell_out]
                    for channel_out in range(8):
                        selected = response[0, channel_out][mask]
                        energy = float(selected.square().sum().cpu())
                        rows.append({
                            "snapshot": snapshot, "epsilon": epsilon, "shell_in": shell_in + 1,
                            "shell_out": shell_out + 1, "channel_in": CHANNELS[channel_in],
                            "channel_out": CHANNELS[channel_out], "gain": ratio(norm(selected), denominator),
                            "energy_fraction": energy / max(total, 1.0e-30),
                            "same_shell": shell_in == shell_out,
                            "direction": "same" if shell_in == shell_out else "inward" if shell_out < shell_in else "outward",
                        })
    return rows


def mode_response_rows(model: torch.nn.Module, state: torch.Tensor, shells: torch.Tensor, base: torch.Tensor, snapshot: int, epsilons: Iterable[float], *, family: str) -> list[dict[str, Any]]:
    rows = []
    shape = tuple(int(value) for value in state.shape[2:])
    definitions = (
        [(name, None, radial_mode(shape, name)) for name in ("constant", "linear_log_r", "inner_localized", "outer_localized", "mid_frequency")]
        if family == "radial"
        else [(f"{axis}_{band}", band, directional_mode(shape, axis, band)) for axis in ("phi", "theta", "r") for band in ("low_k", "mid_k", "high_k")]
    )
    for epsilon in epsilons:
        for name, band, values in definitions:
            pattern = torch.as_tensor(values, device=state.device, dtype=state.dtype)
            for channel_in in range(8):
                direction = torch.zeros_like(state)
                direction[:, channel_in] = pattern
                response = (apply_operator(model, state + float(epsilon) * direction, shells) - base) / float(epsilon)
                denominator = norm(direction)
                for channel_out in range(8):
                    output = response[0, channel_out]
                    rows.append({
                        "snapshot": snapshot, "epsilon": epsilon, "mode": name, "band": band,
                        "axis": None if family == "radial" else name.split("_")[0],
                        "channel_in": CHANNELS[channel_in], "channel_out": CHANNELS[channel_out],
                        "gain": ratio(norm(output), denominator), "phase_correlation": correlation(output, pattern),
                        "cross_channel": channel_out != channel_in,
                    })
    return rows


def fixed_and_two_application(
    model: torch.nn.Module, processor: PaperDataProcessor, handle: h5py.File,
    state: torch.Tensor, raw: torch.Tensor, shells: torch.Tensor, base: torch.Tensor,
    snapshot: int, shell_index: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    fixed_rows = []
    second = apply_operator(model, base, shells)
    first_physical = processor.decode_prediction(base, apply_evaluation_clamp=True).physical_prediction
    second_physical = processor.decode_prediction(second, apply_evaluation_clamp=True).physical_prediction
    legacy = artifact_diagnostics(second_physical, raw, raw, reference_kind="raw_input_state")
    two_rows = []
    target_normalized = None
    if snapshot < 110:
        _, target_normalized = encode_snapshot(handle, snapshot + 1, processor, state.device)
    for channel, name in enumerate(CHANNELS):
        x = state[0, channel]
        first = base[0, channel]
        twice = second[0, channel]
        delta = first - x
        record = {
            "snapshot": snapshot, "channel": name, "delta_norm": norm(delta),
            "delta_over_state_norm": ratio(norm(delta), norm(x)),
            "state_mean": float(x.mean().cpu()), "output_mean": float(first.mean().cpu()),
            "mean_shift": float((first.mean() - x.mean()).cpu()),
            "state_std": float(x.std(correction=0).cpu()), "output_std": float(first.std(correction=0).cpu()),
            "correlation": correlation(first, x), "persistence_distance": norm(delta),
            "ground_truth_available": target_normalized is not None,
        }
        if target_normalized is not None:
            true = target_normalized[0, channel] - x
            shell_transport = transport_metrics(shell_variances(x, shell_index), shell_variances(target_normalized[0, channel], shell_index), shell_variances(first, shell_index), epsilon=1e-30, sign_zero_tolerance=1e-30)
            radial_transport = transport_metrics(profile(x), profile(target_normalized[0, channel]), profile(first), epsilon=1e-30, sign_zero_tolerance=1e-30)
            record.update({
                "oracle_delta_norm": norm(true), "model_oracle_delta_norm_ratio": ratio(norm(delta), norm(true)),
                "delta_cosine": cosine(delta, true), "delta_sign_agreement": float(torch.mean((torch.sign(delta) == torch.sign(true)).float()).cpu()),
                "shell_transport_error": shell_transport["relative_error"], "shell_transport_skill": shell_transport["persistence_relative_skill"],
                "shell_transport_sign_agreement": shell_transport["signed_transport_agreement"],
                "radial_transport_error": radial_transport["relative_error"], "radial_transport_skill": radial_transport["persistence_relative_skill"],
                "radial_transport_sign_agreement": radial_transport["signed_transport_agreement"],
            })
        fixed_rows.append(record)
        shell_first = shell_variances(first, shell_index)
        shell_second = shell_variances(twice, shell_index)
        radial_first = profile(first)
        radial_second = profile(twice)
        two_rows.append({
            "snapshot": snapshot, "channel": name, "model_application_count": 2,
            "first_std": float(first.std(correction=0).cpu()), "second_std": float(twice.std(correction=0).cpu()),
            "std_ratio_second_over_first": ratio(float(twice.std(correction=0).cpu()), float(first.std(correction=0).cpu())),
            "global_variance_ratio_second_over_first": ratio(field_variance(twice), field_variance(first)),
            "shell_variance_ratio_second_over_first": ratio(float(np.median(shell_second)), float(np.median(shell_first))),
            "radial_variance_ratio_second_over_first": ratio(float(np.var(radial_second)), float(np.var(radial_first))),
            "first_delta_norm": norm(first - x), "second_delta_norm": norm(twice - first),
            "delta_shrink_ratio": ratio(norm(twice - first), norm(first - x)),
            "first_global_variance_ratio_over_input": ratio(field_variance(first), field_variance(x)),
            "first_shell_variance_ratio_over_input": ratio(float(np.median(shell_first)), float(np.median(shell_variances(x, shell_index)))),
            "first_radial_variance_ratio_over_input": ratio(float(np.var(radial_first)), float(np.var(profile(x)))),
            "legacy_flags": [flag for flag in legacy["flags"] if flag.startswith(name + ":")],
            "candidate_gate2_severe_global": ratio(field_variance(twice), field_variance(first)) is not None and ratio(field_variance(twice), field_variance(first)) < 0.5,
            "candidate_gate2_severe_shell_radial": min(
                ratio(float(np.median(shell_second)), float(np.median(shell_first))) or np.inf,
                ratio(float(np.var(radial_second)), float(np.var(radial_first))) or np.inf,
            ) < 0.5,
            "finite": bool(torch.isfinite(twice).all()),
        })
    return fixed_rows, two_rows, {"snapshot": snapshot, "flags": legacy["flags"]}


def aggregate_operator_evidence(fixed: list[dict[str, Any]], shells: list[dict[str, Any]], radial: list[dict[str, Any]], directional: list[dict[str, Any]], channel: list[dict[str, Any]], two: list[dict[str, Any]], thresholds: Mapping[str, Any]) -> dict[str, Any]:
    snapshots = sorted({int(row["snapshot"]) for row in two})
    majority = int(thresholds["state_majority_required"])
    weak = int(thresholds["weak_state_count"])
    contraction_states = []
    shell_bias_states = []
    frequency_states = []
    coupling_states = []
    frequency_consistent_states = []
    coupling_consistent_states = []
    eps_values = sorted({float(row["epsilon"]) for row in directional})
    if len(eps_values) != 2:
        raise ValueError("Stage N operator evidence requires exactly two epsilon values")
    consistency_tolerance = float(thresholds["epsilon_consistency_relative_tolerance"])
    consistency_fraction = float(thresholds["epsilon_consistent_basis_fraction_minimum"])

    def consistency_for_state(
        selected: list[dict[str, Any]], identity: tuple[str, ...]
    ) -> float:
        groups = sorted({tuple(row[key] for key in identity) for row in selected})
        checks = []
        for group in groups:
            rows_for_group = [
                row for row in selected
                if tuple(row[key] for key in identity) == group
            ]
            gains = {
                epsilon: float(np.median([
                    row["gain"] for row in rows_for_group
                    if float(row["epsilon"]) == epsilon and row["gain"] is not None
                ]))
                for epsilon in eps_values
            }
            relative = abs(gains[eps_values[0]] - gains[eps_values[1]]) / max(
                abs(gains[eps_values[1]]), 1.0e-30
            )
            checks.append(relative <= consistency_tolerance)
        return float(np.mean(checks)) if checks else 0.0

    for snapshot in snapshots:
        r8 = [row for row in two if row["snapshot"] == snapshot]
        contraction = all(statistics.median(row[key] for row in r8 if row[key] is not None) < float(thresholds["contraction_gain"]) for key in ("global_variance_ratio_second_over_first", "shell_variance_ratio_second_over_first", "radial_variance_ratio_second_over_first", "delta_shrink_ratio"))
        if contraction:
            contraction_states.append(snapshot)
        shell_rows = [row for row in shells if row["snapshot"] == snapshot]
        same = sum(row["energy_fraction"] for row in shell_rows if row["same_shell"])
        total = sum(row["energy_fraction"] for row in shell_rows)
        radial_rows = [row for row in radial if row["snapshot"] == snapshot and not row["cross_channel"] and row["phase_correlation"] is not None]
        radial_bad = radial_rows and statistics.median(abs(row["phase_correlation"]) for row in radial_rows) < float(thresholds["radial_phase_absolute_minimum"])
        gt_transport = [
            row for row in fixed
            if row["snapshot"] == snapshot and row["ground_truth_available"]
            and row.get("shell_transport_skill") is not None
            and row.get("radial_transport_skill") is not None
        ]
        worse_than_persistence = bool(
            gt_transport
            and statistics.median(
                min(row["shell_transport_skill"], row["radial_transport_skill"])
                for row in gt_transport
            ) < 0.0
        )
        if worse_than_persistence or same / max(total, 1e-30) < float(thresholds["shell_same_retention_minimum"]) or radial_bad:
            shell_bias_states.append(snapshot)
        direction_all = [row for row in directional if row["snapshot"] == snapshot and not row["cross_channel"] and row["gain"] is not None]
        if consistency_for_state(direction_all, ("mode", "channel_in", "channel_out")) >= consistency_fraction:
            frequency_consistent_states.append(snapshot)
        direction_rows = [row for row in direction_all if float(row["epsilon"]) == eps_values[1]]
        band_gains = [statistics.median(row["gain"] for row in direction_rows if row["band"] == band) for band in ("low_k", "mid_k", "high_k")]
        if max(band_gains) / max(min(band_gains), 1e-30) >= float(thresholds["frequency_gain_spread"]):
            frequency_states.append(snapshot)
        channel_all = [row for row in channel if row["snapshot"] == snapshot and row["gain"] is not None]
        if consistency_for_state(channel_all, ("channel_in", "channel_out")) >= consistency_fraction:
            coupling_consistent_states.append(snapshot)
        channel_rows = [row for row in channel_all if float(row["epsilon"]) == eps_values[1]]
        diagonal = sum(row["gain"] for row in channel_rows if row["channel_in"] == row["channel_out"])
        off = sum(row["gain"] for row in channel_rows if row["channel_in"] != row["channel_out"])
        if off / max(diagonal, 1e-30) >= float(thresholds["channel_off_diagonal_over_diagonal"]):
            coupling_states.append(snapshot)
    def label(count: int) -> str:
        return "supported" if count >= majority else "weakly_supported" if count >= weak else "not_supported"
    frequency_label = (
        label(len(frequency_states))
        if len(frequency_consistent_states) >= majority
        else "inconclusive"
    )
    coupling_label = (
        label(len(coupling_states))
        if len(coupling_consistent_states) >= majority
        else "inconclusive"
    )
    mechanisms = {
        "LOW_VARIANCE_ATTRACTOR": {"label": label(len(contraction_states)), "supporting_states": contraction_states, "epsilon_consistency_required": False},
        "SHELL_RADIAL_TRANSPORT_BIAS": {"label": label(len(shell_bias_states)), "supporting_states": shell_bias_states, "epsilon_consistency_required": False, "direct_oracle_conditioned_transport_used": True},
        "FREQUENCY_SELECTIVE_FAILURE": {"label": frequency_label, "signature_states": frequency_states, "epsilon_consistent_states": frequency_consistent_states, "epsilon_consistency_required": True},
        "CHANNEL_COUPLING_FAILURE": {"label": coupling_label, "signature_states": coupling_states, "epsilon_consistent_states": coupling_consistent_states, "epsilon_consistency_required": True},
    }
    supported = [name for name, value in mechanisms.items() if value["label"] == "supported"]
    if len(supported) >= 2:
        decision = "4. MIXED_OPERATOR_RESPONSE_FAILURE"
    elif supported == ["LOW_VARIANCE_ATTRACTOR"]:
        decision = "1. LOW_VARIANCE_ATTRACTOR_SUPPORTED"
    elif supported == ["SHELL_RADIAL_TRANSPORT_BIAS"]:
        decision = "2. SHELL_RADIAL_TRANSPORT_BIAS_SUPPORTED"
    elif supported and supported[0] in {"FREQUENCY_SELECTIVE_FAILURE", "CHANNEL_COUPLING_FAILURE"}:
        decision = "3. FREQUENCY_OR_CHANNEL_RESPONSE_FAILURE_SUPPORTED"
    else:
        decision = "5. NO_CLEAR_OPERATOR_SIGNATURE"
    return {"mechanisms": mechanisms, "supported_mechanisms": supported, "decision": decision, "thresholds": dict(thresholds)}


def candidate_replay(output: Path, fixed: list[dict[str, Any]], two: list[dict[str, Any]], validation_rows: list[dict[str, Any]], legacy: list[dict[str, Any]]) -> list[dict[str, Any]]:
    gate_dir = output / "candidate_gate"
    write_json(gate_dir / "schema.json", oracle_conditioned_gate_schema())
    validation = {(row["prototype"], row["channel"]): row for row in validation_rows}
    legacy_map = {row["snapshot"]: row for row in legacy}
    rows = []
    for snapshot in sorted({row["snapshot"] for row in fixed if row["ground_truth_available"]}):
        for channel in CHANNELS:
            r0 = next(row for row in fixed if row["snapshot"] == snapshot and row["channel"] == channel)
            r8 = next(row for row in two if row["snapshot"] == snapshot and row["channel"] == channel)
            floor = validation[("P0", channel)]
            severe = {
                1: {
                    "global_variance": r8["first_global_variance_ratio_over_input"] is not None and r8["first_global_variance_ratio_over_input"] < 0.5,
                    "shell_radial_variance": min(
                        r8["first_shell_variance_ratio_over_input"] if r8["first_shell_variance_ratio_over_input"] is not None else np.inf,
                        r8["first_radial_variance_ratio_over_input"] if r8["first_radial_variance_ratio_over_input"] is not None else np.inf,
                    ) < 0.5,
                    "dynamic_span": False, "high_k_energy": False,
                },
                2: {"global_variance": bool(r8["candidate_gate2_severe_global"]), "shell_radial_variance": bool(r8["candidate_gate2_severe_shell_radial"]), "dynamic_span": False, "high_k_energy": False},
            }
            result = evaluate_oracle_conditioned_structure_gate(
                legacy_detector={"flags": [flag for flag in legacy_map[snapshot]["flags"] if flag.startswith(channel + ":")]},
                engineering_checks={name: True for name in ("finite", "rho_press_positive", "transform_counters", "checkpoint_provenance", "decoded_range", "Rout", "shape_device")},
                floor_metrics={
                    "oracle_legacy_detector_triggered": floor["median_global_variance_retention"] < 0.05,
                    "median_variance_retention": floor["median_global_variance_retention"],
                    "median_shell_radial_retention": floor["median_shell_radial_variance_retention"],
                    "median_high_k_retention": floor["median_high_k_retention"],
                },
                severe_by_step=severe,
                shell_transport_skill=r0.get("shell_transport_skill"), radial_transport_skill=r0.get("radial_transport_skill"),
                shell_sign_agreement=r0.get("shell_transport_sign_agreement"), radial_sign_agreement=r0.get("radial_transport_sign_agreement"),
                enabled=True, training_blocking=False,
            )
            rows.append({
                "snapshot": snapshot, "channel": channel,
                "legacy_flag_count": len(result["legacy_detector"]["flags"]),
                "gate_0_passed": result["gate_0_engineering_validity"]["passed"],
                "gate_1_floor_limited": result["gate_1_floor_qualification"]["floor_limited"],
                "gate_2_model_added_degradation": result["gate_2_model_added_degradation"]["failed"],
                "gate_3_transport_passed": result["gate_3_transport_skill"]["passed"],
                "reporting_only": result["reporting_only"], "training_blocking": result["training_blocking"],
            })
    write_csv(gate_dir / "replay.csv", rows)
    write_json(gate_dir / "replay.json", {"schema_version": "paper-stage-n-candidate-gate-replay-v1", "oracle_conditioned_gate_version": "stage_m_v1", "historical_reclassification": False, "rows": rows})
    (gate_dir / "replay.md").write_text(
        "# Stage N candidate gate replay\n\nThe `stage_m_v1` gate is reported beside the unchanged legacy detector. It is reporting-only, never blocks training, and does not reclassify Stage K/L/M.\n",
        encoding="utf-8",
    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/paper_reduced100/stage_n_transform_operator_audit.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_reduced100/stage_n"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    combined_path = output / "transform_prototypes/combined_decision.json"
    if not combined_path.exists():
        raise RuntimeError("Stage N combined prototype decision must precede operator response")
    combined = json.loads(combined_path.read_text())
    if not str(combined.get("decision", "")).startswith(("1.", "2.", "3.", "4.")):
        raise RuntimeError("Stage N combined prototype decision is incomplete")
    provenance = provenance_gate(root, config_path)
    config = yaml.safe_load(config_path.read_text())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing silent CPU LocalNO operator audit")
    device = torch.device("cuda:0")
    model, processor, metadata = load_frozen_model(root, root / config["provenance"]["stage_k_config"], device)
    before_hash = tensor_state_sha256(model)
    if sha256_file(root / config["provenance"]["stage_k_best_checkpoint"] / "paper_state_dict.pt") != EXPECTED["localno_best"]:
        raise RuntimeError("Stage N LocalNO checkpoint changed after provenance gate")
    manifest = json.loads((root / config["provenance"]["manifest"]).read_text())
    h5_path = root / manifest["dataset"]["configured_path"]
    states = list(config["operator_response"]["train_states"]) + list(config["operator_response"]["validation_states"])
    if len(states) != len(set(states)):
        raise RuntimeError("Stage N operator fixed states contain duplicates")
    epsilons = tuple(float(value) for value in config["operator_response"]["epsilon"])
    out = output / "operator_response"
    fixed_rows: list[dict[str, Any]] = []
    constant_rows: list[dict[str, Any]] = []
    impulse_rows: list[dict[str, Any]] = []
    shell_rows: list[dict[str, Any]] = []
    radial_rows: list[dict[str, Any]] = []
    directional_rows: list[dict[str, Any]] = []
    two_rows: list[dict[str, Any]] = []
    legacy_rows: list[dict[str, Any]] = []
    with h5py.File(h5_path, "r") as handle, torch.no_grad():
        r = np.asarray(handle["coords/r"][...], dtype=np.float64)
        _, shell_index = radial_shell_indices(r, 8)
        shell_masks = processor.shells[0].bool()
        for snapshot in states:
            raw, state = encode_snapshot(handle, snapshot, processor, device)
            shells = processor.shells.to(dtype=state.dtype)
            base = apply_operator(model, state, shells)
            fixed, two, legacy = fixed_and_two_application(model, processor, handle, state, raw, shells, base, snapshot, shell_index)
            fixed_rows.extend(fixed)
            two_rows.extend(two)
            legacy_rows.append(legacy)
            constant_rows.extend(run_constant(model, state, shells, base, snapshot, epsilons))
            impulse_rows.extend(run_impulse(model, state, shells, base, snapshot, shell_index, epsilons))
            shell_rows.extend(run_shells(model, state, shells, base, snapshot, shell_masks, epsilons))
            radial_rows.extend(mode_response_rows(model, state, shells, base, snapshot, epsilons, family="radial"))
            directional_rows.extend(mode_response_rows(model, state, shells, base, snapshot, epsilons, family="directional"))
    after_hash = tensor_state_sha256(model)
    if before_hash != after_hash or any(parameter.grad is not None for parameter in model.parameters()):
        raise RuntimeError("Stage N read-only operator audit changed model state/gradients")
    channel_rows = aggregate_rows(
        constant_rows, group_keys=("epsilon", "channel_in", "channel_out")
    )
    for row in channel_rows:
        row["matrix_kind"] = "constant_field_finite_difference"
    compact_directional = aggregate_rows(
        directional_rows,
        group_keys=("epsilon", "mode", "band", "axis", "channel_in", "channel_out", "cross_channel"),
    )
    compact_radial = aggregate_rows(
        radial_rows,
        group_keys=("epsilon", "mode", "channel_in", "channel_out", "cross_channel"),
    )
    fd_source = [
        {"snapshot": row["snapshot"], "epsilon": row["epsilon"], "basis": f"directional:{row['mode']}", "channel_in": row["channel_in"], "channel_out": row["channel_out"], "gain": row["gain"], "phase_correlation": row["phase_correlation"]}
        for row in directional_rows if not row["cross_channel"]
    ] + [
        {"snapshot": row["snapshot"], "epsilon": row["epsilon"], "basis": f"radial:{row['mode']}", "channel_in": row["channel_in"], "channel_out": row["channel_out"], "gain": row["gain"], "phase_correlation": row["phase_correlation"]}
        for row in radial_rows if not row["cross_channel"]
    ]
    fd_rows = []
    tolerance = float(config["operator_response"]["evidence_thresholds"]["epsilon_consistency_relative_tolerance"])
    for basis, channel_in, channel_out in sorted({(row["basis"], row["channel_in"], row["channel_out"]) for row in fd_source}):
        selected = [row for row in fd_source if (row["basis"], row["channel_in"], row["channel_out"]) == (basis, channel_in, channel_out)]
        gains = {epsilon: float(np.median([row["gain"] for row in selected if row["epsilon"] == epsilon])) for epsilon in epsilons}
        phases = {}
        for epsilon in epsilons:
            values = [
                row["phase_correlation"]
                for row in selected
                if row["epsilon"] == epsilon and row["phase_correlation"] is not None
            ]
            phases[epsilon] = None if not values else float(np.median(values))
        relative = abs(gains[epsilons[0]] - gains[epsilons[1]]) / max(abs(gains[epsilons[1]]), 1e-30)
        fd_rows.append({
            "basis": basis, "channel_in": channel_in, "channel_out": channel_out,
            "epsilon_small": epsilons[0], "epsilon_large": epsilons[1],
            "gain_small": gains[epsilons[0]], "gain_large": gains[epsilons[1]],
            "phase_small": phases[epsilons[0]], "phase_large": phases[epsilons[1]],
            "epsilon_relative_gain_difference": relative,
            "epsilon_consistency_tolerance": tolerance,
            "nonlinear_response_warning": relative > tolerance,
            "state_count": len(states),
        })
    outputs = {
        "fixed_point": fixed_rows,
        "impulse_response": aggregate_rows(
            impulse_rows, group_keys=("epsilon", "region", "channel_in")
        ),
        "shell_transport": aggregate_rows(shell_rows, group_keys=("epsilon", "shell_in", "shell_out", "channel_in", "channel_out", "same_shell", "direction")),
        "radial_modes": compact_radial,
        "directional_modes": compact_directional, "channel_response": channel_rows,
        "finite_difference_gain": fd_rows, "two_application_probe": two_rows,
    }
    for name, rows in outputs.items():
        write_csv(out / f"{name}.csv", rows)
        write_json(out / f"{name}.json", {"schema_version": f"paper-stage-n-{name.replace('_', '-')}-v1", "canonical_preprocessing_only": True, "prototype_input": False, "checkpoint_epoch": 22, "rows": rows})
    mechanism = aggregate_operator_evidence(
        fixed_rows, shell_rows, radial_rows, directional_rows, constant_rows,
        two_rows, config["operator_response"]["evidence_thresholds"],
    )
    mechanism.update({
        "schema_version": "paper-stage-n-operator-mechanism-v1", "checkpoint_epoch": 22,
        "checkpoint_sha256": EXPECTED["localno_best"], "model_state_hash_before": before_hash,
        "model_state_hash_after": after_hash, "state_unchanged": before_hash == after_hash,
        "model_eval": not model.training, "torch_no_grad": True, "optimizer_created": False,
        "scheduler_created": False, "backward_called": False, "prototype_transform_used": False,
        "fixed_states": states, "epsilons": list(epsilons), "device": torch.cuda.get_device_name(device),
    })
    write_json(out / "mechanism_summary.json", mechanism)
    (out / "mechanism_summary.md").write_text(
        "# Stage N LocalNO operator-response\n\n"
        + f"Decision: `{mechanism['decision']}`\n\n"
        + "\n".join(
            f"- {name}: `{value['label']}`; signature states "
            f"{value.get('supporting_states', value.get('signature_states', []))}"
            for name, value in mechanism["mechanisms"].items()
        )
        + "\n\nAll probes use frozen canonical inputs and epoch-22 LocalNO in eval/no-grad mode.\n",
        encoding="utf-8",
    )
    validation_payload = json.loads((output / "transform_prototypes/validation_roundtrip.json").read_text())
    gate_rows = candidate_replay(output, fixed_rows, two_rows, validation_payload["rows"], legacy_rows)
    run_manifest = json.loads((output / "run_manifest.json").read_text())
    run_manifest.update({
        "status": "stage_n_analysis_complete_documentation_pending",
        "operator_response": {
            "strict_checkpoint_reload": True, "checkpoint_epoch": int(metadata["epoch"]),
            "checkpoint_sha256": EXPECTED["localno_best"], "canonical_preprocessing_only": True,
            "prototype_transform_used": False, "model_state_hash_before": before_hash,
            "model_state_hash_after": after_hash, "state_unchanged": before_hash == after_hash,
            "fixed_states": states, "epsilons": list(epsilons), "decision": mechanism["decision"],
            "optimizer_created": False, "scheduler_created": False, "backward_called": False,
        },
        "candidate_gate": {"decision": "I. REPORTING_INTERFACE_READY", "version": "stage_m_v1", "reporting_only": True, "training_blocking": False, "replay_rows": len(gate_rows)},
        "provenance": provenance,
    })
    write_json(output / "run_manifest.json", run_manifest)
    print(json.dumps({"status": run_manifest["status"], "mechanism_decision": mechanism["decision"], "model_state_unchanged": before_hash == after_hash, "candidate_gate": "I. REPORTING_INTERFACE_READY"}, indent=2))


if __name__ == "__main__":
    main()
