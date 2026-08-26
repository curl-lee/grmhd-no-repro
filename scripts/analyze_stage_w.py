#!/usr/bin/env python3
"""Generate Stage W spectral, boundary, shell, radial, and temporal comparisons."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from grmhd import CHANNELS
from grmhd.stage_w_mixed_basis import mixed_basis_inverse, mixed_basis_transform

from evaluate_stage_t import model_from_checkpoint
from evaluate_stage_w import load_model as load_stage_w_model
from train_stage_s import StageSBatchPath
from train_stage_t import atomic_csv, verify_frozen_contract


ROOT = Path(__file__).resolve().parents[1]
DISPLAY = {
    "stage_t": "Stage-T",
    "mixed_basis": "W1",
    "mixed_basis_boundary": "W2",
}
FOCUS = ("Bcc2", "Bcc3", "vel3")
REGIONS = {"inner": slice(0, 21), "middle": slice(21, 42), "outer": slice(42, 64)}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fft_lowpass_stage_t(x: torch.Tensor) -> torch.Tensor:
    """Reconstruct Stage-T's exact requested 8x8x8 modal window without weights."""

    sizes = tuple(int(value) for value in x.shape[-3:])
    coefficients = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="forward")
    coefficients = torch.fft.fftshift(coefficients, dim=(-3, -2))
    output = torch.zeros_like(coefficients)
    phi_center, theta_center = coefficients.shape[-3] // 2, coefficients.shape[-2] // 2
    output[
        ..., phi_center - 4 : phi_center + 4,
        theta_center - 4 : theta_center + 4, :5,
    ] = coefficients[
        ..., phi_center - 4 : phi_center + 4,
        theta_center - 4 : theta_center + 4, :5,
    ]
    output = torch.fft.ifftshift(output, dim=(-3, -2))
    output = torch.fft.ifftn(output, s=sizes[:-1], dim=(-3, -2), norm="forward")
    output = torch.cat(
        (torch.complex(output[..., :1].real, torch.zeros_like(output[..., :1].real)), output[..., 1:]),
        dim=-1,
    )
    return torch.fft.irfft(output, n=sizes[-1], dim=-1, norm="forward")


def mixed_lowpass(x: torch.Tensor) -> torch.Tensor:
    coefficients = mixed_basis_transform(x)
    output = torch.zeros_like(coefficients)
    output[..., :5, :8, :8] = coefficients[..., :5, :8, :8]
    return mixed_basis_inverse(output, n_phi=x.shape[-3])


def natural_spectral_profiles(x: torch.Tensor, variant: str) -> dict[str, np.ndarray]:
    """Return per-channel energy by nonnegative natural mode for each logical axis."""

    if variant == "stage_t":
        coefficients = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="ortho")
        coefficients = torch.fft.fftshift(coefficients, dim=(-3, -2))
        energy = coefficients.abs().square()
        radial_weights = torch.full(
            (energy.shape[-1],), 2.0, dtype=energy.dtype, device=energy.device
        )
        radial_weights[0] = radial_weights[-1] = 1.0
        weighted = energy * radial_weights.reshape(1, 1, 1, 1, -1)
        phi_raw = weighted.sum(dim=(-2, -1))[0]
        theta_raw = weighted.sum(dim=(-3, -1))[0]
        radial = weighted.sum(dim=(-3, -2))[0]
        frequencies = torch.fft.fftshift(torch.fft.fftfreq(x.shape[-3], device=x.device)) * x.shape[-3]
        absolute = frequencies.abs().round().to(torch.int64)
        phi = torch.zeros((x.shape[1], x.shape[-3] // 2 + 1), device=x.device)
        theta = torch.zeros_like(phi)
        for mode in range(phi.shape[-1]):
            phi[:, mode] = phi_raw[:, absolute == mode].sum(dim=-1)
            theta[:, mode] = theta_raw[:, absolute == mode].sum(dim=-1)
        values = {"phi": phi, "theta": theta, "log_r": radial}
    else:
        coefficients = mixed_basis_transform(x)
        energy = coefficients.abs().square()
        phi_weights = torch.full(
            (energy.shape[-3],), 2.0, dtype=energy.dtype, device=energy.device
        )
        phi_weights[0] = phi_weights[-1] = 1.0
        weighted = energy * phi_weights.reshape(1, 1, -1, 1, 1)
        values = {
            "phi": weighted.sum(dim=(-2, -1))[0],
            "theta": weighted.sum(dim=(-3, -1))[0],
            "log_r": weighted.sum(dim=(-3, -2))[0],
        }
    return {key: value.detach().cpu().numpy().astype(np.float64) for key, value in values.items()}


def accumulate_boundary(
    accumulator: dict[tuple[str, str, int, str, str], list[float]],
    variant: str,
    subject: str,
    values: torch.Tensor,
    *, axis_name: str, width: int,
) -> None:
    axis = -2 if axis_name == "theta" else -1
    boundary = torch.cat(
        (values.narrow(axis, 0, width), values.narrow(axis, values.shape[axis] - width, width)),
        dim=axis,
    )
    interior = values.narrow(axis, width, values.shape[axis] - 2 * width)
    boundary_sum = boundary.square().sum(dim=(0, 2, 3, 4)).detach().cpu().numpy()
    interior_sum = interior.square().sum(dim=(0, 2, 3, 4)).detach().cpu().numpy()
    boundary_count = boundary[0, 0].numel()
    interior_count = interior[0, 0].numel()
    for channel, name in enumerate(CHANNELS):
        item = accumulator[(variant, axis_name, width, name, subject)]
        item[0] += float(boundary_sum[channel]); item[1] += boundary_count
        item[2] += float(interior_sum[channel]); item[3] += interior_count
    item = accumulator[(variant, axis_name, width, "all", subject)]
    item[0] += float(boundary_sum.sum()); item[1] += boundary_count * len(CHANNELS)
    item[2] += float(interior_sum.sum()); item[3] += interior_count * len(CHANNELS)


def skill_rows(
    metrics_by_variant: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shell_values: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    radial_values: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    epsilon = 1e-30
    for variant, metrics in metrics_by_variant.items():
        for row in metrics["transport_rows"]:
            channel = row["channel"]
            for family, target in (("shell", shell_values), ("radial", radial_values)):
                delta_model = np.asarray(row[family]["delta_model"], dtype=np.float64)
                delta_true = np.asarray(row[family]["delta_true"], dtype=np.float64)
                skill = 1.0 - np.abs(delta_model - delta_true) / np.maximum(np.abs(delta_true), epsilon)
                for index, value in enumerate(skill):
                    if np.isfinite(value):
                        target[(variant, channel, index)].append(float(value))
                        if channel in FOCUS:
                            target[(variant, "aggregate_focus", index)].append(float(value))
    shell_rows = []
    for (variant, channel, index), values in sorted(shell_values.items()):
        shell_rows.append({
            "variant": variant, "display_name": DISPLAY[variant], "channel": channel,
            "shell": index, "region": "inner" if index < 3 else ("middle" if index < 6 else "outer"),
            "skill_median": float(np.median(values)), "skill_mean": float(np.mean(values)),
            "sample_count": len(values),
        })
    radial_rows = []
    for (variant, channel, index), values in sorted(radial_values.items()):
        radial_rows.append({
            "variant": variant, "display_name": DISPLAY[variant], "channel": channel,
            "radial_index": index,
            "region": "inner" if index < 21 else ("middle" if index < 42 else "outer"),
            "skill_median": float(np.median(values)), "skill_mean": float(np.mean(values)),
            "sample_count": len(values),
        })
    return shell_rows, radial_rows


def one_step_comparison(metrics: Mapping[str, Mapping[str, Any]], rollouts: Mapping[str, Mapping[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    channel_rows = []
    basis = {
        "stage_t": ("FFT/FFT/FFT", "periodic/all"),
        "mixed_basis": ("FFT/DCT/DCT", "Stage-T periodic/all"),
        "mixed_basis_boundary": ("FFT/DCT/DCT", "phi periodic, theta/r replicate"),
    }
    for variant, item in metrics.items():
        model = item["model"]
        rows.append({
            "variant": variant, "display_name": DISPLAY[variant],
            "spectral_basis": basis[variant][0], "fd_boundary": basis[variant][1],
            "formal_best_epoch": item.get("formal_best_epoch", item.get("checkpoint_epoch", 150)),
            "state_l2": model["normalized_relative_l2"]["arithmetic_average"],
            "persistence_ratio": item["model_error_over_persistence_error"],
            "residual_l2": model["residual"]["arithmetic_average_relative_l2"],
            "cosine": model["residual"]["global_cosine"],
            "shell_skill": item["transport"]["shell_skill_median"],
            "radial_skill": item["transport"]["radial_skill_median"],
            "physical_l2": model["physical_relative_l2"]["arithmetic_average"],
            "finite": item["finite"], "rho_press_positive": item["rho_press_positive"],
            "first10x": rollouts[variant]["FIRST_10X_PHYSICAL_RANGE_STEP"],
            "rollout_completed_steps": rollouts[variant]["completed_steps"],
            "rollout_finite": rollouts[variant]["finite"],
            "rollout_rho_press_positive": rollouts[variant]["rho_press_positive"],
        })
        for channel in CHANNELS:
            tail = item.get("preprocessing_tail", {}).get(channel, {})
            physical = item["physical_prediction_distribution"][channel]
            channel_rows.append({
                "variant": variant, "display_name": DISPLAY[variant], "channel": channel,
                "state_l2": model["normalized_relative_l2"]["per_channel"][channel],
                "residual_l2": model["residual"]["per_channel_relative_l2"][channel],
                "residual_cosine": model["residual"]["per_channel_cosine"][channel],
                "shell_skill": item["transport"]["per_channel"][channel]["shell_skill_median"],
                "radial_skill": item["transport"]["per_channel"][channel]["radial_skill_median"],
                "physical_l2": model["physical_relative_l2"]["per_channel"][channel],
                "physical_q001": physical["pair_median_q001"],
                "physical_q999": physical["pair_median_q999"],
                "decoder_tail_fraction": tail.get("extreme_decoder_tail_fraction"),
            })
    return rows, channel_rows


def plots(
    root: Path,
    comparison: list[dict[str, Any]],
    shell_rows: list[dict[str, Any]],
    radial_rows: list[dict[str, Any]],
    boundary_rows: list[dict[str, Any]],
    spectral_rows: list[dict[str, Any]],
    rollouts: Mapping[str, Mapping[str, Any]],
) -> None:
    output = root / "figures"; output.mkdir(parents=True, exist_ok=True)
    variants = [row["variant"] for row in comparison]
    labels = [DISPLAY[value] for value in variants]
    for metric, filename, ylabel in (
        ("state_l2", "state_l2_comparison.png", "normalized state relative L2"),
        ("residual_l2", "residual_l2_comparison.png", "normalized residual relative L2"),
    ):
        fig, axis = plt.subplots(figsize=(6, 4)); axis.bar(labels, [float(row[metric]) for row in comparison])
        axis.set_ylabel(ylabel); axis.grid(axis="y", alpha=.3); fig.tight_layout(); fig.savefig(output / filename, dpi=150); plt.close(fig)
    for axis_name in ("phi", "theta", "log_r"):
        fig, axis = plt.subplots(figsize=(7, 4))
        for variant in variants:
            selected = [row for row in spectral_rows if row["variant"] == variant and row["channel"] == "aggregate_focus" and row["axis"] == axis_name]
            axis.semilogy([int(row["mode"]) for row in selected], [max(float(row["energy_fraction"]), 1e-16) for row in selected], label=DISPLAY[variant])
        axis.set(xlabel="natural mode", ylabel="energy fraction", title=f"Predicted residual {axis_name} spectrum"); axis.legend(); axis.grid(True, alpha=.3)
        fig.tight_layout(); fig.savefig(output / f"spectral_energy_{axis_name}.png", dpi=150); plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4))
    for variant in variants:
        selected = [row for row in shell_rows if row["variant"] == variant and row["channel"] == "aggregate_focus"]
        axis.plot([row["shell"] for row in selected], [row["skill_median"] for row in selected], marker="o", label=DISPLAY[variant])
    axis.axhline(0, color="black", lw=.8); axis.set(xlabel="radial shell", ylabel="median skill", title="Shell transport skill"); axis.legend(); axis.grid(True, alpha=.3)
    fig.tight_layout(); fig.savefig(output / "shell_skill_vs_shell.png", dpi=150); plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4))
    for variant in variants:
        selected = [row for row in radial_rows if row["variant"] == variant and row["channel"] == "aggregate_focus"]
        axis.plot([row["radial_index"] for row in selected], [row["skill_median"] for row in selected], label=DISPLAY[variant])
    axis.axhline(0, color="black", lw=.8); axis.set(xlabel="radial index", ylabel="median skill", title="Radial-profile transport skill"); axis.legend(); axis.grid(True, alpha=.3)
    fig.tight_layout(); fig.savefig(output / "radial_skill_vs_r.png", dpi=150); plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4))
    selected = [row for row in boundary_rows if row["channel"] == "all" and row["subject"] == "prediction_error" and int(row["width"]) == 4]
    x = np.arange(len(variants)); width = .35
    theta = [next(float(row["boundary_over_interior"]) for row in selected if row["variant"] == variant and row["axis"] == "theta") for variant in variants]
    radial = [next(float(row["boundary_over_interior"]) for row in selected if row["variant"] == variant and row["axis"] == "r") for variant in variants]
    axis.bar(x-width/2, theta, width, label="theta"); axis.bar(x+width/2, radial, width, label="r")
    axis.set_xticks(x, labels); axis.set_ylabel("boundary/interior RMS error"); axis.legend(); axis.grid(axis="y", alpha=.3)
    fig.tight_layout(); fig.savefig(output / "boundary_error_comparison.png", dpi=150); plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4))
    for variant in variants:
        records = [row for row in rollouts[variant]["records"] if row["ground_truth_available"]]
        axis.semilogy([row["step"] for row in records], [row["normalized_relative_l2_average"] for row in records], label=DISPLAY[variant])
    axis.set(xlabel="rollout step", ylabel="normalized state error", title="Closed-loop rollout error"); axis.legend(); axis.grid(True, alpha=.3)
    fig.tight_layout(); fig.savefig(output / "rollout_error_comparison.png", dpi=150); plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4))
    for variant in variants:
        records = rollouts[variant]["records"]
        values = [max(max(abs(row["physical_range"][name]["minimum"]), abs(row["physical_range"][name]["maximum"])) for name in ("rho", "press")) for row in records]
        axis.semilogy([row["step"] for row in records], values, label=DISPLAY[variant])
    axis.set(xlabel="rollout step", ylabel="max |rho/press|", title="Physical range amplification"); axis.legend(); axis.grid(True, alpha=.3)
    fig.tight_layout(); fig.savefig(output / "physical_range_comparison.png", dpi=150); plt.close(fig)


def main() -> None:
    config = yaml.safe_load((ROOT / "configs/stage_w/mixed_basis.yaml").read_text())
    _, stage_s, _, _, _, validation_pairs = verify_frozen_contract(
        ROOT / config["frozen_stage_t_config"]
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Stage W attribution requires CUDA; refusing CPU fallback")
    device = torch.device("cuda:0")
    data = StageSBatchPath(
        ROOT / stage_s["data"]["dataset"], ROOT / stage_s["preprocessing"]["artifact"], device
    )
    root = ROOT / "artifacts/stage_w"
    (root / "comparison").mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    metrics = {
        "stage_t": read_json(ROOT / "artifacts/stage_t/full_long/checkpoint_metrics_epoch_0150.json"),
        "mixed_basis": read_json(root / "variants/mixed_basis/one_step_metrics.json"),
        "mixed_basis_boundary": read_json(root / "variants/mixed_basis_boundary/one_step_metrics.json"),
    }
    rollouts = {
        "stage_t": read_json(ROOT / "artifacts/stage_t/full_long/rollout/epoch_0150_rollout.json"),
        "mixed_basis": read_json(root / "variants/mixed_basis/rollout_metrics.json"),
        "mixed_basis_boundary": read_json(root / "variants/mixed_basis_boundary/rollout_metrics.json"),
    }
    comparison, channel_rows = one_step_comparison(metrics, rollouts)
    shell_rows, radial_rows = skill_rows(metrics)
    atomic_csv(root / "comparison/stage_t_vs_stage_w.csv", comparison)
    atomic_csv(root / "comparison/per_channel_metrics.csv", channel_rows)
    atomic_csv(root / "comparison/per_shell_metrics.csv", shell_rows)
    atomic_csv(root / "comparison/radial_profile_metrics.csv", radial_rows)

    models = {}
    stage_t_model, _, _ = model_from_checkpoint(
        ROOT / config["baseline"]["checkpoint"], stage_s, device, expected_updates=6300
    )
    models["stage_t"] = stage_t_model
    for variant in ("mixed_basis", "mixed_basis_boundary"):
        selection = read_json(root / "variants" / variant / "checkpoint_selection.json")
        epoch = int(selection["formal_best_state_l2"]["epoch"])
        models[variant], _, _ = load_stage_w_model(
            root / "variants" / variant / "checkpoints" / f"epoch_{epoch:04d}.pt",
            stage_s, variant, device,
        )
    boundary_accumulator: dict[tuple[str, str, int, str, str], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
    spectral_accumulator: dict[tuple[str, str, str], np.ndarray] = {}
    region_energy: dict[tuple[str, str, str], float] = defaultdict(float)
    for variant, model in models.items():
        lowpass = fft_lowpass_stage_t if variant == "stage_t" else mixed_lowpass
        for source in validation_pairs:
            with torch.no_grad():
                result = data.predict(model, int(source))
                truth = result["residual_target"]
                prediction = result["predicted_residual"]
                error = prediction - truth
                truth_low = lowpass(truth)
                prediction_low = lowpass(prediction)
                subjects = {
                    "truth_residual": truth,
                    "predicted_residual": prediction,
                    "prediction_error": error,
                    "truth_spectral_reconstruction_error": truth - truth_low,
                    "prediction_spectral_reconstruction_error": prediction - prediction_low,
                }
                for axis_name in ("theta", "r"):
                    for width in config["evaluation"]["boundary_zone_widths"]:
                        for subject, values in subjects.items():
                            accumulate_boundary(
                                boundary_accumulator, variant, subject, values,
                                axis_name=axis_name, width=int(width),
                            )
                profiles = natural_spectral_profiles(prediction, variant)
                for axis_name, values in profiles.items():
                    for channel, name in enumerate(CHANNELS):
                        key = (variant, name, axis_name)
                        spectral_accumulator[key] = spectral_accumulator.get(key, np.zeros_like(values[channel])) + values[channel]
                    aggregate = values[[CHANNELS.index(name) for name in FOCUS]].sum(axis=0)
                    key = (variant, "aggregate_focus", axis_name)
                    spectral_accumulator[key] = spectral_accumulator.get(key, np.zeros_like(aggregate)) + aggregate
                for region, section in REGIONS.items():
                    energy = prediction_low[..., section].square().sum(dim=(0, 2, 3, 4)).cpu().numpy()
                    for channel, name in enumerate(CHANNELS):
                        region_energy[(variant, name, region)] += float(energy[channel])
                    region_energy[(variant, "aggregate_focus", region)] += float(sum(energy[CHANNELS.index(name)] for name in FOCUS))
    boundary_rows = []
    for (variant, axis_name, width, channel, subject), values in sorted(boundary_accumulator.items()):
        boundary_rms = np.sqrt(values[0] / values[1]); interior_rms = np.sqrt(values[2] / values[3])
        boundary_rows.append({
            "variant": variant, "display_name": DISPLAY[variant], "axis": axis_name,
            "width": width, "channel": channel, "subject": subject,
            "boundary_rms": boundary_rms, "interior_rms": interior_rms,
            "boundary_over_interior": boundary_rms / max(interior_rms, 1e-300),
        })
    atomic_csv(root / "comparison/boundary_metrics.csv", boundary_rows)
    spectral_rows = []
    for (variant, channel, axis_name), values in sorted(spectral_accumulator.items()):
        total = max(float(values.sum()), 1e-300)
        retained = 5 if axis_name == "phi" else 8
        for mode, value in enumerate(values):
            spectral_rows.append({
                "kind": "mode_energy", "variant": variant, "display_name": DISPLAY[variant],
                "channel": channel, "axis": axis_name, "mode": mode,
                "energy_fraction": float(value / total),
                "retained_low_half": mode < retained // 2,
                "retained_high_half": retained // 2 <= mode < retained,
            })
    for (variant, channel, region), value in sorted(region_energy.items()):
        spectral_rows.append({
            "kind": "regional_lowpass_response", "variant": variant,
            "display_name": DISPLAY[variant], "channel": channel, "axis": "all",
            "mode": "", "energy_fraction": "", "region": region,
            "response_squared_sum": value,
        })
    atomic_csv(root / "comparison/spectral_response.csv", spectral_rows)

    temporal_rows = []
    with (ROOT / "artifacts/stage_v/comparison/shift_comparison.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["variant"] == "stage_t_plain":
                temporal_rows.append({
                    "variant": "stage_t", "display_name": "Stage-T", "stratum": row["stratum"],
                    "pair_count": row["pair_count"], "state_l2": row["state_relative_l2"],
                    "residual_l2": row["residual_relative_l2"], "cosine": row["residual_global_cosine"],
                    "shell_skill": row["shell_skill"], "radial_skill": row["radial_skill"],
                    "late_over_early_state_l2": row["late_over_early_state_relative_l2"],
                    "late_over_early_residual_l2": row["late_over_early_residual_relative_l2"],
                })
    for variant in ("mixed_basis", "mixed_basis_boundary"):
        with (root / "variants" / variant / "temporal_shift_metrics.csv").open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["display_name"] = DISPLAY[variant]; temporal_rows.append(row)
    atomic_csv(root / "comparison/temporal_shift_metrics.csv", temporal_rows)
    plots(root, comparison, shell_rows, radial_rows, boundary_rows, spectral_rows, rollouts)
    data.close()
    print(json.dumps({"comparison_rows": len(comparison), "boundary_rows": len(boundary_rows), "spectral_rows": len(spectral_rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
