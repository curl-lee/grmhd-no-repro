#!/usr/bin/env python3
"""Evaluate Stage AD checkpoints, attribution, paired validation, and rollout."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from grmhd import CHANNELS
from grmhd.paper_losses import PlainL2Loss
from grmhd.paper_stage_g_evaluation import radial_shell_indices
from grmhd.paper_stage_g import tensor_state_sha256

from evaluate_stage_s import build_train_reference
from evaluate_stage_t import (
    checkpoint_row,
    evaluate_one_step,
    model_from_checkpoint as stage_t_model_from_checkpoint,
    rollout,
)
from preflight_stage_ad import build_model_and_hashes, load_contract
from train_stage_s import StageSBatchPath, tensor_sha256


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/stage_ad"
TRAIN = OUT / "training/disco3d_localno"
CHECKPOINT_EPOCHS = (2, 10, 30, 75, 150, 300)
SELECTED_STEPS = (1, 2, 3, 5, 10, 19, 25, 42, 50, 75, 100)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write empty Stage AD table")
    fields = sorted({name for row in rows for name in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def model_from_checkpoint(
    path: Path,
    config: dict,
    stage_s: dict,
    device: torch.device,
    expected_updates: int,
) -> tuple[torch.nn.Module, dict, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model, hashes = build_model_and_hashes(config, stage_s)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    completed = int(payload["scheduler_state"]["completed_updates"])
    if completed != expected_updates or completed != int(payload["training_state"]["optimizer_updates"]):
        raise ValueError("Stage AD checkpoint scheduler/update mismatch")
    if payload["metadata"]["common_tensor_state_sha256"] != hashes["common_tensor_state_sha256"]:
        raise ValueError("Stage AD common provenance changed")
    model.to(device).eval()
    return model, payload, {
        "strict_model_reload": True,
        "optimizer_reload": True,
        "scheduler_reload": True,
    }


def compact_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        "state_l2": float(metrics["model"]["normalized_relative_l2"]["arithmetic_average"]),
        "residual_l2": float(metrics["model"]["residual"]["arithmetic_average_relative_l2"]),
        "cosine": float(metrics["model"]["residual"]["global_cosine"]),
        "shell_skill": float(metrics["transport"]["shell_skill_median"]),
        "radial_skill": float(metrics["transport"]["radial_skill_median"]),
        "physical_l2": float(metrics["model"]["physical_relative_l2"]["arithmetic_average"]),
    }


def paired_bootstrap(rows: list[dict[str, Any]], *, seed: int = 42) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    output = []
    for metric in ("state_l2", "residual_l2", "cosine", "shell_skill", "radial_skill"):
        delta = np.asarray([float(row[f"disco_{metric}"]) - float(row[f"stage_t_{metric}"]) for row in rows])
        samples = np.empty(10_000, dtype=np.float64)
        for index in range(len(samples)):
            samples[index] = float(np.mean(delta[rng.integers(0, len(delta), len(delta))]))
        higher_is_better = metric in {"cosine", "shell_skill", "radial_skill"}
        output.append({
            "metric": metric,
            "delta_definition": "DISCO3D_minus_StageT",
            "mean_paired_delta": float(np.mean(delta)),
            "median_paired_delta": float(np.median(delta)),
            "win_fraction": float(np.mean(delta > 0 if higher_is_better else delta < 0)),
            "bootstrap_seed": seed,
            "bootstrap_samples": len(samples),
            "ci95_low": float(np.quantile(samples, 0.025)),
            "ci95_high": float(np.quantile(samples, 0.975)),
            "higher_is_better": higher_is_better,
        })
    return output


def per_channel_rows(stage_t: Mapping[str, Any], disco: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for channel in CHANNELS:
        for model_name, metrics in (("Stage-T differential LocalNO", stage_t), ("Adapted 3D-DISCO LocalNO", disco)):
            rows.append({
                "model": model_name,
                "channel": channel,
                "state_l2": metrics["model"]["normalized_relative_l2"]["per_channel"][channel],
                "residual_l2": metrics["model"]["residual"]["per_channel_relative_l2"][channel],
                "cosine": metrics["model"]["residual"]["per_channel_cosine"][channel],
                "shell_skill": metrics["transport"]["per_channel"][channel]["shell_skill_median"],
                "radial_skill": metrics["transport"]["per_channel"][channel]["radial_skill_median"],
            })
        rows.append({
            "model": "Persistence", "channel": channel,
            "state_l2": disco["persistence"]["normalized_relative_l2"]["per_channel"][channel],
            "residual_l2": 1.0, "cosine": 0.0, "shell_skill": 0.0, "radial_skill": 0.0,
        })
    return rows


def branch_activity(
    model: torch.nn.Module,
    initial_model: torch.nn.Module,
    data: StageSBatchPath,
) -> tuple[list[dict[str, Any]], bool]:
    records = {index: {} for index in range(4)}
    handles = []
    blocks = model.local_no_blocks

    def capture(layer: int, branch: str):
        def hook(_module, _inputs, output):
            records[layer][f"{branch}_output_norm"] = float(
                torch.linalg.vector_norm(output.detach()).cpu()
            )
        return hook

    for layer in range(4):
        handles.append(blocks.convs[layer].register_forward_hook(capture(layer, "spectral")))
        handles.append(blocks.differential[layer].register_forward_hook(capture(layer, "differential")))
        handles.append(blocks.local_convs[layer].register_forward_hook(capture(layer, "disco3d")))
    model.zero_grad(set_to_none=True)
    result = data.predict(model, 169)
    PlainL2Loss().to(data.device)(result["predicted_residual"], result["residual_target"]).backward()
    for handle in handles:
        handle.remove()
    rows = []
    for layer in range(4):
        for branch, modules in (
            ("spectral", blocks.convs),
            ("differential", blocks.differential),
            ("disco3d", blocks.local_convs),
        ):
            gradients = [
                torch.sum(torch.abs(parameter.grad.detach()).float().square())
                for parameter in modules[layer].parameters() if parameter.grad is not None
            ]
            records[layer][f"{branch}_parameter_gradient_norm"] = float(
                torch.sqrt(torch.stack(gradients).sum()).cpu()
            )
        current = blocks.local_convs[layer]
        initial = initial_model.local_no_blocks.local_convs[layer]
        difference = torch.sqrt(torch.stack([
            torch.sum((left.detach().cpu() - right.detach().cpu()).float().square())
            for left, right in zip(current.parameters(), initial.parameters(), strict=True)
        ]).sum())
        records[layer]["disco_parameter_update_norm"] = float(difference)
        rows.append({"layer": layer, **records[layer]})
    active = all(
        row["disco3d_output_norm"] > 0
        and row["disco3d_parameter_gradient_norm"] > 0
        and row["disco_parameter_update_norm"] > 0
        for row in rows
    )
    return rows, active


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage AD evaluation refuses CPU fallback")
    config, _, stage_s = load_contract()
    device = torch.device("cuda:0")
    data = StageSBatchPath(
        ROOT / stage_s["data"]["dataset"],
        ROOT / stage_s["preprocessing"]["artifact"],
        device,
    )
    split = json.loads((ROOT / stage_s["data"]["split"]).read_text(encoding="utf-8"))
    validation_pairs = [int(value) for value in split["validation_pair_source_indices"]]
    if validation_pairs != list(range(169, 211)):
        raise ValueError("Stage AD validation population changed")
    r = np.asarray(data.handle["coords/r"], dtype=np.float64)
    _, shell_index = radial_shell_indices(r, int(stage_s["representation"]["shell_count"]))
    reference_path = ROOT / "artifacts/stage_s/diagnostic_reference_train_only.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8")) \
        if reference_path.exists() else build_train_reference(data, ROOT / "artifacts/stage_s")

    checkpoint_rows = []
    metrics_by_epoch = {}
    for epoch in CHECKPOINT_EPOCHS:
        path = TRAIN / "checkpoints" / f"epoch_{epoch:04d}.pt"
        model, payload, reload = model_from_checkpoint(
            path, config, stage_s, device, expected_updates=epoch * 42
        )
        with torch.no_grad():
            probe_a = tensor_sha256(data.predict(model, 169)["z_prediction"])
            probe_b = tensor_sha256(data.predict(model, 169)["z_prediction"])
        if probe_a != probe_b:
            raise RuntimeError("Stage AD checkpoint prediction is nondeterministic")
        reload.update({"deterministic_probe_prediction": True, "probe_prediction_sha256": probe_a})
        metrics = evaluate_one_step(model, data, validation_pairs, shell_index, reference)
        metrics.update({
            "schema_version": "stage-ad-one-step-v1",
            "checkpoint_epoch": epoch,
            "optimizer_updates": epoch * 42,
            "checkpoint_reload": reload,
        })
        atomic_json(TRAIN / f"validation_epoch_{epoch:04d}.json", metrics)
        checkpoint_rows.append(checkpoint_row(epoch, metrics, reload, epoch * 42))
        metrics_by_epoch[epoch] = metrics
        print(json.dumps({"epoch": epoch, **compact_metrics(metrics)}, sort_keys=True), flush=True)
        del model, payload
        torch.cuda.empty_cache()
    atomic_csv(TRAIN / "checkpoint_metrics.csv", checkpoint_rows)
    best_row = min(checkpoint_rows, key=lambda row: float(row["normalized_relative_l2"]))
    best_epoch = int(best_row["epoch"])
    best_metrics = metrics_by_epoch[best_epoch]
    atomic_json(TRAIN / "one_step_metrics.json", {
        "formal_selector": config["evaluation"]["selector"],
        "formal_best_epoch": best_epoch,
        "metrics": best_metrics,
    })

    best_model, best_payload, best_reload = model_from_checkpoint(
        TRAIN / "checkpoints" / f"epoch_{best_epoch:04d}.pt",
        config, stage_s, device, expected_updates=best_epoch * 42,
    )
    recomputed = evaluate_one_step(
        best_model, data, validation_pairs, shell_index, reference, include_transport_rows=False
    )
    original_compact = compact_metrics(best_metrics)
    recomputed_compact = compact_metrics(recomputed)
    maximum_recompute_difference = max(
        abs(original_compact[name] - recomputed_compact[name]) for name in original_compact
    )
    if maximum_recompute_difference > 1e-12:
        raise RuntimeError("Stage AD validation metrics do not recompute exactly")
    best_reload["validation_metrics_recomputed"] = True
    best_reload["maximum_metric_difference"] = maximum_recompute_difference

    stage_t_metrics = json.loads(
        (ROOT / "artifacts/stage_t/full_long/checkpoint_metrics_epoch_0150.json").read_text(encoding="utf-8")
    )
    atomic_csv(OUT / "comparison/per_channel_comparison.csv", per_channel_rows(stage_t_metrics, best_metrics))

    stage_t_config = ROOT / "configs/stage_t/optimization_convergence.yaml"
    stage_t_model, _, _ = stage_t_model_from_checkpoint(
        ROOT / "artifacts/stage_t/full_long/checkpoints/epoch_0150.pt",
        stage_s, device, expected_updates=6300,
    )
    paired = []
    for source in validation_pairs:
        stage_t_pair = compact_metrics(evaluate_one_step(
            stage_t_model, data, [source], shell_index, reference, include_transport_rows=False
        ))
        disco_pair = compact_metrics(evaluate_one_step(
            best_model, data, [source], shell_index, reference, include_transport_rows=False
        ))
        paired.append({
            "source_snapshot": source,
            **{f"stage_t_{name}": value for name, value in stage_t_pair.items()},
            **{f"disco_{name}": value for name, value in disco_pair.items()},
        })
    atomic_csv(OUT / "comparison/per_pair_comparison.csv", paired)
    bootstrap = paired_bootstrap(paired)
    atomic_csv(OUT / "comparison/paired_bootstrap.csv", bootstrap)

    initial_model, _ = build_model_and_hashes(config, stage_s)
    initial_model.to(device).eval()
    activity_rows, active = branch_activity(best_model, initial_model, data)
    atomic_csv(OUT / "attribution/branch_activity.csv", activity_rows)

    full_probe = data.predict(best_model, 169)
    original_indices = list(best_model.local_no_blocks.disco_idx_list)
    best_model.local_no_blocks.disco_idx_list = [-1] * 4
    minus_metrics = evaluate_one_step(best_model, data, validation_pairs, shell_index, reference)
    minus_probe = data.predict(best_model, 169)
    best_model.local_no_blocks.disco_idx_list = original_indices
    contribution_norm = float(torch.linalg.vector_norm(
        full_probe["predicted_residual"] - minus_probe["predicted_residual"]
    ).detach().cpu())
    full_compact = compact_metrics(best_metrics)
    minus_compact = compact_metrics(minus_metrics)
    ablation_rows = []
    for metric in full_compact:
        ablation_rows.append({
            "metric": metric,
            "full": full_compact[metric],
            "full_minus_disco": minus_compact[metric],
            "full_minus_ablation": full_compact[metric] - minus_compact[metric],
        })
    ablation_rows.append({
        "metric": "probe_residual_disco_contribution_norm",
        "full": contribution_norm,
        "full_minus_disco": 0.0,
        "full_minus_ablation": contribution_norm,
    })
    atomic_csv(OUT / "attribution/branch_ablation.csv", ablation_rows)

    rollout_result = rollout(
        best_model, data, shell_index, reference, epoch=best_epoch, steps=100
    )
    rollout_result["schema_version"] = "stage-ad-rollout-v1"
    rollout_result["formal_best_epoch"] = best_epoch
    atomic_json(TRAIN / "rollout_metrics.json", rollout_result)
    stage_t_rollout = json.loads(
        (ROOT / "artifacts/stage_t/full_long/rollout/epoch_0150_rollout.json").read_text(encoding="utf-8")
    )
    rollout_rows = []
    for model_name, result in (
        ("Stage-T differential LocalNO", stage_t_rollout),
        ("Adapted 3D-DISCO LocalNO", rollout_result),
    ):
        by_step = {int(row["step"]): row for row in result["records"]}
        for step in SELECTED_STEPS:
            if step not in by_step:
                continue
            row = by_step[step]
            rollout_rows.append({
                "model": model_name, "step": step,
                "ground_truth_available": row["ground_truth_available"],
                "state_l2": row.get("normalized_relative_l2_average"),
                "residual_cosine": row.get("residual_cosine"),
                "shell_skill": None if row.get("transport") is None else row["transport"]["shell_skill_median"],
                "radial_skill": None if row.get("transport") is None else row["transport"]["radial_skill_median"],
                "finite": row["finite"],
                "rho_press_positive": row["rho_press_positive"],
                "physical_range_explosion": row["physical_range_explosion"],
            })
    atomic_csv(OUT / "comparison/rollout_comparison.csv", rollout_rows)

    persistence = best_metrics["persistence"]
    stage_t_compact = compact_metrics(stage_t_metrics)
    comparison = [
        {
            "model": "Persistence", "params": 0, "local_integral": "no",
            "state_l2": persistence["normalized_relative_l2"]["arithmetic_average"],
            "residual_l2": 1.0, "cosine": 0.0, "shell_skill": 0.0,
            "radial_skill": 0.0, "first10x": "stable",
        },
        {
            "model": "Stage-T differential LocalNO", "params": 358296, "local_integral": "no",
            **stage_t_compact,
            "first10x": stage_t_rollout["FIRST_10X_PHYSICAL_RANGE_STEP"],
        },
        {
            "model": "Adapted 3D-DISCO LocalNO", "params": 363480, "local_integral": "yes",
            **full_compact,
            "first10x": rollout_result["FIRST_10X_PHYSICAL_RANGE_STEP"],
        },
    ]
    atomic_csv(OUT / "comparison/persistence_stage_t_disco.csv", comparison)
    training_summary = json.loads((TRAIN / "training_summary.json").read_text(encoding="utf-8"))
    with (TRAIN / "train_log.csv").open(newline="", encoding="utf-8") as handle:
        stage_ad_epoch_150 = next(
            row for row in csv.DictReader(handle) if int(row["epoch"]) == 150
        )
    with (ROOT / "artifacts/stage_t/full_long/train_log.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        stage_t_epoch_150 = next(
            row for row in csv.DictReader(handle) if int(row["epoch"]) == 150
        )
    atomic_csv(OUT / "comparison/efficiency_comparison.csv", [
        {
            "model": "Stage-T differential LocalNO", "params": 358296,
            "epoch": 150,
            "runtime_seconds": float(stage_t_epoch_150["cumulative_runtime_seconds"]),
            "peak_allocated_mib": float(stage_t_epoch_150["gpu_peak_allocated_mib"]),
            "optimizer_updates": 6300,
        },
        {
            "model": "Adapted 3D-DISCO LocalNO", "params": 363480,
            "epoch": 150,
            "runtime_seconds": float(stage_ad_epoch_150["cumulative_runtime_seconds"]),
            "peak_allocated_mib": training_summary["peak_allocated_mib_max"],
            "optimizer_updates": 6300,
        },
    ])

    extension = (
        float(metrics_by_epoch[30]["model"]["normalized_relative_l2"]["arithmetic_average"])
        > float(metrics_by_epoch[75]["model"]["normalized_relative_l2"]["arithmetic_average"])
        > float(metrics_by_epoch[150]["model"]["normalized_relative_l2"]["arithmetic_average"])
        and (
            float(metrics_by_epoch[75]["model"]["normalized_relative_l2"]["arithmetic_average"])
            - float(metrics_by_epoch[150]["model"]["normalized_relative_l2"]["arithmetic_average"])
        ) / float(metrics_by_epoch[75]["model"]["normalized_relative_l2"]["arithmetic_average"]) > 0.03
    )
    summary = {
        "formal_best_epoch": best_epoch,
        "formal_selector": config["evaluation"]["selector"],
        "checkpoint_reload": best_reload,
        "one_step": full_compact,
        "stage_t": stage_t_compact,
        "persistence_state_l2": persistence["normalized_relative_l2"]["arithmetic_average"],
        "disco_branch_active": active,
        "branch_ablation_probe_contribution_norm": contribution_norm,
        "rollout": {
            "finite": rollout_result["finite"],
            "rho_press_positive": rollout_result["rho_press_positive"],
            "completed_steps": rollout_result["completed_steps"],
            "first_negative_residual_cosine": rollout_result["FIRST_NEGATIVE_RESIDUAL_COSINE_STEP"],
            "first_negative_shell_skill": rollout_result["FIRST_NEGATIVE_SHELL_SKILL_STEP"],
            "first_negative_radial_skill": rollout_result["FIRST_NEGATIVE_RADIAL_SKILL_STEP"],
            "first_10x": rollout_result["FIRST_10X_PHYSICAL_RANGE_STEP"],
        },
        "paired_bootstrap": bootstrap,
        "extension_to_300_authorized_by_frozen_rule": extension,
    }
    atomic_json(OUT / "comparison/evaluation_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    data.close()


if __name__ == "__main__":
    main()
