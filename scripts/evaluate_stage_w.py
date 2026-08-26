#!/usr/bin/env python3
"""Select and evaluate Stage W checkpoints under the frozen Stage-T metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from grmhd import CHANNELS
from grmhd.paper_stage_g_evaluation import radial_shell_indices
from grmhd.stage_t_training import validation_thirds
from grmhd.stage_w_training import VARIANTS, build_stage_w_model

from evaluate_stage_s import build_train_reference
from evaluate_stage_t import checkpoint_row, evaluate_one_step, rollout
from train_stage_s import StageSBatchPath, tensor_sha256
from train_stage_t import atomic_csv, atomic_json, verify_frozen_contract


ROOT = Path(__file__).resolve().parents[1]


def load_model(
    checkpoint: Path,
    stage_s: Mapping[str, Any],
    variant: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    initial = torch.load(
        ROOT / stage_s["frozen_pairing"]["initial_state"], map_location="cpu", weights_only=True
    )
    model, _ = build_stage_w_model(stage_s, variant, initial_state=initial)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(stage_s["optimizer"]["learning_rate"]),
        weight_decay=float(stage_s["optimizer"]["weight_decay"]),
    )
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    expected = int(payload["training_state"]["completed_epoch"]) * 42
    observed = int(payload["scheduler_state"]["completed_updates"])
    if observed != expected or observed != int(payload["training_state"]["optimizer_updates"]):
        raise ValueError(f"Stage W checkpoint scheduler/update mismatch: {checkpoint}")
    model.to(device).eval()
    return model, payload, {
        "strict_model_reload": True,
        "optimizer_reload": True,
        "scheduler_reload": True,
    }


def temporal_rows(
    model: torch.nn.Module,
    data: StageSBatchPath,
    validation_pairs: list[int],
    shell_index: np.ndarray,
    reference: Mapping[str, Any],
    *, variant: str, epoch: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stratum, pairs in validation_thirds(validation_pairs).items():
        result = evaluate_one_step(model, data, pairs, shell_index, reference)
        rows.append({
            "variant": variant,
            "formal_best_epoch": epoch,
            "stratum": stratum,
            "pair_count": len(pairs),
            "state_l2": result["model"]["normalized_relative_l2"]["arithmetic_average"],
            "residual_l2": result["model"]["residual"]["arithmetic_average_relative_l2"],
            "cosine": result["model"]["residual"]["global_cosine"],
            "shell_skill": result["transport"]["shell_skill_median"],
            "radial_skill": result["transport"]["radial_skill_median"],
        })
    early, middle, late = rows
    for row in rows:
        row["late_over_early_state_l2"] = late["state_l2"] / early["state_l2"]
        row["late_over_early_residual_l2"] = late["residual_l2"] / early["residual_l2"]
    return rows


def evaluate_variant(
    variant: str,
    config: Mapping[str, Any],
    stage_s: Mapping[str, Any],
    validation_pairs: list[int],
    data: StageSBatchPath,
    shell_index: np.ndarray,
    reference: Mapping[str, Any],
    root: Path,
    device: torch.device,
) -> None:
    output = root / "variants" / variant
    training = json.loads((output / "training_summary.json").read_text())
    if training["status"] != "passed" or int(training["completed_epoch"]) != 150:
        raise RuntimeError(f"Cannot evaluate incomplete Stage W variant {variant}")
    checkpoint_epochs = [int(value) for value in config["pilots"]["checkpoint_epochs"]]
    history: list[dict[str, Any]] = []
    candidate_metrics: dict[int, dict[str, Any]] = {}
    for epoch in checkpoint_epochs:
        checkpoint = output / "checkpoints" / f"epoch_{epoch:04d}.pt"
        model, _, reload = load_model(checkpoint, stage_s, variant, device)
        with torch.no_grad():
            probe_a = tensor_sha256(data.predict(model, 169)["z_prediction"])
            probe_b = tensor_sha256(data.predict(model, 169)["z_prediction"])
        if probe_a != probe_b:
            raise RuntimeError(f"Stage W {variant} epoch {epoch} is nondeterministic")
        reload.update({"deterministic_probe_prediction": True, "probe_prediction_sha256": probe_a})
        metrics = evaluate_one_step(model, data, validation_pairs, shell_index, reference)
        metrics.update({
            "schema_version": "stage-w-checkpoint-evaluation-v1",
            "variant": variant,
            "checkpoint_epoch": epoch,
            "optimizer_updates": epoch * 42,
            "checkpoint_reload": reload,
        })
        candidate_metrics[epoch] = metrics
        atomic_json(output / "validation" / f"epoch_{epoch:04d}.json", metrics)
        history.append(checkpoint_row(epoch, metrics, reload, epoch * 42))
        print(json.dumps({
            "variant": variant, "validation_epoch": epoch,
            "state_l2": history[-1]["normalized_relative_l2"],
            "residual_l2": history[-1]["residual_relative_l2"],
            "shell": history[-1]["shell_skill"], "radial": history[-1]["radial_skill"],
        }), flush=True)
        del model; torch.cuda.empty_cache()
    atomic_csv(output / "validation_history.csv", history)
    formal = min(history, key=lambda row: float(row["normalized_relative_l2"]))
    formal_epoch = int(formal["epoch"])
    selection = {
        "formal_selector": config["evaluation"]["formal_selector"],
        "formal_best_state_l2": {
            "epoch": formal_epoch,
            "value": float(formal["normalized_relative_l2"]),
        },
        "diagnostic_best_residual_l2": min(history, key=lambda row: float(row["residual_relative_l2"])),
        "diagnostic_best_residual_cosine": max(history, key=lambda row: float(row["residual_global_cosine"])),
        "diagnostic_best_shell_skill": max(history, key=lambda row: float(row["shell_skill"])),
        "diagnostic_best_radial_skill": max(history, key=lambda row: float(row["radial_skill"])),
        "diagnostic_selectors_do_not_select_formal_checkpoint": True,
    }
    atomic_json(output / "checkpoint_selection.json", selection)
    source_checkpoint = output / "checkpoints" / f"epoch_{formal_epoch:04d}.pt"
    shutil.copyfile(source_checkpoint, output / "formal_best.pt")
    model, payload, reload = load_model(source_checkpoint, stage_s, variant, device)
    with torch.no_grad():
        probe_a = tensor_sha256(data.predict(model, 169)["z_prediction"])
        probe_b = tensor_sha256(data.predict(model, 169)["z_prediction"])
    reload.update({
        "deterministic_probe_prediction": probe_a == probe_b,
        "probe_prediction_sha256": probe_a,
        "epoch": formal_epoch,
        "optimizer_updates": int(payload["training_state"]["optimizer_updates"]),
    })
    metrics = evaluate_one_step(model, data, validation_pairs, shell_index, reference)
    recorded = candidate_metrics[formal_epoch]
    reload["validation_recomputed_matches_recorded"] = bool(np.isclose(
        metrics["model"]["normalized_relative_l2"]["arithmetic_average"],
        recorded["model"]["normalized_relative_l2"]["arithmetic_average"],
        rtol=1e-10, atol=1e-12,
    ))
    if not all(reload[key] for key in (
        "strict_model_reload", "optimizer_reload", "scheduler_reload",
        "deterministic_probe_prediction", "validation_recomputed_matches_recorded",
    )):
        raise RuntimeError(f"Stage W formal checkpoint reload failed: {reload}")
    metrics.update({
        "schema_version": "stage-w-one-step-v1", "variant": variant,
        "formal_best_epoch": formal_epoch,
        "optimizer_updates": int(payload["training_state"]["optimizer_updates"]),
        "checkpoint_reload": reload,
    })
    atomic_json(output / "one_step_metrics.json", metrics)
    atomic_json(output / "formal_best_reload.json", reload)
    atomic_csv(
        output / "temporal_shift_metrics.csv",
        temporal_rows(
            model, data, validation_pairs, shell_index, reference,
            variant=variant, epoch=formal_epoch,
        ),
    )
    try:
        rollout_metrics = rollout(
            model, data, shell_index, reference, epoch=formal_epoch, steps=100
        )
        rollout_metrics.update({"schema_version": "stage-w-rollout-v1", "variant": variant})
    except FloatingPointError as exc:
        rollout_metrics = {
            "schema_version": "stage-w-rollout-v1", "variant": variant,
            "checkpoint_epoch": formal_epoch, "requested_steps": 100,
            "completed_steps": 0, "finite": False, "rho_press_positive": False,
            "FIRST_10X_PHYSICAL_RANGE_STEP": 1,
            "FIRST_NEGATIVE_RESIDUAL_COSINE_STEP": None,
            "FIRST_NEGATIVE_SHELL_SKILL_STEP": None,
            "FIRST_NEGATIVE_RADIAL_SKILL_STEP": None,
            "failure": type(exc).__name__, "failure_message": str(exc),
            "failure_context": "closed-loop physical decode; no clamp or fallback",
            "selected_steps": [], "records": [],
        }
    atomic_json(output / "rollout_metrics.json", rollout_metrics)
    print(json.dumps({
        "variant": variant, "formal_epoch": formal_epoch,
        "state_l2": metrics["model"]["normalized_relative_l2"]["arithmetic_average"],
        "first10x": rollout_metrics["FIRST_10X_PHYSICAL_RANGE_STEP"],
        "completed_steps": rollout_metrics["completed_steps"],
    }), flush=True)
    del model; torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=(*VARIANTS, "all"), default="all")
    parser.add_argument("--config", type=Path, default=Path("configs/stage_w/mixed_basis.yaml"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/stage_w"))
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text())
    _, stage_s, _, _, _, validation_pairs = verify_frozen_contract(
        ROOT / config["frozen_stage_t_config"]
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Stage W evaluation requires CUDA; refusing CPU fallback")
    device = torch.device("cuda:0")
    data = StageSBatchPath(
        ROOT / stage_s["data"]["dataset"],
        ROOT / stage_s["preprocessing"]["artifact"], device,
    )
    r = np.asarray(data.handle["coords/r"], dtype=np.float64)
    _, shell_index = radial_shell_indices(r, int(stage_s["representation"]["shell_count"]))
    reference_path = ROOT / "artifacts/stage_s/diagnostic_reference_train_only.json"
    reference = (
        json.loads(reference_path.read_text())
        if reference_path.exists() else build_train_reference(data, ROOT / "artifacts/stage_s")
    )
    variants = VARIANTS if args.variant == "all" else (args.variant,)
    for variant in variants:
        evaluate_variant(
            variant, config, stage_s, validation_pairs, data,
            shell_index, reference, args.root, device,
        )
    data.close()


if __name__ == "__main__":
    main()
