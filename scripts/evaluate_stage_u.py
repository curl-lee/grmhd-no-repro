#!/usr/bin/env python3
"""Evaluate frozen Stage U epoch-150 checkpoints using Stage T metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from grmhd import CHANNELS
from grmhd.paper_stage_g_evaluation import radial_shell_indices
from grmhd.stage_t_training import validation_thirds
from grmhd.stage_u_training import VARIANTS, build_stage_u_model, load_coords

from evaluate_stage_s import build_train_reference
from evaluate_stage_t import evaluate_one_step, rollout, write_csv, write_json
from train_stage_s import StageSBatchPath, tensor_sha256
from train_stage_t import verify_frozen_contract


ROOT = Path(__file__).resolve().parents[1]


def load_model(
    checkpoint: Path,
    stage_s: dict[str, Any],
    variant: str,
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    coords = load_coords(ROOT / stage_s["data"]["dataset"])
    initial = torch.load(
        ROOT / stage_s["frozen_pairing"]["initial_state"], map_location="cpu", weights_only=True
    )
    padding = json.loads((ROOT / "artifacts/stage_u/audit_summary.json").read_text())["synthetic_selected_padding"]
    model, _ = build_stage_u_model(
        stage_s, variant, coords=coords, initial_state=initial, padding=padding
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(stage_s["optimizer"]["learning_rate"]),
        weight_decay=float(stage_s["optimizer"]["weight_decay"]),
    )
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    completed = int(payload["scheduler_state"]["completed_updates"])
    if completed != 6300 or completed != int(payload["training_state"]["optimizer_updates"]):
        raise ValueError("Stage U checkpoint scheduler/update mismatch")
    model.to(device).eval()
    return model, payload, {
        "strict_model_reload": True,
        "optimizer_reload": True,
        "scheduler_reload": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--root", type=Path, default=Path("artifacts/stage_u"))
    args = parser.parse_args()
    stage_u = yaml.safe_load((ROOT / "configs/stage_u/geometry_ablation.yaml").read_text())
    stage_t, stage_s, _, _, _, validation_pairs = verify_frozen_contract(
        ROOT / stage_u["frozen_stage_t_config"]
    )
    if not torch.cuda.is_available():
        raise RuntimeError("Stage U evaluation requires CUDA; refusing CPU fallback")
    device = torch.device("cuda:0")
    data = StageSBatchPath(
        ROOT / stage_s["data"]["dataset"], ROOT / stage_s["preprocessing"]["artifact"], device
    )
    r = np.asarray(data.handle["coords/r"], dtype=np.float64)
    _, shell_index = radial_shell_indices(r, int(stage_s["representation"]["shell_count"]))
    reference_path = ROOT / "artifacts/stage_s/diagnostic_reference_train_only.json"
    reference = json.loads(reference_path.read_text()) if reference_path.exists() else build_train_reference(data, ROOT / "artifacts/stage_s")
    output = args.root / "variants" / args.variant
    model, payload, reload = load_model(
        output / "checkpoints/epoch_0150.pt", stage_s, args.variant, device
    )
    with torch.no_grad():
        probe_a = tensor_sha256(data.predict(model, 169)["z_prediction"])
        probe_b = tensor_sha256(data.predict(model, 169)["z_prediction"])
    if probe_a != probe_b:
        raise RuntimeError("Stage U checkpoint prediction is nondeterministic")
    reload.update({"deterministic_probe_prediction": True, "probe_prediction_sha256": probe_a})
    metrics = evaluate_one_step(model, data, validation_pairs, shell_index, reference)
    metrics.update({
        "schema_version": "stage-u-one-step-v1",
        "variant": args.variant,
        "checkpoint_epoch": 150,
        "optimizer_updates": 6300,
        "checkpoint_reload": reload,
    })
    write_json(output / "one_step_metrics.json", metrics)
    rows=[]
    for stratum, pairs in validation_thirds(validation_pairs).items():
        result = evaluate_one_step(model, data, pairs, shell_index, reference, include_transport_rows=False)
        for channel in CHANNELS:
            rows.append({
                "variant": args.variant, "epoch": 150, "stratum": stratum, "channel": channel,
                "model_normalized_relative_l2": result["model"]["normalized_relative_l2"]["per_channel"][channel],
                "persistence_normalized_relative_l2": result["persistence"]["normalized_relative_l2"]["per_channel"][channel],
                "residual_relative_l2": result["model"]["residual"]["per_channel_relative_l2"][channel],
                "residual_cosine": result["model"]["residual"]["per_channel_cosine"][channel],
            })
    write_csv(output / "validation_time_stratification.csv", rows)
    try:
        result = rollout(model, data, shell_index, reference, epoch=150, steps=100)
        result["schema_version"] = "stage-u-rollout-v1"
        result["variant"] = args.variant
    except FloatingPointError as exc:
        # Preserve an explicit engineering/scientific failure artifact.  The
        # frozen Stage T rollout helper intentionally raises at decode instead
        # of returning a partially decoded nonfinite physical state.
        result = {
            "schema_version": "stage-u-rollout-v1",
            "variant": args.variant,
            "checkpoint_epoch": 150,
            "requested_steps": 100,
            "completed_steps": 0,
            "finite": False,
            "rho_press_positive": False,
            "FIRST_10X_PHYSICAL_RANGE_STEP": 1,
            "FIRST_NEGATIVE_RESIDUAL_COSINE_STEP": None,
            "failure": type(exc).__name__,
            "failure_message": str(exc),
            "failure_context": "closed-loop physical decode; no clipping or fallback applied",
            "selected_steps": [],
            "records": [],
        }
    write_json(output / "rollout_metrics.json", result)
    print(json.dumps({
        "variant": args.variant,
        "state_l2": metrics["model"]["normalized_relative_l2"]["arithmetic_average"],
        "residual_l2": metrics["model"]["residual"]["arithmetic_average_relative_l2"],
        "cosine": metrics["model"]["residual"]["global_cosine"],
        "shell": metrics["transport"]["shell_skill_median"],
        "radial": metrics["transport"]["radial_skill_median"],
        "first_10x": result["FIRST_10X_PHYSICAL_RANGE_STEP"],
        "finite": result["finite"],
    }, sort_keys=True), flush=True)
    data.close()


if __name__ == "__main__":
    main()
