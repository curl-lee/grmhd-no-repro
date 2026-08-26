#!/usr/bin/env python3
"""Strictly reload and evaluate the completed Stage AB Z96 experiment."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np
import torch

from evaluate_stage_s import build_train_reference
from evaluate_stage_t import evaluate_one_step, model_from_checkpoint, rollout, write_json
from train_stage_s import tensor_sha256, validation_score
from train_stage_z import frozen_shell_data, load_contract


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/stage_ab/resume"
CONFIG = ROOT / "configs/stage_z/higher_resolution.yaml"


def close_enough(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    paths = (
        ("normalized_relative_l2", "arithmetic_average"),
        ("normalized_relative_l2", "global"),
        ("residual", "arithmetic_average_relative_l2"),
        ("residual", "global_cosine"),
    )
    return all(
        np.isclose(float(left[a][b]), float(right[a][b]), rtol=1.0e-7, atol=1.0e-9)
        for a, b in paths
    )


def reload_and_evaluate(
    checkpoint: Path,
    *,
    epoch: int,
    stage_s: Mapping[str, Any],
    data: Any,
    validation_pairs: list[int],
    shell_index: np.ndarray,
    train_reference: Mapping[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, dict[str, Any], dict[str, Any]]:
    model, payload, reload_status = model_from_checkpoint(
        checkpoint,
        stage_s,
        device,
        expected_updates=epoch * 42,
    )
    if int(payload["metadata"]["resolution"]) != 96:
        raise ValueError("Stage AB checkpoint resolution changed")
    if int(payload["metadata"]["parameter_count"]) != 358296:
        raise ValueError("Stage AB checkpoint parameter count changed")
    if payload["metadata"]["dataset_sha256"] != "292d3fe62297e91fda8e0f81ab156ca7ceb127ea02a004ca79180705e7411564":
        raise ValueError("Stage AB checkpoint dataset provenance changed")
    if payload["metadata"]["normalizer_sha256"] != "63c5dda86601cbececd45cc8695adfa272fa279ec032f5ebe2db39697cc97127":
        raise ValueError("Stage AB checkpoint normalizer provenance changed")
    with torch.no_grad():
        probe_a = tensor_sha256(data.predict(model, validation_pairs[0])["z_prediction"])
        probe_b = tensor_sha256(data.predict(model, validation_pairs[0])["z_prediction"])
    if probe_a != probe_b:
        raise ValueError("Stage AB checkpoint prediction is nondeterministic")
    recomputed_validation = validation_score(model, data, validation_pairs)
    if not close_enough(recomputed_validation, payload["validation"]):
        raise ValueError("Stage AB checkpoint validation did not reproduce")
    metrics = evaluate_one_step(model, data, validation_pairs, shell_index, train_reference)
    metrics.update({
        "checkpoint_epoch": epoch,
        "optimizer_updates": epoch * 42,
        "checkpoint_reload": {
            **reload_status,
            "deterministic_probe_prediction": True,
            "probe_prediction_sha256": probe_a,
            "saved_validation_recomputed": True,
        },
    })
    return model, payload, metrics


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage AB evaluation requires CUDA; refusing CPU fallback")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    stage_z, stage_s_base, dataset, normalizer, _, _, validation_pairs = load_contract(
        CONFIG, 96
    )
    stage_s = copy.deepcopy(stage_s_base)
    stage_s["model"]["default_in_shape"] = [96, 96, 96]
    data = frozen_shell_data(
        dataset,
        normalizer,
        device,
        ROOT / "artifacts/stage_z/shell_contract.json",
    )
    shell_index = np.argmax(
        data.shells[0, :, 0, 0, :].detach().cpu().numpy(), axis=0
    ).astype(np.int64)
    reference_path = OUT / "diagnostic_reference_train_only.json"
    reference = (
        json.loads(reference_path.read_text(encoding="utf-8"))
        if reference_path.is_file()
        else build_train_reference(data, OUT)
    )
    validation_records = json.loads(
        (OUT / "checkpoint_validation.json").read_text(encoding="utf-8")
    )
    best_record = min(
        validation_records,
        key=lambda item: float(item["normalized_relative_l2"]["arithmetic_average"]),
    )
    best_epoch = int(best_record["epoch"])
    if best_epoch != 75:
        raise ValueError("Stage AB formal checkpoint selector changed")

    best_model, _, best_metrics = reload_and_evaluate(
        OUT / "checkpoints" / f"epoch_{best_epoch:04d}.pt",
        epoch=best_epoch,
        stage_s=stage_s,
        data=data,
        validation_pairs=validation_pairs,
        shell_index=shell_index,
        train_reference=reference,
        device=device,
    )
    write_json(OUT / "one_step_metrics_best.json", best_metrics)
    best_rollout = rollout(
        best_model,
        data,
        shell_index,
        reference,
        epoch=best_epoch,
        steps=100,
    )
    best_rollout.update({
        "reproduction_scope": stage_z["reproduction_scope"],
        "resolution": 96,
        "formal_checkpoint_epoch": best_epoch,
    })
    write_json(OUT / "rollout_metrics.json", best_rollout)
    del best_model
    torch.cuda.empty_cache()

    last_model, _, last_metrics = reload_and_evaluate(
        OUT / "checkpoints/epoch_0150.pt",
        epoch=150,
        stage_s=stage_s,
        data=data,
        validation_pairs=validation_pairs,
        shell_index=shell_index,
        train_reference=reference,
        device=device,
    )
    write_json(OUT / "one_step_metrics_last.json", last_metrics)
    del last_model
    torch.cuda.empty_cache()

    write_json(OUT / "one_step_metrics.json", {
        "schema_version": "stage-ab-z96-one-step-v1",
        "reproduction_scope": stage_z["reproduction_scope"],
        "resolution": 96,
        "formal_checkpoint_selector": stage_z["training"]["formal_selector"],
        "formal_best_epoch": best_epoch,
        "best": best_metrics,
        "last": last_metrics,
    })
    shutil.copyfile(OUT / "resolved_config.json", OUT / "z96_resolved_config.json")
    shutil.copyfile(OUT / "train_log.csv", OUT / "z96_train_log.csv")
    print(json.dumps({
        "status": "passed",
        "formal_best_epoch": best_epoch,
        "best_state_l2": best_metrics["model"]["normalized_relative_l2"]["arithmetic_average"],
        "best_residual_l2": best_metrics["model"]["residual"]["arithmetic_average_relative_l2"],
        "best_residual_cosine": best_metrics["model"]["residual"]["global_cosine"],
        "best_shell_skill": best_metrics["transport"]["shell_skill_median"],
        "best_radial_skill": best_metrics["transport"]["radial_skill_median"],
        "rollout_finite": best_rollout["finite"],
        "rollout_first_10x": best_rollout["FIRST_10X_PHYSICAL_RANGE_STEP"],
    }, indent=2, sort_keys=True))
    data.close()


if __name__ == "__main__":
    main()
