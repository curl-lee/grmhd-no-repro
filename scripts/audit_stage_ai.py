#!/usr/bin/env python3
"""Create the no-training Stage AI comparability and artifact audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import h5py
import yaml

from grmhd.dataset import sha256_file
from grmhd.stage_ai import build_canonical_cnn


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage_ai/final_benchmark.yaml"
OUT = ROOT / "artifacts/stage_ai/audit"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def common(model: str, stage: str, code: str, checkpoint: str) -> dict[str, Any]:
    return {
        "model": model,
        "artifact_stage": stage,
        "code_path": code,
        "checkpoint_path": checkpoint,
        "dataset_sha256": "3582a5c4f52f4fef305578ffd5a63c51423f8583f35d137c288d5c20203b50da",
        "resolution": "64^3",
        "input_contract": "8 P3 state + 8 frozen physical-r shells",
        "target_contract": "normalized residual z[t+1]-z[t]",
        "preprocessing": "Z64 P3 expanded train-only",
        "train_split": "pairs 0..167",
        "validation_split": "pairs 169..210; no shuffle",
        "optimizer": "Adam lr=1e-3 weight_decay=1e-4 clip=1",
        "scheduler": "50400-update warmup-cosine; warmup=3150; min_lr=1e-6",
        "epoch_budget": "",
        "pair_order": "seed=42 natural_epoch_orders",
        "checkpoint_selector": "42-pair normalized per-channel state relative-L2 arithmetic mean",
        "evaluation_metrics_available": "one-step, residual, transport, physical, rollout",
        "comparable_to_final_contract": "",
        "reuse_or_retrain": "",
        "reason": "",
    }


def rows() -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    row = common("Persistence", "analytic", "scripts/evaluate_stage_ai.py", "not applicable")
    row.update(
        optimizer="not applicable", scheduler="not applicable", epoch_budget="0",
        pair_order="not applicable; evaluated on all 42 validation pairs",
        checkpoint_selector="not applicable", comparable_to_final_contract="true",
        reuse_or_retrain="reuse", reason="deterministic fixed-state reference",
    )
    result.append(row)
    row = common("FNO", "G/U", "src/grmhd/stage_u_training.py", "artifacts/stage_u/variants/spectral_only/checkpoints/epoch_0150.pt")
    row.update(epoch_budget="150 diagnostic; early FNO runs use reduced100/30 epochs", comparable_to_final_contract="false", reuse_or_retrain="retrain", reason="no canonical 300-epoch formal-selector checkpoint set; Stage-U scheduler explicitly diagnostic/truncated")
    result.append(row)
    row = common("3D CNN/U-Net", "not previously implemented", "src/grmhd/stage_ai.py", "not available before Stage AI")
    row.update(epoch_budget="none", comparable_to_final_contract="false", reuse_or_retrain="train", reason="no existing canonical CNN checkpoint")
    result.append(row)
    row = common("Differential LocalNO", "T", "src/grmhd/models.py", "artifacts/stage_t/full_long/checkpoints/epoch_0150.pt")
    row.update(epoch_budget="formal best epoch 150 (restricted to <=300)", comparable_to_final_contract="true", reuse_or_retrain="reuse", reason="frozen canonical contract and formal selector; strict reload recorded")
    result.append(row)
    row = common("Index-space DISCO3D LocalNO", "AD", "src/grmhd/localno3d_disco.py", "artifacts/stage_ad/training/disco3d_localno/checkpoints/epoch_0300.pt")
    row.update(epoch_budget="300", comparable_to_final_contract="true", reuse_or_retrain="reuse", reason="frozen canonical contract and formal epoch-300 selector; strict reload recorded")
    result.append(row)
    row = common("Anisotropic spherical DISCO3D LocalNO", "AG", "src/grmhd/anisotropic_spherical_disco_localno.py", "artifacts/stage_ag/training/checkpoints/best.pt")
    row.update(epoch_budget="300", comparable_to_final_contract="true", reuse_or_retrain="reuse", reason="controlled canonical contract; formal best epoch 300; strict reload recorded")
    result.append(row)
    return result


def main() -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    dataset = ROOT / config["data"]["dataset"]
    with h5py.File(dataset, "r") as handle:
        shape = list(handle["snapshots"].shape)
    if sha256_file(dataset) != config["data"]["dataset_sha256"] or shape != config["data"]["shape"]:
        raise RuntimeError("canonical Stage AI dataset identity failed")
    if sha256_file(ROOT / config["preprocessing"]["normalizer_file"]) != config["preprocessing"]["normalizer_sha256"]:
        raise RuntimeError("canonical Stage AI normalizer identity failed")
    if sha256_file(ROOT / config["data"]["split"]) != config["data"]["split_sha256"]:
        raise RuntimeError("canonical Stage AI split identity failed")
    upstream = subprocess.check_output(["git", "-C", ROOT / "external/neuraloperator", "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", ROOT / "external/neuraloperator", "status", "--short"], text=True).strip()
    if upstream != config["runtime"]["upstream_commit"] or dirty:
        raise RuntimeError("pinned upstream identity failed")
    for key in ("stage_t", "stage_ad", "stage_ag"):
        path = ROOT / config["reuse"][f"{key}_checkpoint"]
        if sha256_file(path) != config["reuse"][f"{key}_checkpoint_sha256"]:
            raise RuntimeError(f"{key} checkpoint identity failed")
    contract = {
        "schema_version": "stage-ai-canonical-contract-v1",
        "REPRODUCTION_SCOPE": config["reproduction_scope"],
        "EXACT_REPRODUCTION_BLOCKED": True,
        "data": config["data"],
        "preprocessing": config["preprocessing"],
        "prediction": config["prediction"],
        "training": config["training"],
        "models": config["models"],
        "runtime": config["runtime"],
        "FINAL_FNO_REUSE": False,
        "FINAL_CNN_REUSE": False,
        "LOCALNO_RETRAINED": False,
    }
    audit_rows = rows()
    atomic_csv(OUT / "comparability_audit.csv", audit_rows)
    # Keep the root-level path requested by the initial Stage AI audit contract
    # as an identical convenience copy; the structured publication lives in audit/.
    atomic_csv(OUT.parent / "comparability_audit.csv", audit_rows)
    inventory = []
    for row in audit_rows:
        checkpoint = str(row["checkpoint_path"])
        path = ROOT / checkpoint
        inventory.append({
            "model": row["model"],
            "artifact_stage": row["artifact_stage"],
            "checkpoint_path": checkpoint,
            "checkpoint_exists": path.is_file(),
            "checkpoint_size_bytes": path.stat().st_size if path.is_file() else "not_available",
            "checkpoint_sha256": sha256_file(path) if path.is_file() else "not_available",
            "reuse_or_retrain": row["reuse_or_retrain"],
        })
    atomic_csv(OUT / "artifact_inventory.csv", inventory)
    cnn, identity = build_canonical_cnn(seed=int(config["runtime"]["seed"]))
    parameter_rows = []
    for module_name, module in cnn.named_modules():
        direct = sum(parameter.numel() for parameter in module.parameters(recurse=False))
        if direct:
            parameter_rows.append({
                "module": module_name,
                "module_type": type(module).__name__,
                "direct_trainable_parameters": direct,
            })
    if sum(int(row["direct_trainable_parameters"]) for row in parameter_rows) != identity["parameter_count"]:
        raise RuntimeError("CNN parameter-count derivation does not sum to the frozen total")
    atomic_csv(OUT / "cnn_parameter_count_derivation.csv", parameter_rows)
    atomic_json(OUT / "cnn_architecture_contract.json", identity)
    checkpoint_rows = []
    for model in ("fno", "cnn"):
        for path in sorted((OUT.parent / "training" / model / "checkpoints").glob("*.pt")):
            checkpoint_rows.append({
                "model": model,
                "checkpoint_filename": path.name,
                "repository_relative_local_path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "published_to_git": False,
                "regeneration_command": f"python scripts/train_stage_ai_baseline.py --model {model}",
            })
    if checkpoint_rows:
        atomic_csv(ROOT / "docs/stage_ai_excluded_checkpoint_sha256.csv", checkpoint_rows)
    atomic_json(OUT / "canonical_contract.json", contract)
    print(json.dumps({"audit_rows": len(audit_rows), "FINAL_FNO_REUSE": False, "FINAL_CNN_REUSE": False, "dataset_sha256": config["data"]["dataset_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
