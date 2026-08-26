#!/usr/bin/env python3
"""Run the controlled 150-epoch adapted DISCO3D LocalNO experiment."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import numpy as np
import torch

from grmhd.models import trainable_parameter_count
from grmhd.paper_losses import PlainL2Loss
from grmhd.paper_stage_g import tensor_state_sha256
from grmhd.stage_s_training import (
    accumulation_groups,
    natural_epoch_orders,
    order_sha256,
    warmup_cosine_learning_rate,
)

from preflight_stage_ad import build_model_and_hashes, load_contract
from train_stage_s import StageSBatchPath, gradient_norm


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts/stage_ad/training/disco3d_localno"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty Stage AD log")
    fields = sorted({name for row in rows for name in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def tensor_parameters(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value for name, value in model.state_dict().items() if torch.is_tensor(value)}


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def named_groups(model: torch.nn.Module) -> dict[str, list[torch.nn.Parameter]]:
    groups = {"spectral": [], "differential": [], "disco3d": []}
    for name, parameter in model.named_parameters():
        if ".local_convs." in name:
            groups["disco3d"].append(parameter)
        elif ".differential." in name:
            groups["differential"].append(parameter)
        elif ".convs." in name:
            groups["spectral"].append(parameter)
    if any(not values for values in groups.values()):
        raise RuntimeError("Stage AD branch parameter grouping is incomplete")
    return groups


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    state: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "stage-ad-checkpoint-v1",
        "model_state_dict": tensor_parameters(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state": {
            "kind": "stage_t_explicit_update_warmup_cosine",
            "completed_updates": int(state["optimizer_updates"]),
            "total_updates": int(metadata["scheduler_total_updates"]),
            "warmup_updates": int(metadata["scheduler_warmup_updates"]),
        },
        "training_state": dict(state),
        "metadata": dict(metadata),
    }


def train_group(
    *,
    sources: Iterable[int],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: PlainL2Loss,
    data: StageSBatchPath,
    all_parameters: list[torch.nn.Parameter],
    branches: dict[str, list[torch.nn.Parameter]],
    clip: float,
    update: int,
    scheduler_total_updates: int,
    scheduler_warmup_updates: int,
    base_lr: float,
    min_lr: float,
) -> dict[str, Any]:
    batch_sources = [int(value) for value in sources]
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for source in batch_sources:
        result = data.predict(model, source)
        loss = loss_fn(result["predicted_residual"], result["residual_target"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Stage AD training loss became NaN/Inf")
        (loss / len(batch_sources)).backward()
        losses.append(float(loss.detach().cpu()))
    before = gradient_norm(all_parameters)
    branch_gradients = {name: gradient_norm(parameters) for name, parameters in branches.items()}
    if not math.isfinite(before) or before <= 0:
        raise FloatingPointError("Stage AD total gradient is nonfinite or zero")
    if not math.isfinite(branch_gradients["disco3d"]) or branch_gradients["disco3d"] <= 0:
        raise FloatingPointError("Stage AD DISCO gradient is nonfinite or zero")
    torch.nn.utils.clip_grad_norm_(all_parameters, clip)
    after = gradient_norm(all_parameters)
    learning_rate = warmup_cosine_learning_rate(
        update,
        total_updates=scheduler_total_updates,
        warmup_updates=scheduler_warmup_updates,
        base_learning_rate=base_lr,
        min_learning_rate=min_lr,
    )
    for values in optimizer.param_groups:
        values["lr"] = learning_rate
    optimizer.step()
    if not all(torch.isfinite(parameter).all() for parameter in all_parameters):
        raise FloatingPointError("Stage AD optimizer produced NaN/Inf parameters")
    return {
        "loss_mean": float(np.mean(losses)),
        "loss_min": float(np.min(losses)),
        "loss_max": float(np.max(losses)),
        "gradient_norm_before_clip": before,
        "gradient_norm_after_clip": after,
        "spectral_gradient_norm": branch_gradients["spectral"],
        "differential_gradient_norm": branch_gradients["differential"],
        "disco_gradient_norm": branch_gradients["disco3d"],
        "clipping_triggered": before > clip,
        "accumulation_count": len(batch_sources),
        "learning_rate": learning_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--extend-to-300", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Stage AD training refuses CPU fallback")
    config, stage_t, stage_s = load_contract()
    cuda_preflight = json.loads(
        (ROOT / "artifacts/stage_ad/preflight/cuda_preflight.json").read_text(encoding="utf-8")
    )
    runtime = json.loads(
        (ROOT / "artifacts/stage_ad/preflight/runtime_feasibility.json").read_text(encoding="utf-8")
    )
    if cuda_preflight["status"] != "passed" or not runtime["dense_backend_feasible"]:
        raise RuntimeError("Stage AD CUDA/runtime gates are not PASS")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model, hashes = build_model_and_hashes(config, stage_s)
    if trainable_parameter_count(model) != 363_480:
        raise RuntimeError("Stage AD trainable parameter count changed")
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    loss_fn = PlainL2Loss().to(device)
    data = StageSBatchPath(
        ROOT / stage_s["data"]["dataset"],
        ROOT / stage_s["preprocessing"]["artifact"],
        device,
    )
    split = json.loads((ROOT / stage_s["data"]["split"]).read_text(encoding="utf-8"))
    train_pairs = [int(value) for value in split["train_pair_source_indices"]]
    if train_pairs != list(range(168)):
        raise ValueError("Stage AD frozen 168-pair training population changed")
    epochs = 300 if args.extend_to_300 else int(config["training"]["epochs"])
    if args.extend_to_300:
        extension_path = ROOT / "artifacts/stage_ad/comparison/evaluation_summary.json"
        if not extension_path.exists():
            raise FileNotFoundError("Stage AD convergence-extension decision is missing")
        extension = json.loads(extension_path.read_text(encoding="utf-8"))
        if extension["extension_to_300_authorized_by_frozen_rule"] is not True:
            raise RuntimeError("Stage AD frozen rule did not authorize 300 epochs")
        if not args.resume:
            raise ValueError("Stage AD 300-epoch extension must resume the controlled run")
    orders = natural_epoch_orders(
        train_pairs, epochs=epochs, seed=int(stage_s["runtime"]["seed"])
    )
    observed_epoch_hashes = [order_sha256(order) for order in orders]
    stage_t_resolved = json.loads(
        (ROOT / "artifacts/stage_t/full_long/resolved_config.json").read_text(encoding="utf-8")
    )
    if observed_epoch_hashes != stage_t_resolved["epoch_order_sha256"][:epochs]:
        raise ValueError("Stage AD pair order differs from Stage-T first 150 epochs")
    mandatory = {int(value) for value in config["training"]["mandatory_checkpoint_epochs"]}
    if args.extend_to_300:
        mandatory.add(300)
    scheduler_total = int(config["training"]["scheduler_total_updates"])
    scheduler_warmup = int(config["training"]["scheduler_warmup_updates"])
    expected_updates = epochs * 42
    metadata = {
        "contract": "Stage AD adapted isotropic radial DISCO3D",
        "reproduction_scope": config["reproduction_scope"],
        "operator": config["operator"],
        "common_initialization_match": hashes["common_initialization_match"],
        "common_tensor_state_sha256": hashes["common_tensor_state_sha256"],
        "disco_initial_state_sha256": hashes["disco_parameter_state_sha256"],
        "full_initial_state_sha256": hashes["full_initial_state_sha256"],
        "pair_order_sha256": order_sha256(value for order in orders for value in order),
        "epoch_order_sha256": observed_epoch_hashes,
        "pair_order_matches_stage_t_first_150": True,
        "train_pairs": 168,
        "validation_pairs": 42,
        "dropped_boundary_pair": [168, 169],
        "epochs": epochs,
        "microbatches_per_epoch": 168,
        "updates_per_epoch": 42,
        "expected_optimizer_updates": expected_updates,
        "scheduler_total_updates": scheduler_total,
        "scheduler_warmup_updates": scheduler_warmup,
        "loss": "PlainL2Loss_on_normalized_residual",
        "optimizer": config["training"]["optimizer"],
        "learning_rate": config["training"]["learning_rate"],
        "weight_decay": config["training"]["weight_decay"],
        "gradient_clip_norm": config["training"]["gradient_clip_norm"],
        "mixed_precision": False,
        "formal_selector": config["evaluation"]["selector"],
        "validation_not_used_for_training_or_stopping": True,
    }
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    rows = load_rows(output / "train_log.csv") if args.resume else []
    state: dict[str, Any] = {
        "completed_epoch": 0,
        "optimizer_updates": 0,
        "microbatches": 0,
        "runtime_seconds": 0.0,
    }
    resume_path = output / "resume.pt"
    if args.resume:
        if not resume_path.exists():
            raise FileNotFoundError("Stage AD resume checkpoint does not exist")
        payload = torch.load(resume_path, map_location="cpu", weights_only=True)
        if args.extend_to_300:
            prior = payload["metadata"]
            if int(prior["epochs"]) != 150 or int(payload["training_state"]["completed_epoch"]) != 150:
                raise ValueError("Stage AD extension must begin at the completed 150-epoch gate")
            if prior["epoch_order_sha256"] != metadata["epoch_order_sha256"][:150]:
                raise ValueError("Stage AD extension pair-order prefix changed")
            frozen_keys = (
                "contract", "reproduction_scope", "operator",
                "common_initialization_match", "common_tensor_state_sha256",
                "disco_initial_state_sha256", "full_initial_state_sha256",
                "train_pairs", "validation_pairs", "dropped_boundary_pair",
                "microbatches_per_epoch", "updates_per_epoch",
                "scheduler_total_updates", "scheduler_warmup_updates",
                "loss", "optimizer", "learning_rate", "weight_decay",
                "gradient_clip_norm", "mixed_precision", "formal_selector",
                "validation_not_used_for_training_or_stopping",
            )
            if any(prior[key] != metadata[key] for key in frozen_keys):
                raise ValueError("Stage AD scientific metadata changed during extension")
        elif payload["metadata"] != metadata:
            raise ValueError("Stage AD resume metadata changed")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        state = dict(payload["training_state"])
        if len(rows) != int(state["completed_epoch"]):
            raise ValueError("Stage AD resume log/checkpoint mismatch")
        expected_completed_updates = int(state["completed_epoch"]) * 42
        if int(state["optimizer_updates"]) != expected_completed_updates:
            raise ValueError("Stage AD resume update/epoch mismatch")
    elif resume_path.exists() or rows:
        raise FileExistsError("Stage AD output already contains training state; use --resume")

    all_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    branches = named_groups(model)
    initial_disco_hash = hashes["disco_parameter_state_sha256"]
    model.train()
    for epoch_index in range(int(state["completed_epoch"]), epochs):
        epoch = epoch_index + 1
        groups = accumulation_groups(
            orders[epoch_index], accumulation=int(config["training"]["gradient_accumulation"])
        )
        if len(groups) != 42 or any(len(group) != 4 for group in groups):
            raise RuntimeError("Stage AD accumulation contract changed")
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        records = []
        for group in groups:
            update = int(state["optimizer_updates"]) + 1
            record = train_group(
                sources=group,
                model=model,
                optimizer=optimizer,
                loss_fn=loss_fn,
                data=data,
                all_parameters=all_parameters,
                branches=branches,
                clip=float(config["training"]["gradient_clip_norm"]),
                update=update,
                scheduler_total_updates=scheduler_total,
                scheduler_warmup_updates=scheduler_warmup,
                base_lr=float(config["training"]["learning_rate"]),
                min_lr=float(config["training"]["scheduler_min_learning_rate"]),
            )
            state["optimizer_updates"] = update
            state["microbatches"] = int(state["microbatches"]) + record["accumulation_count"]
            records.append(record)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        state["completed_epoch"] = epoch
        state["runtime_seconds"] = float(state["runtime_seconds"]) + elapsed
        row = {
            "epoch": epoch,
            "optimizer_updates": state["optimizer_updates"],
            "microbatches_seen": state["microbatches"],
            "microbatches_this_epoch": 168,
            "optimizer_updates_this_epoch": 42,
            "train_loss_mean": float(np.mean([record["loss_mean"] for record in records])),
            "train_loss_min": float(np.min([record["loss_min"] for record in records])),
            "train_loss_max": float(np.max([record["loss_max"] for record in records])),
            "gradient_norm_before_clip_mean": float(np.mean([record["gradient_norm_before_clip"] for record in records])),
            "gradient_norm_before_clip_max": float(np.max([record["gradient_norm_before_clip"] for record in records])),
            "gradient_norm_after_clip_mean": float(np.mean([record["gradient_norm_after_clip"] for record in records])),
            "spectral_gradient_norm_mean": float(np.mean([record["spectral_gradient_norm"] for record in records])),
            "differential_gradient_norm_mean": float(np.mean([record["differential_gradient_norm"] for record in records])),
            "disco_gradient_norm_mean": float(np.mean([record["disco_gradient_norm"] for record in records])),
            "disco_gradient_norm_min": float(np.min([record["disco_gradient_norm"] for record in records])),
            "disco_weight_norm": gradient_norm(branches["disco3d"]),
            "clipping_fraction": float(np.mean([record["clipping_triggered"] for record in records])),
            "learning_rate_start": records[0]["learning_rate"],
            "learning_rate_end": records[-1]["learning_rate"],
            "nonfinite_count": 0,
            "epoch_runtime_seconds": elapsed,
            "cumulative_runtime_seconds": state["runtime_seconds"],
            "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        }
        # ``gradient_norm`` reads .grad; weight norm must read parameters directly.
        row["disco_weight_norm"] = float(torch.sqrt(torch.stack([
            torch.sum(parameter.detach().float().square()) for parameter in branches["disco3d"]
        ]).sum()).cpu())
        rows.append(row)
        atomic_csv(output / "train_log.csv", rows)
        payload = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            state=state,
            metadata=metadata,
        )
        save_checkpoint(resume_path, payload)
        if epoch in mandatory:
            save_checkpoint(output / "checkpoints" / f"epoch_{epoch:04d}.pt", payload)
        print(json.dumps({
            "epoch": epoch,
            "updates": state["optimizer_updates"],
            "loss": row["train_loss_mean"],
            "lr": row["learning_rate_end"],
            "clip_fraction": row["clipping_fraction"],
            "disco_gradient": row["disco_gradient_norm_mean"],
            "epoch_seconds": elapsed,
            "peak_allocated_mib": row["gpu_peak_allocated_mib"],
        }, sort_keys=True), flush=True)

    if int(state["optimizer_updates"]) != expected_updates:
        raise RuntimeError("Stage AD completed update count changed")
    if int(state["microbatches"]) != epochs * 168:
        raise RuntimeError("Stage AD completed microbatch count changed")
    missing = [
        epoch for epoch in mandatory
        if not (output / "checkpoints" / f"epoch_{epoch:04d}.pt").exists()
    ]
    if missing:
        raise RuntimeError(f"Stage AD mandatory checkpoints missing: {missing}")
    disco_final = {
        name: parameter for name, parameter in model.named_parameters()
        if ".local_convs." in name
    }
    final_disco_hash = tensor_state_sha256(disco_final)
    if final_disco_hash == initial_disco_hash:
        raise RuntimeError("Stage AD DISCO branch did not learn a nonzero update")
    summary = {
        "status": "passed",
        "all_finite": True,
        **state,
        "parameter_count": trainable_parameter_count(model),
        "initial_disco_state_sha256": initial_disco_hash,
        "final_disco_state_sha256": final_disco_hash,
        "disco_weights_changed": True,
        "disco_gradient_nonzero_every_update": min(
            float(row["disco_gradient_norm_min"]) for row in rows
        ) > 0,
        "mandatory_checkpoints": sorted(mandatory),
        "peak_allocated_mib_max": max(float(row["gpu_peak_allocated_mib"]) for row in rows),
        "peak_reserved_mib_max": max(float(row["gpu_peak_reserved_mib"]) for row in rows),
    }
    atomic_json(output / "training_summary.json", summary)
    data.close()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
