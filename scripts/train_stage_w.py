#!/usr/bin/env python3
"""Run one sequential frozen 150-epoch Stage W GPU pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
import yaml

from grmhd.dataset import sha256_file
from grmhd.models import trainable_parameter_count
from grmhd.paper_losses import PlainL2Loss
from grmhd.paper_stage_g import tensor_state_sha256
from grmhd.stage_s_training import (
    accumulation_groups,
    natural_epoch_orders,
    order_sha256,
    warmup_cosine_learning_rate,
)
from grmhd.stage_w_training import VARIANTS, build_stage_w_model, sha256_json

from train_stage_s import StageSBatchPath, gradient_norm
from train_stage_t import (
    atomic_csv,
    atomic_json,
    checkpoint_payload,
    load_rows,
    save_checkpoint,
    verify_frozen_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def load_stage_w(path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[int]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    stage_t_path = ROOT / config["frozen_stage_t_config"]
    if sha256_file(stage_t_path) != config["frozen_stage_t_config_sha256"]:
        raise ValueError("Frozen Stage T config changed")
    _, stage_s, freeze, train_pairs, _, _ = verify_frozen_contract(stage_t_path)
    transform = json.loads((ROOT / "artifacts/stage_w/mixed_transform/transform_unit_tests.json").read_text())
    coordinate = json.loads((ROOT / "artifacts/stage_w/coordinate_basis_audit.json").read_text())
    if not transform["all_core_tests_passed"]:
        raise RuntimeError("Stage W transform tests did not authorize training")
    if not coordinate["log_r"]["LOG_R_GRID_UNIFORM"]:
        raise RuntimeError("Stored log-r grid does not authorize the frozen DCT implementation")
    freeze.update({
        "DATASET_FROZEN": True,
        "SPLIT_FROZEN": True,
        "PREPROCESSING_FROZEN": True,
        "RESIDUAL_CONTRACT_FROZEN": True,
        "LOSS_FROZEN": True,
        "DIFFERENTIAL_BRANCH_FROZEN_EXCEPT_W2_BOUNDARY": True,
        "OPTIMIZER_FROZEN": True,
        "TRAINING_ORDER_FROZEN": True,
    })
    return config, stage_s, freeze, train_pairs


def prepare(
    variant: str, stage_s: Mapping[str, Any], output: Path,
) -> tuple[torch.device, str, torch.nn.Module, torch.optim.Optimizer, PlainL2Loss, StageSBatchPath, dict[str, Any]]:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage W requires CUDA and refuses silent CPU fallback")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    device_name = torch.cuda.get_device_name(device)
    if stage_s["runtime"]["required_device_substring"] not in device_name:
        raise RuntimeError(f"Expected RTX 5070, found {device_name!r}")
    torch.cuda.reset_peak_memory_stats(device)
    seed = int(stage_s["runtime"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    initial = torch.load(
        ROOT / stage_s["frozen_pairing"]["initial_state"], map_location="cpu", weights_only=True
    )
    if tensor_state_sha256(initial) != stage_s["frozen_pairing"]["initial_tensor_state_sha256"]:
        raise ValueError("Frozen Stage-T common initial state changed")
    model, initialization = build_stage_w_model(stage_s, variant, initial_state=initial)
    if abs(float(initialization["parameter_count_relative_difference"])) >= 0.10:
        raise ValueError("Stage W parameter-count delta exceeds 10 percent")
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(stage_s["optimizer"]["learning_rate"]),
        weight_decay=float(stage_s["optimizer"]["weight_decay"]),
    )
    loss = PlainL2Loss().to(device)
    data = StageSBatchPath(
        ROOT / stage_s["data"]["dataset"],
        ROOT / stage_s["preprocessing"]["artifact"], device,
    )
    output.mkdir(parents=True, exist_ok=True)
    return device, device_name, model, optimizer, loss, data, initialization


def preflight(config_path: Path, root: Path) -> None:
    _, stage_s, _, _ = load_stage_w(config_path)
    output = root / "variants/mixed_basis"
    device, device_name, model, optimizer, loss_fn, data, initialization = prepare(
        "mixed_basis", stage_s, output
    )
    del optimizer
    model.train(); model.zero_grad(set_to_none=True)
    batch = data.pair(0)
    # Warm cuFFT plans separately so the recorded throughput represents the
    # steady pilot path rather than one-time plan construction.
    warm_residual = model(x=batch["model_input"])
    warm_loss = loss_fn(warm_residual, batch["residual_target"])
    warm_loss.backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device); started = time.perf_counter()
    residual = model(x=batch["model_input"])
    loss = loss_fn(residual, batch["residual_target"])
    loss.backward()
    torch.cuda.synchronize(device); elapsed = time.perf_counter() - started
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    payload = {
        "schema_version": "stage-w-gpu-preflight-v1",
        "device": device_name,
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "capability": list(torch.cuda.get_device_capability(device)),
        "model_parameters_on_cuda": all(parameter.is_cuda for parameter in model.parameters()),
        "input_on_cuda": batch["model_input"].is_cuda,
        "output_on_cuda": residual.is_cuda,
        "loss_on_cuda": loss.is_cuda,
        "forward_finite": bool(torch.isfinite(residual).all()),
        "loss_finite": bool(torch.isfinite(loss)),
        "gradients_finite": bool(gradients and all(torch.isfinite(value).all() for value in gradients)),
        "forward_backward_seconds": elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        "parameter_count": trainable_parameter_count(model),
        "initial_tensor_state_sha256": initialization["full_initial_tensor_state_sha256"],
    }
    if not all(payload[key] for key in (
        "model_parameters_on_cuda", "input_on_cuda", "output_on_cuda", "loss_on_cuda",
        "forward_finite", "loss_finite", "gradients_finite",
    )):
        raise RuntimeError(f"Stage W real 64^3 GPU preflight failed: {payload}")
    atomic_json(root / "gpu_preflight.json", payload)
    data.close()
    print(json.dumps(payload, sort_keys=True))


def train_group(
    *, group: list[int], model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    loss_fn: PlainL2Loss, data: StageSBatchPath, parameters: list[torch.nn.Parameter],
    clip: float, update: int, total_updates: int, warmup_updates: int,
    base_lr: float, min_lr: float,
) -> dict[str, Any]:
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    for source in group:
        batch = data.predict(model, int(source))
        loss = loss_fn(batch["predicted_residual"], batch["residual_target"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Stage W loss became nonfinite")
        (loss / len(group)).backward()
        losses.append(float(loss.detach().cpu()))
    before = gradient_norm(parameters)
    if not np.isfinite(before) or before <= 0:
        raise FloatingPointError("Stage W gradient became nonfinite or zero")
    torch.nn.utils.clip_grad_norm_(parameters, clip)
    after = gradient_norm(parameters)
    lr = warmup_cosine_learning_rate(
        update, total_updates=total_updates, warmup_updates=warmup_updates,
        base_learning_rate=base_lr, min_learning_rate=min_lr,
    )
    for group_spec in optimizer.param_groups:
        group_spec["lr"] = lr
    optimizer.step()
    if not all(torch.isfinite(parameter).all() for parameter in parameters):
        raise FloatingPointError("Stage W optimizer produced nonfinite parameters")
    return {
        "loss": float(np.mean(losses)), "before": before, "after": after,
        "clipped": before > clip, "lr": lr,
    }


def run(variant: str, config_path: Path, root: Path, resume: bool) -> None:
    config, stage_s, freeze, train_pairs = load_stage_w(config_path)
    if variant == "mixed_basis_boundary":
        prerequisite = root / "variants/mixed_basis/training_summary.json"
        if not prerequisite.exists():
            raise RuntimeError("W2 cannot start before W1 completes")
        w1 = json.loads(prerequisite.read_text())
        if w1.get("status") != "passed" or int(w1.get("completed_epoch", 0)) != 150:
            raise RuntimeError("W2 prerequisite W1 is incomplete")
    output = root / "variants" / variant
    device, device_name, model, optimizer, loss_fn, data, initialization = prepare(
        variant, stage_s, output
    )
    pilot = config["pilots"]
    epochs = int(pilot["epochs"])
    accumulation = int(pilot["gradient_accumulation"])
    orders = natural_epoch_orders(train_pairs, epochs=epochs, seed=int(pilot["seed"]))
    updates_per_epoch = len(accumulation_groups(orders[0], accumulation=accumulation))
    scheduler_total_updates = updates_per_epoch * int(pilot["scheduler_total_epochs"])
    warmup_updates = updates_per_epoch * int(pilot["scheduler_warmup_epochs"])
    pilot_updates = updates_per_epoch * epochs
    stage_t_checkpoint = torch.load(
        ROOT / config["baseline"]["checkpoint"], map_location="cpu", weights_only=True
    )
    expected_epoch_hashes = stage_t_checkpoint["metadata"]["epoch_order_sha256"][:epochs]
    actual_epoch_hashes = [order_sha256(order) for order in orders]
    if actual_epoch_hashes != expected_epoch_hashes:
        raise ValueError("Stage W order differs from Stage T first 150 epochs")
    flat = [value for order in orders for value in order]
    contract = {
        **freeze,
        "schema_version": "stage-w-pilot-v1",
        "variant": variant,
        "only_scientific_change": (
            "spectral_basis" if variant == "mixed_basis"
            else "spectral_basis_plus_predeclared_fd_boundary"
        ),
        "initialization": initialization,
        "epochs": epochs,
        "train_pairs": len(train_pairs),
        "validation_pairs": 42,
        "microbatches_per_epoch": len(train_pairs),
        "updates_per_epoch": updates_per_epoch,
        "pilot_optimizer_updates": pilot_updates,
        "scheduler_total_updates": scheduler_total_updates,
        "warmup_updates": warmup_updates,
        "scheduler_semantics": "same Stage T 1200-epoch schedule truncated after epoch 150",
        "pair_order_sha256": order_sha256(flat),
        "epoch_order_sha256": actual_epoch_hashes,
        "stage_t_first_150_order_match": True,
        "seed": int(pilot["seed"]),
        "device": device_name,
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "python": sys.version,
        "platform": platform.platform(),
        "loss": "PlainL2Loss_on_normalized_residual",
        "mixed_precision": False,
        "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "formal_selector": config["evaluation"]["formal_selector"],
        "validation_not_used_for_loss_weights_training_or_stopping": True,
        "coordinate_channels_added": False,
    }
    contract_hash = sha256_json(contract)
    atomic_json(output / "resolved_config.json", contract)
    state: dict[str, Any] = {
        "completed_epoch": 0, "optimizer_updates": 0,
        "microbatches": 0, "runtime_seconds": 0.0,
    }
    rows = load_rows(output / "train_log.csv") if resume else []
    resume_path = output / "resume.pt"
    if resume:
        if not resume_path.exists():
            raise FileNotFoundError("Stage W resume checkpoint does not exist")
        payload = torch.load(resume_path, map_location="cpu", weights_only=True)
        if payload["metadata"]["contract_sha256"] != contract_hash:
            raise ValueError("Stage W resume contract changed")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        state = dict(payload["training_state"])
        completed = int(state["completed_epoch"])
        if len(rows) < completed:
            raise ValueError("Stage W resume log is shorter than checkpoint")
        rows = rows[:completed]
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    clip = float(stage_s["optimizer"]["gradient_clip_norm"])
    checkpoint_epochs = {int(value) for value in pilot["checkpoint_epochs"]}
    model.train()
    for epoch_index in range(int(state["completed_epoch"]), epochs):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device); started = time.perf_counter()
        records = []
        groups = accumulation_groups(orders[epoch_index], accumulation=accumulation)
        for group in groups:
            update = int(state["optimizer_updates"]) + 1
            record = train_group(
                group=group, model=model, optimizer=optimizer, loss_fn=loss_fn,
                data=data, parameters=parameters, clip=clip, update=update,
                total_updates=scheduler_total_updates, warmup_updates=warmup_updates,
                base_lr=float(stage_s["optimizer"]["learning_rate"]),
                min_lr=float(stage_s["scheduler"]["min_learning_rate"]),
            )
            state["optimizer_updates"] = update
            state["microbatches"] += len(group)
            records.append(record)
        torch.cuda.synchronize(device); elapsed = time.perf_counter() - started
        epoch = epoch_index + 1
        state["completed_epoch"] = epoch
        state["runtime_seconds"] += elapsed
        row = {
            "epoch": epoch,
            "optimizer_updates": state["optimizer_updates"],
            "microbatches_seen": state["microbatches"],
            "microbatches_this_epoch": len(orders[epoch_index]),
            "optimizer_updates_this_epoch": len(groups),
            "train_loss_mean": float(np.mean([record["loss"] for record in records])),
            "train_loss_min": float(np.min([record["loss"] for record in records])),
            "train_loss_max": float(np.max([record["loss"] for record in records])),
            "learning_rate_start": records[0]["lr"],
            "learning_rate_end": records[-1]["lr"],
            "gradient_norm_before_clip_mean": float(np.mean([record["before"] for record in records])),
            "gradient_norm_before_clip_max": float(np.max([record["before"] for record in records])),
            "gradient_norm_after_clip_mean": float(np.mean([record["after"] for record in records])),
            "gradient_norm_after_clip_max": float(np.max([record["after"] for record in records])),
            "clipping_fraction": float(np.mean([record["clipped"] for record in records])),
            "nonfinite_count": 0,
            "epoch_runtime_seconds": elapsed,
            "cumulative_runtime_seconds": state["runtime_seconds"],
            "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        }
        rows.append(row); atomic_csv(output / "train_log.csv", rows)
        metadata = {
            **contract, "contract_sha256": contract_hash,
            "total_updates": scheduler_total_updates, "warmup_updates": warmup_updates,
            "epoch_train_loss": row["train_loss_mean"],
            "epoch_learning_rate": row["learning_rate_end"],
        }
        payload = checkpoint_payload(
            model=model, optimizer=optimizer, training_state=state, metadata=metadata
        )
        if epoch in checkpoint_epochs:
            save_checkpoint(output / "checkpoints" / f"epoch_{epoch:04d}.pt", payload)
        if epoch % 10 == 0 or epoch in checkpoint_epochs:
            save_checkpoint(resume_path, payload)
        print(json.dumps({
            "stage": "W", "variant": variant, "epoch": epoch,
            "updates": state["optimizer_updates"], "loss": row["train_loss_mean"],
            "lr": row["learning_rate_end"], "clip": row["clipping_fraction"],
            "seconds": elapsed,
        }), flush=True)
    if state["optimizer_updates"] != pilot_updates:
        raise RuntimeError("Stage W pilot update count changed")
    atomic_json(output / "training_summary.json", {
        "schema_version": "stage-w-training-summary-v1",
        "status": "passed", "all_finite": True, **state,
        "initial_model_state_sha256": initialization["full_initial_tensor_state_sha256"],
        "final_model_state_sha256": tensor_state_sha256(model.state_dict()),
        "parameter_count": trainable_parameter_count(model),
        "checkpoint_epochs": sorted(checkpoint_epochs),
        "peak_allocated_mib_max": max(float(row["gpu_peak_allocated_mib"]) for row in rows),
        "peak_reserved_mib_max": max(float(row["gpu_peak_reserved_mib"]) for row in rows),
    })
    data.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--config", type=Path, default=Path("configs/stage_w/mixed_basis.yaml"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/stage_w"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only:
        preflight(args.config, args.root)
        return
    if args.variant is None:
        parser.error("--variant is required unless --preflight-only is used")
    run(args.variant, args.config, args.root, args.resume)


if __name__ == "__main__":
    main()
