#!/usr/bin/env python3
"""Train frozen Stage-T models with Stage-V objective-only interventions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import yaml

from grmhd.dataset import sha256_file
from grmhd.models import trainable_parameter_count
from grmhd.paper_stage_g import tensor_state_sha256
from grmhd.paper_stage_g_evaluation import radial_shell_indices
from grmhd.stage_s_training import (
    accumulation_groups,
    natural_epoch_orders,
    order_sha256,
    warmup_cosine_learning_rate,
)
from grmhd.stage_v_objectives import ObjectiveWeights, combined_objective, objective_components

from train_stage_s import StageSBatchPath, build_frozen_model, gradient_norm
from train_stage_t import checkpoint_payload, verify_frozen_contract


ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("direction", "transport", "direction_transport")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty Stage V train log")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def json_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tensor_only_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value for key, value in model.state_dict().items() if torch.is_tensor(value)}


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def variant_weights(variant: str, scaling: Mapping[str, Any]) -> ObjectiveWeights:
    if variant == "direction":
        return ObjectiveWeights(direction=float(scaling["direction"]["lambda_single"]))
    if variant == "transport":
        return ObjectiveWeights(transport=float(scaling["transport"]["lambda_single"]))
    if variant == "direction_transport":
        return ObjectiveWeights(
            direction=float(scaling["direction"]["lambda_combined"]),
            transport=float(scaling["transport"]["lambda_combined"]),
        )
    raise ValueError(f"Unknown Stage V variant: {variant}")


def prepare(
    stage_s: Mapping[str, Any], freeze: Mapping[str, Any]
) -> tuple[torch.device, str, torch.nn.Module, torch.optim.Optimizer, StageSBatchPath, np.ndarray]:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage V training requires CUDA; refusing silent CPU fallback")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    device_name = torch.cuda.get_device_name(device)
    if stage_s["runtime"]["required_device_substring"] not in device_name:
        raise RuntimeError(f"Stage V expected RTX 5070, found {device_name!r}")
    torch.cuda.reset_peak_memory_stats(device)
    seed = int(stage_s["runtime"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_frozen_model(stage_s)
    initial = torch.load(ROOT / stage_s["frozen_pairing"]["initial_state"], map_location="cpu", weights_only=True)
    if tensor_state_sha256(initial) != freeze["initial_tensor_state_sha256"]:
        raise ValueError("Frozen shared tensor state changed")
    model.load_state_dict(initial, strict=True)
    if trainable_parameter_count(model) != int(freeze["parameter_count"]):
        raise ValueError("Frozen Stage-T parameter count changed")
    if tensor_state_sha256(model.state_dict()) != freeze["initial_tensor_state_sha256"]:
        raise ValueError("Stage V failed to strictly load shared initialization")
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(stage_s["optimizer"]["learning_rate"]),
        weight_decay=float(stage_s["optimizer"]["weight_decay"]),
    )
    data = StageSBatchPath(
        ROOT / stage_s["data"]["dataset"], ROOT / stage_s["preprocessing"]["artifact"], device
    )
    r = np.asarray(data.handle["coords/r"], dtype=np.float64)
    _, shell_index = radial_shell_indices(r, int(stage_s["representation"]["shell_count"]))
    return device, device_name, model, optimizer, data, shell_index


def train_group(
    *,
    group: Iterable[int],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    data: StageSBatchPath,
    shell_index: torch.Tensor,
    weights: ObjectiveWeights,
    parameters: Sequence[torch.nn.Parameter],
    clip: float,
    update: int,
    scheduler_total_updates: int,
    scheduler_warmup_updates: int,
    base_lr: float,
    min_lr: float,
    epsilon: float,
) -> dict[str, Any]:
    sources = [int(value) for value in group]
    optimizer.zero_grad(set_to_none=True)
    logs: dict[str, list[float]] = {
        name: [] for name in ("plain", "direction", "shell", "radial", "transport", "total")
    }
    enabled = {"plain"}
    if weights.direction:
        enabled.add("direction")
    if weights.transport:
        enabled.update(("shell", "radial", "transport"))
    for source in sources:
        result = data.predict(model, source)
        components = objective_components(
            predicted_residual=result["predicted_residual"],
            residual_target=result["residual_target"],
            normalized_input=result["z_input"],
            normalized_target=result["z_target"],
            preprocessor=data.preprocessor,
            shell_index=shell_index,
            epsilon=epsilon,
            enabled=enabled,
        )
        total = combined_objective(components, weights)
        if not torch.isfinite(total) or not all(torch.isfinite(value) for value in components.values()):
            raise FloatingPointError(f"Stage V objective became NaN/Inf at pair {source}")
        (total / len(sources)).backward()
        logs["total"].append(float(total.detach().cpu()))
        for name, value in components.items():
            logs[name].append(float(value.detach().cpu()))
    before = gradient_norm(parameters)
    if not math.isfinite(before) or before <= 0:
        raise FloatingPointError("Stage V combined gradient is nonfinite or zero")
    torch.nn.utils.clip_grad_norm_(parameters, clip)
    after = gradient_norm(parameters)
    learning_rate = warmup_cosine_learning_rate(
        update,
        total_updates=scheduler_total_updates,
        warmup_updates=scheduler_warmup_updates,
        base_learning_rate=base_lr,
        min_learning_rate=min_lr,
    )
    for group_values in optimizer.param_groups:
        group_values["lr"] = learning_rate
    optimizer.step()
    if not all(torch.isfinite(parameter).all() for parameter in parameters):
        raise FloatingPointError("Stage V optimizer produced NaN/Inf parameters")
    return {
        **{f"{name}_mean": (None if not values else float(np.mean(values))) for name, values in logs.items()},
        "direction_weighted_mean": (
            None if not logs["direction"] else weights.direction * float(np.mean(logs["direction"]))
        ),
        "transport_weighted_mean": (
            None if not logs["transport"] else weights.transport * float(np.mean(logs["transport"]))
        ),
        "gradient_norm_before_clip": before,
        "gradient_norm_after_clip": after,
        "clipping_triggered": before > clip,
        "accumulation_count": len(sources),
        "learning_rate": learning_rate,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/stage_v/objective_alignment.yaml"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/stage_v"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    stage_t, stage_s, freeze, train_pairs, _, _ = verify_frozen_contract(ROOT / config["frozen_stage_t_config"])
    scaling_path = args.root / "loss_weights/train_only_gradient_scaling.json"
    scaling = json.loads(scaling_path.read_text(encoding="utf-8"))
    weights = variant_weights(args.variant, scaling)
    device, device_name, model, optimizer, data, shell_index_np = prepare(stage_s, freeze)
    shell_index = torch.as_tensor(shell_index_np, device=device, dtype=torch.long)
    pilot = config["pilots"]
    epochs = int(pilot["epochs"])
    updates_per_epoch = int(pilot["optimizer_updates_per_epoch"])
    total_updates = int(pilot["total_optimizer_updates"])
    scheduler_total = int(pilot["scheduler_total_updates"])
    scheduler_warmup = int(pilot["scheduler_warmup_updates"])
    accumulation = int(stage_s["runtime"]["gradient_accumulation"])
    orders = natural_epoch_orders(train_pairs, epochs=epochs, seed=int(stage_s["runtime"]["seed"]))
    if any(len(order) != 168 or set(order) != set(train_pairs) for order in orders):
        raise RuntimeError("Stage V epoch orders violate the frozen population")
    if any(len(accumulation_groups(order, accumulation=accumulation)) != updates_per_epoch for order in orders):
        raise RuntimeError("Stage V update count per epoch changed")
    flat_order = [source for order in orders for source in order]
    # Stage T generated all 1200 orders from the same deterministic function;
    # the first 150 hashes must therefore match exactly.
    stage_t_resolved = json.loads((ROOT / "artifacts/stage_t/full_long/resolved_config.json").read_text())
    observed_epoch_hashes = [order_sha256(order) for order in orders]
    if observed_epoch_hashes != stage_t_resolved["epoch_order_sha256"][:epochs]:
        raise ValueError("Stage V pair order differs from Stage T")
    output = args.root / "variants" / args.variant
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_epochs = {int(value) for value in pilot["checkpoint_epochs"]}
    contract = {
        **freeze,
        "schema_version": "stage-v-objective-pilot-v1",
        "variant": args.variant,
        "objective_weights": {"direction": weights.direction, "transport": weights.transport},
        "objective_definitions": config["objectives"],
        "train_only_gradient_scaling_sha256": sha256_file(scaling_path),
        "config_sha256": sha256_file(args.config),
        "epochs": epochs,
        "optimizer_updates": total_updates,
        "microbatches": epochs * len(train_pairs),
        "pair_order_sha256": order_sha256(flat_order),
        "epoch_order_sha256": observed_epoch_hashes,
        "stage_t_first_150_order_match": True,
        "scheduler_total_updates": scheduler_total,
        "scheduler_warmup_updates": scheduler_warmup,
        "formal_selector": config["selection"]["formal"],
        "validation_not_used_for_training_or_weight_selection": True,
        "early_stopping": False,
        "mixed_precision": False,
        "device": device_name,
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "python": sys.version,
        "platform": platform.platform(),
        "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "disabled": ["H1", "ROI", "bounds", "radial_envelope", "dissipation", "rollout_stabilization"],
    }
    contract_hash = json_sha256(contract)
    write_json(output / "resolved_config.json", contract)
    state: dict[str, Any] = {
        "completed_epoch": 0,
        "optimizer_updates": 0,
        "microbatches": 0,
        "runtime_seconds": 0.0,
    }
    rows = read_rows(output / "train_log.csv") if args.resume else []
    resume_path = output / "resume.pt"
    if args.resume:
        if not resume_path.exists():
            raise FileNotFoundError("Stage V resume checkpoint is missing")
        payload = torch.load(resume_path, map_location="cpu", weights_only=True)
        if payload["metadata"]["contract_sha256"] != contract_hash:
            raise ValueError("Stage V resume contract changed")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        state = dict(payload["training_state"])
        if len(rows) != int(state["completed_epoch"]):
            raise ValueError("Stage V resume log/checkpoint mismatch")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    clip = float(stage_s["optimizer"]["gradient_clip_norm"])
    model.train()
    for epoch_index in range(int(state["completed_epoch"]), epochs):
        epoch = epoch_index + 1
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        records: list[dict[str, Any]] = []
        for group in accumulation_groups(orders[epoch_index], accumulation=accumulation):
            update = int(state["optimizer_updates"]) + 1
            record = train_group(
                group=group,
                model=model,
                optimizer=optimizer,
                data=data,
                shell_index=shell_index,
                weights=weights,
                parameters=parameters,
                clip=clip,
                update=update,
                scheduler_total_updates=scheduler_total,
                scheduler_warmup_updates=scheduler_warmup,
                base_lr=float(stage_s["optimizer"]["learning_rate"]),
                min_lr=float(stage_s["scheduler"]["min_learning_rate"]),
                epsilon=float(config["gradient_audit"]["epsilon"]),
            )
            state["optimizer_updates"] = update
            state["microbatches"] = int(state["microbatches"]) + int(record["accumulation_count"])
            records.append(record)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        state["completed_epoch"] = epoch
        state["runtime_seconds"] = float(state["runtime_seconds"]) + elapsed
        row: dict[str, Any] = {
            "epoch": epoch,
            "optimizer_updates": state["optimizer_updates"],
            "microbatches_seen": state["microbatches"],
            "microbatches_this_epoch": len(orders[epoch_index]),
            "optimizer_updates_this_epoch": len(records),
            "learning_rate_start": records[0]["learning_rate"],
            "learning_rate_end": records[-1]["learning_rate"],
            "clipping_fraction": float(np.mean([record["clipping_triggered"] for record in records])),
            "gradient_norm_before_clip_mean": float(np.mean([record["gradient_norm_before_clip"] for record in records])),
            "gradient_norm_before_clip_max": float(np.max([record["gradient_norm_before_clip"] for record in records])),
            "gradient_norm_after_clip_mean": float(np.mean([record["gradient_norm_after_clip"] for record in records])),
            "nonfinite_count": 0,
            "epoch_runtime_seconds": elapsed,
            "cumulative_runtime_seconds": state["runtime_seconds"],
            "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        }
        for name in ("plain_mean", "direction_mean", "shell_mean", "radial_mean", "transport_mean", "total_mean", "direction_weighted_mean", "transport_weighted_mean"):
            values = [float(record[name]) for record in records if record[name] is not None]
            row[f"train_{name}"] = None if not values else float(np.mean(values))
        rows.append(row)
        write_csv(output / "train_log.csv", rows)
        metadata = {
            **contract,
            "contract_sha256": contract_hash,
            "total_updates": scheduler_total,
            "warmup_updates": scheduler_warmup,
            "pilot_stop_updates": total_updates,
            "epoch_train_metrics": row,
        }
        payload = checkpoint_payload(model=model, optimizer=optimizer, training_state=state, metadata=metadata)
        save_checkpoint(resume_path, payload)
        if epoch in checkpoint_epochs:
            save_checkpoint(output / "checkpoints" / f"epoch_{epoch:04d}.pt", payload)
        print(json.dumps({
            "stage": "V",
            "variant": args.variant,
            "epoch": epoch,
            "updates": state["optimizer_updates"],
            "plain": row["train_plain_mean"],
            "direction": row["train_direction_mean"],
            "transport": row["train_transport_mean"],
            "total": row["train_total_mean"],
            "clip_fraction": row["clipping_fraction"],
            "epoch_seconds": elapsed,
        }, sort_keys=True), flush=True)
    if int(state["optimizer_updates"]) != total_updates or int(state["microbatches"]) != epochs * 168:
        raise RuntimeError("Stage V completed budget differs from 150 epochs / 6300 updates")
    missing = [epoch for epoch in checkpoint_epochs if not (output / "checkpoints" / f"epoch_{epoch:04d}.pt").exists()]
    if missing:
        raise RuntimeError(f"Stage V checkpoints missing: {missing}")
    final_hash = tensor_state_sha256(model.state_dict())
    if final_hash == freeze["initial_tensor_state_sha256"]:
        raise RuntimeError("Stage V model did not learn a nonzero parameter update")
    write_json(output / "training_summary.json", {
        "status": "passed",
        "all_finite": True,
        **state,
        "final_model_state_sha256": final_hash,
        "initial_model_state_sha256": freeze["initial_tensor_state_sha256"],
        "objective_weights": {"direction": weights.direction, "transport": weights.transport},
        "checkpoint_epochs": sorted(checkpoint_epochs),
        "peak_allocated_mib_max": max(float(row["gpu_peak_allocated_mib"]) for row in rows),
        "peak_reserved_mib_max": max(float(row["gpu_peak_reserved_mib"]) for row in rows),
    })
    data.close()


if __name__ == "__main__":
    main()
