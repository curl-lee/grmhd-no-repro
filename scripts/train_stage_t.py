#!/usr/bin/env python3
"""Run frozen Stage T causal-control and long-horizon optimization experiments."""

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
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml

from grmhd.dataset import sha256_file
from grmhd.models import trainable_parameter_count
from grmhd.paper_losses import PlainL2Loss
from grmhd.paper_stage_g import tensor_state_sha256
from grmhd.stage_s_training import (
    accumulation_groups,
    matched_microbatch_order,
    natural_epoch_orders,
    order_sha256,
    warmup_cosine_learning_rate,
)
from grmhd.stage_t_training import resolve_optimization_budget

from train_stage_s import StageSBatchPath, build_frozen_model, gradient_norm


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty Stage T log")
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def tensor_only_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value for key, value in model.state_dict().items() if torch.is_tensor(value)
    }


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def verify_frozen_contract(
    stage_t_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[int], list[int], list[int]]:
    stage_t = yaml.safe_load(stage_t_path.read_text(encoding="utf-8"))
    stage_s_path = ROOT / stage_t["frozen_stage_s_config"]
    if sha256_file(stage_s_path) != stage_t["frozen_stage_s_config_sha256"]:
        raise ValueError("Frozen Stage S config checksum changed")
    stage_s = yaml.safe_load(stage_s_path.read_text(encoding="utf-8"))
    upstream = subprocess.check_output(
        ["git", "-C", str(ROOT / "external/neuraloperator"), "rev-parse", "HEAD"], text=True
    ).strip()
    upstream_dirty = subprocess.check_output(
        ["git", "-C", str(ROOT / "external/neuraloperator"), "status", "--short"], text=True
    ).strip()
    if upstream != stage_s["frozen_pairing"]["upstream_commit"] or upstream_dirty:
        raise ValueError("Pinned neuraloperator provenance changed")
    dataset = ROOT / stage_s["data"]["dataset"]
    normalizer = ROOT / stage_s["preprocessing"]["artifact"] / "normalizer.npz"
    split_path = ROOT / stage_s["data"]["split"]
    small_path = ROOT / stage_s["data"]["small_pairs"]
    initial_path = ROOT / stage_s["frozen_pairing"]["initial_state"]
    expected = {
        dataset: stage_s["data"]["dataset_sha256"],
        normalizer: stage_s["preprocessing"]["normalizer_sha256"],
        initial_path: stage_s["frozen_pairing"]["initial_state_file_sha256"],
        split_path: "4a909a4294a36d50b7effab093d711b28d5da650b9942f29658d97f5a1a36160",
        small_path: "980d3f551334c77c9916dc786af6152e917a75bc0abb7e77e8c94f575ac276d0",
    }
    observed = {str(path.relative_to(ROOT)): sha256_file(path) for path in expected}
    for path, digest in expected.items():
        if sha256_file(path) != digest:
            raise ValueError(f"Frozen Stage S artifact changed: {path}")
    split = read_json(split_path)
    train_pairs = [int(value) for value in split["train_pair_source_indices"]]
    validation_pairs = [int(value) for value in split["validation_pair_source_indices"]]
    small_pairs = [
        int(line.split()[0]) for line in small_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if train_pairs != list(range(168)) or validation_pairs != list(range(169, 211)):
        raise ValueError("Frozen Stage S train/validation split changed")
    if len(small_pairs) != 79 or not set(small_pairs) <= set(train_pairs):
        raise ValueError("Frozen 79-pair population changed")
    freeze = {
        "DATASET_FROZEN": True,
        "SPLIT_FROZEN": True,
        "PREPROCESSING_FROZEN": True,
        "MODEL_FROZEN": True,
        "checksums": observed,
        "upstream_commit": upstream,
        "shell_definition": stage_s["representation"],
        "model_resolved_config": stage_s["model"],
        "parameter_count": stage_s["frozen_pairing"]["expected_parameter_count"],
        "optimizer": stage_s["optimizer"],
        "initial_tensor_state_sha256": stage_s["frozen_pairing"]["initial_tensor_state_sha256"],
        "train_pair_count": len(train_pairs),
        "small_pair_count": len(small_pairs),
        "validation_pair_count": len(validation_pairs),
        "dropped_boundary_pair": split["dropped_boundary_pair"],
    }
    return stage_t, stage_s, freeze, train_pairs, small_pairs, validation_pairs


def prepare_runtime(
    stage_s: Mapping[str, Any], freeze: Mapping[str, Any], output: Path,
) -> tuple[torch.device, str, torch.nn.Module, torch.optim.Optimizer, PlainL2Loss, StageSBatchPath]:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage T requires CUDA; refusing silent CPU fallback")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    device_name = torch.cuda.get_device_name(device)
    if stage_s["runtime"]["required_device_substring"] not in device_name:
        raise RuntimeError(f"Stage T expected RTX 5070, found {device_name!r}")
    torch.cuda.reset_peak_memory_stats(device)
    seed = int(stage_s["runtime"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_frozen_model(stage_s)
    initial = torch.load(
        ROOT / stage_s["frozen_pairing"]["initial_state"], map_location="cpu", weights_only=True
    )
    if tensor_state_sha256(initial) != stage_s["frozen_pairing"]["initial_tensor_state_sha256"]:
        raise ValueError("Frozen initial tensor state changed")
    model.load_state_dict(initial, strict=True)
    if trainable_parameter_count(model) != int(freeze["parameter_count"]):
        raise ValueError("Frozen LocalNO parameter count changed")
    if tensor_state_sha256(model.state_dict()) != freeze["initial_tensor_state_sha256"]:
        raise ValueError("Model did not strictly load the frozen shared initial state")
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
    return device, device_name, model, optimizer, loss, data


def train_group(
    *, group: Iterable[int], model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    loss_fn: PlainL2Loss, data: StageSBatchPath, parameters: list[torch.nn.Parameter],
    clip: float, update: int, total_updates: int, warmup_updates: int,
    base_lr: float, min_lr: float,
) -> dict[str, Any]:
    sources = [int(value) for value in group]
    optimizer.zero_grad(set_to_none=True)
    losses: list[float] = []
    for source in sources:
        result = data.predict(model, source)
        loss = loss_fn(result["predicted_residual"], result["residual_target"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Stage T training loss became NaN/Inf")
        (loss / len(sources)).backward()
        losses.append(float(loss.detach().cpu()))
    before = gradient_norm(parameters)
    if not math.isfinite(before) or before <= 0:
        raise FloatingPointError("Stage T gradient is nonfinite or zero")
    torch.nn.utils.clip_grad_norm_(parameters, clip)
    after = gradient_norm(parameters)
    lr = warmup_cosine_learning_rate(
        update, total_updates=total_updates, warmup_updates=warmup_updates,
        base_learning_rate=base_lr, min_learning_rate=min_lr,
    )
    for values in optimizer.param_groups:
        values["lr"] = lr
    optimizer.step()
    if not all(torch.isfinite(parameter).all() for parameter in parameters):
        raise FloatingPointError("Stage T optimizer produced NaN/Inf parameters")
    return {
        "mean_microbatch_loss": float(np.mean(losses)),
        "min_microbatch_loss": float(np.min(losses)),
        "max_microbatch_loss": float(np.max(losses)),
        "gradient_norm_before_clip": before,
        "gradient_norm_after_clip": after,
        "clipping_triggered": before > clip,
        "accumulation_count": len(sources),
        "learning_rate": lr,
    }


def checkpoint_payload(
    *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    training_state: Mapping[str, Any], metadata: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "stage-t-checkpoint-v1",
        "model_state_dict": tensor_only_state(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state": {
            "kind": "explicit_update_warmup_cosine",
            "completed_updates": int(training_state["optimizer_updates"]),
            "total_updates": int(metadata["total_updates"]),
            "warmup_updates": int(metadata["warmup_updates"]),
        },
        "training_state": dict(training_state),
        "metadata": dict(metadata),
    }


def run_small(
    *, stage_t: Mapping[str, Any], stage_s: Mapping[str, Any], freeze: Mapping[str, Any],
    small_pairs: list[int], output: Path, resume: bool,
) -> None:
    device, device_name, model, optimizer, loss_fn, data = prepare_runtime(stage_s, freeze, output)
    spec = stage_t["causal_control"]
    accumulation = int(stage_s["runtime"]["gradient_accumulation"])
    total_updates = int(spec["optimizer_updates"])
    warmup_updates = int(spec["warmup_updates"])
    flat_order = matched_microbatch_order(
        small_pairs, optimizer_updates=total_updates, accumulation=accumulation,
        seed=int(stage_s["runtime"]["seed"]),
    )
    groups = accumulation_groups(flat_order, accumulation=accumulation)
    if len(groups) != total_updates or any(len(group) != accumulation for group in groups):
        raise RuntimeError("T-small-1260 order violates the matched update contract")
    contract = {
        **freeze,
        "experiment": "T-small-1260",
        "train_pairs": len(small_pairs),
        "optimizer_updates": total_updates,
        "total_updates": total_updates,
        "effective_microbatches": len(flat_order),
        "scheduler_total_updates": total_updates,
        "warmup_updates": warmup_updates,
        "order_sha256": order_sha256(flat_order),
        "device": device_name,
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "mixed_precision": False,
        "loss": "PlainL2Loss_on_normalized_residual",
        "disabled": ["H1", "physics_loss", "rollout_stabilization"],
        "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    contract_hash = sha256_json(contract)
    atomic_json(output / "small_1260_config.json", contract)
    state: dict[str, Any] = {"optimizer_updates": 0, "microbatches": 0, "runtime_seconds": 0.0}
    rows = load_rows(output / "small_1260_train_log.csv") if resume else []
    resume_path = output / "resume.pt"
    if resume:
        if not resume_path.exists():
            raise FileNotFoundError("T-small resume checkpoint does not exist")
        payload = torch.load(resume_path, map_location="cpu", weights_only=True)
        if payload["metadata"]["contract_sha256"] != contract_hash:
            raise ValueError("T-small resume contract changed")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        state = dict(payload["training_state"])
        if len(rows) != int(state["optimizer_updates"]):
            raise ValueError("T-small resume log/checkpoint mismatch")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    clip = float(stage_s["optimizer"]["gradient_clip_norm"])
    started = time.perf_counter()
    model.train()
    for group_index in range(int(state["optimizer_updates"]), total_updates):
        update = group_index + 1
        record = train_group(
            group=groups[group_index], model=model, optimizer=optimizer, loss_fn=loss_fn,
            data=data, parameters=parameters, clip=clip, update=update,
            total_updates=total_updates, warmup_updates=warmup_updates,
            base_lr=float(stage_s["optimizer"]["learning_rate"]),
            min_lr=float(stage_s["scheduler"]["min_learning_rate"]),
        )
        state["optimizer_updates"] = update
        state["microbatches"] = update * accumulation
        rows.append({"optimizer_update": update, "microbatches_seen": state["microbatches"], **record})
        if update % 100 == 0 or update == total_updates:
            state["runtime_seconds"] = float(state.get("runtime_seconds", 0.0)) + time.perf_counter() - started
            metadata = {**contract, "contract_sha256": contract_hash, "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20}
            save_checkpoint(resume_path, checkpoint_payload(
                model=model, optimizer=optimizer, training_state=state, metadata=metadata,
            ))
            atomic_csv(output / "small_1260_train_log.csv", rows)
            print(json.dumps({
                "experiment": "T-small-1260", "update": update,
                "loss": record["mean_microbatch_loss"], "lr": record["learning_rate"],
                "peak_allocated_mib": metadata["peak_allocated_mib"],
            }, sort_keys=True), flush=True)
            started = time.perf_counter()
    final_metadata = {
        **contract, "contract_sha256": contract_hash,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
    }
    final = checkpoint_payload(model=model, optimizer=optimizer, training_state=state, metadata=final_metadata)
    save_checkpoint(output / "small_1260.pt", final)
    if tensor_state_sha256(final["model_state_dict"]) == freeze["initial_tensor_state_sha256"]:
        raise RuntimeError("T-small-1260 did not learn a non-zero update")
    atomic_json(output / "training_summary.json", {
        "status": "passed", "all_finite": True, **state,
        "final_model_state_sha256": tensor_state_sha256(final["model_state_dict"]),
        "peak_allocated_mib": final_metadata["peak_allocated_mib"],
        "peak_reserved_mib": final_metadata["peak_reserved_mib"],
    })
    data.close()


def run_full(
    *, stage_t: Mapping[str, Any], stage_s: Mapping[str, Any], freeze: Mapping[str, Any],
    train_pairs: list[int], output: Path, resume: bool,
) -> None:
    device, device_name, model, optimizer, loss_fn, data = prepare_runtime(stage_s, freeze, output)
    spec = stage_t["full_long"]
    budget = resolve_optimization_budget(
        pair_count=len(train_pairs), batch_size=int(stage_s["runtime"]["batch_size"]),
        gradient_accumulation=int(stage_s["runtime"]["gradient_accumulation"]),
        total_epochs=int(spec["total_epochs"]), warmup_epochs=int(spec["warmup_epochs"]),
    )
    orders = natural_epoch_orders(
        train_pairs, epochs=budget.total_epochs, seed=int(stage_s["runtime"]["seed"])
    )
    if any(len(order) != budget.microbatches_per_epoch or set(order) != set(train_pairs) for order in orders):
        raise RuntimeError("Stage T full-long epoch order is incomplete")
    flat_order = [value for order in orders for value in order]
    mandatory = {int(value) for value in spec["mandatory_checkpoint_epochs"]}
    contract = {
        **freeze,
        "experiment": "T-full-long-1200epoch",
        "budget": budget.as_dict(),
        "scheduler": {
            "name": "warmup_cosine_by_optimizer_update",
            "warmup_epochs_equivalent": budget.warmup_epochs,
            "warmup_updates": budget.warmup_updates,
            "total_epochs": budget.total_epochs,
            "total_updates": budget.total_updates,
            "base_learning_rate": float(stage_s["optimizer"]["learning_rate"]),
            "minimum_learning_rate": float(stage_s["scheduler"]["min_learning_rate"]),
        },
        "mandatory_checkpoint_epochs": sorted(mandatory),
        "pair_order_sha256": order_sha256(flat_order),
        "epoch_order_sha256": [order_sha256(order) for order in orders],
        "device": device_name,
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "python": sys.version,
        "platform": platform.platform(),
        "mixed_precision": False,
        "loss": "PlainL2Loss_on_normalized_residual",
        "formal_selector": stage_t["evaluation"]["formal_checkpoint_selector"],
        "validation_not_used_for_training_or_stopping": True,
        "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    contract_hash = sha256_json(contract)
    atomic_json(output / "resolved_config.json", contract)
    rows = load_rows(output / "train_log.csv") if resume else []
    state: dict[str, Any] = {
        "completed_epoch": 0, "optimizer_updates": 0, "microbatches": 0,
        "runtime_seconds": 0.0,
    }
    resume_path = output / "resume.pt"
    if resume:
        if not resume_path.exists():
            raise FileNotFoundError("Stage T full-long resume checkpoint does not exist")
        payload = torch.load(resume_path, map_location="cpu", weights_only=True)
        if payload["metadata"]["contract_sha256"] != contract_hash:
            raise ValueError("Stage T full-long resume contract changed")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        state = dict(payload["training_state"])
        if len(rows) != int(state["completed_epoch"]):
            raise ValueError("Stage T full-long resume log/checkpoint mismatch")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    clip = float(stage_s["optimizer"]["gradient_clip_norm"])
    model.train()
    for epoch_index in range(int(state["completed_epoch"]), budget.total_epochs):
        epoch = epoch_index + 1
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        losses: list[float] = []
        before: list[float] = []
        after: list[float] = []
        clipped: list[bool] = []
        learning_rates: list[float] = []
        groups = accumulation_groups(orders[epoch_index], accumulation=budget.gradient_accumulation)
        if len(groups) != budget.updates_per_epoch:
            raise RuntimeError("Stage T full-long update count changed")
        for group in groups:
            update = int(state["optimizer_updates"]) + 1
            record = train_group(
                group=group, model=model, optimizer=optimizer, loss_fn=loss_fn,
                data=data, parameters=parameters, clip=clip, update=update,
                total_updates=budget.total_updates, warmup_updates=budget.warmup_updates,
                base_lr=float(stage_s["optimizer"]["learning_rate"]),
                min_lr=float(stage_s["scheduler"]["min_learning_rate"]),
            )
            state["optimizer_updates"] = update
            state["microbatches"] = int(state["microbatches"]) + int(record["accumulation_count"])
            losses.append(float(record["mean_microbatch_loss"]))
            before.append(float(record["gradient_norm_before_clip"]))
            after.append(float(record["gradient_norm_after_clip"]))
            clipped.append(bool(record["clipping_triggered"]))
            learning_rates.append(float(record["learning_rate"]))
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        state["completed_epoch"] = epoch
        state["runtime_seconds"] = float(state["runtime_seconds"]) + elapsed
        epoch_row = {
            "epoch": epoch,
            "optimizer_updates": state["optimizer_updates"],
            "microbatches_seen": state["microbatches"],
            "microbatches_this_epoch": len(orders[epoch_index]),
            "optimizer_updates_this_epoch": len(groups),
            "train_loss_mean": float(np.mean(losses)),
            "train_loss_min": float(np.min(losses)),
            "train_loss_max": float(np.max(losses)),
            "learning_rate_start": learning_rates[0],
            "learning_rate_end": learning_rates[-1],
            "gradient_norm_before_clip_mean": float(np.mean(before)),
            "gradient_norm_before_clip_max": float(np.max(before)),
            "gradient_norm_after_clip_mean": float(np.mean(after)),
            "gradient_norm_after_clip_max": float(np.max(after)),
            "clipping_fraction": float(np.mean(clipped)),
            "nonfinite_count": 0,
            "epoch_runtime_seconds": elapsed,
            "cumulative_runtime_seconds": state["runtime_seconds"],
            "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        }
        rows.append(epoch_row)
        atomic_csv(output / "train_log.csv", rows)
        metadata = {
            **contract,
            "contract_sha256": contract_hash,
            "total_updates": budget.total_updates,
            "warmup_updates": budget.warmup_updates,
            "epoch_gradient_statistics": {
                key: epoch_row[key] for key in (
                    "gradient_norm_before_clip_mean", "gradient_norm_before_clip_max",
                    "gradient_norm_after_clip_mean", "gradient_norm_after_clip_max",
                    "clipping_fraction", "nonfinite_count",
                )
            },
            "epoch_train_loss": epoch_row["train_loss_mean"],
            "epoch_learning_rate": epoch_row["learning_rate_end"],
        }
        payload = checkpoint_payload(
            model=model, optimizer=optimizer, training_state=state, metadata=metadata,
        )
        if epoch in mandatory:
            save_checkpoint(output / "checkpoints" / f"epoch_{epoch:04d}.pt", payload)
        if epoch % 10 == 0 or epoch in mandatory or epoch == budget.total_epochs:
            save_checkpoint(resume_path, payload)
        print(json.dumps({
            "experiment": "T-full-long", "epoch": epoch,
            "updates": state["optimizer_updates"], "train_loss": epoch_row["train_loss_mean"],
            "lr": epoch_row["learning_rate_end"], "clip_fraction": epoch_row["clipping_fraction"],
            "epoch_seconds": elapsed,
        }, sort_keys=True), flush=True)
    if int(state["optimizer_updates"]) != budget.total_updates:
        raise RuntimeError("Stage T full-long completed update count differs from budget")
    missing = [epoch for epoch in mandatory if not (output / "checkpoints" / f"epoch_{epoch:04d}.pt").exists()]
    if missing:
        raise RuntimeError(f"Stage T mandatory checkpoints are missing: {missing}")
    final_hash = tensor_state_sha256(model.state_dict())
    if final_hash == freeze["initial_tensor_state_sha256"]:
        raise RuntimeError("Stage T full-long did not learn a non-zero update")
    atomic_json(output / "training_summary.json", {
        "status": "passed", "all_finite": True, **state,
        "final_model_state_sha256": final_hash,
        "mandatory_checkpoints": sorted(mandatory),
        "peak_allocated_mib_max": max(float(row["gpu_peak_allocated_mib"]) for row in rows),
        "peak_reserved_mib_max": max(float(row["gpu_peak_reserved_mib"]) for row in rows),
    })
    data.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("small-1260", "full-long"), required=True)
    parser.add_argument(
        "--config", type=Path,
        default=Path("configs/stage_t/optimization_convergence.yaml"),
    )
    parser.add_argument("--root", type=Path, default=Path("artifacts/stage_t"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    stage_t, stage_s, freeze, train_pairs, small_pairs, _ = verify_frozen_contract(args.config)
    output = args.root / ("causal_control" if args.mode == "small-1260" else "full_long")
    if args.mode == "small-1260":
        run_small(
            stage_t=stage_t, stage_s=stage_s, freeze=freeze,
            small_pairs=small_pairs, output=output, resume=args.resume,
        )
    else:
        run_full(
            stage_t=stage_t, stage_s=stage_s, freeze=freeze,
            train_pairs=train_pairs, output=output, resume=args.resume,
        )


if __name__ == "__main__":
    main()
