#!/usr/bin/env python3
"""Run the frozen Stage Z resolution pilot, with a mandatory CUDA preflight."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
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
from grmhd.stage_z import REPRODUCTION_SCOPE, fixed_shell_tensor
from train_stage_s import (
    StageSBatchPath,
    build_frozen_model,
    gradient_norm,
    validation_score,
)


ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty Stage Z train log")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def tensor_only_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value for key, value in model.state_dict().items() if torch.is_tensor(value)}


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def load_contract(
    config_path: Path, resolution: int,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, dict[str, Any], list[int], list[int]]:
    stage_z = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if stage_z["reproduction_scope"] != REPRODUCTION_SCOPE:
        raise ValueError("Stage Z adapted scope changed")
    if resolution not in (64, 96, 128):
        raise ValueError("Stage Z resolution must be 64, 96, or 128")
    stage_s = yaml.safe_load((ROOT / "configs/stage_s/expanded_localno_p3_residual.yaml").read_text(encoding="utf-8"))
    upstream = subprocess.check_output(
        ["git", "-C", str(ROOT / "external/neuraloperator"), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(ROOT / "external/neuraloperator"), "status", "--short"], text=True
    ).strip()
    if upstream != stage_z["frozen"]["upstream_commit"] or dirty:
        raise ValueError("pinned neuraloperator provenance changed")
    dataset = ROOT / stage_z["data"]["resolutions"][resolution]
    p3_summary_path = ROOT / f"artifacts/stage_z/preprocessing/p3_{resolution}.json"
    p3 = json.loads(p3_summary_path.read_text(encoding="utf-8"))
    if sha256_file(dataset) != p3["dataset_sha256"]:
        raise ValueError("Stage Z dataset checksum changed")
    normalizer = ROOT / p3["artifact_directory"]
    if sha256_file(normalizer / "normalizer.npz") != p3["normalizer_sha256"]:
        raise ValueError("Stage Z normalizer checksum changed")
    split = json.loads((ROOT / stage_z["frozen"]["split"]).read_text(encoding="utf-8"))
    train_pairs = [int(value) for value in split["train_pair_source_indices"]]
    validation_pairs = [int(value) for value in split["validation_pair_source_indices"]]
    if train_pairs != list(range(168)) or validation_pairs != list(range(169, 211)):
        raise ValueError("Stage Z chronological split changed")
    return stage_z, stage_s, dataset, normalizer, p3, train_pairs, validation_pairs


def frozen_shell_data(
    dataset: Path, normalizer: Path, device: torch.device, shell_contract_path: Path,
) -> StageSBatchPath:
    data = StageSBatchPath(dataset, normalizer, device)
    shell_contract = json.loads(shell_contract_path.read_text(encoding="utf-8"))
    shape = tuple(int(value) for value in data.snapshots.shape[2:])
    r = np.asarray(data.handle["coords/r"], dtype=np.float64)
    shells = fixed_shell_tensor(
        r, shape[0], shape[1], shell_contract["physical_edges"], dtype=np.float32
    )
    data.shells = torch.from_numpy(shells).unsqueeze(0).to(device=device)
    data.shell_metadata = shell_contract
    return data


def build_model_at_resolution(
    stage_s: Mapping[str, Any], stage_z: Mapping[str, Any], resolution: int,
) -> tuple[torch.nn.Module, str]:
    resolved = copy.deepcopy(stage_s)
    resolved["model"]["default_in_shape"] = [resolution, resolution, resolution]
    model = build_frozen_model(resolved)
    initial_path = ROOT / stage_z["model"]["shared_initial_state"]
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    observed_initial = tensor_state_sha256(initial)
    if observed_initial != stage_z["model"]["shared_initial_tensor_sha256"]:
        raise ValueError("shared initial tensor state changed")
    model.load_state_dict(initial, strict=True)
    if tensor_state_sha256(model.state_dict()) != observed_initial:
        raise ValueError("resolution model did not strictly load the shared initial state")
    if trainable_parameter_count(model) != int(stage_z["model"]["parameter_count"]):
        raise ValueError("Stage Z trainable parameter count changed")
    return model, observed_initial


def gpu_diagnostic() -> dict[str, Any]:
    smi = subprocess.run(["nvidia-smi"], text=True, capture_output=True)
    return {
        "nvidia_smi_returncode": smi.returncode,
        "nvidia_smi_stdout": smi.stdout.strip(),
        "nvidia_smi_stderr": smi.stderr.strip(),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }


def preflight(
    *, stage_z: Mapping[str, Any], stage_s: Mapping[str, Any], dataset: Path,
    normalizer: Path, resolution: int, output: Path,
) -> bool:
    diagnosis = gpu_diagnostic()
    payload: dict[str, Any] = {
        "schema_version": "stage-z-gpu-preflight-v1",
        "reproduction_scope": REPRODUCTION_SCOPE,
        "resolution": resolution,
        **diagnosis,
        "cpu_fallback": False,
    }
    if not torch.cuda.is_available():
        payload.update({
            "status": "blocked_gpu_unavailable",
            "training_resource_feasible": False,
            "forward_finite": None,
            "loss_finite": None,
            "gradients_finite": None,
            "reason": "WSL/operating system blocked CUDA device access; no CPU fallback authorized",
        })
        atomic_json(output / "gpu_preflight.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return False
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    device_name = torch.cuda.get_device_name(device)
    if "RTX 5070" not in device_name:
        payload.update({
            "status": "blocked_wrong_gpu", "training_resource_feasible": False,
            "device_name": device_name,
        })
        atomic_json(output / "gpu_preflight.json", payload)
        return False
    model, state_hash = build_model_at_resolution(stage_s, stage_z, resolution)
    model.to(device).train()
    data = frozen_shell_data(
        dataset, normalizer, device, ROOT / "artifacts/stage_z/shell_contract.json"
    )
    loss_fn = PlainL2Loss().to(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = data.predict(model, 0)
    loss = loss_fn(result["predicted_residual"], result["residual_target"])
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - started
    backward_started = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(device)
    backward_seconds = time.perf_counter() - backward_started
    elapsed = forward_seconds + backward_seconds
    gradients_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )
    preflight_gradient_norm = gradient_norm(
        [parameter for parameter in model.parameters() if parameter.requires_grad]
    )
    payload.update({
        "status": "passed" if gradients_finite and torch.isfinite(loss) else "failed_nonfinite",
        "training_resource_feasible": bool(gradients_finite and torch.isfinite(loss)),
        "device_name": device_name,
        "device_capability": list(torch.cuda.get_device_capability(device)),
        "model_parameter_count": trainable_parameter_count(model),
        "initial_tensor_state_sha256": state_hash,
        "model_parameters_on_cuda": all(parameter.is_cuda for parameter in model.parameters()),
        "input_on_cuda": result["model_input"].is_cuda,
        "output_on_cuda": result["predicted_residual"].is_cuda,
        "loss_on_cuda": loss.is_cuda,
        "forward_finite": bool(torch.isfinite(result["predicted_residual"]).all()),
        "loss_finite": bool(torch.isfinite(loss)),
        "gradients_finite": bool(gradients_finite),
        "gradient_norm": preflight_gradient_norm,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "forward_backward_seconds": elapsed,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "optimizer_step_executed": False,
    })
    atomic_json(output / "gpu_preflight.json", payload)
    data.close()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return bool(payload["training_resource_feasible"])


def checkpoint_payload(
    *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    state: Mapping[str, Any], metadata: Mapping[str, Any], validation: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": "stage-z-checkpoint-v1",
        "model_state_dict": tensor_only_state(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_state": dict(state),
        "scheduler_state": {
            "kind": "stage_t_1200_epoch_warmup_cosine_prefix",
            "completed_updates": int(state["optimizer_updates"]),
            "total_updates": int(metadata["scheduler_total_updates"]),
            "warmup_updates": int(metadata["scheduler_warmup_updates"]),
        },
        "metadata": dict(metadata),
        "validation": dict(validation),
    }


def train(
    *, stage_z: Mapping[str, Any], stage_s: Mapping[str, Any], dataset: Path,
    normalizer: Path, p3: Mapping[str, Any], train_pairs: list[int],
    validation_pairs: list[int], resolution: int, output: Path, resume: bool,
) -> None:
    if not preflight(
        stage_z=stage_z, stage_s=stage_s, dataset=dataset,
        normalizer=normalizer, resolution=resolution, output=output,
    ):
        raise RuntimeError("Stage Z training blocked by mandatory GPU preflight")
    device = torch.device("cuda:0")
    model, initial_hash = build_model_at_resolution(stage_s, stage_z, resolution)
    model.to(device)
    data = frozen_shell_data(
        dataset, normalizer, device, ROOT / "artifacts/stage_z/shell_contract.json"
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(stage_z["training"]["learning_rate"]),
        weight_decay=float(stage_z["training"]["weight_decay"]),
    )
    loss_fn = PlainL2Loss().to(device)
    epochs = int(stage_z["training"]["epochs"])
    accumulation = int(stage_z["training"]["gradient_accumulation"])
    orders = natural_epoch_orders(train_pairs, epochs=epochs, seed=int(stage_z["training"]["seed"]))
    flat = [value for order in orders for value in order]
    metadata = {
        "reproduction_scope": REPRODUCTION_SCOPE,
        "resolution": resolution,
        "dataset_sha256": p3["dataset_sha256"],
        "normalizer_sha256": p3["normalizer_sha256"],
        "initial_tensor_state_sha256": initial_hash,
        "pair_order_sha256": order_sha256(flat),
        "per_epoch_order_sha256": [order_sha256(order) for order in orders],
        "parameter_count": trainable_parameter_count(model),
        "train_pairs": train_pairs,
        "validation_pairs": validation_pairs,
        "dropped_pair": [168, 169],
        "epochs": epochs,
        "optimizer_updates_per_epoch": 42,
        "optimizer_updates": 6300,
        "scheduler_total_updates": int(stage_z["training"]["scheduler_total_updates"]),
        "scheduler_warmup_updates": int(stage_z["training"]["scheduler_warmup_updates"]),
        "optimizer": "Adam",
        "learning_rate": float(stage_z["training"]["learning_rate"]),
        "weight_decay": float(stage_z["training"]["weight_decay"]),
        "gradient_clip_norm": float(stage_z["training"]["gradient_clip_norm"]),
        "mixed_precision": False,
        "loss": "PlainL2Loss_on_normalized_residual",
        "formal_selector": stage_z["training"]["formal_selector"],
        "validation_not_used_for_training_or_stopping": True,
        "device": torch.cuda.get_device_name(device),
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "python": sys.version,
        "platform": platform.platform(),
    }
    atomic_json(output / "resolved_config.json", metadata)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    state = {"completed_epoch": 0, "optimizer_updates": 0, "microbatches": 0, "runtime_seconds": 0.0}
    checkpoints = {int(value) for value in stage_z["training"]["checkpoint_epochs"]}
    rows: list[dict[str, Any]] = []
    validation_records: list[dict[str, Any]] = []
    resume_path = output / "resume.pt"
    if resume:
        if not resume_path.is_file():
            raise FileNotFoundError("Stage Z resume checkpoint does not exist")
        payload = torch.load(resume_path, map_location="cpu", weights_only=True)
        previous_metadata = payload["metadata"]
        invariant_keys = (
            "reproduction_scope", "resolution", "dataset_sha256", "normalizer_sha256",
            "initial_tensor_state_sha256", "pair_order_sha256", "parameter_count",
            "train_pairs", "validation_pairs", "epochs", "optimizer_updates",
        )
        if any(previous_metadata[key] != metadata[key] for key in invariant_keys):
            raise ValueError("Stage Z resume provenance/configuration changed")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        state = dict(payload["training_state"])
        rows = load_csv(output / "train_log.csv")
        if len(rows) != int(state["completed_epoch"]):
            raise ValueError("Stage Z resume log/checkpoint epoch mismatch")
        validation_path = output / "checkpoint_validation.json"
        if validation_path.is_file():
            validation_records = json.loads(validation_path.read_text(encoding="utf-8"))
        expected_updates = int(state["completed_epoch"]) * 42
        if int(state["optimizer_updates"]) != expected_updates:
            raise ValueError("Stage Z resume optimizer update count changed")
        if int(state["microbatches"]) != int(state["completed_epoch"]) * 168:
            raise ValueError("Stage Z resume microbatch count changed")
    start_epoch = int(state["completed_epoch"])
    model.train()
    for epoch_index, order in enumerate(orders[start_epoch:], start=start_epoch):
        epoch = epoch_index + 1
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        losses: list[float] = []
        gradients_before: list[float] = []
        clipped: list[bool] = []
        groups = accumulation_groups(order, accumulation=accumulation)
        if len(groups) != 42 or any(len(group) != 4 for group in groups):
            raise RuntimeError("Stage Z accumulation/order contract changed")
        for group in groups:
            optimizer.zero_grad(set_to_none=True)
            group_losses: list[float] = []
            for source in group:
                result = data.predict(model, int(source))
                loss = loss_fn(result["predicted_residual"], result["residual_target"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Stage Z loss became NaN/Inf")
                (loss / len(group)).backward()
                group_losses.append(float(loss.detach().cpu()))
            before = gradient_norm(parameters)
            if not math.isfinite(before) or before <= 0:
                raise FloatingPointError("Stage Z gradient is nonfinite or zero")
            torch.nn.utils.clip_grad_norm_(parameters, float(stage_z["training"]["gradient_clip_norm"]))
            update = int(state["optimizer_updates"]) + 1
            lr = warmup_cosine_learning_rate(
                update,
                total_updates=int(stage_z["training"]["scheduler_total_updates"]),
                warmup_updates=int(stage_z["training"]["scheduler_warmup_updates"]),
                base_learning_rate=float(stage_z["training"]["learning_rate"]),
                min_learning_rate=1.0e-6,
            )
            for values in optimizer.param_groups:
                values["lr"] = lr
            optimizer.step()
            if not all(torch.isfinite(parameter).all() for parameter in parameters):
                raise FloatingPointError("Stage Z optimizer produced NaN/Inf")
            state["optimizer_updates"] = update
            state["microbatches"] = int(state["microbatches"]) + len(group)
            losses.extend(group_losses)
            gradients_before.append(before)
            clipped.append(before > float(stage_z["training"]["gradient_clip_norm"]))
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        state["completed_epoch"] = epoch
        state["runtime_seconds"] = float(state["runtime_seconds"]) + elapsed
        row = {
            "epoch": epoch,
            "optimizer_updates": state["optimizer_updates"],
            "microbatches_seen": state["microbatches"],
            "microbatches_this_epoch": len(order),
            "optimizer_updates_this_epoch": len(groups),
            "train_loss_mean": float(np.mean(losses)),
            "train_loss_median": float(np.median(losses)),
            "gradient_norm_before_clip_mean": float(np.mean(gradients_before)),
            "clipping_fraction": float(np.mean(clipped)),
            "learning_rate_end": lr,
            "epoch_runtime_seconds": elapsed,
            "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            "nonfinite_count": 0,
        }
        rows.append(row)
        atomic_csv(output / "train_log.csv", rows)
        if epoch in checkpoints:
            validation = validation_score(model, data, validation_pairs)
            validation_records.append({"epoch": epoch, **validation})
            payload = checkpoint_payload(
                model=model, optimizer=optimizer, state=state,
                metadata=metadata, validation=validation,
            )
            save_checkpoint(output / "checkpoints" / f"epoch_{epoch:04d}.pt", payload)
            atomic_json(output / "checkpoint_validation.json", validation_records)
        save_checkpoint(
            resume_path,
            checkpoint_payload(
                model=model,
                optimizer=optimizer,
                state=state,
                metadata=metadata,
                validation=validation_records[-1] if validation_records else {},
            ),
        )
        print(json.dumps({
            "resolution": resolution, "epoch": epoch,
            "updates": state["optimizer_updates"], "train_loss": row["train_loss_mean"],
            "lr": lr, "epoch_seconds": elapsed,
        }, sort_keys=True), flush=True)
    if int(state["optimizer_updates"]) != 6300:
        raise RuntimeError("Stage Z optimizer update count differs from frozen budget")
    best = min(
        validation_records,
        key=lambda item: float(item["normalized_relative_l2"]["arithmetic_average"]),
    )
    atomic_json(output / "training_summary.json", {
        "status": "passed", **state,
        "all_finite": True,
        "formal_best_epoch": best["epoch"],
        "formal_best_validation_normalized_l2": best["normalized_relative_l2"]["arithmetic_average"],
        "final_model_state_sha256": tensor_state_sha256(model.state_dict()),
    })
    data.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/stage_z/higher_resolution.yaml"))
    parser.add_argument("--resolution", type=int, required=True, choices=(96, 128))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output_dir or Path(f"artifacts/stage_z/training/z{args.resolution}")
    stage_z, stage_s, dataset, normalizer, p3, train_pairs, validation_pairs = load_contract(
        args.config, args.resolution
    )
    if args.preflight_only:
        preflight(
            stage_z=stage_z, stage_s=stage_s, dataset=dataset,
            normalizer=normalizer, resolution=args.resolution, output=output,
        )
        return
    train(
        stage_z=stage_z, stage_s=stage_s, dataset=dataset,
        normalizer=normalizer, p3=p3, train_pairs=train_pairs,
        validation_pairs=validation_pairs, resolution=args.resolution, output=output,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
