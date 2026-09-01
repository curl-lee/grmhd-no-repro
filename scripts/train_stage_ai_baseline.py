#!/usr/bin/env python3
"""Train one missing canonical Stage AI baseline without architecture search."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from grmhd.dataset import sha256_file
from grmhd.models import trainable_parameter_count
from grmhd.paper_losses import PlainL2Loss
from grmhd.paper_stage_g import tensor_state_sha256
from grmhd.stage_ai import (
    build_canonical_cnn,
    build_canonical_fno,
    parameter_match_pass,
    tensor_sha256,
)
from grmhd.stage_s_training import (
    accumulation_groups,
    natural_epoch_orders,
    order_sha256,
    warmup_cosine_learning_rate,
)

from train_stage_s import StageSBatchPath, gradient_norm, validation_score


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage_ai/final_benchmark.yaml"
OUT_ROOT = ROOT / "artifacts/stage_ai/training"
MANDATORY = (2, 10, 30, 75, 150, 300)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True))


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty Stage AI training table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def trainable_hash(model: torch.nn.Module) -> str:
    return tensor_state_sha256(dict(model.named_parameters()))


def load_contract() -> tuple[dict[str, Any], dict[str, Any], list[int], list[int], list[list[int]]]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    stage_s_path = ROOT / "configs/stage_s/expanded_localno_p3_residual.yaml"
    stage_s = yaml.safe_load(stage_s_path.read_text(encoding="utf-8"))
    if sha256_file(stage_s_path) != "cdba8328a66b79cd937a0262e6bc1123f645ba0821ba915481af1d64e551618a":
        raise RuntimeError("frozen Stage S config changed")
    if sha256_file(ROOT / config["data"]["dataset"]) != config["data"]["dataset_sha256"]:
        raise RuntimeError("canonical dataset changed")
    if sha256_file(ROOT / config["preprocessing"]["normalizer_file"]) != config["preprocessing"]["normalizer_sha256"]:
        raise RuntimeError("canonical normalizer changed")
    if sha256_file(ROOT / config["data"]["split"]) != config["data"]["split_sha256"]:
        raise RuntimeError("canonical split changed")
    split = json.loads((ROOT / config["data"]["split"]).read_text(encoding="utf-8"))
    train_pairs = [int(value) for value in split["train_pair_source_indices"]]
    validation_pairs = [int(value) for value in split["validation_pair_source_indices"]]
    if train_pairs != list(range(168)) or validation_pairs != list(range(169, 211)):
        raise RuntimeError("canonical temporal split changed")
    orders = natural_epoch_orders(train_pairs, epochs=300, seed=42)
    flattened = order_sha256(value for order in orders for value in order)
    if flattened != config["training"]["pair_order_sha256"]:
        raise RuntimeError("canonical 300-epoch pair order changed")
    upstream = subprocess.check_output(
        ["git", "-C", ROOT / "external/neuraloperator", "rev-parse", "HEAD"], text=True
    ).strip()
    upstream_dirty = subprocess.check_output(
        ["git", "-C", ROOT / "external/neuraloperator", "status", "--short"], text=True
    ).strip()
    if upstream != config["runtime"]["upstream_commit"] or upstream_dirty:
        raise RuntimeError("pinned upstream changed or is dirty")
    return config, stage_s, train_pairs, validation_pairs, orders


def build_model(name: str, config: Mapping[str, Any], stage_s: Mapping[str, Any]) -> tuple[torch.nn.Module, dict[str, Any]]:
    if name == "fno":
        initial_path = ROOT / config["models"]["fno"]["initialization_state"]
        if sha256_file(initial_path) != config["models"]["fno"]["initialization_state_file_sha256"]:
            raise RuntimeError("canonical FNO shared initial state changed")
        initial = torch.load(initial_path, map_location="cpu", weights_only=True)
        model, identity = build_canonical_fno(
            stage_s,
            dataset=str(ROOT / config["data"]["dataset"]),
            initial_state=initial,
        )
    elif name == "cnn":
        model, identity = build_canonical_cnn(seed=int(config["runtime"]["seed"]))
    else:
        raise ValueError(f"unknown Stage AI baseline {name!r}")
    expected = int(config["models"][name]["expected_parameter_count"])
    observed = trainable_parameter_count(model)
    if observed != expected:
        raise RuntimeError(f"{name} parameter count changed: {observed} != {expected}")
    if name == "cnn" and not parameter_match_pass(observed):
        raise RuntimeError("CNN parameter match exceeds the frozen 10% limit")
    identity = {**identity, "model": name, "observed_parameter_count": observed, "initial_trainable_state_sha256": trainable_hash(model)}
    return model, identity


def checkpoint_payload(
    *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    state: Mapping[str, Any], metadata: Mapping[str, Any], validation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": "stage-ai-baseline-checkpoint-v1",
        "model_state_dict": {
            key: value for key, value in model.state_dict().items() if torch.is_tensor(value)
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "training_state": dict(state),
        "scheduler_state": {
            "kind": "explicit_update_warmup_cosine",
            "completed_updates": int(state["optimizer_updates"]),
            "total_updates": 50400,
            "warmup_updates": 3150,
        },
        "metadata": dict(metadata),
        "validation": dict(validation) if validation is not None else {},
        "trainable_state_sha256": trainable_hash(model),
    }


def selector_record(metrics: Mapping[str, Any], epoch: int, updates: int) -> dict[str, Any]:
    normalized = metrics["normalized_relative_l2"]
    residual = metrics["residual"]
    record: dict[str, Any] = {
        "epoch": int(epoch),
        "optimizer_updates": int(updates),
        "normalized_per_channel_relative_l2_arithmetic_average": float(normalized["arithmetic_average"]),
        "normalized_global_relative_l2": float(normalized["global"]),
        "residual_relative_l2_arithmetic_average": float(residual["arithmetic_average_relative_l2"]),
        "residual_cosine_arithmetic_average": float(residual["arithmetic_average_cosine"]),
        "finite": bool(metrics["finite"]),
    }
    record.update({f"channel_{key}": float(value) for key, value in normalized["per_channel"].items()})
    return record


def strict_reload(
    *, name: str, path: Path, expected_metric: float, config: Mapping[str, Any],
    stage_s: Mapping[str, Any], validation_pairs: Sequence[int], device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model, _ = build_model(name, config, stage_s)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    model.to(device).eval()
    data = StageSBatchPath(
        ROOT / config["data"]["dataset"], ROOT / config["preprocessing"]["artifact"], device
    )
    with torch.no_grad():
        probe_a = data.predict(model, 169)["predicted_residual"]
        probe_b = data.predict(model, 169)["predicted_residual"]
    metrics = validation_score(model, data, validation_pairs)
    observed = float(metrics["normalized_relative_l2"]["arithmetic_average"])
    observed_hash = trainable_hash(model)
    record = {
        "model": name,
        "checkpoint": str(path.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(path),
        "epoch": int(payload["training_state"]["completed_epoch"]),
        "optimizer_updates": int(payload["training_state"]["optimizer_updates"]),
        "strict_model_reload": True,
        "optimizer_reload": True,
        "scheduler_reload": int(payload["scheduler_state"]["completed_updates"]) == int(payload["training_state"]["optimizer_updates"]),
        "stored_trainable_state_sha256": payload["trainable_state_sha256"],
        "reloaded_trainable_state_sha256": observed_hash,
        "trainable_hash_match": observed_hash == payload["trainable_state_sha256"],
        "deterministic_probe_sha256": tensor_sha256(probe_a),
        "deterministic_probe_pass": tensor_sha256(probe_a) == tensor_sha256(probe_b),
        "stored_selector_metric": float(expected_metric),
        "recomputed_selector_metric": observed,
        "selector_absolute_difference": abs(observed - float(expected_metric)),
    }
    record["pass"] = bool(
        record["scheduler_reload"] and record["trainable_hash_match"]
        and record["deterministic_probe_pass"] and record["selector_absolute_difference"] <= 1e-12
    )
    data.close()
    return record


def preflight(
    name: str, model: torch.nn.Module, identity: Mapping[str, Any],
    config: Mapping[str, Any], device: torch.device,
) -> dict[str, Any]:
    data = StageSBatchPath(
        ROOT / config["data"]["dataset"], ROOT / config["preprocessing"]["artifact"], device
    )
    model.to(device).train()
    model.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = data.predict(model, 0)
    loss = PlainL2Loss().to(device)(result["predicted_residual"], result["residual_target"])
    loss.backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    gradients_finite = all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    gradient = gradient_norm(model.parameters())
    record = {
        "model": name,
        "device": torch.cuda.get_device_name(device),
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "identity": dict(identity),
        "input_shape": list(result["model_input"].shape),
        "output_shape": list(result["predicted_residual"].shape),
        "forward_finite": bool(torch.isfinite(result["predicted_residual"]).all()),
        "loss": float(loss.detach().cpu()),
        "loss_finite": bool(torch.isfinite(loss)),
        "gradients_finite": bool(gradients_finite),
        "gradient_norm": float(gradient),
        "gradient_nonzero": bool(gradient > 0),
        "forward_backward_seconds": elapsed,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        "optimizer_created": False,
        "optimizer_update_performed": False,
    }
    record["pass"] = all((record["forward_finite"], record["loss_finite"], record["gradients_finite"], record["gradient_nonzero"]))
    model.zero_grad(set_to_none=True)
    data.close()
    return record


def run(name: str, *, preflight_only: bool, resume: bool) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage AI baseline training refuses CPU fallback")
    config, stage_s, train_pairs, validation_pairs, orders = load_contract()
    device = torch.device("cuda:0")
    if config["runtime"]["required_device_substring"] not in torch.cuda.get_device_name(device):
        raise RuntimeError("Stage AI baseline training requires RTX 5070")
    torch.cuda.set_device(device)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.cuda.manual_seed_all(42)
    model, identity = build_model(name, config, stage_s)
    output = OUT_ROOT / name
    record = preflight(name, model, identity, config, device)
    atomic_json(output / "preflight.json", record)
    if not record["pass"]:
        raise RuntimeError(f"{name} no-update GPU preflight failed")
    if preflight_only:
        print(json.dumps(record, indent=2, sort_keys=True))
        return

    model.zero_grad(set_to_none=True)
    model.to(device).train()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    loss_fn = PlainL2Loss().to(device)
    data = StageSBatchPath(
        ROOT / config["data"]["dataset"], ROOT / config["preprocessing"]["artifact"], device
    )
    metadata = {
        "contract": "Stage AI canonical missing-baseline training",
        "model": name,
        "model_identity": identity,
        "dataset_sha256": config["data"]["dataset_sha256"],
        "normalizer_sha256": config["preprocessing"]["normalizer_sha256"],
        "split_sha256": config["data"]["split_sha256"],
        "pair_order_sha256": config["training"]["pair_order_sha256"],
        "loss": "PlainL2Loss_on_normalized_residual",
        "formal_selector": config["training"]["selector"],
        "validation_not_used_for_tuning_or_stopping": True,
        "project_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "upstream_commit": config["runtime"]["upstream_commit"],
        "environment": {
            "python": sys.version,
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "platform": platform.platform(),
        },
    }
    atomic_json(output / "resolved_config.json", {"config": config, "metadata": metadata})
    state: dict[str, Any] = {
        "completed_epoch": 0,
        "optimizer_updates": 0,
        "microbatches": 0,
        "runtime_seconds": 0.0,
        "nonfinite_count": 0,
    }
    rows = load_csv(output / "train_log.csv") if resume else []
    validations = load_csv(output / "validation_metrics.csv") if resume else []
    resume_path = output / "checkpoints/resume.pt"
    if resume:
        payload = torch.load(resume_path, map_location="cpu", weights_only=True)
        if payload["metadata"] != metadata:
            raise RuntimeError("Stage AI resume metadata changed")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        state = dict(payload["training_state"])
        rows = rows[: int(state["completed_epoch"])]
    elif resume_path.exists() or rows:
        raise FileExistsError(f"{name} training output exists; use --resume")

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    best_metric = min(
        (float(row["normalized_per_channel_relative_l2_arithmetic_average"]) for row in validations),
        default=math.inf,
    )
    best_epoch = next((int(row["epoch"]) for row in validations if float(row["normalized_per_channel_relative_l2_arithmetic_average"]) == best_metric), None)
    for epoch_index in range(int(state["completed_epoch"]), 300):
        epoch = epoch_index + 1
        groups = accumulation_groups(orders[epoch_index], accumulation=4)
        if len(groups) != 42 or any(len(group) != 4 for group in groups):
            raise RuntimeError("Stage AI accumulation contract changed")
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        update_rows = []
        model.train()
        for sources in groups:
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for source in sources:
                result = data.predict(model, source)
                loss = loss_fn(result["predicted_residual"], result["residual_target"])
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{name} training loss became NaN/Inf")
                (loss / len(sources)).backward()
                losses.append(float(loss.detach().cpu()))
            before = gradient_norm(parameters)
            if not math.isfinite(before) or before <= 0:
                raise FloatingPointError(f"{name} gradients became nonfinite or zero")
            torch.nn.utils.clip_grad_norm_(parameters, float(config["training"]["gradient_clip_norm"]))
            after = gradient_norm(parameters)
            update = int(state["optimizer_updates"]) + 1
            lr = warmup_cosine_learning_rate(
                update,
                total_updates=int(config["training"]["scheduler_total_updates"]),
                warmup_updates=int(config["training"]["scheduler_warmup_updates"]),
                base_learning_rate=float(config["training"]["learning_rate"]),
                min_learning_rate=float(config["training"]["scheduler_min_learning_rate"]),
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            if not all(torch.isfinite(parameter).all() for parameter in parameters):
                raise FloatingPointError(f"{name} optimizer produced NaN/Inf")
            state["optimizer_updates"] = update
            state["microbatches"] = int(state["microbatches"]) + len(sources)
            update_rows.append({"losses": losses, "before": before, "after": after, "lr": lr, "clipped": before > 1.0})
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        state["completed_epoch"] = epoch
        state["runtime_seconds"] = float(state["runtime_seconds"]) + elapsed
        losses = [value for row in update_rows for value in row["losses"]]
        row = {
            "model": name,
            "epoch": epoch,
            "optimizer_updates": state["optimizer_updates"],
            "optimizer_updates_this_epoch": 42,
            "microbatches_completed": state["microbatches"],
            "microbatches_this_epoch": 168,
            "train_loss_mean": float(np.mean(losses)),
            "train_loss_median": float(np.median(losses)),
            "train_loss_min": float(np.min(losses)),
            "train_loss_max": float(np.max(losses)),
            "learning_rate_start": update_rows[0]["lr"],
            "learning_rate": update_rows[-1]["lr"],
            "gradient_norm_before_clip_mean": float(np.mean([entry["before"] for entry in update_rows])),
            "gradient_norm_before_clip_max": float(np.max([entry["before"] for entry in update_rows])),
            "gradient_norm_after_clip_mean": float(np.mean([entry["after"] for entry in update_rows])),
            "clipping_fraction": float(np.mean([entry["clipped"] for entry in update_rows])),
            "epoch_runtime_seconds": elapsed,
            "cumulative_runtime_seconds": state["runtime_seconds"],
            "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            "nonfinite_count": state["nonfinite_count"],
        }
        rows.append(row)
        atomic_csv(output / "train_log.csv", rows)
        validation = None
        if epoch in MANDATORY:
            metrics = validation_score(model, data, validation_pairs)
            validation = selector_record(metrics, epoch, int(state["optimizer_updates"]))
            validations.append(validation)
            atomic_csv(output / "validation_metrics.csv", validations)
        payload = checkpoint_payload(model=model, optimizer=optimizer, state=state, metadata=metadata, validation=validation)
        save_checkpoint(resume_path, payload)
        if epoch in MANDATORY:
            save_checkpoint(output / "checkpoints" / f"epoch_{epoch:04d}.pt", payload)
            metric = float(validation["normalized_per_channel_relative_l2_arithmetic_average"])
            if metric < best_metric:
                best_metric = metric
                best_epoch = epoch
                save_checkpoint(output / "checkpoints/best.pt", payload)
                atomic_json(output / "best_selector_metric.json", validation)
        if epoch == 300:
            save_checkpoint(output / "checkpoints/last.pt", payload)
            atomic_json(output / "last_selector_metric.json", validation)
        print(json.dumps({
            "model": name,
            "epoch": f"{epoch}/300",
            "train_loss": row["train_loss_mean"],
            "learning_rate": row["learning_rate"],
            "clip_fraction": row["clipping_fraction"],
            "epoch_seconds": elapsed,
            "elapsed_seconds": state["runtime_seconds"],
            "gpu_peak_allocated_mib": row["gpu_peak_allocated_mib"],
            "gpu_peak_reserved_mib": row["gpu_peak_reserved_mib"],
        }, sort_keys=True), flush=True)

    if state["completed_epoch"] != 300 or state["optimizer_updates"] != 12600 or state["microbatches"] != 50400:
        raise RuntimeError(f"{name} final training counts changed")
    if best_epoch is None:
        raise RuntimeError(f"{name} formal best selector was not produced")
    best_expected = float(json.loads((output / "best_selector_metric.json").read_text())["normalized_per_channel_relative_l2_arithmetic_average"])
    last_expected = float(json.loads((output / "last_selector_metric.json").read_text())["normalized_per_channel_relative_l2_arithmetic_average"])
    best_reload = strict_reload(name=name, path=output / "checkpoints/best.pt", expected_metric=best_expected, config=config, stage_s=stage_s, validation_pairs=validation_pairs, device=device)
    last_reload = strict_reload(name=name, path=output / "checkpoints/last.pt", expected_metric=last_expected, config=config, stage_s=stage_s, validation_pairs=validation_pairs, device=device)
    atomic_json(output / "best_reload.json", best_reload)
    atomic_json(output / "last_reload.json", last_reload)
    if not best_reload["pass"] or not last_reload["pass"]:
        raise RuntimeError(f"{name} strict checkpoint reload failed")
    summary = {
        "status": "passed",
        "model": name,
        "parameter_count": trainable_parameter_count(model),
        "initial_trainable_state_sha256": identity["initial_trainable_state_sha256"],
        "formal_best_epoch": int(best_epoch),
        "formal_best_metric": best_metric,
        **state,
        "peak_allocated_mib_max": max(float(row["gpu_peak_allocated_mib"]) for row in rows),
        "peak_reserved_mib_max": max(float(row["gpu_peak_reserved_mib"]) for row in rows),
        "best_checkpoint_reload_pass": True,
        "last_checkpoint_reload_pass": True,
        "all_finite": True,
    }
    atomic_json(output / "training_summary.json", summary)
    data.close()
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("fno", "cnn"), required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = OUT_ROOT / args.model
    try:
        run(args.model, preflight_only=args.preflight_only, resume=args.resume)
    except Exception as error:
        atomic_text(output / "FAILURE_STATE.md", f"""# Stage AI Baseline Failure

- model: `{args.model}`
- time: `{time.strftime('%Y-%m-%dT%H:%M:%S%z')}`
- error type: `{type(error).__name__}`
- error: `{error}`
- CPU fallback attempted: `false`
- contract changed automatically: `false`

```text
{traceback.format_exc()}
```
""")
        raise


if __name__ == "__main__":
    main()
