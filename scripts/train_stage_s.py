#!/usr/bin/env python3
"""Run one frozen Stage S LocalNO/P3/residual training experiment."""

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

import h5py
import numpy as np
import torch
import yaml

from grmhd import CHANNELS
from grmhd.dataset import sha256_file
from grmhd.models import build_model, trainable_parameter_count
from grmhd.paper_losses import PlainL2Loss
from grmhd.paper_stage_g import tensor_state_sha256
from grmhd.paper_stage_n import NO_SOFTCLIP, PrototypePreprocessor, prototype_specs
from grmhd.shells import radial_shells_tensor
from grmhd.stage_s_training import (
    accumulation_groups,
    matched_microbatch_order,
    natural_epoch_orders,
    order_sha256,
    warmup_cosine_learning_rate,
)


EXPERIMENTS = ("smoke_small", "smoke_full", "s_small_matched", "s_full_matched", "s_full_30epoch")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Refusing to write an empty Stage S training log")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(str(tuple(tensor.shape)).encode())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def build_frozen_model(config: Mapping[str, Any]) -> torch.nn.Module:
    model = config["model"]
    return build_model(
        "localno_differential_3d",
        in_channels=int(model["in_channels"]),
        out_channels=int(model["out_channels"]),
        n_modes=tuple(int(value) for value in model["n_modes"]),
        hidden_channels=int(model["hidden_channels"]),
        n_layers=int(model["n_layers"]),
        default_in_shape=tuple(int(value) for value in model["default_in_shape"]),
        positional_embedding=model["positional_embedding"],
        fin_diff_kernel_size=int(model["fin_diff_kernel_size"]),
        mix_derivatives=bool(model["mix_derivatives"]),
        conv_padding_mode=str(model["conv_padding_mode"]),
        use_channel_mlp=bool(model["use_channel_mlp"]),
        local_no_skip=model["local_no_skip"],
        norm=model["normalization"],
        enforce_hermitian_symmetry=bool(model["enforce_hermitian_symmetry"]),
    )


class StageSBatchPath:
    def __init__(self, dataset: Path, normalizer: Path, device: torch.device) -> None:
        self.dataset = dataset
        self.handle = h5py.File(dataset, "r")
        self.snapshots = self.handle["snapshots"]
        self.times = np.asarray(self.handle["times"], dtype=np.float64)
        self.preprocessor = PrototypePreprocessor.load(
            normalizer, spec=prototype_specs(NO_SOFTCLIP)["P3"]
        )
        r = np.asarray(self.handle["coords/r"], dtype=np.float64)
        shape = tuple(int(value) for value in self.snapshots.shape[2:])
        shells, metadata = radial_shells_tensor(r, shape[0], shape[1], n_shells=8)
        self.shells = shells.unsqueeze(0).to(device=device, dtype=torch.float32)
        self.shell_metadata = metadata.as_dict()
        self.device = device

    def close(self) -> None:
        self.handle.close()

    def raw(self, index: int) -> torch.Tensor:
        value = torch.from_numpy(np.asarray(self.snapshots[index], dtype=np.float32)).unsqueeze(0)
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Raw snapshot {index} contains NaN/Inf")
        return value

    def pair(self, source: int) -> dict[str, torch.Tensor]:
        raw_input = self.raw(source).to(self.device, non_blocking=True)
        raw_target = self.raw(source + 1).to(self.device, non_blocking=True)
        z_input = self.preprocessor.encode_tensor(raw_input, channel_axis=1)
        z_target = self.preprocessor.encode_tensor(raw_target, channel_axis=1)
        model_input = torch.cat((z_input, self.shells), dim=1)
        return {
            "raw_input": raw_input,
            "raw_target": raw_target,
            "z_input": z_input,
            "z_target": z_target,
            "residual_target": z_target - z_input,
            "model_input": model_input,
        }

    def predict(self, model: torch.nn.Module, source: int) -> dict[str, torch.Tensor]:
        batch = self.pair(source)
        residual = model(x=batch["model_input"])
        if residual.shape != batch["residual_target"].shape:
            raise ValueError("LocalNO output shape changed")
        prediction = batch["z_input"] + residual
        return {**batch, "predicted_residual": residual, "z_prediction": prediction}


def gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    squares = [
        torch.sum(torch.abs(parameter.grad.detach()).float().square())
        for parameter in parameters if parameter.grad is not None
    ]
    return 0.0 if not squares else float(torch.sqrt(torch.stack(squares).sum()).cpu())


def validation_score(
    model: torch.nn.Module, data: StageSBatchPath, pair_indices: Iterable[int]
) -> dict[str, Any]:
    model.eval()
    numerator = np.zeros(len(CHANNELS), dtype=np.float64)
    denominator = np.zeros(len(CHANNELS), dtype=np.float64)
    residual_num = np.zeros(len(CHANNELS), dtype=np.float64)
    residual_den = np.zeros(len(CHANNELS), dtype=np.float64)
    cos_dot = np.zeros(len(CHANNELS), dtype=np.float64)
    cos_left = np.zeros(len(CHANNELS), dtype=np.float64)
    cos_right = np.zeros(len(CHANNELS), dtype=np.float64)
    finite = True
    with torch.no_grad():
        for source in pair_indices:
            result = data.predict(model, int(source))
            difference = (result["z_prediction"] - result["z_target"]).double()
            target = result["z_target"].double()
            residual_difference = (
                result["predicted_residual"] - result["residual_target"]
            ).double()
            true_residual = result["residual_target"].double()
            predicted_residual = result["predicted_residual"].double()
            reduce = (0, 2, 3, 4)
            numerator += torch.sum(difference.square(), dim=reduce).cpu().numpy()
            denominator += torch.sum(target.square(), dim=reduce).cpu().numpy()
            residual_num += torch.sum(residual_difference.square(), dim=reduce).cpu().numpy()
            residual_den += torch.sum(true_residual.square(), dim=reduce).cpu().numpy()
            cos_dot += torch.sum(predicted_residual * true_residual, dim=reduce).cpu().numpy()
            cos_left += torch.sum(predicted_residual.square(), dim=reduce).cpu().numpy()
            cos_right += torch.sum(true_residual.square(), dim=reduce).cpu().numpy()
            finite &= bool(
                torch.isfinite(result["predicted_residual"]).all()
                and torch.isfinite(result["z_prediction"]).all()
            )
    per_channel = np.sqrt(numerator / np.maximum(denominator, 1e-300))
    residual_per_channel = np.sqrt(residual_num / np.maximum(residual_den, 1e-300))
    cosine = cos_dot / np.maximum(np.sqrt(cos_left * cos_right), 1e-300)
    return {
        "finite": finite,
        "normalized_relative_l2": {
            "per_channel": dict(zip(CHANNELS, per_channel.tolist(), strict=True)),
            "arithmetic_average": float(np.mean(per_channel)),
            "global": float(math.sqrt(numerator.sum() / max(denominator.sum(), 1e-300))),
        },
        "residual": {
            "per_channel_relative_l2": dict(zip(CHANNELS, residual_per_channel.tolist(), strict=True)),
            "arithmetic_average_relative_l2": float(np.mean(residual_per_channel)),
            "per_channel_cosine": dict(zip(CHANNELS, cosine.tolist(), strict=True)),
            "arithmetic_average_cosine": float(np.mean(cosine)),
            "global_cosine": float(cos_dot.sum() / max(math.sqrt(cos_left.sum() * cos_right.sum()), 1e-300)),
        },
    }


def checkpoint_payload(
    *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    state: Mapping[str, Any], metadata: Mapping[str, Any], validation: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": "stage-s-checkpoint-v1",
        # Match the frozen Stage-K shared state: LocalNO's dynamic output-shape
        # ``_extra_state`` is not a learned tensor and is rebuilt by forward.
        "model_state_dict": {
            key: value for key, value in model.state_dict().items()
            if torch.is_tensor(value)
        },
        "optimizer_state_dict": optimizer.state_dict(),
        "training_state": dict(state),
        "scheduler_state": {
            "kind": "explicit_update_warmup_cosine",
            "completed_updates": int(state["optimizer_updates"]),
        },
        "metadata": dict(metadata),
        "validation": dict(validation),
    }


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def strict_reload_check(
    path: Path, *, config: Mapping[str, Any], data: StageSBatchPath,
    expected_probe_hash: str, device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    reloaded = build_frozen_model(config)
    reloaded.load_state_dict(payload["model_state_dict"], strict=True)
    optimizer = torch.optim.Adam(
        reloaded.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    reloaded.to(device).eval()
    with torch.no_grad():
        probe = data.predict(reloaded, 169)["z_prediction"]
    observed = tensor_sha256(probe)
    if observed != expected_probe_hash:
        raise ValueError("Strict checkpoint reload prediction mismatch")
    return {
        "strict_model_reload": True,
        "optimizer_reload": True,
        "scheduler_state_reload": payload["scheduler_state"]["completed_updates"]
        == payload["training_state"]["optimizer_updates"],
        "probe_prediction_sha256": observed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=EXPERIMENTS, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/stage_s/expanded_localno_p3_residual.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    upstream = subprocess.check_output(
        ["git", "-C", "external/neuraloperator", "rev-parse", "HEAD"], text=True
    ).strip()
    upstream_dirty = subprocess.check_output(
        ["git", "-C", "external/neuraloperator", "status", "--short"], text=True
    ).strip()
    if upstream != config["frozen_pairing"]["upstream_commit"] or upstream_dirty:
        raise ValueError("Pinned upstream provenance changed")
    if not torch.cuda.is_available():
        raise RuntimeError("Stage S requires CUDA; refusing silent CPU fallback")
    device_name = torch.cuda.get_device_name(0)
    if config["runtime"]["required_device_substring"] not in device_name:
        raise RuntimeError(f"Stage S expected RTX 5070, found {device_name!r}")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)

    dataset = root / config["data"]["dataset"]
    if sha256_file(dataset) != config["data"]["dataset_sha256"]:
        raise ValueError("Expanded dataset checksum changed")
    normalizer_dir = root / config["preprocessing"]["artifact"]
    if sha256_file(normalizer_dir / "normalizer.npz") != config["preprocessing"]["normalizer_sha256"]:
        raise ValueError("Expanded P3 normalizer checksum changed")
    split = read_json(root / config["data"]["split"])
    full_pairs = [int(value) for value in split["train_pair_source_indices"]]
    validation_pairs = [int(value) for value in split["validation_pair_source_indices"]]
    small_pairs = []
    for line in (root / config["data"]["small_pairs"]).read_text().splitlines():
        if not line.strip():
            continue
        columns = [int(value) for value in line.split()]
        if len(columns) != 2 or columns[1] != columns[0] + 1:
            raise ValueError("Small control manifest must contain adjacent source/target pairs")
        small_pairs.append(columns[0])
    if len(full_pairs) != 168 or len(small_pairs) != 79 or len(validation_pairs) != 42:
        raise ValueError("Stage S pair counts changed")
    if not set(small_pairs) <= set(full_pairs) or set(validation_pairs) & set(full_pairs):
        raise ValueError("Stage S pair populations overlap or escape training")

    seed = int(config["runtime"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = build_frozen_model(config)
    initial_path = root / config["frozen_pairing"]["initial_state"]
    if sha256_file(initial_path) != config["frozen_pairing"]["initial_state_file_sha256"]:
        raise ValueError("Frozen initial-state file checksum changed")
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    if tensor_state_sha256(initial) != config["frozen_pairing"]["initial_tensor_state_sha256"]:
        raise ValueError("Frozen initial tensor-state hash changed")
    model.load_state_dict(initial, strict=True)
    if trainable_parameter_count(model) != int(config["frozen_pairing"]["expected_parameter_count"]):
        raise ValueError("Stage S model parameter count changed")
    if tensor_state_sha256(model.state_dict()) != config["frozen_pairing"]["initial_tensor_state_sha256"]:
        raise ValueError("Stage S model did not strictly load the shared state")
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["optimizer"]["learning_rate"]),
        weight_decay=float(config["optimizer"]["weight_decay"]),
    )
    loss_fn = PlainL2Loss().to(device)
    data = StageSBatchPath(dataset, normalizer_dir, device)

    smoke = args.experiment.startswith("smoke_")
    small = args.experiment in ("smoke_small", "s_small_matched")
    population = small_pairs if small else full_pairs
    accumulation = int(config["runtime"]["gradient_accumulation"])
    if smoke:
        epoch_orders = natural_epoch_orders(population, epochs=2, seed=seed)
        groups_by_interval = [
            accumulation_groups(order, accumulation=accumulation) for order in epoch_orders
        ]
        total_updates = sum(len(groups) for groups in groups_by_interval)
        flat_order = [item for order in epoch_orders for item in order]
        warmup_updates = total_updates
        interval_labels = [1, 2]
    elif args.experiment in ("s_small_matched", "s_full_matched"):
        total_updates = int(config["scheduler"]["matched_total_updates"])
        flat_order = matched_microbatch_order(
            population, optimizer_updates=total_updates, accumulation=accumulation, seed=seed
        )
        all_groups = accumulation_groups(flat_order, accumulation=accumulation)
        groups_by_interval = [all_groups[start : start + 20] for start in range(0, total_updates, 20)]
        warmup_updates = int(config["scheduler"]["matched_warmup_updates"])
        interval_labels = list(range(1, len(groups_by_interval) + 1))
    else:
        epoch_orders = natural_epoch_orders(population, epochs=30, seed=seed)
        groups_by_interval = [
            accumulation_groups(order, accumulation=accumulation) for order in epoch_orders
        ]
        total_updates = sum(len(groups) for groups in groups_by_interval)
        flat_order = [item for order in epoch_orders for item in order]
        warmup_updates = 2 * len(groups_by_interval[0])
        interval_labels = list(range(1, 31))
    if not smoke and args.experiment.endswith("matched") and total_updates != 600:
        raise ValueError("Matched Stage S budget changed from 600 updates")

    config_payload = {
        **config,
        "resolved_experiment": {
            "name": args.experiment,
            "output_dir": str(output),
            "train_pair_count": len(population),
            "validation_pair_count": len(validation_pairs),
            "optimizer_updates": total_updates,
            "microbatches": len(flat_order),
            "effective_samples_seen": len(flat_order),
            "population_epochs_equivalent": len(flat_order) / len(population),
            "gradient_accumulation": accumulation,
            "warmup_updates": warmup_updates,
            "order_sha256": order_sha256(flat_order),
            "initial_tensor_state_sha256": tensor_state_sha256(initial),
            "parameter_count": trainable_parameter_count(model),
            "device": device_name,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "mixed_precision": False,
            "validation_interval_updates": None if smoke else len(groups_by_interval[0]),
            "checkpoint_selection": "validation_normalized_arithmetic_average_relative_l2",
            "no_hyperparameter_search": True,
        },
        "provenance": {
            "project_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "project_worktree_dirty": bool(subprocess.check_output(["git", "status", "--short"], text=True).strip()),
            "upstream_commit": upstream,
            "dataset_sha256": sha256_file(dataset),
            "normalizer_sha256": sha256_file(normalizer_dir / "normalizer.npz"),
            "split_sha256": sha256_file(root / config["data"]["split"]),
            "small_pairs_sha256": sha256_file(root / config["data"]["small_pairs"]),
            "config_sha256": sha256_file(args.config),
            "python": sys.version,
            "platform": platform.platform(),
        },
    }
    write_json(output / "resolved_config.json", config_payload)
    write_json(output / "pair_order.json", {
        "schema_version": "stage-s-pair-order-v1",
        "experiment": args.experiment,
        "population": population,
        "microbatch_order": flat_order,
        "sha256": order_sha256(flat_order),
    })

    state = {"optimizer_updates": 0, "microbatches": 0, "intervals_complete": 0}
    rows: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    best_value = math.inf
    best_interval: int | None = None
    started = time.perf_counter()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    initial_model_state_sha256 = tensor_state_sha256(model.state_dict())
    initial_probe_hash: str | None = None
    with torch.no_grad():
        initial_probe_hash = tensor_sha256(data.predict(model, 169)["z_prediction"])

    for interval_index, (label, groups) in enumerate(zip(interval_labels, groups_by_interval, strict=True)):
        interval_losses: list[float] = []
        model.train()
        for group in groups:
            optimizer.zero_grad(set_to_none=True)
            group_losses: list[float] = []
            for source in group:
                result = data.predict(model, source)
                loss = loss_fn(result["predicted_residual"], result["residual_target"])
                if not torch.isfinite(loss):
                    raise FloatingPointError("Stage S training loss became NaN/Inf")
                (loss / len(group)).backward()
                group_losses.append(float(loss.detach().cpu()))
            before_clip = gradient_norm(parameters)
            if not math.isfinite(before_clip) or before_clip <= 0:
                raise FloatingPointError("Stage S gradients are nonfinite or zero")
            clip = float(config["optimizer"]["gradient_clip_norm"])
            torch.nn.utils.clip_grad_norm_(parameters, clip)
            after_clip = gradient_norm(parameters)
            update = state["optimizer_updates"] + 1
            learning_rate = warmup_cosine_learning_rate(
                update, total_updates=total_updates, warmup_updates=warmup_updates,
                base_learning_rate=float(config["optimizer"]["learning_rate"]),
                min_learning_rate=float(config["scheduler"]["min_learning_rate"]),
            )
            for group_values in optimizer.param_groups:
                group_values["lr"] = learning_rate
            optimizer.step()
            if not all(torch.isfinite(parameter).all() for parameter in parameters):
                raise FloatingPointError("Stage S optimizer produced NaN/Inf parameters")
            state["optimizer_updates"] = update
            state["microbatches"] += len(group)
            mean_loss = float(np.mean(group_losses))
            interval_losses.extend(group_losses)
            rows.append({
                "optimizer_update": update,
                "interval": label,
                "accumulation_count": len(group),
                "microbatches_seen": state["microbatches"],
                "mean_microbatch_loss": mean_loss,
                "gradient_norm_before_clip": before_clip,
                "gradient_norm_after_clip": after_clip,
                "clipping_triggered": before_clip > clip,
                "learning_rate": learning_rate,
                "nonfinite": False,
            })
        state["intervals_complete"] = interval_index + 1
        scored_pairs = validation_pairs[:2] if smoke else validation_pairs
        validation = validation_score(model, data, scored_pairs)
        value = float(validation["normalized_relative_l2"]["arithmetic_average"])
        validation_record = {
            "interval": label,
            "optimizer_updates": state["optimizer_updates"],
            "microbatches": state["microbatches"],
            "train_loss_mean": float(np.mean(interval_losses)),
            "train_loss_first": interval_losses[0],
            "train_loss_last": interval_losses[-1],
            **validation,
        }
        validations.append(validation_record)
        with torch.no_grad():
            probe_hash = tensor_sha256(data.predict(model, 169)["z_prediction"])
        metadata = {
            "experiment": args.experiment,
            "initial_tensor_state_sha256": config["frozen_pairing"]["initial_tensor_state_sha256"],
            "pair_order_sha256": order_sha256(flat_order),
            "parameter_count": trainable_parameter_count(model),
            "loss": "strict_plain_l2_normalized_residual",
            "disabled": ["H1", "ROI", "bounds", "radial", "physics"],
            "gradient_clip_norm": clip,
            "probe_prediction_sha256": probe_hash,
            "provenance": config_payload["provenance"],
        }
        payload = checkpoint_payload(
            model=model, optimizer=optimizer, state=state,
            metadata=metadata, validation=validation_record,
        )
        save_checkpoint(output / "last.pt", payload)
        if value < best_value:
            best_value = value
            best_interval = int(label)
            save_checkpoint(output / "best.pt", payload)
        write_csv(output / "train_log.csv", rows)
        write_json(output / "validation_log.json", validations)
        print(json.dumps({
            "experiment": args.experiment,
            "interval": label,
            "updates": state["optimizer_updates"],
            "train_loss_mean": validation_record["train_loss_mean"],
            "validation_average": value,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        }, sort_keys=True), flush=True)

    if state["optimizer_updates"] != total_updates or state["microbatches"] != len(flat_order):
        raise RuntimeError("Stage S completed budget differs from resolved budget")
    final_hash = tensor_state_sha256(model.state_dict())
    if final_hash == initial_model_state_sha256:
        raise RuntimeError("Stage S model did not learn a non-zero update")
    best_payload = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    last_payload = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
    best_check = strict_reload_check(
        output / "best.pt", config=config, data=data,
        expected_probe_hash=best_payload["metadata"]["probe_prediction_sha256"], device=device,
    )
    last_check = strict_reload_check(
        output / "last.pt", config=config, data=data,
        expected_probe_hash=last_payload["metadata"]["probe_prediction_sha256"], device=device,
    )
    smoke_loss_decreased = validations[-1]["train_loss_mean"] < validations[0]["train_loss_mean"]
    if smoke and not smoke_loss_decreased:
        raise RuntimeError("Two-epoch Stage S smoke loss did not decrease numerically")
    summary = {
        "schema_version": "stage-s-training-summary-v1",
        "experiment": args.experiment,
        "status": "passed",
        "smoke": smoke,
        "train_pairs": len(population),
        "validation_pairs_evaluated": 2 if smoke else len(validation_pairs),
        "optimizer_updates": state["optimizer_updates"],
        "microbatches": state["microbatches"],
        "effective_samples_seen": state["microbatches"],
        "population_epochs_equivalent": state["microbatches"] / len(population),
        "best_interval": best_interval,
        "best_validation_normalized_average": best_value,
        "initial_probe_sha256": initial_probe_hash,
        "initial_model_state_sha256": initial_model_state_sha256,
        "final_model_state_sha256": final_hash,
        "learned_nonzero_update": True,
        "all_finite": True,
        "gradients_nonzero": True,
        "smoke_loss_decreased": smoke_loss_decreased if smoke else None,
        "best_checkpoint_reload": best_check,
        "last_checkpoint_reload": last_check,
        "runtime_seconds": time.perf_counter() - started,
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
    }
    write_json(output / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    data.close()


if __name__ == "__main__":
    main()
