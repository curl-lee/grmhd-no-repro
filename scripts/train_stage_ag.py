#!/usr/bin/env python3
"""Stage AG controlled 300-epoch training and artifact-integrity checks only."""

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
from typing import Any, Iterable, Mapping

import h5py
import numpy as np
import torch
import yaml

from grmhd.anisotropic_spherical_disco_localno import attach_anisotropic_spherical_disco3d
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

from train_stage_s import StageSBatchPath, build_frozen_model, gradient_norm, validation_score


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/stage_ag"
CONFIG = ROOT / "configs/stage_ag/controlled_training.yaml"
PINNED_UPSTREAM = "86a8bc7812a31b42c4f7895693cf4ac11521c066"
EXPECTED_INITIAL_HASH = "77252855f054a199500fa779f7340b33b6a2ce546b2082f17f93666275f4c588"


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True))


def atomic_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty Stage AG table {path}")
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
        return [dict(row) for row in csv.DictReader(handle)]


def save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), temporary)
    temporary.replace(path)


def tensor_parameters(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {name: value for name, value in model.state_dict().items() if torch.is_tensor(value)}


def trainable_hash(model: torch.nn.Module) -> str:
    return tensor_state_sha256({name: value for name, value in model.named_parameters()})


def tensor_hash(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    import hashlib
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def directory_sha256(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    for entry in sorted(value for value in path.rglob("*") if value.is_file()):
        digest.update(str(entry.relative_to(path)).encode())
        digest.update(bytes.fromhex(sha256_file(entry)))
    return digest.hexdigest()


def parameter_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    # Spectral weights are complex; abs-square retains both real and imaginary
    # components before the final real-valued reduction.
    values = [torch.sum(parameter.detach().abs().float().square()) for parameter in parameters]
    return 0.0 if not values else float(torch.sqrt(torch.stack(values).sum()).cpu())


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
        raise RuntimeError("Stage AG branch parameter grouping is incomplete")
    return groups


def load_contract() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    frozen = (
        ("frozen_stage_af_config", "frozen_stage_af_config_sha256"),
        ("frozen_stage_af_decision", "frozen_stage_af_decision_sha256"),
        ("frozen_stage_af_parameter_identity", "frozen_stage_af_parameter_identity_sha256"),
        ("frozen_stage_s_config", "frozen_stage_s_config_sha256"),
        ("frozen_stage_t_resolved", "frozen_stage_t_resolved_sha256"),
        ("frozen_stage_ad_epoch300", "frozen_stage_ad_epoch300_sha256"),
    )
    for path_key, hash_key in frozen:
        if sha256_file(ROOT / config[path_key]) != config[hash_key]:
            raise ValueError(f"frozen object changed: {path_key}")
    decision = json.loads((ROOT / config["frozen_stage_af_decision"]).read_text(encoding="utf-8"))
    if decision["PRIMARY_DECISION"] != "A" or decision["AUTHORIZE_NEXT_STAGE"] != "anisotropic_spherical_disco_controlled_training":
        raise RuntimeError("Stage AF does not authorize Stage AG training")
    if any(decision[key] is not True for key in (
        "ALL_BASES_ACTIVE", "CONSTANT_FIELD_PASS", "ANISOTROPIC_DENSE_REFERENCE_MATCH",
        "PHI_EQUIVARIANCE_PASS", "GRADCHECK_PASS", "ANISOTROPIC_DISCO_UNIT_TESTS_PASS",
        "ANISOTROPIC_GEOMETRY_DISTINCT_FROM_STAGE_AD", "TRAINABLE_PARAMETER_COUNT_MATCH",
        "TRAINABLE_INITIALIZATION_HASH_MATCH", "PRODUCTION_FORWARD_PASS", "PRODUCTION_BACKWARD_PASS",
        "DISCO_BRANCH_FORWARD_ACTIVE", "DISCO_BRANCH_GRADIENT_ACTIVE",
    )):
        raise RuntimeError("one or more frozen Stage AF readiness gates changed")
    stage_s = yaml.safe_load((ROOT / config["frozen_stage_s_config"]).read_text(encoding="utf-8"))
    stage_t = json.loads((ROOT / config["frozen_stage_t_resolved"]).read_text(encoding="utf-8"))
    upstream = subprocess.check_output(
        ["git", "-C", str(ROOT / "external/neuraloperator"), "rev-parse", "HEAD"], text=True
    ).strip()
    upstream_status = subprocess.check_output(
        ["git", "-C", str(ROOT / "external/neuraloperator"), "status", "--short"], text=True
    ).strip()
    if upstream != PINNED_UPSTREAM or upstream_status:
        raise ValueError("pinned upstream changed or is dirty")
    return config, stage_s, stage_t


def read_coordinates(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(ROOT / config["data"]["dataset"], "r") as handle:
        return tuple(np.asarray(handle[f"coords/{name}"], dtype=np.float64) for name in ("r", "theta", "phi"))


def build_initial_model(config: Mapping[str, Any], stage_s: Mapping[str, Any]) -> torch.nn.Module:
    common_path = ROOT / config["initialization"]["common_state"]
    if sha256_file(common_path) != config["initialization"]["common_state_file_sha256"]:
        raise ValueError("common initial-state file changed")
    common = torch.load(common_path, map_location="cpu", weights_only=True)
    if tensor_state_sha256(common) != config["initialization"]["common_tensor_state_sha256"]:
        raise ValueError("common initial tensor state changed")
    torch.manual_seed(int(config["pair_order"]["seed"]))
    np.random.seed(int(config["pair_order"]["seed"]))
    model = build_frozen_model(stage_s)
    model.load_state_dict(common, strict=True)
    r, theta, phi = read_coordinates(config)
    attach_anisotropic_spherical_disco3d(
        model, r=r, theta=theta, phi=phi,
        seed=int(config["initialization"]["disco_seed"]),
    )
    observed = trainable_hash(model)
    expected = config["initialization"]["expected_trainable_state_sha256"]
    if observed != expected or observed != EXPECTED_INITIAL_HASH:
        raise RuntimeError(f"initial trainable-state hash mismatch: {observed}")
    if trainable_parameter_count(model) != int(config["model"]["parameter_count"]):
        raise RuntimeError("Stage AG parameter count changed")
    return model


def pair_orders(
    config: Mapping[str, Any], stage_s: Mapping[str, Any], stage_t: Mapping[str, Any]
) -> tuple[list[list[int]], dict[str, Any], list[int], list[int]]:
    split_path = ROOT / config["data"]["split"]
    if sha256_file(split_path) != config["data"]["split_sha256"]:
        raise ValueError("frozen train/validation split changed")
    split = json.loads(split_path.read_text(encoding="utf-8"))
    train_pairs = [int(value) for value in split["train_pair_source_indices"]]
    validation_pairs = [int(value) for value in split["validation_pair_source_indices"]]
    if train_pairs != list(range(168)) or validation_pairs != list(range(169, 211)):
        raise ValueError("Stage AG chronological split changed")
    epochs = int(config["training"]["epochs"])
    orders = natural_epoch_orders(train_pairs, epochs=epochs, seed=int(config["pair_order"]["seed"]))
    hashes = [order_sha256(order) for order in orders]
    if hashes != stage_t["epoch_order_sha256"][:epochs]:
        raise ValueError("Stage AG epoch orders differ from frozen Stage T/AD orders")
    flat_hash = order_sha256(value for order in orders for value in order)
    if flat_hash != config["pair_order"]["expected_flat_sha256"]:
        raise ValueError("Stage AG flattened pair-order hash changed")
    record = {
        "algorithm": config["pair_order"]["algorithm"], "seed": config["pair_order"]["seed"],
        "epochs": epochs, "pairs_per_epoch": len(train_pairs), "flat_pair_count": epochs * len(train_pairs),
        "observed_flat_sha256": flat_hash, "expected_flat_sha256": config["pair_order"]["expected_flat_sha256"],
        "epoch_order_sha256": hashes, "matches_stage_t_first_300": True,
        "matches_stage_ad_epoch300_metadata": True, "validation_shuffle": False,
        "dropped_boundary_pair": [168, 169], "pass": True,
    }
    return orders, record, train_pairs, validation_pairs


def dataset_record(config: Mapping[str, Any]) -> dict[str, Any]:
    path = ROOT / config["data"]["dataset"]
    observed_hash = sha256_file(path)
    with h5py.File(path, "r") as handle:
        shape = list(handle["snapshots"].shape)
        metadata = json.loads(handle.attrs["metadata_json"])
    expected_shape = list(config["data"]["shape"])
    result = {
        "path": config["data"]["dataset"], "observed_sha256": observed_hash,
        "expected_sha256": config["data"]["dataset_sha256"], "shape": shape,
        "expected_shape": expected_shape, "axis_order": config["data"]["axis_order"],
        "channels": metadata["channels"], "train_snapshot_indices": [0, 168],
        "train_pair_source_indices": [0, 167], "validation_snapshot_indices": [169, 211],
        "validation_pair_source_indices": [169, 210], "dropped_boundary_pair": [168, 169],
    }
    result["pass"] = bool(
        observed_hash == result["expected_sha256"] and shape == expected_shape
        and metadata["channels"] == config["data"]["channels"]
    )
    return result


def pretrain_gate(
    model: torch.nn.Module, data: StageSBatchPath, config: Mapping[str, Any],
    stage_s: Mapping[str, Any], pair_record: Mapping[str, Any], device: torch.device,
) -> None:
    dataset = dataset_record(config)
    normalizer_path = ROOT / config["preprocessing"]["normalizer_file"]
    normalizer = {
        "artifact": config["preprocessing"]["artifact"],
        "file": config["preprocessing"]["normalizer_file"],
        "observed_sha256": sha256_file(normalizer_path),
        "expected_sha256": config["preprocessing"]["normalizer_sha256"],
        "directory_sha256": directory_sha256(ROOT / config["preprocessing"]["artifact"]),
        "validation_used_for_fit": False,
    }
    normalizer["pass"] = normalizer["observed_sha256"] == normalizer["expected_sha256"]
    initial_hash = trainable_hash(model)
    model_identity = {
        "name": config["model"]["name"], "geometry": config["model"]["geometry"],
        "branches": ["SpectralConv", "FiniteDifferenceConv", "AnisotropicSphericalDISCO3D", "linear_skip"],
        "in_channels": 16, "out_channels": 8, "hidden_width": 16,
        "n_modes": [8, 8, 8], "n_layers": 4,
        "basis_count": 5, "candidate_stencil_shape": [7, 7, 7],
        "directional_radius_multiplier": 3, "parameter_count": trainable_parameter_count(model),
        "phi_boundary": "PERIODIC", "theta_boundary": "TRUNCATED_RENORMALIZED",
        "r_boundary": "TRUNCATED_RENORMALIZED", "pass": trainable_parameter_count(model) == 363480,
    }
    initialization = {
        "observed_trainable_state_sha256": initial_hash,
        "expected_trainable_state_sha256": EXPECTED_INITIAL_HASH,
        "matches_stage_af_and_stage_ad": initial_hash == EXPECTED_INITIAL_HASH,
        "parameter_count": trainable_parameter_count(model), "pass": initial_hash == EXPECTED_INITIAL_HASH,
    }
    nvidia = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        text=True, capture_output=True, check=True,
    )
    gpu = {
        "nvidia_smi": nvidia.stdout.strip(), "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(), "device_name": torch.cuda.get_device_name(device),
        "capability": list(torch.cuda.get_device_capability(device)), "cpu_fallback": False,
    }
    gpu["pass"] = bool(gpu["cuda_available"] and gpu["device_count"] >= 1 and "RTX 5070" in gpu["device_name"])

    model.to(device).train()
    model.zero_grad(set_to_none=True)
    result = data.predict(model, 0)
    loss = PlainL2Loss().to(device)(result["predicted_residual"], result["residual_target"])
    loss.backward()
    disco_parameters = [parameter for name, parameter in model.named_parameters() if ".local_convs." in name]
    disco_gradient = gradient_norm(disco_parameters)
    gradients_finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in model.parameters()
    )
    no_update = {
        "pair_source_index": 0, "input_shape": list(result["model_input"].shape),
        "output_shape": list(result["predicted_residual"].shape),
        "output_finite": bool(torch.isfinite(result["predicted_residual"]).all()),
        "loss": float(loss.detach().cpu()), "loss_finite": bool(torch.isfinite(loss)),
        "gradients_finite": gradients_finite, "disco_gradient_norm": disco_gradient,
        "disco_gradient_nonzero": disco_gradient > 0, "optimizer_created": False,
        "parameter_update_performed": False,
    }
    model.zero_grad(set_to_none=True)
    no_update["trainable_hash_after_zero_grad"] = trainable_hash(model)
    no_update["hash_unchanged"] = no_update["trainable_hash_after_zero_grad"] == initial_hash
    no_update["pass"] = bool(
        no_update["output_finite"] and no_update["loss_finite"] and gradients_finite
        and disco_gradient > 0 and no_update["hash_unchanged"]
    )
    records = {
        "dataset_identity.json": dataset, "normalizer_identity.json": normalizer,
        "model_identity.json": model_identity, "initialization_identity.json": initialization,
        "pair_order_identity.json": dict(pair_record), "gpu_gate.json": gpu,
        "no_update_preflight.json": no_update,
    }
    for name, record in records.items():
        atomic_json(OUT / "pretrain" / name, record)
    atomic_json(OUT / "pair_order_identity.json", pair_record)
    if not all(record["pass"] for record in records.values()):
        raise RuntimeError("Stage AG pretrain gate failed")
    atomic_text(OUT / "scope/training_contract.md", f"""# Stage AG Frozen Training Contract

- training only; no Stage AD/AG scientific comparison or rollout
- dataset `{config['data']['dataset']}` SHA256 `{config['data']['dataset_sha256']}`
- chronological train pairs 0..167; drop 168->169; validation pairs 169..210
- frozen expanded-data train-only P3 normalizer; normalized residual Plain L2
- anisotropic spherical DISCO LocalNO, 363480 parameters, K=5, 7x7x7, multiplier=3
- 300 epochs, 168 microbatches/epoch, accumulation=4, 42 updates/epoch
- Adam lr=1e-3, weight decay=1e-4, global clip=1, AMP=false
- frozen 1200-epoch Stage T scheduler horizon: 50400 updates, warmup=3150
- no early stopping and no validation-driven modification
""")
    atomic_text(OUT / "PRETRAIN_AUDIT.md", f"""# Stage AG Pretrain Audit

`STAGE_AG_PRETRAIN_READY`

- dataset identity: PASS (`{dataset['observed_sha256']}`; shape `{dataset['shape']}`)
- train/validation split and dropped boundary: PASS
- frozen P3 normalizer identity: PASS (`{normalizer['observed_sha256']}`)
- model identity and 363480 parameters: PASS
- initial trainable SHA256: `{initial_hash}` (exact Stage AF/AD match)
- 300-epoch pair order: PASS (`{pair_record['observed_flat_sha256']}`)
- GPU gate: PASS (`{gpu['device_name']}`)
- no-update forward/loss/backward: PASS; DISCO gradient `{disco_gradient}`
- hash after backward/zero-grad unchanged: PASS
- optimizer created: false; parameter update performed: false

All frozen gates passed before the first optimizer update.
""")
    print("STAGE_AG_PRETRAIN_READY", flush=True)


def train_group(
    *, sources: Iterable[int], model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    loss_fn: PlainL2Loss, data: StageSBatchPath, all_parameters: list[torch.nn.Parameter],
    branches: Mapping[str, list[torch.nn.Parameter]], clip: float, update: int,
    scheduler_total_updates: int, scheduler_warmup_updates: int,
    base_lr: float, min_lr: float,
) -> dict[str, Any]:
    batch_sources = [int(value) for value in sources]
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for source in batch_sources:
        result = data.predict(model, source)
        loss = loss_fn(result["predicted_residual"], result["residual_target"])
        if not torch.isfinite(loss):
            raise FloatingPointError("Stage AG training loss became NaN/Inf")
        (loss / len(batch_sources)).backward()
        losses.append(float(loss.detach().cpu()))
    before = gradient_norm(all_parameters)
    branch_gradients = {name: gradient_norm(parameters) for name, parameters in branches.items()}
    if not math.isfinite(before) or before <= 0:
        raise FloatingPointError("Stage AG total gradient is nonfinite or zero")
    if not math.isfinite(branch_gradients["disco3d"]) or branch_gradients["disco3d"] <= 0:
        raise FloatingPointError("Stage AG DISCO gradient is nonfinite or zero")
    torch.nn.utils.clip_grad_norm_(all_parameters, clip)
    after = gradient_norm(all_parameters)
    learning_rate = warmup_cosine_learning_rate(
        update, total_updates=scheduler_total_updates,
        warmup_updates=scheduler_warmup_updates,
        base_learning_rate=base_lr, min_learning_rate=min_lr,
    )
    for group in optimizer.param_groups:
        group["lr"] = learning_rate
    optimizer.step()
    if not all(torch.isfinite(parameter).all() for parameter in all_parameters):
        raise FloatingPointError("Stage AG optimizer produced NaN/Inf parameters")
    return {
        "losses": losses, "gradient_norm_before_clip": before,
        "gradient_norm_after_clip": after, "spectral_gradient_norm": branch_gradients["spectral"],
        "differential_gradient_norm": branch_gradients["differential"],
        "disco_gradient_norm": branch_gradients["disco3d"],
        "clipping_triggered": before > clip, "learning_rate": learning_rate,
    }


def checkpoint_payload(
    *, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
    state: Mapping[str, Any], metadata: Mapping[str, Any], resolved: Mapping[str, Any],
    validation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": "stage-ag-checkpoint-v1",
        "model_state_dict": tensor_parameters(model),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state": {
            "kind": "stage_t_explicit_update_warmup_cosine",
            "completed_updates": int(state["optimizer_updates"]),
            "total_updates": int(metadata["scheduler_total_updates"]),
            "warmup_updates": int(metadata["scheduler_warmup_updates"]),
        },
        "training_state": dict(state), "metadata": dict(metadata),
        "resolved_config": dict(resolved), "validation": None if validation is None else dict(validation),
        "trainable_state_sha256": trainable_hash(model),
    }


def selector_record(metrics: Mapping[str, Any], epoch: int) -> dict[str, Any]:
    normalized = metrics["normalized_relative_l2"]
    return {
        "epoch": epoch,
        "optimizer_updates": epoch * 42,
        "normalized_per_channel_relative_l2_arithmetic_average": normalized["arithmetic_average"],
        "normalized_global_relative_l2": normalized["global"],
        **{f"channel_{name}": value for name, value in normalized["per_channel"].items()},
    }


def fresh_reload(
    path: Path, expected_metric: float, config: Mapping[str, Any], stage_s: Mapping[str, Any],
    validation_pairs: list[int], device: torch.device,
) -> tuple[dict[str, Any], torch.nn.Module]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = build_initial_model(config, stage_s)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    data = StageSBatchPath(
        ROOT / stage_s["data"]["dataset"], ROOT / stage_s["preprocessing"]["artifact"], device
    )
    with torch.no_grad():
        probe_a = data.predict(model, 169)["predicted_residual"]
        probe_b = data.predict(model, 169)["predicted_residual"]
    probe_hash_a, probe_hash_b = tensor_hash(probe_a), tensor_hash(probe_b)
    metrics = validation_score(model, data, validation_pairs)
    observed_metric = float(metrics["normalized_relative_l2"]["arithmetic_average"])
    difference = abs(observed_metric - expected_metric)
    observed_hash = trainable_hash(model)
    record = {
        "checkpoint": str(path.relative_to(ROOT)),
        "epoch": int(payload["training_state"]["completed_epoch"]),
        "optimizer_updates": int(payload["training_state"]["optimizer_updates"]),
        "scheduler_completed_updates": int(payload["scheduler_state"]["completed_updates"]),
        "stored_trainable_state_sha256": payload["trainable_state_sha256"],
        "reloaded_trainable_state_sha256": observed_hash,
        "trainable_hash_match": observed_hash == payload["trainable_state_sha256"],
        "optimizer_state_reload": True, "scheduler_state_reload": True,
        "deterministic_probe_sha256": probe_hash_a,
        "deterministic_probe_pass": probe_hash_a == probe_hash_b,
        "stored_selector_metric": expected_metric, "recomputed_selector_metric": observed_metric,
        "selector_absolute_difference": difference, "selector_recompute_pass": difference <= 1e-12,
    }
    record["pass"] = bool(
        record["trainable_hash_match"] and record["deterministic_probe_pass"]
        and record["selector_recompute_pass"]
    )
    data.close()
    return record, model


def cache_predictions(
    model: torch.nn.Module, stage_s: Mapping[str, Any], validation_pairs: list[int], device: torch.device
) -> None:
    destination = OUT / "raw_validation_predictions"
    destination.mkdir(parents=True, exist_ok=True)
    data = StageSBatchPath(
        ROOT / stage_s["data"]["dataset"], ROOT / stage_s["preprocessing"]["artifact"], device
    )
    model.eval()
    rows = []
    with torch.no_grad():
        for source in validation_pairs:
            result = data.predict(model, source)
            path = destination / f"pair_{source:03d}_{source + 1:03d}.npz"
            np.savez_compressed(
                path,
                predicted_normalized_residual=result["predicted_residual"].cpu().numpy(),
                predicted_next_normalized_state=result["z_prediction"].cpu().numpy(),
                target_normalized_residual=result["residual_target"].cpu().numpy(),
                target_next_normalized_state=result["z_target"].cpu().numpy(),
            )
            rows.append({"source": source, "target": source + 1, "path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)})
    atomic_json(destination / "manifest.json", {
        "schema_version": "stage-ag-raw-validation-predictions-v1",
        "checkpoint": "best.pt", "scientific_analysis_performed": False,
        "pairs": rows,
    })
    data.close()


def run(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage AG refuses CPU fallback")
    config, stage_s, stage_t = load_contract()
    orders, pair_record, train_pairs, validation_pairs = pair_orders(config, stage_s, stage_t)
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model = build_initial_model(config, stage_s)
    data = StageSBatchPath(
        ROOT / stage_s["data"]["dataset"], ROOT / stage_s["preprocessing"]["artifact"], device
    )
    pretrain_gate(model, data, config, stage_s, pair_record, device)
    if args.pretrain_only:
        data.close()
        return

    resolved = {
        "schema_version": "stage-ag-resolved-config-v1",
        "config": config, "stage_s_model": stage_s["model"],
        "stage_s_representation": stage_s["representation"],
        "initial_trainable_state_sha256": EXPECTED_INITIAL_HASH,
        "pair_order_sha256": pair_record["observed_flat_sha256"],
        "epoch_order_sha256": pair_record["epoch_order_sha256"],
        "project_git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "upstream_commit": PINNED_UPSTREAM,
        "environment": {"python": sys.version, "torch": str(torch.__version__), "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(device), "platform": platform.platform()},
    }
    atomic_json(OUT / "training/resolved_config.json", resolved)
    metadata = {
        "contract": "Stage AG anisotropic spherical DISCO3D controlled training only",
        "data_sha256": config["data"]["dataset_sha256"],
        "normalizer_sha256": config["preprocessing"]["normalizer_sha256"],
        "initial_trainable_state_sha256": EXPECTED_INITIAL_HASH,
        "pair_order_sha256": pair_record["observed_flat_sha256"],
        "geometry_contract": config["model"], "epochs": 300,
        "microbatches_per_epoch": 168, "updates_per_epoch": 42,
        "expected_optimizer_updates": 12600,
        "scheduler_total_updates": int(config["training"]["scheduler_total_updates"]),
        "scheduler_warmup_updates": int(config["training"]["scheduler_warmup_updates"]),
        "formal_selector": config["validation"]["selector"],
        "validation_not_used_for_stopping_or_tuning": True,
        "git_commit": resolved["project_git_commit"], "upstream_commit": PINNED_UPSTREAM,
    }
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    loss_fn = PlainL2Loss().to(device)
    rows = load_csv(OUT / "training/train_log.csv") if args.resume else []
    validation_rows = load_csv(OUT / "validation/fixed_checkpoint_metrics.csv") if args.resume else []
    state: dict[str, Any] = {"completed_epoch": 0, "optimizer_updates": 0, "microbatches": 0, "runtime_seconds": 0.0, "nonfinite_count": 0}
    resume_path = OUT / "training/resume.pt"
    if args.resume:
        if not resume_path.exists():
            raise FileNotFoundError("Stage AG resume checkpoint is missing")
        payload = torch.load(resume_path, map_location="cpu", weights_only=True)
        if payload["metadata"] != metadata:
            raise ValueError("Stage AG resume metadata changed")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        state = dict(payload["training_state"])
        if len(rows) != int(state["completed_epoch"]):
            raise ValueError("Stage AG resume log/checkpoint mismatch")
        # Recompute the last checkpoint-bound weight norms. This also repairs
        # telemetry from the initial epoch if an older runner discarded the
        # imaginary part of complex spectral weights; model/optimizer state is
        # unchanged.
        resume_groups = named_groups(model)
        rows[-1]["spectral_weight_norm"] = parameter_norm(resume_groups["spectral"])
        rows[-1]["differential_weight_norm"] = parameter_norm(resume_groups["differential"])
        rows[-1]["disco_weight_norm"] = parameter_norm(resume_groups["disco3d"])
        atomic_csv(OUT / "training/train_log.csv", rows)
        atomic_csv(OUT / "training/epoch_summary.csv", rows)
    elif resume_path.exists() or rows:
        raise FileExistsError("Stage AG output already contains training state; use --resume")

    model.to(device).train()
    all_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    branches = named_groups(model)
    mandatory = {2, 10, 30, 75, 150, 300}
    best_metric = min(
        (float(row["normalized_per_channel_relative_l2_arithmetic_average"]) for row in validation_rows),
        default=float("inf"),
    )
    best_epoch = next(
        (int(row["epoch"]) for row in validation_rows if float(row["normalized_per_channel_relative_l2_arithmetic_average"]) == best_metric),
        None,
    )
    total_started_wall = time.time() - float(state["runtime_seconds"])
    for epoch_index in range(int(state["completed_epoch"]), 300):
        epoch = epoch_index + 1
        groups = accumulation_groups(orders[epoch_index], accumulation=4)
        if len(groups) != 42 or any(len(group) != 4 for group in groups):
            raise RuntimeError("Stage AG accumulation contract changed")
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        update_records = []
        for group in groups:
            update = int(state["optimizer_updates"]) + 1
            record = train_group(
                sources=group, model=model, optimizer=optimizer, loss_fn=loss_fn, data=data,
                all_parameters=all_parameters, branches=branches,
                clip=float(config["training"]["gradient_clip_norm"]), update=update,
                scheduler_total_updates=int(config["training"]["scheduler_total_updates"]),
                scheduler_warmup_updates=int(config["training"]["scheduler_warmup_updates"]),
                base_lr=float(config["training"]["learning_rate"]),
                min_lr=float(config["training"]["scheduler_min_learning_rate"]),
            )
            state["optimizer_updates"] = update
            state["microbatches"] = int(state["microbatches"]) + 4
            update_records.append(record)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        state["completed_epoch"] = epoch
        state["runtime_seconds"] = float(state["runtime_seconds"]) + elapsed
        losses = [value for record in update_records for value in record["losses"]]
        row = {
            "epoch": epoch, "learning_rate": update_records[-1]["learning_rate"],
            "learning_rate_start": update_records[0]["learning_rate"],
            "train_loss_mean": float(np.mean(losses)), "train_loss_median": float(np.median(losses)),
            "train_loss_min": float(np.min(losses)), "train_loss_max": float(np.max(losses)),
            "gradient_norm_before_clip_mean": float(np.mean([item["gradient_norm_before_clip"] for item in update_records])),
            "gradient_norm_before_clip_max": float(np.max([item["gradient_norm_before_clip"] for item in update_records])),
            "gradient_norm_after_clip_mean": float(np.mean([item["gradient_norm_after_clip"] for item in update_records])),
            "clip_fraction": float(np.mean([item["clipping_triggered"] for item in update_records])),
            "spectral_gradient_norm_mean": float(np.mean([item["spectral_gradient_norm"] for item in update_records])),
            "differential_gradient_norm_mean": float(np.mean([item["differential_gradient_norm"] for item in update_records])),
            "disco_gradient_norm_mean": float(np.mean([item["disco_gradient_norm"] for item in update_records])),
            "disco_gradient_norm_min": float(np.min([item["disco_gradient_norm"] for item in update_records])),
            "spectral_weight_norm": parameter_norm(branches["spectral"]),
            "differential_weight_norm": parameter_norm(branches["differential"]),
            "disco_weight_norm": parameter_norm(branches["disco3d"]),
            "optimizer_updates": state["optimizer_updates"], "optimizer_updates_this_epoch": 42,
            "microbatches_completed": state["microbatches"], "microbatches_this_epoch": 168,
            "epoch_runtime_seconds": elapsed, "cumulative_runtime_seconds": state["runtime_seconds"],
            "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            "nonfinite_count": state["nonfinite_count"],
        }
        rows.append(row)
        atomic_csv(OUT / "training/train_log.csv", rows)
        atomic_csv(OUT / "training/epoch_summary.csv", rows)
        validation = None
        if epoch in mandatory:
            model.eval()
            metrics = validation_score(model, data, validation_pairs)
            model.train()
            validation = selector_record(metrics, epoch)
            validation_rows.append(validation)
            atomic_csv(OUT / "validation/fixed_checkpoint_metrics.csv", validation_rows)
        payload = checkpoint_payload(
            model=model, optimizer=optimizer, state=state, metadata=metadata,
            resolved=resolved, validation=validation,
        )
        save_checkpoint(resume_path, payload)
        if epoch in mandatory:
            checkpoint = OUT / "training/checkpoints" / f"epoch_{epoch:04d}.pt"
            save_checkpoint(checkpoint, payload)
            metric = float(validation["normalized_per_channel_relative_l2_arithmetic_average"])
            if metric < best_metric:
                best_metric, best_epoch = metric, epoch
                save_checkpoint(OUT / "training/checkpoints/best.pt", payload)
                atomic_json(OUT / "validation/best_selector_metric.json", validation)
        if epoch == 300:
            save_checkpoint(OUT / "training/checkpoints/last.pt", payload)
            atomic_json(OUT / "validation/last_selector_metric.json", validation)
        average_epoch = float(state["runtime_seconds"]) / epoch
        eta = average_epoch * (300 - epoch)
        print(json.dumps({
            "epoch": f"{epoch}/300", "train_loss": row["train_loss_mean"], "lr": row["learning_rate"],
            "epoch_seconds": elapsed, "elapsed_seconds": state["runtime_seconds"],
            "estimated_remaining_seconds": eta, "gpu_peak_allocated_mib": row["gpu_peak_allocated_mib"],
            "gpu_peak_reserved_mib": row["gpu_peak_reserved_mib"],
        }, sort_keys=True), flush=True)

    if int(state["completed_epoch"]) != 300 or int(state["optimizer_updates"]) != 12600 or int(state["microbatches"]) != 50400:
        raise RuntimeError("Stage AG completed budget changed")
    if best_epoch is None:
        raise RuntimeError("Stage AG formal best checkpoint was not selected")
    best_expected = float(json.loads((OUT / "validation/best_selector_metric.json").read_text())["normalized_per_channel_relative_l2_arithmetic_average"])
    last_expected = float(json.loads((OUT / "validation/last_selector_metric.json").read_text())["normalized_per_channel_relative_l2_arithmetic_average"])
    best_reload, best_model = fresh_reload(
        OUT / "training/checkpoints/best.pt", best_expected, config, stage_s, validation_pairs, device
    )
    atomic_json(OUT / "integrity/best_reload_test.json", best_reload)
    last_reload, last_model = fresh_reload(
        OUT / "training/checkpoints/last.pt", last_expected, config, stage_s, validation_pairs, device
    )
    atomic_json(OUT / "integrity/last_reload_test.json", last_reload)
    if not best_reload["pass"] or not last_reload["pass"]:
        raise RuntimeError("Stage AG strict checkpoint reload failed")
    del last_model
    torch.cuda.empty_cache()
    if not args.skip_prediction_cache:
        cache_predictions(best_model, stage_s, validation_pairs, device)
    peak_allocated = max(float(row["gpu_peak_allocated_mib"]) for row in rows)
    peak_reserved = max(float(row["gpu_peak_reserved_mib"]) for row in rows)
    atomic_text(OUT / "TRAINING_COMPLETE.md", f"""# Stage AG Training Complete

```text
TRAINING_STARTED = true
TRAINING_COMPLETED = true

EPOCHS_COMPLETED = 300
MICROBATCHES_COMPLETED = 50400
OPTIMIZER_UPDATES_COMPLETED = 12600

MODEL =
ANISOTROPIC_SPHERICAL_DISCO_LOCALNO

PARAMETER_COUNT = 363480

INITIAL_TRAINABLE_STATE_SHA256 =
{EXPECTED_INITIAL_HASH}

FORMAL_BEST_EPOCH = {best_epoch}

BEST_CHECKPOINT = artifacts/stage_ag/training/checkpoints/best.pt
LAST_CHECKPOINT = artifacts/stage_ag/training/checkpoints/last.pt

BEST_CHECKPOINT_RELOAD_PASS = true
LAST_CHECKPOINT_RELOAD_PASS = true

NONFINITE_COUNT = {state['nonfinite_count']}

TOTAL_TRAINING_SECONDS = {state['runtime_seconds']}
PEAK_ALLOCATED_VRAM_MIB = {peak_allocated}
PEAK_RESERVED_VRAM_MIB = {peak_reserved}

SCIENTIFIC_ANALYSIS_PERFORMED = false

AUTHORIZE_NEXT_STAGE =
anisotropic_spherical_disco_result_analysis
```
""")
    atomic_text(OUT / "STAGE_AG_STATUS.md", f"""# Stage AG Status

`TRAINING_COMPLETED = true`

The frozen 300-epoch controlled training budget completed with 50,400 microbatches and 12,600 optimizer updates. Formal best epoch: {best_epoch}. Best and last checkpoint reload tests pass. No Stage AD comparison, rollout, attribution, or scientific interpretation was performed.
""")
    data.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-prediction-cache", action="store_true")
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        atomic_text(OUT / "FAILURE_STATE.md", f"""# Stage AG Failure State

- time: `{time.strftime('%Y-%m-%dT%H:%M:%S%z')}`
- error type: `{type(error).__name__}`
- error: `{error}`
- CPU fallback attempted: false
- frozen contract changed automatically: false
- latest recoverable checkpoint: `artifacts/stage_ag/training/resume.pt` if present

```text
{traceback.format_exc()}
```
""")
        raise


if __name__ == "__main__":
    main()
