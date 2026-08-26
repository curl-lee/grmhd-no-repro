#!/usr/bin/env python3
"""Run the gated CUDA/model/runtime preflight for Stage AD."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
import yaml

from grmhd.dataset import sha256_file
from grmhd.localno3d_disco import attach_adapted_disco3d, disco_parameter_names
from grmhd.models import trainable_parameter_count
from grmhd.paper_losses import PlainL2Loss
from grmhd.paper_stage_g import tensor_state_sha256
from grmhd.operators.disco3d import AdaptedRadialDISCO3d

from train_stage_s import StageSBatchPath, build_frozen_model, gradient_norm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/stage_ad"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_contract() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config_path = ROOT / "configs/stage_ad/disco3d_localno.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stage_t_path = ROOT / config["frozen_stage_t_config"]
    if sha256_file(stage_t_path) != config["frozen_stage_t_config_sha256"]:
        raise ValueError("frozen Stage-T config changed")
    stage_t = yaml.safe_load(stage_t_path.read_text(encoding="utf-8"))
    stage_s_path = ROOT / stage_t["frozen_stage_s_config"]
    if sha256_file(stage_s_path) != stage_t["frozen_stage_s_config_sha256"]:
        raise ValueError("frozen Stage-S config changed")
    stage_s = yaml.safe_load(stage_s_path.read_text(encoding="utf-8"))
    implementation = json.loads(
        (OUT / "implementation/implementation_gates.json").read_text(encoding="utf-8")
    )
    mandatory = [
        "ALL_BASES_ACTIVE",
        "BASIS_PARTITION_OF_UNITY_PASS",
        "QUADRATURE_NORMALIZATION_PASS",
        "COMPACT_SUPPORT_PASS",
        "DENSE_REFERENCE_MATCH",
        "KERNEL_ORIENTATION_PASS",
        "GRADCHECK_PASS",
    ]
    if not all(implementation[name] is True for name in mandatory):
        raise RuntimeError("Stage AD implementation gates are not all PASS")
    unit_tests = json.loads(
        (OUT / "implementation/unit_tests.json").read_text(encoding="utf-8")
    )
    if unit_tests["status"] != "passed" or int(unit_tests["failed"]) != 0:
        raise RuntimeError("Stage AD repository/unit-test gate is not PASS")
    return config, stage_t, stage_s


def build_model_and_hashes(config: dict, stage_s: dict) -> tuple[torch.nn.Module, dict]:
    common_state = torch.load(
        ROOT / config["initialization"]["common_state"],
        map_location="cpu",
        weights_only=True,
    )
    expected_common = config["initialization"]["common_tensor_state_sha256"]
    if tensor_state_sha256(common_state) != expected_common:
        raise ValueError("frozen common tensor state changed")
    torch.manual_seed(int(stage_s["runtime"]["seed"]))
    np.random.seed(int(stage_s["runtime"]["seed"]))
    model = build_frozen_model(stage_s)
    model.load_state_dict(common_state, strict=True)
    attach_adapted_disco3d(
        model,
        grid_shape=config["operator"]["grid_shape"],
        domain_length=config["operator"]["domain_length"],
        radius_cells=config["operator"]["radius_cells"],
        basis_count=config["operator"]["basis_count"],
        groups=config["operator"]["groups"],
        bias=config["operator"]["bias"],
        seed=config["initialization"]["disco_seed"],
    )
    state = model.state_dict()
    common_observed = {name: state[name] for name in common_state}
    observed_common_hash = tensor_state_sha256(common_observed)
    if observed_common_hash != expected_common:
        raise RuntimeError("common initialization changed while attaching DISCO3D")
    names = disco_parameter_names(model)
    disco_state = {name: dict(model.named_parameters())[name] for name in names}
    hashes = {
        "common_initialization_match": True,
        "common_tensor_state_sha256": observed_common_hash,
        "disco_parameter_state_sha256": tensor_state_sha256(disco_state),
        "full_initial_state_sha256": tensor_state_sha256(state),
        "disco_seed": config["initialization"]["disco_seed"],
        "disco_parameter_names": names,
    }
    return model, hashes


def branch_hooks(model: torch.nn.Module):
    records: dict[str, dict[str, float]] = {str(index): {} for index in range(4)}
    handles = []
    blocks = model.local_no_blocks

    def capture(layer: int, name: str):
        def hook(_module, _inputs, output):
            records[str(layer)][name] = float(torch.linalg.vector_norm(output.detach()).cpu())
        return hook

    for layer in range(4):
        handles.append(blocks.convs[layer].register_forward_hook(capture(layer, "spectral")))
        handles.append(blocks.differential[layer].register_forward_hook(capture(layer, "differential")))
        handles.append(blocks.local_convs[layer].register_forward_hook(capture(layer, "disco3d")))
        handles.append(blocks.local_no_skips[layer].register_forward_hook(capture(layer, "skip")))
    return records, handles


def disco_timing_hooks(model: torch.nn.Module):
    events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
    handles = []
    for module in model.local_no_blocks.local_convs:
        pair = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        events.append(pair)
        handles.append(module.register_forward_pre_hook(
            lambda _module, _inputs, start=pair[0]: start.record()
        ))
        handles.append(module.register_forward_hook(
            lambda _module, _inputs, output, end=pair[1]: end.record()
        ))
    return events, handles


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage AD CUDA preflight refuses CPU fallback")
    config, stage_t, stage_s = load_contract()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    device_name = torch.cuda.get_device_name(device)
    if stage_s["runtime"]["required_device_substring"] not in device_name:
        raise RuntimeError(f"expected RTX 5070, found {device_name!r}")
    model, hashes = build_model_and_hashes(config, stage_s)
    stage_t_params = int(stage_s["frozen_pairing"]["expected_parameter_count"])
    total_params = trainable_parameter_count(model)
    disco_params = total_params - stage_t_params
    if (stage_t_params, disco_params, total_params) != (358_296, 5_184, 363_480):
        raise RuntimeError("Stage AD parameter audit changed")
    parameter_audit = {
        "stage_t_parameters": stage_t_params,
        "disco3d_parameters": disco_params,
        "total_parameters": total_params,
        "relative_increase": disco_params / stage_t_params,
        "layers": [
            {
                "layer": index,
                "weight_shape": list(branch.weight.shape),
                "bias_shape": list(branch.bias.shape) if branch.bias is not None else None,
            }
            for index, branch in enumerate(model.local_no_blocks.local_convs)
        ],
    }
    atomic_json(OUT / "training/disco3d_localno/initialization_hashes.json", hashes)

    model.to(device)
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise RuntimeError("not all model parameters are on CUDA")
    data = StageSBatchPath(
        ROOT / stage_s["data"]["dataset"],
        ROOT / stage_s["preprocessing"]["artifact"],
        device,
    )
    loss_fn = PlainL2Loss().to(device)
    batch = data.pair(169)
    branch_records, branch_handles = branch_hooks(model)
    model.eval()
    with torch.no_grad():
        warm = model(x=batch["model_input"])
    for handle in branch_handles:
        handle.remove()
    if not torch.isfinite(warm).all():
        raise FloatingPointError("branch-scale audit output is nonfinite")
    for record in branch_records.values():
        record["disco_over_spectral"] = record["disco3d"] / record["spectral"]
        record["disco_over_differential"] = record["disco3d"] / record["differential"]
        if not 1e-6 <= record["disco_over_spectral"] <= 100.0:
            raise RuntimeError("initial DISCO/spectral scale gate failed")
    atomic_json(OUT / "preflight/branch_norms_initial.json", {
        "validation_like_pair": 169,
        "layers": branch_records,
        "gate": "1e-6 <= DISCO/spectral <= 100",
        "pass": True,
    })

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    model.zero_grad(set_to_none=True)
    timing_events, timing_handles = disco_timing_hooks(model)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = data.predict(model, 169)
    loss = loss_fn(result["predicted_residual"], result["residual_target"])
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - started
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(device)
    backward_seconds = time.perf_counter() - started
    for handle in timing_handles:
        handle.remove()
    disco_forward_seconds = sum(start.elapsed_time(end) for start, end in timing_events) / 1000.0
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    disco_parameters = [
        parameter for name, parameter in model.named_parameters() if ".local_convs." in name
    ]
    total_gradient = gradient_norm(parameters)
    disco_gradient = gradient_norm(disco_parameters)
    finite = bool(
        torch.isfinite(result["predicted_residual"]).all()
        and torch.isfinite(loss)
        and all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in parameters
        )
    )
    if not finite or disco_gradient <= 0:
        raise FloatingPointError("Stage AD CUDA forward/backward or DISCO gradient gate failed")

    torch.manual_seed(123)
    # The fixed operator contract expects 64 cubed, so use separately constructed
    # tiny instances for the implementation-level CPU/CUDA agreement.
    cpu_tiny = AdaptedRadialDISCO3d(
        2, 3, grid_shape=(7, 7, 7), domain_length=(2.0, 2.0, 2.0)
    )
    cuda_tiny = AdaptedRadialDISCO3d(
        2, 3, grid_shape=(7, 7, 7), domain_length=(2.0, 2.0, 2.0)
    ).cuda()
    cuda_tiny.load_state_dict(cpu_tiny.state_dict(), strict=True)
    tiny_input = torch.randn(1, 2, 7, 7, 7)
    cpu_value = cpu_tiny(tiny_input)
    cuda_value = cuda_tiny(tiny_input.cuda()).cpu()
    agreement_relative = float((
        torch.linalg.vector_norm(cpu_value - cuda_value)
        / torch.linalg.vector_norm(cpu_value)
    ).detach())
    agreement_max = float(torch.max(torch.abs(cpu_value - cuda_value)).detach())
    agreement_pass = bool(torch.allclose(cpu_value, cuda_value, rtol=2e-5, atol=2e-5))
    if not agreement_pass:
        raise RuntimeError("DISCO3D CPU/CUDA agreement failed")

    peak_allocated = torch.cuda.max_memory_allocated(device) / 2**20
    peak_reserved = torch.cuda.max_memory_reserved(device) / 2**20
    measured_microbatch_seconds = forward_seconds + backward_seconds
    estimated_epoch_seconds = measured_microbatch_seconds * 168 * 1.15
    estimated_150_hours = estimated_epoch_seconds * 150 / 3600.0
    runtime_pass = estimated_150_hours <= 12.0
    preflight = {
        "status": "passed" if runtime_pass else "valid_but_dense_runtime_exceeds_gate",
        "device": device_name,
        "capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "model_parameters_on_cuda": True,
        "input_on_cuda": batch["model_input"].device.type == "cuda",
        "output_on_cuda": result["predicted_residual"].device.type == "cuda",
        "loss_on_cuda": loss.device.type == "cuda",
        "forward_finite": bool(torch.isfinite(result["predicted_residual"]).all()),
        "loss_finite": bool(torch.isfinite(loss)),
        "gradients_finite": finite,
        "total_gradient_norm": total_gradient,
        "disco_gradient_norm": disco_gradient,
        "disco_gradient_nonzero": disco_gradient > 0,
        "forward_seconds": forward_seconds,
        "backward_seconds": backward_seconds,
        "disco_forward_seconds": disco_forward_seconds,
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
        "cpu_cuda_agreement": {
            "relative_l2": agreement_relative,
            "max_abs": agreement_max,
            "rtol": 2e-5,
            "atol": 2e-5,
            "pass": agreement_pass,
        },
    }
    atomic_json(OUT / "preflight/cuda_preflight.json", preflight)
    runtime = {
        "execution_backend": config["operator"]["execution_backend"],
        "measured_training_microbatch_seconds": measured_microbatch_seconds,
        "safety_multiplier": 1.15,
        "estimated_epoch_seconds": estimated_epoch_seconds,
        "estimated_150_epoch_hours": estimated_150_hours,
        "dense_gate_hours": 12.0,
        "dense_backend_feasible": runtime_pass,
        "requires_sparse_equivalent": not runtime_pass,
    }
    atomic_json(OUT / "preflight/runtime_feasibility.json", runtime)

    resolved = {
        "schema_version": "stage-ad-resolved-config-v1",
        "reproduction_scope": config["reproduction_scope"],
        "operator": config["operator"],
        "model_common": stage_s["model"],
        "data": stage_s["data"],
        "preprocessing": stage_s["preprocessing"],
        "prediction": stage_s["prediction"],
        "loss": stage_s["loss"],
        "training": config["training"],
        "evaluation": config["evaluation"],
        "parameter_audit": parameter_audit,
        "initialization": hashes,
        "pair_order": "identical_to_Stage-T_first_150_natural_epoch_orders_seed_42",
        "environment": {
            "project_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "upstream_commit": subprocess.check_output(
                ["git", "-C", str(ROOT / "external/neuraloperator"), "rev-parse", "HEAD"],
                text=True,
            ).strip(),
            "python": sys.version,
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "device": device_name,
            "platform": platform.platform(),
        },
    }
    resolved["contract_sha256"] = sha256_json(resolved)
    atomic_json(OUT / "training/disco3d_localno/resolved_config.json", resolved)
    data.close()
    print(json.dumps({
        "cuda_preflight": preflight,
        "runtime_feasibility": runtime,
        "parameter_audit": parameter_audit,
    }, indent=2, sort_keys=True))
    if not runtime_pass:
        raise RuntimeError("dense 7x7x7 backend exceeds the 12-hour feasibility gate")


if __name__ == "__main__":
    main()
