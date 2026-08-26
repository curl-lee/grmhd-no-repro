#!/usr/bin/env python3
"""Stage AF full-model CUDA forward/backward feasibility audit; no training."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping

import h5py
import numpy as np
import torch
import yaml

from grmhd.anisotropic_spherical_disco_localno import (
    anisotropic_disco_parameter_names,
    attach_anisotropic_spherical_disco3d,
)
from grmhd.dataset import sha256_file
from grmhd.localno3d_disco import attach_adapted_disco3d
from grmhd.models import trainable_parameter_count
from grmhd.operators.anisotropic_spherical_disco3d import AnisotropicSphericalDISCO3d
from grmhd.paper_losses import PlainL2Loss
from grmhd.paper_stage_g import tensor_state_sha256

from train_stage_s import StageSBatchPath, build_frozen_model, gradient_norm


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/stage_af"
CONFIG = ROOT / "configs/stage_af/anisotropic_spherical_disco3d.yaml"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def parameter_hash(model: torch.nn.Module) -> str:
    return tensor_state_sha256({name: value for name, value in model.named_parameters()})


def load_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    geometry = json.loads((OUT / "geometry_audit_summary.json").read_text(encoding="utf-8"))
    required = {
        "zero_directional_scale_count": 0,
        "zero_z_count": 0,
        "all_bases_active": True,
        "constant_field_pass": True,
        "dense_reference_match": True,
        "phi_equivariance_pass": True,
        "gradcheck_pass": True,
        "geometry_distinct_from_stage_ad": True,
    }
    for key, expected in required.items():
        if geometry[key] != expected:
            raise RuntimeError(f"Stage AF preflight blocked by geometry gate {key}")
    for stage in ("ae", "ad", "t", "s"):
        path = ROOT / config[f"frozen_stage_{stage}_config"]
        if sha256_file(path) != config[f"frozen_stage_{stage}_config_sha256"]:
            raise ValueError(f"frozen Stage {stage.upper()} config changed")
    stage_s = yaml.safe_load((ROOT / config["frozen_stage_s_config"]).read_text(encoding="utf-8"))
    return config, stage_s


def coordinates(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = ROOT / config["data"]["dataset"]
    if sha256_file(dataset) != config["data"]["dataset_sha256"]:
        raise ValueError("frozen Z64 dataset changed")
    with h5py.File(dataset, "r") as handle:
        group = handle[config["data"]["coordinate_group"]]
        return tuple(np.asarray(group[key], dtype=np.float64) for key in ("r", "theta", "phi"))


def common_model(config: Mapping[str, Any], stage_s: Mapping[str, Any]) -> torch.nn.Module:
    common = torch.load(ROOT / config["initialization"]["common_state"], map_location="cpu", weights_only=True)
    if tensor_state_sha256(common) != config["initialization"]["common_tensor_state_sha256"]:
        raise ValueError("frozen common initialization changed")
    torch.manual_seed(int(stage_s["runtime"]["seed"]))
    np.random.seed(int(stage_s["runtime"]["seed"]))
    model = build_frozen_model(stage_s)
    model.load_state_dict(common, strict=True)
    return model


def branch_hooks(model: torch.nn.Module) -> tuple[dict[str, dict[str, float]], list[Any]]:
    records: dict[str, dict[str, float]] = {str(index): {} for index in range(4)}
    handles = []

    def capture(layer: int, name: str):
        def hook(_module, _inputs, output):
            records[str(layer)][name] = float(torch.linalg.vector_norm(output.detach()).cpu())
        return hook

    blocks = model.local_no_blocks
    for layer in range(4):
        handles.append(blocks.convs[layer].register_forward_hook(capture(layer, "spectral")))
        handles.append(blocks.differential[layer].register_forward_hook(capture(layer, "differential")))
        handles.append(blocks.local_convs[layer].register_forward_hook(capture(layer, "disco3d")))
        handles.append(blocks.local_no_skips[layer].register_forward_hook(capture(layer, "skip")))
    return records, handles


def timing_hooks(model: torch.nn.Module):
    events, handles = [], []
    for module in model.local_no_blocks.local_convs:
        pair = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        events.append(pair)
        handles.append(module.register_forward_pre_hook(lambda _m, _i, start=pair[0]: start.record()))
        handles.append(module.register_forward_hook(lambda _m, _i, _o, end=pair[1]: end.record()))
    return events, handles


def category_gradient(model: torch.nn.Module, fragment: str) -> float:
    return gradient_norm(
        parameter for name, parameter in model.named_parameters() if fragment in name
    )


def gradients_finite(parameters: Iterable[torch.nn.Parameter]) -> bool:
    return all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in parameters)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage AF CUDA preflight refuses CPU fallback")
    config, stage_s = load_contract()
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    device_name = torch.cuda.get_device_name(device)
    if stage_s["runtime"]["required_device_substring"] not in device_name:
        raise RuntimeError(f"expected RTX 5070, found {device_name!r}")
    r, theta, phi = coordinates(config)

    # Construct the frozen Stage AD trainable state, then independently construct
    # AF from the same common state and RNG seed. Geometry buffers are excluded.
    ad_model = common_model(config, stage_s)
    attach_adapted_disco3d(ad_model, seed=int(config["initialization"]["disco_seed"]))
    ad_parameters = {name: value.detach().clone() for name, value in ad_model.named_parameters()}
    ad_hash = parameter_hash(ad_model)
    del ad_model

    model = common_model(config, stage_s)
    attach_anisotropic_spherical_disco3d(
        model, r=r, theta=theta, phi=phi,
        groups=int(config["operator"]["groups"]), bias=bool(config["operator"]["bias"]),
        seed=int(config["initialization"]["disco_seed"]),
    )
    af_parameters = dict(model.named_parameters())
    names_match = list(ad_parameters) == list(af_parameters)
    tensor_matches = {
        name: bool(torch.equal(ad_parameters[name], af_parameters[name]))
        for name in ad_parameters if name in af_parameters
    }
    initialization_match = names_match and all(tensor_matches.values())
    total_parameters = trainable_parameter_count(model)
    expected_parameters = int(config["architecture"]["expected_parameter_count"])
    af_hash = parameter_hash(model)
    identity = {
        "stage_ad_parameter_count": expected_parameters,
        "stage_af_parameter_count": total_parameters,
        "parameter_count_match": total_parameters == expected_parameters,
        "parameter_names_match": names_match,
        "trainable_initialization_hash_match": initialization_match,
        "stage_ad_trainable_state_sha256": ad_hash,
        "stage_af_trainable_state_sha256": af_hash,
        "hash_match": ad_hash == af_hash,
        "disco_parameter_names": anisotropic_disco_parameter_names(model),
        "per_tensor_exact_match": tensor_matches,
    }
    if not all((identity["parameter_count_match"], initialization_match, identity["hash_match"])):
        write_json(OUT / "preflight/model_parameter_identity.json", identity)
        raise RuntimeError("Stage AF architecture/initialization identity gate failed")
    write_json(OUT / "preflight/model_parameter_identity.json", identity)

    model.to(device)
    if any(parameter.device.type != "cuda" for parameter in model.parameters()):
        raise RuntimeError("not all model parameters are on CUDA")
    data = StageSBatchPath(
        ROOT / stage_s["data"]["dataset"], ROOT / stage_s["preprocessing"]["artifact"], device
    )
    forward_records: dict[str, Any] = {}
    model.eval()
    for label, pair_index in (("training_like", 0), ("validation_like", 169)):
        batch = data.pair(pair_index)
        records, handles = branch_hooks(model)
        with torch.no_grad():
            output = model(x=batch["model_input"])
        for handle in handles:
            handle.remove()
        for record in records.values():
            record["disco_over_spectral"] = record["disco3d"] / record["spectral"]
            record["disco_over_differential"] = record["disco3d"] / record["differential"]
        forward_records[label] = {
            "pair_index": pair_index, "output_shape": list(output.shape),
            "output_norm": float(torch.linalg.vector_norm(output).cpu()),
            "output_finite": bool(torch.isfinite(output).all()), "layers": records,
        }
    forward_active = all(
        layer["disco3d"] > 0 and np.isfinite(layer["disco3d"])
        for sample in forward_records.values() for layer in sample["layers"].values()
    )
    write_json(OUT / "preflight/branch_forward_norms.json", {
        "samples": forward_records, "spherical_disco_branch_forward_active": forward_active,
        "scientific_validation_metric_computed": False,
    })

    # Exactly one production forward/backward feasibility pass. No optimizer or
    # scheduler is constructed, and model parameters are not updated.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    model.train()
    model.zero_grad(set_to_none=True)
    batch = data.pair(0)
    loss_fn = PlainL2Loss().to(device)
    events, handles = timing_hooks(model)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    predicted_residual = model(x=batch["model_input"])
    loss = loss_fn(predicted_residual, batch["residual_target"])
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - started
    started = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(device)
    backward_seconds = time.perf_counter() - started
    for handle in handles:
        handle.remove()
    disco_forward_seconds = sum(start.elapsed_time(end) for start, end in events) / 1000.0
    all_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradients = {
        "spectral_gradient_norm": category_gradient(model, ".convs."),
        "differential_gradient_norm": category_gradient(model, ".differential."),
        "disco_gradient_norm": category_gradient(model, ".local_convs."),
        "skip_gradient_norm": category_gradient(model, ".local_no_skips."),
        "total_gradient_norm": gradient_norm(all_parameters),
    }
    finite = bool(torch.isfinite(predicted_residual).all() and torch.isfinite(loss) and gradients_finite(all_parameters))
    gradient_active = finite and gradients["disco_gradient_norm"] > 0
    write_json(OUT / "preflight/branch_backward_gradients.json", {
        **gradients, "loss": float(loss.detach().cpu()), "all_gradients_finite": finite,
        "spherical_disco_branch_gradient_active": gradient_active,
        "parameter_update_performed": False,
    })

    # Tiny identical-state CPU/CUDA agreement for the operator itself.
    tiny_r = np.geomspace(1.0, 3.0, 5)
    tiny_theta = np.linspace(0.4, np.pi - 0.4, 6)
    tiny_phi = (np.arange(8) + 0.5) * 2 * np.pi / 8
    torch.manual_seed(77)
    cpu_operator = AnisotropicSphericalDISCO3d(1, 1, r=tiny_r, theta=tiny_theta, phi=tiny_phi).float()
    cuda_operator = AnisotropicSphericalDISCO3d(1, 1, r=tiny_r, theta=tiny_theta, phi=tiny_phi).float().cuda()
    cuda_operator.load_state_dict(cpu_operator.state_dict(), strict=True)
    tiny_input = torch.randn(1, 1, 8, 6, 5)
    cpu_output = cpu_operator(tiny_input)
    cuda_output = cuda_operator(tiny_input.cuda()).cpu()
    agreement_relative = float(
        (torch.linalg.vector_norm(cpu_output - cuda_output) / torch.linalg.vector_norm(cpu_output)).detach()
    )
    agreement_max = float(torch.max(torch.abs(cpu_output - cuda_output)).detach())
    agreement_pass = bool(torch.allclose(cpu_output, cuda_output, rtol=2e-5, atol=2e-5))

    peak_allocated = torch.cuda.max_memory_allocated(device) / 2**20
    peak_reserved = torch.cuda.max_memory_reserved(device) / 2**20
    feasibility = {
        "status": "passed" if finite and gradient_active and agreement_pass else "failed",
        "device": device_name, "capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": str(torch.__version__), "torch_cuda_version": torch.version.cuda,
        "model_parameters_on_cuda": True, "input_on_cuda": batch["model_input"].device.type == "cuda",
        "output_on_cuda": predicted_residual.device.type == "cuda", "loss_on_cuda": loss.device.type == "cuda",
        "forward_finite": bool(torch.isfinite(predicted_residual).all()),
        "loss_finite": bool(torch.isfinite(loss)), "gradients_finite": finite,
        "forward_seconds": forward_seconds, "backward_seconds": backward_seconds,
        "disco_forward_seconds": disco_forward_seconds,
        "peak_allocated_mib": peak_allocated, "peak_reserved_mib": peak_reserved,
        "cpu_cuda_agreement": {"relative_l2": agreement_relative, "max_abs": agreement_max, "rtol": 2e-5, "atol": 2e-5, "pass": agreement_pass},
        "batch": 1, "input_shape": list(batch["model_input"].shape),
        "optimizer_created": False, "parameter_update_performed": False,
    }
    write_json(OUT / "preflight/cuda_feasibility.json", feasibility)
    model.zero_grad(set_to_none=True)

    ad_preflight = json.loads((ROOT / "artifacts/stage_ad/preflight/cuda_preflight.json").read_text(encoding="utf-8"))
    ad_training = json.loads((ROOT / "artifacts/stage_ad/training/disco3d_localno/training_summary.json").read_text(encoding="utf-8"))
    af_microbatch = forward_seconds + backward_seconds
    ad_microbatch = ad_preflight["forward_seconds"] + ad_preflight["backward_seconds"]
    runtime_ratio = af_microbatch / ad_microbatch
    memory_ratio = peak_allocated / ad_preflight["peak_allocated_mib"]
    estimate = {
        "method": "Stage AD measured 300-epoch runtime scaled by AF/AD no-update preflight microbatch ratio",
        "stage_ad_actual_300_epoch_seconds": ad_training["runtime_seconds"],
        "stage_ad_preflight_microbatch_seconds": ad_microbatch,
        "stage_af_preflight_microbatch_seconds": af_microbatch,
        "af_over_ad_runtime_ratio": runtime_ratio,
        "af_over_ad_peak_allocated_memory_ratio": memory_ratio,
        "estimated_300_epoch_seconds": ad_training["runtime_seconds"] * runtime_ratio,
        "estimated_300_epoch_hours": ad_training["runtime_seconds"] * runtime_ratio / 3600,
        "estimate_only": True, "training_run": False,
    }
    write_json(OUT / "preflight/estimated_training_cost.json", estimate)
    write_csv(OUT / "comparison/computational_cost.csv", [
        {"stage": "AD", "backend": "DENSE_CONV3D", "forward_seconds": ad_preflight["forward_seconds"], "backward_seconds": ad_preflight["backward_seconds"], "disco_forward_seconds": ad_preflight["disco_forward_seconds"], "peak_allocated_mib": ad_preflight["peak_allocated_mib"], "peak_reserved_mib": ad_preflight["peak_reserved_mib"], "source": "frozen_stage_ad_preflight"},
        {"stage": "AE", "backend": "SHIFT_WINDOW_EINSUM", "forward_seconds": "not_available", "backward_seconds": "not_available", "disco_forward_seconds": "not_available", "peak_allocated_mib": "not_available", "peak_reserved_mib": "not_available", "source": "production_normalization_gate_failed"},
        {"stage": "AF", "backend": "OFFSET_VECTORIZED_ACCUMULATION", "forward_seconds": forward_seconds, "backward_seconds": backward_seconds, "disco_forward_seconds": disco_forward_seconds, "peak_allocated_mib": peak_allocated, "peak_reserved_mib": peak_reserved, "source": "no_update_cuda_preflight"},
    ])
    data.close()
    print(json.dumps({"identity": identity, "feasibility": feasibility, "estimate": estimate}, indent=2, sort_keys=True))
    if feasibility["status"] != "passed":
        raise RuntimeError("Stage AF CUDA feasibility gate failed")


if __name__ == "__main__":
    main()
