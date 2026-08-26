#!/usr/bin/env python
"""Run the mandatory CUDA and real-batch Full-loss preflight for Stage G."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import torch
from torch.utils.data import DataLoader

from grmhd.paper_config import (
    build_paper_model,
    build_paper_training_loss,
    load_paper_experiment_config,
)
from grmhd.paper_data_processor import PaperDataProcessor
from grmhd.paper_protocol import PaperReduced100Protocol
from grmhd.paper_trainer import PaperTrainerLossAdapter
from grmhd.upstream_adapters import GRMHDNextStepDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/paper_reduced100/full_fno_proxy.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/paper_reduced100/stage_g/gpu_preflight.json"),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    output_path = args.output if args.output.is_absolute() else root / args.output
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing Stage G CPU fallback")
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(device)
    if "RTX 5070" not in device_name:
        raise RuntimeError(f"Stage G expected RTX 5070, found {device_name!r}")

    config = load_paper_experiment_config(config_path, project_root=root)
    protocol = PaperReduced100Protocol.from_yaml(
        root / "configs/data/paper_reduced100.yaml", project_root=root
    )
    train_dataset = GRMHDNextStepDataset(protocol.make_datasets()["train"])
    batch = next(iter(DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)))

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    processor = PaperDataProcessor.from_config(config).to(device)
    processor.set_epoch(0)
    model = build_paper_model(config).to(device).train()
    loss_adapter = PaperTrainerLossAdapter(build_paper_training_loss(config)).to(device)
    loss_adapter.set_epoch(0)
    model.zero_grad(set_to_none=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    sample = processor.preprocess(batch)
    torch.cuda.synchronize(device)

    forward_started = time.perf_counter()
    prediction = model(x=sample["x"])
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - forward_started

    loss_started = time.perf_counter()
    prediction, loss_fields = processor.postprocess(prediction, sample)
    loss = loss_adapter(prediction, **loss_fields)
    torch.cuda.synchronize(device)
    loss_seconds = time.perf_counter() - loss_started

    backward_started = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(device)
    backward_seconds = time.perf_counter() - backward_started
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    nvidia_smi = subprocess.run(
        ["nvidia-smi"], capture_output=True, text=True, check=True
    ).stdout
    result = {
        "schema_version": "paper-stage-g-gpu-preflight-v1",
        "status": "passed",
        "nvidia_smi": nvidia_smi,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": device_name,
        "capability": list(torch.cuda.get_device_capability(device)),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "model_parameters_on_cuda": all(
            parameter.device.type == "cuda" for parameter in model.parameters()
        ),
        "input_shape": list(sample["x"].shape),
        "input_on_cuda": sample["x"].device.type == "cuda",
        "output_shape": list(prediction.shape),
        "output_on_cuda": prediction.device.type == "cuda",
        "loss_on_cuda": loss.device.type == "cuda",
        "forward_finite": bool(torch.isfinite(prediction).all()),
        "loss": float(loss.detach().cpu()),
        "loss_finite": bool(torch.isfinite(loss)),
        "gradients_present": bool(gradients),
        "gradients_finite": bool(
            gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        ),
        "forward_seconds": forward_seconds,
        "loss_seconds": loss_seconds,
        "backward_seconds": backward_seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "optimizer_step_executed": False,
    }
    required = (
        result["cuda_available"],
        result["model_parameters_on_cuda"],
        result["input_on_cuda"],
        result["output_on_cuda"],
        result["loss_on_cuda"],
        result["forward_finite"],
        result["loss_finite"],
        result["gradients_present"],
        result["gradients_finite"],
    )
    if not all(required):
        raise RuntimeError(f"Stage G GPU preflight failed: {result}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
