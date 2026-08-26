#!/usr/bin/env python
"""Run the mandatory real-64^3 CUDA preflight for Stage K LocalNO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from neuralop.layers.differential_conv import FiniteDifferenceConvolution
from neuralop.layers.discrete_continuous_convolution import (
    EquidistantDiscreteContinuousConv2d,
)
from neuralop.layers.spectral_convolution import SpectralConv

from grmhd.paper_config import (
    build_paper_model,
    build_paper_training_loss,
    load_paper_experiment_config,
    model_tensor_state_sha256,
)
from grmhd.paper_data_processor import PaperDataProcessor
from grmhd.paper_protocol import PaperReduced100Protocol
from grmhd.paper_trainer import PaperTrainerLossAdapter
from grmhd.upstream_adapters import GRMHDNextStepDataset


def _markdown(result: dict[str, object]) -> str:
    return "\n".join(
        [
            "# Stage K 3D differential LocalNO GPU preflight",
            "",
            f"- Status: `{result['status']}`",
            "- Classification: `adapted_method_reproduction`",
            "- Backbone: `3D differential LocalNO`",
            "- DISCO integral: `disabled`",
            "- Exact paper backbone: `false`",
            f"- GPU: `{result['device_name']}`",
            f"- Input shape: `{result['input_shape']}`",
            f"- Output shape: `{result['output_shape']}`",
            f"- Parameters: `{result['parameter_count']}`",
            f"- Differential/spectral/DISCO modules: "
            f"`{result['differential_module_count']}/"
            f"{result['spectral_module_count']}/{result['disco_module_count']}`",
            f"- Forward/loss/backward finite: "
            f"`{result['forward_finite']}/{result['loss_finite']}/"
            f"{result['all_trainable_gradients_finite']}`",
            f"- Forward/backward seconds: "
            f"`{result['forward_seconds']:.6f}/{result['backward_seconds']:.6f}`",
            f"- Peak allocated/reserved MiB: "
            f"`{result['peak_allocated_mib']:.2f}/{result['peak_reserved_mib']:.2f}`",
            f"- Tensor state unchanged by backward: `{result['state_hash_unchanged']}`",
            "- Optimizer/scheduler/checkpoint created: `false/false/false`",
            "",
            "This preflight uses one real reduced100 training pair and the frozen "
            "PaperDataProcessor/Plain-L2 contract. It is not a training run.",
        ]
    ) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/paper_reduced100/stage_k_localno_differential_plain.yaml"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper_reduced100/stage_k"),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing Stage K CPU fallback")
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(device)
    if "RTX 5070" not in device_name:
        raise RuntimeError(f"Stage K expected RTX 5070, found {device_name!r}")

    config = load_paper_experiment_config(config_path, project_root=root)
    if not config.stage_k:
        raise ValueError("Stage K preflight requires the differential LocalNO config")
    protocol = PaperReduced100Protocol.from_yaml(
        root / "configs/data/paper_reduced100.yaml", project_root=root
    )
    train_dataset = GRMHDNextStepDataset(protocol.make_datasets()["train"])
    batch = next(
        iter(DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0))
    )

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    processor = PaperDataProcessor.from_config(config).to(device)
    processor.set_epoch(0)
    model = build_paper_model(config).to(device).train()
    loss_adapter = PaperTrainerLossAdapter(build_paper_training_loss(config)).to(device)
    loss_adapter.set_epoch(0)
    model.zero_grad(set_to_none=True)

    differential_count = sum(
        isinstance(module, FiniteDifferenceConvolution) for module in model.modules()
    )
    spectral_count = sum(isinstance(module, SpectralConv) for module in model.modules())
    disco_count = sum(
        isinstance(module, EquidistantDiscreteContinuousConv2d)
        for module in model.modules()
    )
    conv2d_count = sum(isinstance(module, nn.Conv2d) for module in model.modules())
    conv3d_count = sum(isinstance(module, nn.Conv3d) for module in model.modules())
    if differential_count <= 0 or spectral_count <= 0:
        raise RuntimeError("Stage K differential or spectral branch was not constructed")
    if disco_count != 0 or conv2d_count != 0:
        raise RuntimeError("Stage K unexpectedly constructed a 2D/DISCO module")

    state_hash_before = model_tensor_state_sha256(model)
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
    state_hash_after = model_tensor_state_sha256(model)

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradients = [parameter.grad for parameter in trainable]
    result: dict[str, object] = {
        "schema_version": "paper-stage-k-localno-gpu-preflight-v1",
        "status": "passed",
        "classification": "adapted_method_reproduction",
        "exact_paper_backbone": False,
        "architecture": "localno_differential_3d",
        "n_dim": 3,
        "differential_enabled": True,
        "spectral_enabled": True,
        "disco_integral_enabled": False,
        "nvidia_smi": subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, check=True
        ).stdout,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": device_name,
        "capability": list(torch.cuda.get_device_capability(device)),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "differential_module_count": differential_count,
        "spectral_module_count": spectral_count,
        "disco_module_count": disco_count,
        "conv2d_module_count": conv2d_count,
        "conv3d_module_count": conv3d_count,
        "model_parameters_on_cuda": all(
            parameter.device.type == "cuda" for parameter in model.parameters()
        ),
        "input_shape": list(sample["x"].shape),
        "target_shape": list(sample["y"].shape),
        "input_on_cuda": sample["x"].device.type == "cuda",
        "output_shape": list(prediction.shape),
        "output_on_cuda": prediction.device.type == "cuda",
        "loss_on_cuda": loss.device.type == "cuda",
        "forward_finite": bool(torch.isfinite(prediction).all()),
        "loss": float(loss.detach().cpu()),
        "loss_finite": bool(torch.isfinite(loss)),
        "trainable_parameter_tensor_count": len(trainable),
        "gradients_present_for_all_trainable": bool(
            gradients and all(gradient is not None for gradient in gradients)
        ),
        "all_trainable_gradients_finite": bool(
            gradients
            and all(
                gradient is not None and torch.isfinite(gradient).all()
                for gradient in gradients
            )
        ),
        "state_hash_before_backward": state_hash_before,
        "state_hash_after_backward": state_hash_after,
        "state_hash_unchanged": state_hash_before == state_hash_after,
        "forward_seconds": forward_seconds,
        "loss_seconds": loss_seconds,
        "backward_seconds": backward_seconds,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
        "optimizer_created": False,
        "optimizer_step_executed": False,
        "scheduler_created": False,
        "checkpoint_written": False,
    }
    required = (
        result["cuda_available"],
        result["device_name"] == "NVIDIA GeForce RTX 5070",
        result["model_parameters_on_cuda"],
        result["input_shape"] == [1, 16, 64, 64, 64],
        result["output_shape"] == [1, 8, 64, 64, 64],
        result["input_on_cuda"],
        result["output_on_cuda"],
        result["loss_on_cuda"],
        result["forward_finite"],
        result["loss_finite"],
        result["gradients_present_for_all_trainable"],
        result["all_trainable_gradients_finite"],
        result["state_hash_unchanged"],
        result["disco_module_count"] == 0,
        result["conv2d_module_count"] == 0,
    )
    if not all(required):
        raise RuntimeError(f"Stage K GPU preflight failed: {result}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "localno_preflight.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "localno_preflight.md").write_text(
        _markdown(result), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
