#!/usr/bin/env python
"""Run the mandatory real-64^3 Stage R residual-contract CUDA preflight."""

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
from grmhd.paper_stage_g import tensor_state_sha256
from grmhd.paper_stage_o import EXPECTED_INITIAL_TENSOR_STATE_SHA256
from grmhd.paper_stage_r import validate_contract_config
from grmhd.paper_trainer import PaperTrainerLossAdapter
from grmhd.upstream_adapters import GRMHDNextStepDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/paper_reduced100/stage_r_localno_p3_residual_plain.yaml"),
    )
    parser.add_argument(
        "--initial-state",
        type=Path,
        default=Path("outputs/paper_reduced100/stage_k/shared_localno_initial_state.pt"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper_reduced100/stage_r"),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    initial_path = args.initial_state if args.initial_state.is_absolute() else root / args.initial_state
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing Stage R CPU fallback")
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(device)
    if "RTX 5070" not in device_name:
        raise RuntimeError(f"Stage R expected RTX 5070, found {device_name!r}")
    config = load_paper_experiment_config(config_path, project_root=root)
    validate_contract_config(config.values)
    if not config.stage_r:
        raise ValueError("Stage R preflight requires normalized_residual mode")
    protocol = PaperReduced100Protocol.from_yaml(
        root / "configs/data/paper_reduced100.yaml", project_root=root
    )
    dataset = GRMHDNextStepDataset(protocol.make_datasets()["train"])
    batch = next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))
    processor = PaperDataProcessor.from_config(config).to(device)
    processor.set_epoch(0)
    model = build_paper_model(config)
    state = torch.load(initial_path, map_location="cpu", weights_only=True)
    state_hash = tensor_state_sha256(state)
    if state_hash != EXPECTED_INITIAL_TENSOR_STATE_SHA256:
        raise ValueError("Stage K initial tensor state changed before Stage R preflight")
    model.load_state_dict(state, strict=True)
    model = model.to(device).train()
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
    state_before = model_tensor_state_sha256(model)
    if state_before != state_hash:
        raise ValueError("Stage R CUDA model differs from the shared initial state")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    sample = processor.preprocess(batch)
    torch.cuda.synchronize(device)
    forward_start = time.perf_counter()
    predicted_residual = model(x=sample["x"])
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - forward_start
    loss_start = time.perf_counter()
    reconstructed, fields = processor.postprocess(predicted_residual, sample)
    loss = loss_adapter(reconstructed, **fields)
    torch.cuda.synchronize(device)
    loss_seconds = time.perf_counter() - loss_start
    backward_start = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(device)
    backward_seconds = time.perf_counter() - backward_start
    decoded = processor.preprocessor.decode(reconstructed.detach(), channel_axis=1)
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    state_after = model_tensor_state_sha256(model)
    equivalence = loss_adapter.last_log["loss_equivalence"] if loss_adapter.last_log else None
    result = {
        "schema_version": "paper-stage-r-residual-localno-gpu-preflight-v1",
        "status": "passed",
        "classification": "adapted_residual_contract_model_pilot",
        "prediction_mode": "normalized_residual",
        "identity_state_skip": True,
        "residual_scale": 1.0,
        "clipping": False,
        "initial_state_tensor_sha256": state_hash,
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
        "model_parameters_on_cuda": all(
            parameter.device.type == "cuda" for parameter in model.parameters()
        ),
        "input_shape": list(sample["x"].shape),
        "target_residual_shape": list(fields["normalized_residual_target"].shape),
        "predicted_residual_shape": list(predicted_residual.shape),
        "reconstructed_state_shape": list(reconstructed.shape),
        "input_on_cuda": sample["x"].device.type == "cuda",
        "residual_output_on_cuda": predicted_residual.device.type == "cuda",
        "reconstructed_state_on_cuda": reconstructed.device.type == "cuda",
        "loss_on_cuda": loss.device.type == "cuda",
        "input_finite": bool(torch.isfinite(sample["x"]).all()),
        "residual_finite": bool(torch.isfinite(predicted_residual).all()),
        "reconstructed_state_finite": bool(torch.isfinite(reconstructed).all()),
        "loss": float(loss.detach().cpu()),
        "loss_finite": bool(torch.isfinite(loss)),
        "loss_equivalence": equivalence,
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
        "decoded_prediction_finite": bool(torch.isfinite(decoded).all()),
        "decoded_rho_press_positive": bool(torch.all(decoded[:, 3:5] > 0)),
        "state_hash_before_backward": state_before,
        "state_hash_after_backward": state_after,
        "state_hash_unchanged": state_before == state_after,
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
        "RTX 5070" in str(result["device_name"]),
        result["model_parameters_on_cuda"],
        result["input_shape"] == [1, 16, 64, 64, 64],
        result["predicted_residual_shape"] == [1, 8, 64, 64, 64],
        result["reconstructed_state_shape"] == [1, 8, 64, 64, 64],
        result["input_on_cuda"],
        result["residual_output_on_cuda"],
        result["reconstructed_state_on_cuda"],
        result["loss_on_cuda"],
        result["input_finite"],
        result["residual_finite"],
        result["reconstructed_state_finite"],
        result["loss_finite"],
        bool(equivalence and equivalence["passed"]),
        result["gradients_present_for_all_trainable"],
        result["all_trainable_gradients_finite"],
        result["decoded_prediction_finite"],
        result["decoded_rho_press_positive"],
        result["state_hash_unchanged"],
        result["disco_module_count"] == 0,
        result["differential_module_count"] > 0,
    )
    if not all(required):
        raise RuntimeError(f"Stage R GPU preflight failed: {result}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "preflight.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "preflight.md").write_text(
        "\n".join(
            [
                "# Stage R residual LocalNO GPU preflight",
                "",
                f"- Status: `{result['status']}`.",
                f"- GPU: `{result['device_name']}`; capability `{result['capability']}`.",
                f"- Input/residual/state: `{result['input_shape']}` -> "
                f"`{result['predicted_residual_shape']}` -> `{result['reconstructed_state_shape']}`.",
                f"- Parameters: `{result['parameter_count']}`; differential/DISCO: `{differential_count}/{disco_count}`.",
                f"- Forward/backward: `{forward_seconds:.6f}/{backward_seconds:.6f}` s.",
                f"- Peak allocated/reserved: `{result['peak_allocated_mib']:.2f}/{result['peak_reserved_mib']:.2f}` MiB.",
                "- Residual/state Plain L2 equivalence, finite gradients/decode, and positivity: passed.",
                "- Optimizer, scheduler, optimizer step, and checkpoint write: absent.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
