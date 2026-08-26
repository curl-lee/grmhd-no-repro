#!/usr/bin/env python
"""Run and record random/real forward-backward compatibility checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import traceback

import torch

from grmhd.dataset import make_temporal_datasets
from grmhd.models import build_model, parameters_without_grad, trainable_parameter_count
from grmhd.normalizer import GRMHDNormalizer
from grmhd.shells import radial_shells_tensor


def downsample(state: torch.Tensor, factor: int) -> torch.Tensor:
    return state[..., ::factor, ::factor, ::factor] if factor > 1 else state


def check_one(
    model_type: str,
    input_tensor: torch.Tensor,
    shape: tuple[int, int, int],
    device: torch.device,
    n_modes: tuple[int, int, int],
    hidden_channels: int,
    n_layers: int,
) -> dict[str, object]:
    result: dict[str, object] = {"model_type": model_type, "input_shape": list(input_tensor.shape)}
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        model = build_model(
            model_type,
            in_channels=input_tensor.shape[1],
            out_channels=8,
            default_in_shape=shape,
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            n_layers=n_layers,
            positional_embedding=None,
            **({"conv_padding_mode": "zeros"} if model_type.startswith("localno") else {}),
        ).to(device)
        x = input_tensor.detach().clone().to(device)
        prediction = model(x)
        loss = torch.mean(prediction.square())
        has_trainable_parameters = any(parameter.requires_grad for parameter in model.parameters())
        if has_trainable_parameters:
            loss.backward()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        missing_grad = parameters_without_grad(model)
        result.update(
            {
                "status": "passed" if not missing_grad else "failed_unused_parameters",
                "output_shape": list(prediction.shape),
                "output_finite": bool(torch.isfinite(prediction).all().item()),
                "loss": float(loss.detach().cpu()),
                "trainable_parameters": trainable_parameter_count(model),
                "parameters_without_grad": missing_grad,
                "backward_applicable": has_trainable_parameters,
                "elapsed_seconds": elapsed,
                "peak_cuda_memory_mib": (
                    torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None
                ),
            }
        )
        if prediction.shape != (x.shape[0], 8, *shape):
            result["status"] = "failed_shape"
        if not result["output_finite"]:
            result["status"] = "failed_nonfinite"
        del model, x, prediction, loss
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("outputs/model_compatibility.json"))
    parser.add_argument("--resolution", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hidden_channels", type=int, default=16)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_modes", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    datasets = make_temporal_datasets(args.data)
    normalizer = GRMHDNormalizer.fit(datasets["train"], max_samples_per_channel=50_000)
    raw = datasets["train"][0]["x"].unsqueeze(0)
    source_resolution = raw.shape[-1]
    if source_resolution % args.resolution:
        raise ValueError("Requested resolution must evenly divide source resolution")
    factor = source_resolution // args.resolution
    physical = downsample(normalizer.encode(raw, channel_axis=1), factor)
    r = datasets["train"].coords["r"][::factor]
    shells, shell_metadata = radial_shells_tensor(r, args.resolution, args.resolution)
    actual = torch.cat([physical, shells.unsqueeze(0)], dim=1)
    random = torch.randn_like(actual)
    shape = (args.resolution,) * 3
    modes = (min(args.n_modes, args.resolution // 2),) * 3

    results = {
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "resolution": args.resolution,
        "n_modes": modes,
        "shell_metadata": shell_metadata.as_dict(),
        "checks": [],
    }
    for source_name, tensor in (("random", random), ("real_batch", actual)):
        for model_type in ("persistence", "fno", "localno_diff"):
            check = check_one(
                model_type,
                tensor,
                shape,
                device,
                modes,
                args.hidden_channels,
                args.n_layers,
            )
            check["input_source"] = source_name
            results["checks"].append(check)
            print(json.dumps(check, indent=2), flush=True)
    # DISCO is attempted once with the exact 3D construction and full traceback.
    disco = check_one(
        "localno_disco", random, shape, device, modes, args.hidden_channels, args.n_layers
    )
    disco["input_source"] = "random"
    results["checks"].append(disco)
    print(json.dumps(disco, indent=2), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
