#!/usr/bin/env python
"""Run the mandatory three-way RTX 5070 Stage I preflight without training."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import subprocess
import time
from typing import Any, Iterable

import torch
from torch.utils.data import DataLoader

from grmhd import CHANNELS
from grmhd.paper_config import (
    build_paper_model,
    build_paper_training_loss,
    load_paper_experiment_config,
    model_tensor_state_sha256,
)
from grmhd.paper_data_processor import PaperDataProcessor
from grmhd.paper_h1_diagnostics import (
    flatten_optional_gradients,
    gradient_pair_metrics,
    spherical_proxy_volume_weights,
)
from grmhd.paper_h1_extensions import DiagnosticPaperCompositeLoss
from grmhd.paper_protocol import PaperReduced100Protocol
from grmhd.paper_stage_g import tensor_state_sha256
from grmhd.upstream_adapters import GRMHDNextStepDataset


EXPECTED_SHARED_HASH = (
    "02d83278d5e74a27c83130e99b488ca18eae5477dad9732a4552b694056b6cb1"
)
EXPECTED_DEVICE = "NVIDIA GeForce RTX 5070"
EXPECTED_CAPABILITY = (12, 0)
NON_H1_COMPONENTS = (
    "base_fidelity_raw",
    "base_fidelity_weighted",
    "roi_raw",
    "roi_ramp",
    "roi_weighted",
    "bounds_rho_raw",
    "bounds_press_raw",
    "bounds_weighted",
    "envelope_rho_raw",
    "envelope_press_raw",
    "envelope_weighted",
    "dissipation_raw",
    "dissipation_weighted",
)


def tensor_sha256(value: torch.Tensor) -> str:
    return tensor_state_sha256({"tensor": value})


def optional_gradient_vector(
    component: torch.Tensor,
    parameters: tuple[torch.nn.Parameter, ...],
) -> torch.Tensor:
    gradients = torch.autograd.grad(
        component,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    return flatten_optional_gradients(gradients, parameters)


def prediction_gradient(
    component: torch.Tensor,
    prediction: torch.Tensor,
) -> torch.Tensor:
    gradient = torch.autograd.grad(
        component,
        prediction,
        retain_graph=True,
        allow_unused=True,
    )[0]
    return torch.zeros_like(prediction) if gradient is None else gradient


def finite_scalar(value: torch.Tensor) -> bool:
    return value.ndim == 0 and bool(torch.isfinite(value))


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def all_finite(values: Iterable[torch.Tensor]) -> bool:
    return all(bool(torch.isfinite(value).all()) for value in values)


def run_variant(
    *,
    name: str,
    config_path: Path,
    root: Path,
    device: torch.device,
    batch: dict[str, Any],
    shared_state: dict[str, torch.Tensor],
    reference_loss: torch.nn.Module,
    reference_input_hash: str | None,
    reference_output_hash: str | None,
) -> tuple[dict[str, Any], str, str]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    config = load_paper_experiment_config(config_path, project_root=root)
    processor = PaperDataProcessor.from_config(config).to(device)
    processor.set_epoch(0)
    model = build_paper_model(config).to(device).train()
    model.load_state_dict(shared_state, strict=True)
    if model_tensor_state_sha256(model) != EXPECTED_SHARED_HASH:
        raise RuntimeError(f"{name}: strict shared-state hash mismatch")
    loss_module = build_paper_training_loss(config).to(device).eval()
    if not isinstance(loss_module, DiagnosticPaperCompositeLoss):
        raise TypeError(f"{name}: extension config did not build diagnostic loss")
    if loss_module.diagnostic_h1_mode.value != name:
        raise ValueError(f"{name}: configured H1 mode differs from preflight order")

    model.zero_grad(set_to_none=True)
    sample = processor.preprocess(batch)
    input_hash = tensor_sha256(sample["x"])
    if reference_input_hash is not None and input_hash != reference_input_hash:
        raise RuntimeError(f"{name}: model input differs from the first extension")
    before_state_hash = model_tensor_state_sha256(model)
    torch.cuda.synchronize(device)
    started = time.perf_counter()

    forward_started = time.perf_counter()
    prediction = model(x=sample["x"])
    torch.cuda.synchronize(device)
    forward_seconds = time.perf_counter() - forward_started
    prediction, loss_fields = processor.postprocess(prediction, sample)
    output_hash = tensor_sha256(prediction)
    if reference_output_hash is not None and output_hash != reference_output_hash:
        raise RuntimeError(f"{name}: epoch-zero model output differs")

    loss_started = time.perf_counter()
    result = loss_module.components(
        prediction,
        context=loss_fields["context"],
    )
    torch.cuda.synchronize(device)
    loss_seconds = time.perf_counter() - loss_started

    with torch.no_grad():
        reference = reference_loss.components(
            prediction.detach(),
            context=loss_fields["context"],
        )
    non_h1_parity = {
        component: torch.equal(
            getattr(result, component),
            getattr(reference, component),
        )
        for component in NON_H1_COMPONENTS
    }

    parameters = tuple(
        parameter for parameter in model.parameters() if parameter.requires_grad
    )
    other = (
        result.roi_weighted
        + result.bounds_weighted
        + result.envelope_weighted
        + result.dissipation_weighted
    )
    gradient_started = time.perf_counter()
    base_prediction_gradient = prediction_gradient(
        result.base_fidelity_weighted,
        prediction,
    )
    h1_prediction_gradient = prediction_gradient(
        result.h1_weighted,
        prediction,
    )
    base_parameter_gradient = optional_gradient_vector(
        result.base_fidelity_weighted,
        parameters,
    )
    h1_parameter_gradient = optional_gradient_vector(
        result.h1_weighted,
        parameters,
    )
    other_parameter_gradient = optional_gradient_vector(other, parameters)
    torch.cuda.synchronize(device)
    component_gradient_seconds = time.perf_counter() - gradient_started

    backward_started = time.perf_counter()
    result.total.backward()
    torch.cuda.synchronize(device)
    backward_seconds = time.perf_counter() - backward_started
    parameter_gradients = tuple(
        parameter.grad
        for parameter in parameters
        if parameter.grad is not None
    )
    total_parameter_gradient = flatten_optional_gradients(
        tuple(parameter.grad for parameter in parameters),
        parameters,
    )
    total_gradient_norm = torch.linalg.vector_norm(total_parameter_gradient)
    clip_scale = torch.clamp(
        total_gradient_norm.new_tensor(1.0)
        / total_gradient_norm.clamp_min(1.0e-30),
        max=1.0,
    )
    gradient_sum = (
        base_parameter_gradient
        + h1_parameter_gradient
        + other_parameter_gradient
    )
    gradient_sum_relative_residual = torch.linalg.vector_norm(
        gradient_sum - total_parameter_gradient
    ) / total_gradient_norm.clamp_min(1.0e-30)
    pair_metrics = gradient_pair_metrics(
        base_parameter_gradient,
        h1_parameter_gradient,
    )
    current_upstream_h1 = result.diagnostics[
        "diagnostic_current_upstream_h1_raw"
    ]
    if not torch.is_tensor(current_upstream_h1):
        raise TypeError(f"{name}: current upstream H1 diagnostic is not a tensor")
    unit_index_h1 = result.diagnostics["diagnostic_unit_index_h1_raw"]
    if not torch.is_tensor(unit_index_h1):
        raise TypeError(f"{name}: unit-index H1 diagnostic is not a tensor")
    if current_upstream_h1.requires_grad or unit_index_h1.requires_grad:
        raise RuntimeError(f"{name}: detached H1 diagnostic requires gradients")
    selected = loss_module.h1.last_result
    if selected is None:
        raise RuntimeError(f"{name}: selected H1 result was not retained")

    selected_raw = result.h1_raw
    selected_weighted = result.h1_weighted
    base_value = result.base_fidelity_weighted
    h1_base_value_ratio = selected_weighted / base_value.clamp_min(1.0e-30)
    base_prediction_norm = torch.linalg.vector_norm(base_prediction_gradient)
    h1_prediction_norm = torch.linalg.vector_norm(h1_prediction_gradient)
    base_parameter_norm = torch.linalg.vector_norm(base_parameter_gradient)
    h1_parameter_norm = torch.linalg.vector_norm(h1_parameter_gradient)
    total_seconds = time.perf_counter() - started
    after_state_hash = model_tensor_state_sha256(model)

    mode_checks: dict[str, Any] = {}
    if name == "no_h1":
        mode_checks = {
            "selected_h1_raw_exact_zero": scalar(selected_raw) == 0.0,
            "selected_h1_weighted_exact_zero": scalar(selected_weighted) == 0.0,
            "selected_h1_prediction_gradient_exact_zero": int(
                torch.count_nonzero(h1_prediction_gradient)
            )
            == 0,
            "selected_h1_parameter_gradient_exact_zero": int(
                torch.count_nonzero(h1_parameter_gradient)
            )
            == 0,
        }
    elif name == "unit_index":
        ratio = current_upstream_h1 / selected_raw.clamp_min(1.0e-30)
        mode_checks = {
            "current_to_unit_index_ratio": scalar(ratio),
            "stage_h_4096_parity": bool(
                torch.isclose(
                    ratio,
                    ratio.new_tensor(4096.0),
                    rtol=2.0e-5,
                    atol=2.0e-3,
                )
            ),
            "uniform_reduction": selected.metadata["reduction"]
            == "uniform_voxel_mean",
        }
    else:
        volume_weights = spherical_proxy_volume_weights(
            loss_module.h1.coordinates
        )
        coordinate_differences = {
            axis: torch.diff(coordinate)
            for axis, coordinate in loss_module.h1.coordinates.items()
        }
        radial_differences = torch.diff(loss_module.h1.coordinate_r)
        with torch.no_grad():
            duplicated_selected = loss_module.h1.components(
                torch.cat((prediction.detach(), prediction.detach()), dim=0),
                torch.cat(
                    (
                        loss_fields["context"].normalized_target.detach(),
                        loss_fields["context"].normalized_target.detach(),
                    ),
                    dim=0,
                ),
            ).raw_h1
        mode_checks = {
            "volume_weights_finite": bool(torch.isfinite(volume_weights).all()),
            "volume_weights_nonnegative": bool(torch.all(volume_weights >= 0)),
            "volume_weights_normalized": bool(
                torch.isclose(
                    volume_weights.sum(),
                    volume_weights.new_tensor(1.0),
                    rtol=0.0,
                    atol=1.0e-12,
                )
            ),
            "volume_weights_shape_excludes_batch": list(volume_weights.shape)
            == [64, 64, 64],
            "batch_reduction_independent": bool(
                torch.isclose(
                    duplicated_selected,
                    selected_raw.detach(),
                    rtol=2.0e-6,
                    atol=1.0e-7,
                )
            ),
            "all_coordinate_spacings_positive": all(
                bool(torch.all(values > 0))
                for values in coordinate_differences.values()
            ),
            "phi_spacing_uniform": bool(
                torch.allclose(
                    coordinate_differences["phi"],
                    coordinate_differences["phi"][0].expand_as(
                        coordinate_differences["phi"]
                    ),
                    rtol=1.0e-5,
                    atol=1.0e-6,
                )
            ),
            "radial_spacing_positive": bool(torch.all(radial_differences > 0)),
            "radial_spacing_nonuniform": not bool(
                torch.allclose(
                    radial_differences,
                    radial_differences[0].expand_as(radial_differences),
                )
            ),
            "diagnostic_proxy_only": selected.metadata["diagnostic_proxy_only"],
            "covariant_GRMHD_H1": selected.metadata["covariant_GRMHD_H1"],
            "proper_Kerr_Schild_volume": selected.metadata[
                "proper_Kerr_Schild_volume"
            ],
            "stored_components_covariant_derivative": selected.metadata[
                "stored_components_covariant_derivative"
            ],
            "stored_vector_covariant_derivative": selected.metadata[
                "stored_vector_covariant_derivative"
            ],
            "pole_sin_floor": selected.metadata["pole_sin_floor"],
            "pole_handling": selected.metadata["pole_handling"],
            "volume_weight_normalization": selected.metadata[
                "volume_weight_normalization"
            ],
        }

    scalar_components = {
        "total": result.total,
        "base": result.base_fidelity_weighted,
        "selected_h1_raw": selected_raw,
        "selected_h1_weighted": selected_weighted,
        "diagnostic_current_upstream_h1_raw": current_upstream_h1,
        "diagnostic_unit_index_h1_raw": unit_index_h1,
        "roi": result.roi_weighted,
        "bounds": result.bounds_weighted,
        "envelope": result.envelope_weighted,
        "dissipation": result.dissipation_weighted,
    }
    record = {
        "mode": name,
        "config": str(config_path.relative_to(root)),
        "status": "pending",
        "device": {
            "model": "cuda"
            if all(parameter.device.type == "cuda" for parameter in parameters)
            else "mixed",
            "input": sample["x"].device.type,
            "output": prediction.device.type,
            "loss": result.total.device.type,
        },
        "shape": {
            "input": list(sample["x"].shape),
            "output": list(prediction.shape),
        },
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "shared_state_hash_before": before_state_hash,
        "shared_state_hash_after": after_state_hash,
        "model_state_unchanged": before_state_hash == after_state_hash,
        "input_sha256": input_hash,
        "output_sha256": output_hash,
        "forward_finite": bool(torch.isfinite(prediction).all()),
        "loss_components_finite": all(
            finite_scalar(value) for value in scalar_components.values()
        ),
        "parameter_gradients_present_for_all": len(parameter_gradients)
        == len(parameters),
        "parameter_gradients_finite": len(parameter_gradients) == len(parameters)
        and all_finite(parameter_gradients),
        "non_h1_component_parity": non_h1_parity,
        "loss": {key: scalar(value) for key, value in scalar_components.items()},
        "selected_h1_decomposition": {
            "per_channel": {
                channel: scalar(selected.per_channel[index])
                for index, channel in enumerate(CHANNELS)
            },
            "per_direction": {
                direction: scalar(value)
                for direction, value in selected.per_direction.items()
            },
            "per_shell": [scalar(value) for value in selected.per_shell],
        },
        "ratios": {
            "selected_h1_to_base_value": scalar(h1_base_value_ratio),
            "selected_h1_to_base_prediction_gradient": scalar(
                h1_prediction_norm / base_prediction_norm.clamp_min(1.0e-30)
            ),
            "selected_h1_to_base_parameter_gradient": scalar(
                h1_parameter_norm / base_parameter_norm.clamp_min(1.0e-30)
            ),
            "base_selected_h1_parameter_gradient_cosine": scalar(
                pair_metrics["cosine"]
            ),
        },
        "gradient": {
            "base_prediction_norm": scalar(base_prediction_norm),
            "selected_h1_prediction_norm": scalar(h1_prediction_norm),
            "base_parameter_norm": scalar(base_parameter_norm),
            "selected_h1_parameter_norm": scalar(h1_parameter_norm),
            "other_prior_parameter_norm": scalar(
                torch.linalg.vector_norm(other_parameter_gradient)
            ),
            "total_parameter_norm_before_clip": scalar(total_gradient_norm),
            "total_parameter_norm_after_simulated_clip": scalar(
                clip_scale * total_gradient_norm
            ),
            "simulated_clip_scale": scalar(clip_scale),
            "simulated_clipping_triggered": bool(clip_scale < 1.0),
            "component_sum_relative_residual": scalar(
                gradient_sum_relative_residual
            ),
        },
        "timing_seconds": {
            "forward": forward_seconds,
            "loss": loss_seconds,
            "component_gradients": component_gradient_seconds,
            "backward": backward_seconds,
            "total": total_seconds,
        },
        "memory_bytes": {
            "peak_allocated": torch.cuda.max_memory_allocated(device),
            "peak_reserved": torch.cuda.max_memory_reserved(device),
            "device_total": torch.cuda.get_device_properties(device).total_memory,
        },
        "mode_checks": mode_checks,
        "stored_geometry": (
            {
                "coordinate_sha256": {
                    axis: tensor_state_sha256({axis: coordinate})
                    for axis, coordinate in loss_module.h1.coordinates.items()
                },
                "coordinate_combined_sha256": tensor_state_sha256(
                    loss_module.h1.coordinates
                ),
                "volume_weight_sha256": tensor_state_sha256(
                    {"spherical_coordinate_volume_proxy": volume_weights}
                ),
                "volume_weight_sum": scalar(volume_weights.sum()),
                "volume_weight_minimum": scalar(volume_weights.min()),
                "volume_weight_maximum": scalar(volume_weights.max()),
                "pole_sin_floor": selected.metadata["pole_sin_floor"],
                "pole_handling": selected.metadata["pole_handling"],
            }
            if name == "stored_coordinate_volume_proxy"
            else None
        ),
        "safety": {
            "model_train_mode": model.training,
            "batch_size": 1,
            "mixed_precision": False,
            "optimizer_created": False,
            "optimizer_step": False,
            "scheduler_step": False,
            "checkpoint_write": False,
            "gradient_clip_simulation_only": True,
        },
    }
    required = (
        record["device"]
        == {"model": "cuda", "input": "cuda", "output": "cuda", "loss": "cuda"},
        record["shape"]["input"] == [1, 16, 64, 64, 64],
        record["shape"]["output"] == [1, 8, 64, 64, 64],
        record["parameter_count"] == 331832,
        record["model_state_unchanged"],
        record["forward_finite"],
        record["loss_components_finite"],
        record["parameter_gradients_present_for_all"],
        record["parameter_gradients_finite"],
        all(non_h1_parity.values()),
        scalar(gradient_sum_relative_residual) < 1.0e-3,
        all(
            value
            for key, value in mode_checks.items()
            if key
            not in {
                "current_to_unit_index_ratio",
                "proper_Kerr_Schild_volume",
                "covariant_GRMHD_H1",
                "stored_components_covariant_derivative",
                "stored_vector_covariant_derivative",
                "pole_sin_floor",
                "pole_handling",
                "volume_weight_normalization",
            }
        ),
        mode_checks.get("covariant_GRMHD_H1", False) is False,
        mode_checks.get("stored_components_covariant_derivative", False) is False,
        mode_checks.get("stored_vector_covariant_derivative", False) is False,
        mode_checks.get("proper_Kerr_Schild_volume", "unverified")
        == "unverified",
        mode_checks.get("pole_sin_floor", 0.0) == 0.0,
        mode_checks.get(
            "pole_handling", "theta_center_sin_clamp_min_0"
        )
        == "theta_center_sin_clamp_min_0",
        mode_checks.get(
            "volume_weight_normalization", "explicit_sum_to_one"
        )
        == "explicit_sum_to_one",
    )
    if not all(required):
        raise RuntimeError(f"{name}: Stage I preflight gate failed: {record}")
    record["status"] = "passed"
    del (
        result,
        reference,
        prediction,
        sample,
        processor,
        model,
        loss_module,
    )
    torch.cuda.empty_cache()
    return record, input_hash, output_hash


def write_outputs(
    output_root: Path,
    payload: dict[str, Any],
    *,
    stem: str = "gpu_preflight",
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / f"{stem}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = []
    for record in payload["extensions"]:
        rows.append(
            {
                "mode": record["mode"],
                "status": record["status"],
                "total_loss": record["loss"]["total"],
                "base": record["loss"]["base"],
                "selected_h1_raw": record["loss"]["selected_h1_raw"],
                "selected_h1_weighted": record["loss"][
                    "selected_h1_weighted"
                ],
                "current_upstream_h1": record["loss"][
                    "diagnostic_current_upstream_h1_raw"
                ],
                "h1_base_value_ratio": record["ratios"][
                    "selected_h1_to_base_value"
                ],
                "h1_base_parameter_gradient_ratio": record["ratios"][
                    "selected_h1_to_base_parameter_gradient"
                ],
                "clip_scale": record["gradient"]["simulated_clip_scale"],
                "forward_seconds": record["timing_seconds"]["forward"],
                "loss_seconds": record["timing_seconds"]["loss"],
                "backward_seconds": record["timing_seconds"]["backward"],
                "peak_allocated_bytes": record["memory_bytes"]["peak_allocated"],
                "peak_reserved_bytes": record["memory_bytes"]["peak_reserved"],
                "finite": (
                    record["forward_finite"]
                    and record["loss_components_finite"]
                    and record["parameter_gradients_finite"]
                ),
            }
        )
    with (output_root / f"{stem}.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Stage I RTX 5070 GPU preflight",
        "",
        f"- Status: `{payload['status']}`",
        f"- Device: `{payload['environment']['device_name']}`",
        f"- Capability: `{payload['environment']['capability']}`",
        f"- PyTorch/CUDA: `{payload['environment']['torch_version']}` / "
        f"`{payload['environment']['torch_cuda_version']}`",
        "- Optimizer created/step: `false/false`",
        "- Scheduler step: `false`",
        "- Checkpoint write: `false`",
        "",
        "## Extension results",
        "",
    ]
    for row in rows:
        lines.append(
            f"- `{row['mode']}`: status `{row['status']}`, "
            f"loss `{row['total_loss']:.6g}`, "
            f"peak allocated `{row['peak_allocated_bytes'] / 2**30:.3f} GiB`, "
            f"peak reserved `{row['peak_reserved_bytes'] / 2**30:.3f} GiB`"
        )
    (output_root / f"{stem}.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_blocker(
    output_root: Path,
    error: BaseException,
    *,
    stem: str = "gpu_preflight",
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / f"{stem}_blocker.md").write_text(
        "# Stage I GPU preflight blocker\n\n"
        f"- Error type: `{type(error).__name__}`\n"
        f"- Error: `{error}`\n"
        "- CPU fallback: `false`\n"
        "- Optimizer created: `false`\n"
        "- Training started: `false`\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "all",
            "no_h1",
            "unit_index",
            "stored_coordinate_volume_proxy",
        ),
        default="all",
    )
    parser.add_argument("--output-stem", default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output_root = root / "outputs/paper_reduced100/stage_i"
    output_stem = args.output_stem or (
        "gpu_preflight"
        if args.mode == "all"
        else f"{args.mode}_gpu_preflight"
    )
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; Stage I refuses CPU fallback")
        device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(device)
        capability = torch.cuda.get_device_capability(device)
        if device_name != EXPECTED_DEVICE:
            raise RuntimeError(
                f"Expected {EXPECTED_DEVICE!r}, found {device_name!r}"
            )
        if capability != EXPECTED_CAPABILITY:
            raise RuntimeError(
                f"Expected capability {EXPECTED_CAPABILITY}, found {capability}"
            )

        shared_path = (
            root / "outputs/paper_reduced100/stage_g/shared_initial_state.pt"
        )
        shared_state = torch.load(
            shared_path,
            map_location="cpu",
            weights_only=True,
        )
        if tensor_state_sha256(shared_state) != EXPECTED_SHARED_HASH:
            raise RuntimeError("Frozen Stage G shared state hash changed")

        protocol = PaperReduced100Protocol.from_yaml(
            root / "configs/data/paper_reduced100.yaml",
            project_root=root,
        )
        train_dataset = GRMHDNextStepDataset(protocol.make_datasets()["train"])
        batch = next(
            iter(
                DataLoader(
                    train_dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=0,
                )
            )
        )
        full_config = load_paper_experiment_config(
            root / "configs/paper_reduced100/full_fno_proxy.yaml",
            project_root=root,
        )
        reference_loss = build_paper_training_loss(full_config).to(device).eval()
        paths = (
            (
                "no_h1",
                root
                / "configs/paper_reduced100/extensions/no_h1_control.yaml",
            ),
            (
                "unit_index",
                root
                / "configs/paper_reduced100/extensions/unit_index_h1.yaml",
            ),
            (
                "stored_coordinate_volume_proxy",
                root
                / "configs/paper_reduced100/extensions/"
                "stored_coordinate_volume_h1.yaml",
            ),
        )
        if args.mode != "all":
            paths = tuple(item for item in paths if item[0] == args.mode)
        records = []
        input_hash = None
        output_hash = None
        for name, path in paths:
            record, input_hash, output_hash = run_variant(
                name=name,
                config_path=path,
                root=root,
                device=device,
                batch=batch,
                shared_state=shared_state,
                reference_loss=reference_loss,
                reference_input_hash=input_hash,
                reference_output_hash=output_hash,
            )
            records.append(record)

        nvidia_smi = subprocess.run(
            ["nvidia-smi"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        payload = {
            "schema_version": "paper-stage-i-gpu-preflight-v1",
            "status": "passed",
            "environment": {
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "cuda_available": torch.cuda.is_available(),
                "device_count": torch.cuda.device_count(),
                "device_name": device_name,
                "capability": list(capability),
                "nvidia_smi": nvidia_smi,
            },
            "frozen_pairing": {
                "shared_state_tensor_sha256": EXPECTED_SHARED_HASH,
                "parameter_count": 331832,
                "seed": 42,
                "batch_size": 1,
                "mixed_precision": False,
            },
            "extensions": records,
            "safety": {
                "all_three_passed_before_training": args.mode == "all",
                "single_no_h1_preflight": args.mode == "no_h1",
                "optimizer_created": False,
                "optimizer_step": False,
                "scheduler_step": False,
                "checkpoint_write": False,
                "training_started": False,
                "cpu_fallback": False,
            },
        }
        write_outputs(output_root, payload, stem=output_stem)
        print(json.dumps(payload, indent=2, sort_keys=True))
    except BaseException as error:
        write_blocker(output_root, error, stem=output_stem)
        raise


if __name__ == "__main__":
    main()
