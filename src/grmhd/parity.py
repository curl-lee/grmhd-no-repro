"""Numerical parity checks between pinned Trainer paths and local equivalents."""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

import torch

from neuralop.training import Trainer

from .data_processor import GRMHDDataProcessor
from .hybrid import HybridTargetStats
from .normalizer import GRMHDNormalizer
from .upstream_adapters import (
    UpstreamFNOConfig,
    build_upstream_fno,
    build_upstream_loss,
    build_upstream_optimizer,
)


class _RecordingProcessor(GRMHDDataProcessor):
    def preprocess(self, sample, step=None):
        processed = super().preprocess(sample, step=step)
        if processed is not None:
            self.last_preprocessed = {
                "x": processed["x"].detach().cpu().clone(),
                "y": processed["y"].detach().cpu().clone(),
            }
        return processed

    def postprocess(self, raw_output, sample, step=None):
        output, processed = super().postprocess(raw_output, sample, step=step)
        self.last_postprocessed = output.detach().cpu().clone()
        return output, processed


def _clone_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value.detach().clone() if torch.is_tensor(value) else deepcopy(value)
        for key, value in batch.items()
    }


def _gradient_norm(model: torch.nn.Module) -> float:
    total = sum(
        float(parameter.grad.detach().abs().double().square().sum())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    return math.sqrt(total)


def compare_upstream_and_manual_train_batch(
    *,
    batch: dict[str, Any],
    normalizer: GRMHDNormalizer,
    shells: torch.Tensor | None,
    downsample: int,
    model_config: UpstreamFNOConfig,
    learning_rate: float = 3.0e-4,
    weight_decay: float = 1.0e-4,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Run one identical batch through upstream Trainer and a literal manual path."""
    device = torch.device(device)
    torch.manual_seed(42)
    template = build_upstream_fno(model_config)
    initial_state = deepcopy(template.state_dict())
    model_upstream = build_upstream_fno(model_config).to(device)
    model_manual = build_upstream_fno(model_config).to(device)
    model_upstream.load_state_dict(deepcopy(initial_state), strict=True)
    model_manual.load_state_dict(deepcopy(initial_state), strict=True)
    processor_upstream = _RecordingProcessor(
        normalizer=normalizer,
        shells=None if shells is None else shells.detach().clone(),
        target_mode="state",
        downsample=downsample,
        device=device,
    )
    processor_manual = _RecordingProcessor(
        normalizer=normalizer,
        shells=None if shells is None else shells.detach().clone(),
        target_mode="state",
        downsample=downsample,
        device=device,
    )
    optimizer_upstream = build_upstream_optimizer(
        model_upstream, learning_rate=learning_rate, weight_decay=weight_decay
    )
    optimizer_manual = build_upstream_optimizer(
        model_manual, learning_rate=learning_rate, weight_decay=weight_decay
    )
    optimizer_state_identical_before = (
        optimizer_upstream.state_dict() == optimizer_manual.state_dict()
    )
    loss_upstream = build_upstream_loss("l2")
    loss_manual = build_upstream_loss("l2")

    raw_upstream: list[torch.Tensor] = []
    raw_manual: list[torch.Tensor] = []
    hook_upstream = model_upstream.register_forward_hook(
        lambda _module, _args, output: raw_upstream.append(output.detach().cpu().clone())
    )
    hook_manual = model_manual.register_forward_hook(
        lambda _module, _args, output: raw_manual.append(output.detach().cpu().clone())
    )

    trainer = Trainer(
        model=model_upstream,
        n_epochs=1,
        device=device,
        data_processor=processor_upstream,
        mixed_precision=False,
        verbose=False,
    )
    trainer.optimizer = optimizer_upstream
    trainer.scheduler = None
    trainer.regularizer = None
    trainer.epoch = 0
    trainer.n_samples = 0
    trainer.model.train()
    trainer.data_processor.train()
    upstream_value = trainer.train_one_batch(
        0, _clone_batch(batch), loss_upstream
    )
    upstream_value.backward()
    upstream_gradient_norm = _gradient_norm(model_upstream)
    optimizer_upstream.step()

    model_manual.train()
    processor_manual.train()
    optimizer_manual.zero_grad(set_to_none=True)
    processed = processor_manual.preprocess(_clone_batch(batch))
    assert processed is not None
    raw = model_manual(**processed)
    manual_output, processed = processor_manual.postprocess(raw, processed)
    manual_value = loss_manual(manual_output, **processed)
    manual_value.backward()
    manual_gradient_norm = _gradient_norm(model_manual)
    optimizer_manual.step()

    hook_upstream.remove()
    hook_manual.remove()
    parameter_difference = max(
        float(torch.max(torch.abs(left.detach() - right.detach())).cpu())
        for left, right in zip(model_upstream.parameters(), model_manual.parameters())
    )
    report = {
        "preprocess_x_max_abs_difference": float(
            torch.max(
                torch.abs(
                    processor_upstream.last_preprocessed["x"]
                    - processor_manual.last_preprocessed["x"]
                )
            )
        ),
        "preprocess_y_max_abs_difference": float(
            torch.max(
                torch.abs(
                    processor_upstream.last_preprocessed["y"]
                    - processor_manual.last_preprocessed["y"]
                )
            )
        ),
        "raw_output_max_abs_difference": float(
            torch.max(torch.abs(raw_upstream[0] - raw_manual[0]))
        ),
        "postprocess_output_max_abs_difference": float(
            torch.max(
                torch.abs(
                    processor_upstream.last_postprocessed
                    - processor_manual.last_postprocessed
                )
            )
        ),
        "upstream_loss": float(upstream_value.detach().cpu()),
        "manual_loss": float(manual_value.detach().cpu()),
        "loss_abs_difference": abs(
            float(upstream_value.detach().cpu()) - float(manual_value.detach().cpu())
        ),
        "upstream_gradient_norm": upstream_gradient_norm,
        "manual_gradient_norm": manual_gradient_norm,
        "gradient_norm_abs_difference": abs(
            upstream_gradient_norm - manual_gradient_norm
        ),
        "parameter_update_max_abs_difference": parameter_difference,
        "optimizer_state_identical_before": optimizer_state_identical_before,
        "loss_reduction": getattr(loss_upstream, "reduction", None),
        "trainer_class": type(trainer).__module__ + "." + type(trainer).__name__,
        "optimizer_class": type(optimizer_upstream).__module__
        + "."
        + type(optimizer_upstream).__name__,
    }
    compared = [
        report["preprocess_x_max_abs_difference"],
        report["preprocess_y_max_abs_difference"],
        report["raw_output_max_abs_difference"],
        report["postprocess_output_max_abs_difference"],
        report["loss_abs_difference"],
        report["gradient_norm_abs_difference"],
        report["parameter_update_max_abs_difference"],
    ]
    report["tolerance"] = 1.0e-7
    report["status"] = (
        "passed"
        if optimizer_state_identical_before
        and report["loss_reduction"] == "sum"
        and max(compared) <= report["tolerance"]
        else "failed"
    )
    return report


def compare_upstream_and_local_autoregression(
    *,
    sample: dict[str, Any],
    normalizer: GRMHDNormalizer,
    shells: torch.Tensor | None,
    downsample: int,
    model_config: UpstreamFNOConfig,
    model_state_dict: dict[str, Any],
    max_steps: int,
    target_mode: str = "state",
    hybrid_stats: HybridTargetStats | None = None,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Cross-check Trainer autoregression against an explicit local loop."""
    device = torch.device(device)
    model_upstream = build_upstream_fno(model_config).to(device)
    model_local = build_upstream_fno(model_config).to(device)
    model_upstream.load_state_dict(deepcopy(model_state_dict), strict=True)
    model_local.load_state_dict(deepcopy(model_state_dict), strict=True)
    processor_upstream = GRMHDDataProcessor(
        normalizer=normalizer,
        shells=None if shells is None else shells.detach().clone(),
        target_mode=target_mode,
        hybrid_stats=hybrid_stats,
        downsample=downsample,
        device=device,
        record_rollout_trace=True,
    )
    processor_local = GRMHDDataProcessor(
        normalizer=normalizer,
        shells=None if shells is None else shells.detach().clone(),
        target_mode=target_mode,
        hybrid_stats=hybrid_stats,
        downsample=downsample,
        device=device,
        record_rollout_trace=True,
    )
    loss_upstream = build_upstream_loss("l2")
    loss_local = build_upstream_loss("l2")
    trainer = Trainer(
        model=model_upstream,
        n_epochs=1,
        device=device,
        data_processor=processor_upstream,
        mixed_precision=False,
        verbose=False,
    )
    trainer.model.eval()
    trainer.data_processor.eval()
    trainer.n_samples = 0
    with torch.no_grad():
        upstream_losses, upstream_last = trainer.eval_one_batch_autoreg(
            _clone_batch(sample),
            {"l2": loss_upstream},
            return_output=True,
            max_steps=max_steps,
        )

        model_local.eval()
        processor_local.eval()
        local_sample = _clone_batch(sample)
        local_loss = 0.0
        step = 0
        local_last = None
        while step < max_steps:
            processed = processor_local.preprocess(local_sample, step=step)
            if processed is None:
                break
            raw_output = model_local(**processed)
            local_last, local_sample = processor_local.postprocess(
                raw_output, processed, step=step
            )
            local_loss += loss_local(local_last, **local_sample)
            step += 1
        local_loss /= step

    upstream_trace = processor_upstream.rollout_trace
    local_trace = processor_local.rollout_trace
    if len(upstream_trace) != len(local_trace):
        return {
            "status": "failed",
            "reason": "trajectory length mismatch",
            "upstream_steps": len(upstream_trace),
            "local_steps": len(local_trace),
        }
    step_records = []
    finite = True
    positive = True
    for upstream_step, local_step in zip(upstream_trace, local_trace):
        differences = {
            name: float(torch.max(torch.abs(upstream_step[name] - local_step[name])))
            for name in ("physical_input", "encoded_input", "physical_output")
        }
        physical_output = local_step["physical_output"]
        physical_target = local_step["physical_target"]
        relative_l2 = float(
            torch.linalg.vector_norm(physical_output - physical_target)
            / torch.linalg.vector_norm(physical_target).clamp_min(1.0e-12)
        )
        finite &= bool(torch.isfinite(physical_output).all())
        positive &= bool(torch.all(physical_output[:, 3:5] > 0))
        step_records.append(
            {
                "step": upstream_step["step"],
                **{f"{name}_max_abs_difference": value for name, value in differences.items()},
                "decoded_global_relative_l2": relative_l2,
            }
        )
    max_difference = max(
        value
        for record in step_records
        for key, value in record.items()
        if key.endswith("max_abs_difference")
    )
    upstream_loss_value = float(upstream_losses["l2"].detach().cpu())
    local_loss_value = float(local_loss.detach().cpu())
    last_difference = float(
        torch.max(torch.abs(upstream_last.detach().cpu() - local_last.detach().cpu()))
    )
    tolerance = 1.0e-6
    return {
        "status": (
            "passed"
            if max_difference <= tolerance
            and abs(upstream_loss_value - local_loss_value) <= tolerance
            and last_difference <= tolerance
            and finite
            and positive
            else "failed"
        ),
        "tolerance": tolerance,
        "trajectory_length": len(step_records),
        "first_step": step_records[0]["step"],
        "last_step": step_records[-1]["step"],
        "upstream_average_l2": upstream_loss_value,
        "local_average_l2": local_loss_value,
        "average_l2_abs_difference": abs(upstream_loss_value - local_loss_value),
        "last_output_max_abs_difference": last_difference,
        "maximum_trace_abs_difference": max_difference,
        "all_outputs_finite": finite,
        "rho_press_positive": positive,
        "target_mode": target_mode,
        "upstream_transform_counts": processor_upstream.transform_counts,
        "local_transform_counts": processor_local.transform_counts,
        "steps": step_records,
    }
