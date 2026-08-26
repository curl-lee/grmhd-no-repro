#!/usr/bin/env python
"""Train or smoke-test one controlled paper_reduced100 operator configuration.

Stage F smoke outputs are engineering validation artifacts, not scientific
models or evidence for a Full-versus-Plain performance conclusion.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from grmhd import CHANNELS
from grmhd.dataset import sha256_file
from grmhd.paper_checkpoint import (
    build_paper_checkpoint_metadata,
    load_paper_checkpoint,
    save_paper_checkpoint,
    stage_i_selected_h1_definition,
)
from grmhd.paper_config import (
    ResolvedPaperConfig,
    build_paper_model,
    build_paper_training_loss,
    load_paper_experiment_config,
    model_tensor_state_sha256,
)
from grmhd.paper_data_processor import PaperDataProcessor
from grmhd.paper_metrics import ClampMaskAccumulator, PaperMetricAccumulator
from grmhd.paper_protocol import EXPECTED_UPSTREAM_COMMIT, PaperReduced100Protocol
from grmhd.paper_references import PaperReferenceStates, paper_reference_metadata
from grmhd.paper_stage_g import (
    accumulation_count_for_batch,
    is_accumulation_step,
    load_epoch_pair_order,
    tensor_state_sha256,
)
from grmhd.paper_stage_p import ood_diagnostics
from grmhd.paper_trainer import (
    PaperTrainerAdapter,
    PaperTrainerLossAdapter,
    build_paper_optimizer,
    build_warmup_cosine_scheduler,
    parameter_gradient_norm,
)
from grmhd.upstream_adapters import GRMHDNextStepDataset


EXPECTED_STAGE_G_PAIR_ORDER_SHA256 = (
    "5e1e361ea75e01ffd667822d8bfe870d70c5bc69570d52cdf806e637b2df2b52"
)


def validate_run_limits(
    *,
    smoke: bool,
    epochs: int,
    max_train_batches: int | None,
    max_validation_batches: int | None,
) -> None:
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if smoke and epochs > 2:
        raise ValueError("Engineering smoke is capped at two epochs")
    if epochs > 30:
        raise ValueError("This entry point is capped at the resource-scaled 30-epoch protocol")
    for name, value in (
        ("max_train_batches", max_train_batches),
        ("max_validation_batches", max_validation_batches),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive when provided")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def frozen_environment(root: Path) -> dict[str, Any]:
    upstream = root / "external/neuraloperator"
    upstream_commit = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    upstream_status = subprocess.run(
        ["git", "-C", str(upstream), "status", "--short"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if upstream_commit != EXPECTED_UPSTREAM_COMMIT:
        raise ValueError("Pinned neuraloperator commit changed before training")
    if upstream_status:
        raise ValueError("Pinned neuraloperator worktree is dirty before training")
    cuda_available = torch.cuda.is_available()
    return {
        "project_commit": _git(root, "rev-parse", "HEAD"),
        "project_branch": _git(root, "branch", "--show-current"),
        "project_dirty": bool(_git(root, "status", "--short")),
        "upstream_commit": upstream_commit,
        "upstream_worktree_clean": True,
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "uname": platform.uname()._asdict(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
    }


def load_p3_train_envelope(
    root: Path, config: ResolvedPaperConfig
) -> Mapping[str, Any] | None:
    """Load the frozen train-only P3 envelope for diagnostics, never fitting it."""
    if not config.uses_p3:
        return None
    return json.loads(
        (root / "outputs/paper_reduced100/stage_p/train_envelope.json").read_text(
            encoding="utf-8"
        )
    )


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def component_gradient_tensors(
    component: torch.Tensor, parameters: list[torch.nn.Parameter]
) -> list[torch.Tensor | None]:
    gradients = torch.autograd.grad(
        component,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    return [
        None if gradient is None else gradient.detach().clone()
        for gradient in gradients
    ]


def gradient_tensor_norm(gradients: Iterable[torch.Tensor | None]) -> float:
    squares = [
        torch.sum(torch.abs(gradient).float().square())
        for gradient in gradients
        if gradient is not None
    ]
    if not squares:
        return 0.0
    value = torch.sqrt(torch.stack(squares).sum())
    return float(value.cpu()) if torch.isfinite(value) else math.nan


def gradient_tensor_cosine(
    left: Iterable[torch.Tensor | None],
    right: Iterable[torch.Tensor | None],
) -> float | None:
    """Return the cosine between two component-gradient collections."""

    left_values = list(left)
    right_values = list(right)
    if len(left_values) != len(right_values):
        raise ValueError("Component-gradient collection length changed")
    dot_terms = []
    left_squares = []
    right_squares = []
    for left_value, right_value in zip(left_values, right_values):
        if left_value is not None:
            left_squares.append(
                torch.sum(torch.abs(left_value).float().square())
            )
        if right_value is not None:
            right_squares.append(
                torch.sum(torch.abs(right_value).float().square())
            )
        if left_value is not None and right_value is not None:
            dot_terms.append(
                torch.real(
                    torch.sum(torch.conj(left_value) * right_value)
                ).float()
            )
    if not left_squares or not right_squares:
        return None
    left_norm = torch.sqrt(torch.stack(left_squares).sum())
    right_norm = torch.sqrt(torch.stack(right_squares).sum())
    if float(left_norm.cpu()) == 0.0 or float(right_norm.cpu()) == 0.0:
        return None
    dot = (
        torch.stack(dot_terms).sum()
        if dot_terms
        else left_norm.new_zeros(())
    )
    cosine = dot / (left_norm * right_norm)
    return float(cosine.cpu()) if torch.isfinite(cosine) else math.nan


def gradient_tensor_projection(
    component: Iterable[torch.Tensor | None],
    total: Iterable[torch.Tensor | None],
) -> float | None:
    """Project one component gradient onto the normalized total direction."""

    component_values = list(component)
    total_values = list(total)
    if len(component_values) != len(total_values):
        raise ValueError("Component/total gradient collection length changed")
    dot_terms = []
    total_squares = []
    component_present = False
    for component_value, total_value in zip(
        component_values,
        total_values,
    ):
        if component_value is not None:
            component_present = True
        if total_value is not None:
            total_squares.append(
                torch.sum(torch.abs(total_value).float().square())
            )
        if component_value is not None and total_value is not None:
            dot_terms.append(
                torch.real(
                    torch.sum(torch.conj(component_value) * total_value)
                ).float()
            )
    if not component_present or not total_squares:
        return None
    total_norm = torch.sqrt(torch.stack(total_squares).sum())
    if float(total_norm.cpu()) == 0.0:
        return None
    dot = (
        torch.stack(dot_terms).sum()
        if dot_terms
        else total_norm.new_zeros(())
    )
    projection = dot / total_norm
    return float(projection.cpu()) if torch.isfinite(projection) else math.nan


def add_scaled_gradients(
    accumulator: list[torch.Tensor | None] | None,
    gradients: list[torch.Tensor | None],
    *,
    scale: float,
) -> list[torch.Tensor | None]:
    if accumulator is None:
        return [
            None if gradient is None else gradient.mul(scale)
            for gradient in gradients
        ]
    if len(accumulator) != len(gradients):
        raise ValueError("Component-gradient accumulator length changed")
    for index, gradient in enumerate(gradients):
        if gradient is None:
            continue
        if accumulator[index] is None:
            accumulator[index] = gradient.mul(scale)
        else:
            accumulator[index].add_(gradient, alpha=scale)
    return accumulator


def parameter_update_norm(
    before: list[torch.Tensor], parameters: list[torch.nn.Parameter]
) -> float:
    squares = [
        torch.sum(
            torch.abs(parameter.detach() - previous).float().square()
        )
        for previous, parameter in zip(before, parameters)
    ]
    if not squares:
        return 0.0
    value = torch.sqrt(torch.stack(squares).sum())
    return float(value.cpu()) if torch.isfinite(value) else math.nan


def model_displacement_norm(
    model: torch.nn.Module,
    reference_state: Mapping[str, torch.Tensor],
) -> float:
    squares = []
    for name, parameter in model.named_parameters():
        reference = reference_state[name].to(
            device=parameter.device,
            dtype=parameter.dtype,
        )
        squares.append(
            torch.sum(torch.abs(parameter.detach() - reference).float().square())
        )
    if not squares:
        return 0.0
    value = torch.sqrt(torch.stack(squares).sum())
    return float(value.cpu()) if torch.isfinite(value) else math.nan


def numeric_summary(values: Iterable[float]) -> dict[str, float | None]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"mean": None, "median": None, "q95": None, "maximum": None}
    return {
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q95": float(np.quantile(array, 0.95)),
        "maximum": float(np.max(array)),
    }


def selected_h1_gradient_ratio(
    *,
    base_gradient_norm: float,
    selected_h1_gradient_norm: float,
    diagnostic_h1_mode: str | None,
) -> float | None:
    if diagnostic_h1_mode == "no_h1":
        if selected_h1_gradient_norm != 0.0:
            raise ValueError("No-H1 selected gradient norm must be exactly zero")
        return None
    return selected_h1_gradient_norm / max(base_gradient_norm, 1e-12)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _json_value(value: Any) -> Any:
    if torch.is_tensor(value):
        detached = value.detach().cpu()
        return float(detached) if detached.numel() == 1 else detached.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def flatten_record(values: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in values.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            output.update(flatten_record(value, path))
        elif isinstance(value, (tuple, list)):
            output[path] = json.dumps(_json_value(value), sort_keys=True)
        else:
            output[path] = _json_value(value)
    return output


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _make_reference(
    *,
    fields: Mapping[str, Any],
    normalized_prediction: torch.Tensor,
    model_physical_prediction: torch.Tensor,
    processor: PaperDataProcessor,
) -> PaperReferenceStates:
    return PaperReferenceStates(
        raw_physical_target=fields["raw_physical_target"],
        oracle_physical_target=fields["oracle_physical_target"],
        normalized_target=fields["normalized_target"],
        normalized_prediction=normalized_prediction,
        model_physical_prediction=model_physical_prediction,
        metadata=paper_reference_metadata(processor.preprocessor),
    )


def evaluate_one_step(
    *,
    model: torch.nn.Module,
    processor: PaperDataProcessor,
    loss_adapter: PaperTrainerLossAdapter,
    loader: DataLoader,
    max_batches: int,
    epoch: int,
) -> dict[str, Any]:
    model.eval()
    processor.eval()
    processor.set_epoch(epoch)
    loss_adapter.set_epoch(epoch)
    metric = PaperMetricAccumulator()
    clamp = ClampMaskAccumulator(
        gamma=processor.preprocessor.gamma,
        inverse_clamp_fraction=processor.preprocessor.inverse_clamp_fraction,
    )
    losses: list[float] = []
    bound_clamps: list[float] = []
    positive = True
    finite = True
    range_min = {name: math.inf for name in CHANNELS}
    range_max = {name: -math.inf for name in CHANNELS}
    residual_numerator = {name: 0.0 for name in CHANNELS}
    residual_denominator = {name: 0.0 for name in CHANNELS}
    residual_norms: list[float] = []
    true_residual_norms: list[float] = []
    residual_ratios: list[float] = []
    residual_cosines: list[float] = []
    residual_sign_agreements: list[float] = []
    probe_sha256: str | None = None
    started = time.perf_counter()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            model_sample = processor.preprocess(batch)
            prediction = model(x=model_sample["x"])
            prediction, fields = processor.postprocess(prediction, model_sample)
            value = loss_adapter(prediction, **fields)
            if processor.config.stage_r:
                predicted_residual = processor.last_predicted_residual
                if predicted_residual is None:
                    raise RuntimeError("Stage R validation lost predicted residual")
                target_residual = fields["normalized_residual_target"]
                pred64 = predicted_residual.to(torch.float64)
                target64 = target_residual.to(torch.float64)
                pred_norm = torch.linalg.vector_norm(pred64)
                target_norm = torch.linalg.vector_norm(target64)
                residual_norms.append(float(pred_norm.cpu()))
                true_residual_norms.append(float(target_norm.cpu()))
                residual_ratios.append(
                    float((pred_norm / target_norm.clamp_min(1.0e-30)).cpu())
                )
                residual_cosines.append(
                    float(
                        (
                            torch.sum(pred64 * target64)
                            / (pred_norm * target_norm).clamp_min(1.0e-30)
                        ).cpu()
                    )
                )
                residual_sign_agreements.append(
                    float(
                        (torch.signbit(pred64) == torch.signbit(target64))
                        .to(torch.float64)
                        .mean()
                        .cpu()
                    )
                )
                for channel, name in enumerate(CHANNELS):
                    residual_numerator[name] += float(
                        torch.sum((pred64[:, channel] - target64[:, channel]).square()).cpu()
                    )
                    residual_denominator[name] += float(
                        torch.sum(target64[:, channel].square()).cpu()
                    )
            decoded = processor.decode_prediction(
                prediction,
                apply_evaluation_clamp=bool(
                    processor.config.values["evaluation"]["rho_press_eval_clamp"]
                ),
            )
            metric.update(
                _make_reference(
                    fields=fields,
                    normalized_prediction=prediction,
                    model_physical_prediction=decoded.physical_prediction,
                    processor=processor,
                ),
                channel_axis=1,
            )
            clamp.update(
                normalized_target=fields["normalized_target"],
                normalized_prediction=prediction,
                channel_axis=1,
            )
            losses.append(float(value.detach().cpu()))
            bound_clamps.append(float(decoded.clamp_fraction["combined"]))
            physical = decoded.physical_prediction
            finite &= bool(torch.isfinite(prediction).all() and torch.isfinite(physical).all())
            positive &= bool(torch.all(physical[:, 3:5] > 0))
            for channel, name in enumerate(CHANNELS):
                range_min[name] = min(range_min[name], float(physical[:, channel].min().cpu()))
                range_max[name] = max(range_max[name], float(physical[:, channel].max().cpu()))
            if probe_sha256 is None:
                probe_sha256 = tensor_sha256(prediction)
    if not losses:
        raise RuntimeError("Validation loader produced no evaluated batch")
    residual_metrics = None
    if processor.config.stage_r:
        per_channel = {
            name: math.sqrt(
                residual_numerator[name] / max(residual_denominator[name], 1.0e-30)
            )
            for name in CHANNELS
        }
        residual_metrics = {
            "predicted_residual_norm": numeric_summary(residual_norms),
            "true_residual_norm": numeric_summary(true_residual_norms),
            "residual_over_true_residual": numeric_summary(residual_ratios),
            "residual_cosine": numeric_summary(residual_cosines),
            "residual_sign_agreement": numeric_summary(residual_sign_agreements),
            "per_channel_relative_l2": per_channel,
            "arithmetic_average_relative_l2": float(np.mean(list(per_channel.values()))),
            "zero_residual_relative_l2": 1.0,
            "persistence_semantics": "predicted_normalized_residual_equals_zero",
        }
    return {
        "loss": float(np.mean(losses)),
        "oracle_aware": metric.finalize(),
        "inverse_clamp": clamp.finalize(),
        "evaluation_bound_clamp_fraction": float(np.mean(bound_clamps)),
        "rho_press_positive": positive,
        "finite": finite,
        "physical_range": {
            name: {"minimum": range_min[name], "maximum": range_max[name]}
            for name in CHANNELS
        },
        "probe_prediction_sha256": probe_sha256,
        "seconds": time.perf_counter() - started,
        "batches": len(losses),
        "residual_metrics": residual_metrics,
    }


def evaluate_rollout3(
    *,
    model: torch.nn.Module,
    processor: PaperDataProcessor,
    validation_dataset: Any,
    epoch: int,
) -> dict[str, Any]:
    model.eval()
    processor.eval()
    processor.set_epoch(epoch)
    metric = PaperMetricAccumulator()
    clamp = ClampMaskAccumulator(
        gamma=processor.preprocessor.gamma,
        inverse_clamp_fraction=processor.preprocessor.inverse_clamp_fraction,
    )
    state = validation_dataset.load_snapshot(91).unsqueeze(0)
    counts_before = dict(processor.transform_counts)
    records = []
    positive = True
    finite = True
    with torch.no_grad():
        for rollout_step in range(3):
            source_index = 91 + rollout_step
            target_index = source_index + 1
            raw_target = validation_dataset.load_snapshot(target_index).unsqueeze(0)
            batch = {
                "physical_input": state,
                "physical_target": raw_target,
                "source_index": torch.tensor([source_index]),
                "target_index": torch.tensor([target_index]),
                "time": torch.tensor(
                    [validation_dataset.times[source_index]], dtype=torch.float64
                ),
                "target_time": torch.tensor(
                    [validation_dataset.times[target_index]], dtype=torch.float64
                ),
            }
            model_sample = processor.preprocess(batch)
            prediction = model(x=model_sample["x"])
            prediction, fields = processor.postprocess(prediction, model_sample)
            evaluation_clamp = bool(
                processor.config.values["evaluation"]["rho_press_eval_clamp"]
            )
            decoded = processor.decode_prediction(
                prediction, apply_evaluation_clamp=evaluation_clamp
            )
            physical = decoded.physical_prediction
            metric.update(
                _make_reference(
                    fields=fields,
                    normalized_prediction=prediction,
                    model_physical_prediction=physical,
                    processor=processor,
                ),
                channel_axis=1,
            )
            clamp.update(
                normalized_target=fields["normalized_target"],
                normalized_prediction=prediction,
                channel_axis=1,
            )
            step_finite = bool(torch.isfinite(prediction).all() and torch.isfinite(physical).all())
            step_positive = bool(torch.all(physical[:, 3:5] > 0))
            finite &= step_finite
            positive &= step_positive
            records.append(
                {
                    "step": rollout_step + 1,
                    "source_snapshot": source_index,
                    "target_snapshot": target_index,
                    "finite": step_finite,
                    "rho_press_positive": step_positive,
                    "evaluation_bound_clamp_fraction": decoded.clamp_fraction,
                    "physical_range": {
                        name: {
                            "minimum": float(physical[:, channel].min().cpu()),
                            "maximum": float(physical[:, channel].max().cpu()),
                        }
                        for channel, name in enumerate(CHANNELS)
                    },
                }
            )
            # Physical rollout: decode once, then the next step re-encodes once.
            state = physical.detach()
    count_delta = {
        name: processor.transform_counts[name] - counts_before[name]
        for name in counts_before
    }
    return {
        "steps": 3,
        "finite": finite,
        "rho_press_positive": positive,
        "physical_state_rollout": True,
        "reencode_each_step": True,
        "no_double_transform": count_delta
        == {
            "input_encode": 3,
            "target_encode": 3,
            "oracle_decode": 3,
            "prediction_decode": 3,
        },
        "transform_count_delta": count_delta,
        "oracle_aware": metric.finalize(),
        "inverse_clamp": clamp.finalize(),
        "records": records,
        "artifact_flags": {
            "radial_mode": processor.radial.mode,
            "radial_envelope_reference_only": True,
            "evaluation_rho_press_clamp": bool(
                processor.config.values["evaluation"]["rho_press_eval_clamp"]
            ),
            "validation_used_for_fit": False,
            "scientific_result": False,
        },
    }


def _checkpoint_metadata(
    config: ResolvedPaperConfig,
    *,
    environment: Mapping[str, Any],
    config_checksum: str,
    epoch: int,
    experiment_name: str,
    gradient_accumulation: int,
    probe_sha256: str,
    stage_g: Mapping[str, Any] | None = None,
    stage_i: Mapping[str, Any] | None = None,
    stage_k: Mapping[str, Any] | None = None,
    stage_o: Mapping[str, Any] | None = None,
    stage_r: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return build_paper_checkpoint_metadata(
        config,
        project_commit=str(environment["project_commit"]),
        config_checksum=config_checksum,
        epoch=epoch,
        experiment_name=experiment_name,
        gradient_accumulation=gradient_accumulation,
        validation_probe_sha256=probe_sha256,
        stage_g=stage_g,
        stage_i=stage_i,
        stage_k=stage_k,
        stage_o=stage_o,
        stage_r=stage_r,
    )


def verify_checkpoint(
    checkpoint_dir: Path,
    *,
    config: ResolvedPaperConfig,
    config_checksum: str,
    epochs: int,
    warmup_epochs: int,
    validation_batch: Mapping[str, Any],
    device: torch.device,
    expected_shared_initial_state_sha256: str | None = None,
    expected_pair_order_sha256: str | None = None,
) -> tuple[dict[str, Any], torch.nn.Module, PaperDataProcessor]:
    model = build_paper_model(config)
    optimizer = build_paper_optimizer(
        model,
        learning_rate=config.values["optimizer"]["learning_rate"],
        weight_decay=config.values["optimizer"]["weight_decay"],
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_epochs=epochs,
        warmup_epochs=warmup_epochs,
        min_learning_rate=config.values["scheduler"]["min_learning_rate"],
    )
    loaded = load_paper_checkpoint(
        checkpoint_dir,
        config=config,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_config_checksum=config_checksum,
    )
    loaded.model.to(device).eval()
    processor = PaperDataProcessor.from_config(config).to(device)
    processor.eval()
    processor.set_epoch(loaded.epoch)
    with torch.no_grad():
        model_sample = processor.preprocess(validation_batch)
        raw_output = loaded.model(x=model_sample["x"])
        prediction, _ = processor.postprocess(raw_output, model_sample)
    prediction_hash = tensor_sha256(prediction)
    expected_hash = loaded.metadata.get("validation_probe_sha256")
    if prediction_hash != expected_hash:
        raise ValueError("Checkpoint deterministic validation prediction mismatch")
    stage_g = loaded.metadata.get("stage_g")
    stage_i = loaded.metadata.get("stage_i")
    stage_k = loaded.metadata.get("stage_k")
    if expected_shared_initial_state_sha256 is not None:
        pairing = stage_k if config.stage_k else stage_g
        if not isinstance(pairing, Mapping):
            raise ValueError("Controlled checkpoint lacks pairing metadata")
        if (
            pairing.get(
                "initial_state_sha256"
                if config.stage_k
                else "shared_initial_state_sha256"
            )
            != expected_shared_initial_state_sha256
        ):
            raise ValueError("Checkpoint shared initial state hash mismatch")
        if pairing.get("pair_order_sha256") != expected_pair_order_sha256:
            raise ValueError("Checkpoint pair-order hash mismatch")
    return (
        {
            "path": str(checkpoint_dir),
            "strict_model_reload": True,
            "optimizer_reload": loaded.optimizer is not None,
            "scheduler_reload": loaded.scheduler is not None,
            "epoch": loaded.epoch,
            "roi_ramp": loaded.metadata["roi_ramp"],
            "prediction_sha256": prediction_hash,
            "prediction_parity": True,
            "metadata_validated": True,
            "stage_g": stage_g,
            "stage_i": stage_i,
            "stage_k": stage_k,
        },
        loaded.model,
        processor,
    )


def main() -> None:
    main_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-validation-batches", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stage-k-engineering-smoke", action="store_true")
    parser.add_argument("--stage-o-engineering-smoke", action="store_true")
    parser.add_argument("--stage-r-engineering-smoke", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--initial-state", type=Path, default=None)
    parser.add_argument("--pair-order", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = load_paper_experiment_config(config_path, project_root=root)
    stage_k = config.stage_k
    stage_o = config.stage_o
    stage_r = config.stage_r
    stage_k_smoke = bool(args.stage_k_engineering_smoke)
    stage_o_smoke = bool(args.stage_o_engineering_smoke)
    stage_r_smoke = bool(args.stage_r_engineering_smoke)
    localno_smoke = stage_k_smoke or stage_o_smoke or stage_r_smoke
    if stage_k_smoke and (not stage_k or stage_o or stage_r):
        raise ValueError("--stage-k-engineering-smoke requires the Stage K LocalNO config")
    if stage_o_smoke and not stage_o:
        raise ValueError("--stage-o-engineering-smoke requires the Stage O P3 config")
    if stage_r_smoke and not stage_r:
        raise ValueError("--stage-r-engineering-smoke requires the Stage R config")
    if sum((stage_k_smoke, stage_o_smoke, stage_r_smoke)) > 1:
        raise ValueError("Only one LocalNO engineering-smoke flag may be selected")
    if (stage_o or stage_r) and args.resume is not None:
        raise ValueError("Stage O/Stage R runs must start from the frozen initial state")
    if stage_k and args.smoke:
        raise ValueError("Stage K must not use the Stage F --smoke path")
    if stage_k and not localno_smoke and args.epochs is not None and args.epochs != 30:
        raise ValueError("Stage K/Stage O/Stage R formal pilot must run exactly 30 epochs")
    diagnostic_h1_mode = config.diagnostic_h1_mode
    no_h1_diagnostic = diagnostic_h1_mode == "no_h1"
    epochs = (
        int(args.epochs)
        if args.epochs is not None
        else int(config.values["scheduler"]["actual_default_epochs"])
    )
    validate_run_limits(
        smoke=args.smoke or localno_smoke,
        epochs=epochs,
        max_train_batches=args.max_train_batches,
        max_validation_batches=args.max_validation_batches,
    )
    if args.smoke and (args.max_train_batches is None or args.max_validation_batches is None):
        raise ValueError("Stage F smoke requires explicit train/validation batch limits")
    stage_g = not args.smoke and not stage_k
    controlled_paired_run = stage_g or stage_k
    formal_pilot = stage_g or (stage_k and not localno_smoke)
    if stage_g:
        if epochs != 30:
            raise ValueError("Stage G pilot must run exactly 30 epochs")
        if args.max_train_batches is not None or args.max_validation_batches is not None:
            raise ValueError("Stage G pilot must use all train and validation pairs")
        if args.initial_state is None or args.pair_order is None:
            raise ValueError("Stage G pilot requires --initial-state and --pair-order")
    if stage_k:
        expected_epochs = 2 if localno_smoke else 30
        if epochs != expected_epochs:
            raise ValueError(
                f"Stage K/Stage O/Stage R {'engineering smoke' if localno_smoke else 'pilot'} "
                f"must run exactly {expected_epochs} epochs"
            )
        if args.max_train_batches is not None or args.max_validation_batches is not None:
            raise ValueError("Stage K must use all train and validation pairs")
        if args.initial_state is None or args.pair_order is None:
            raise ValueError("Stage K requires --initial-state and --pair-order")
    max_train_batches = args.max_train_batches or 10**9
    max_validation_batches = args.max_validation_batches or 10**9
    gradient_accumulation = 1 if args.smoke else int(
        config.values["runtime"]["gradient_accumulation"]
    )
    warmup_epochs = min(
        epochs, int(config.values["scheduler"]["actual_default_warmup_epochs"])
    )
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    environment = frozen_environment(root)
    device = torch.device("cuda" if environment["cuda_available"] else "cpu")
    if controlled_paired_run and device.type != "cuda":
        raise RuntimeError("Controlled pilot requires CUDA; refusing silent CPU fallback")
    if controlled_paired_run and "RTX 5070" not in str(environment["cuda_device"]):
        raise RuntimeError(
            f"Controlled pilot expected RTX 5070, found {environment['cuda_device']!r}"
        )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    seed = int(config.values["runtime"]["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    config_checksum = sha256_file(config_path)
    initial_state_path = None
    pair_order_path = None
    shared_initial_state_sha256 = None
    pair_order_sha256 = None
    epoch_pair_orders = None
    if controlled_paired_run:
        assert args.initial_state is not None and args.pair_order is not None
        initial_state_path = (
            args.initial_state
            if args.initial_state.is_absolute()
            else root / args.initial_state
        )
        pair_order_path = (
            args.pair_order if args.pair_order.is_absolute() else root / args.pair_order
        )
        _, epoch_pair_orders = load_epoch_pair_order(pair_order_path)
        if len(epoch_pair_orders) < epochs:
            raise ValueError("Frozen pair order does not cover all requested epochs")
        pair_order_sha256 = sha256_file(pair_order_path)
        if stage_k and pair_order_sha256 != EXPECTED_STAGE_G_PAIR_ORDER_SHA256:
            raise ValueError("Stage K frozen Stage G pair-order checksum changed")
        shared_initial_state = torch.load(
            initial_state_path, map_location="cpu", weights_only=True
        )
        if not isinstance(shared_initial_state, Mapping):
            raise ValueError("Initial-state file must contain a model state_dict")
        shared_initial_state_sha256 = tensor_state_sha256(shared_initial_state)
    else:
        shared_initial_state = None

    startup = {
        "experiment_name": args.experiment_name,
        "mode": config.mode,
        "smoke": args.smoke,
        "stage_g_pilot": stage_g,
        "stage_k": stage_k,
        "stage_k_engineering_smoke": stage_k_smoke,
        "stage_o": stage_o,
        "stage_o_engineering_smoke": stage_o_smoke,
        "stage_r": stage_r,
        "stage_r_engineering_smoke": stage_r_smoke,
        "prediction_mode": config.prediction_mode,
        "formal_pilot": formal_pilot,
        "model_architecture": config.architecture,
        "stage_i_diagnostic": diagnostic_h1_mode is not None,
        "diagnostic_h1_mode": diagnostic_h1_mode,
        "reproduction_level": (
            "diagnostic_extension"
            if diagnostic_h1_mode is not None
            else config.values["reproduction_metadata"]["reproduction_level"]
        ),
        "paper_faithful_full": (
            False if diagnostic_h1_mode is not None else None
        ),
        "comparison_parent": (
            "stage_g_paper_adapted_full"
            if diagnostic_h1_mode is not None
            else config.values["reproduction_metadata"].get("comparison_parent")
        ),
        "extension_reason": (
            "isolate_H1_contribution_under_spherical_grid_adaptation"
            if no_h1_diagnostic
            else config.values["reproduction_metadata"].get("extension_reason")
            if diagnostic_h1_mode is not None
            else None
        ),
        "scientific_result": False,
        "epochs": epochs,
        "max_train_batches": max_train_batches,
        "max_validation_batches": max_validation_batches,
        "gradient_accumulation": gradient_accumulation,
        "device": str(device),
        "config_checksum": config_checksum,
        "dataset_checksum": config.values["provenance"]["dataset"]["sha256"],
        "preprocessing_checksum": config.values["provenance"]["preprocessing"]["sha256"],
        "prior_artifact_checksums": {
            name: record["sha256"]
            for name, record in config.values["provenance"]["artifacts"].items()
        },
        "shared_initial_state_sha256": shared_initial_state_sha256,
        "pair_order_sha256": pair_order_sha256,
        "exact_adapted_blocked": config.values["reproduction_metadata"],
        **environment,
    }
    print(json.dumps(startup, indent=2, sort_keys=True), flush=True)
    (output_dir / "environment.json").write_text(
        json.dumps(_json_value(startup), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    resolved = config.as_dict()
    resolved["experiment_name"] = args.experiment_name
    resolved["output_dir"] = str(output_dir)
    resolved["resolved_runtime"] = {
        "epochs": epochs,
        "max_train_batches": max_train_batches,
        "max_validation_batches": max_validation_batches,
        "gradient_accumulation": gradient_accumulation,
        "warmup_epochs": warmup_epochs,
        "device": str(device),
        "smoke": args.smoke,
        "stage_g_pilot": stage_g,
        "stage_k": stage_k,
        "stage_k_engineering_smoke": stage_k_smoke,
        "stage_o": stage_o,
        "stage_o_engineering_smoke": stage_o_smoke,
        "stage_r": stage_r,
        "stage_r_engineering_smoke": stage_r_smoke,
        "prediction_mode": config.prediction_mode,
        "formal_pilot": formal_pilot,
        "model_architecture": config.architecture,
        "stage_i_diagnostic": diagnostic_h1_mode is not None,
        "diagnostic_h1_mode": diagnostic_h1_mode,
        "initial_state": str(initial_state_path) if initial_state_path else None,
        "shared_initial_state_sha256": shared_initial_state_sha256,
        "pair_order": str(pair_order_path) if pair_order_path else None,
        "pair_order_sha256": pair_order_sha256,
    }
    (output_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )

    data_started = time.perf_counter()
    protocol = PaperReduced100Protocol.from_yaml(
        root / "configs/data/paper_reduced100.yaml", project_root=root
    )
    datasets = protocol.make_datasets()
    train_dataset = GRMHDNextStepDataset(datasets["train"])
    validation_dataset = GRMHDNextStepDataset(datasets["validation"])
    effective_pin_memory = bool(config.values["runtime"]["pin_memory"] and device.type == "cuda")
    generator = torch.Generator().manual_seed(seed)

    def make_train_loader(epoch: int) -> DataLoader:
        if controlled_paired_run:
            assert epoch_pair_orders is not None
            return DataLoader(
                train_dataset,
                batch_size=int(config.values["runtime"]["batch_size"]),
                shuffle=False,
                sampler=epoch_pair_orders[epoch],
                num_workers=int(config.values["runtime"]["num_workers"]),
                pin_memory=effective_pin_memory,
            )
        return DataLoader(
            train_dataset,
            batch_size=int(config.values["runtime"]["batch_size"]),
            shuffle=True,
            generator=generator,
            num_workers=int(config.values["runtime"]["num_workers"]),
            pin_memory=effective_pin_memory,
        )

    train_loader = make_train_loader(0)
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=int(config.values["runtime"]["num_workers"]),
        pin_memory=effective_pin_memory,
    )
    validation_probe_batch = next(iter(validation_loader))
    data_setup_seconds = time.perf_counter() - data_started

    processor = PaperDataProcessor.from_config(config).to(device)
    p3_train_envelope = load_p3_train_envelope(root, config)
    model = build_paper_model(config).to(device)
    if controlled_paired_run:
        assert shared_initial_state is not None
        model.load_state_dict(shared_initial_state, strict=True)
    initial_state_hash = model_tensor_state_sha256(model)
    if controlled_paired_run and initial_state_hash != shared_initial_state_sha256:
        raise ValueError("Model did not strictly load the frozen initial state")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = build_paper_optimizer(
        model,
        learning_rate=config.values["optimizer"]["learning_rate"],
        weight_decay=config.values["optimizer"]["weight_decay"],
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_epochs=epochs,
        warmup_epochs=warmup_epochs,
        min_learning_rate=config.values["scheduler"]["min_learning_rate"],
    )
    loss_adapter = PaperTrainerLossAdapter(build_paper_training_loss(config)).to(device)
    trainer = PaperTrainerAdapter(
        model=model,
        data_processor=processor,
        loss=loss_adapter,
        optimizer=optimizer,
        n_epochs=epochs,
        device=device,
        mixed_precision=False,
    )
    start_epoch = 0
    resume_optimizer_step = 0
    if args.resume is not None:
        resume_dir = args.resume if args.resume.is_absolute() else root / args.resume
        loaded = load_paper_checkpoint(
            resume_dir,
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_config_checksum=config_checksum,
        )
        start_epoch = loaded.epoch
        if controlled_paired_run:
            pairing_metadata = loaded.metadata.get("stage_k" if stage_k else "stage_g")
            if not isinstance(pairing_metadata, Mapping):
                raise ValueError("Controlled resume checkpoint lacks pairing metadata")
            if (
                pairing_metadata.get(
                    "initial_state_sha256" if stage_k else "shared_initial_state_sha256"
                )
                != shared_initial_state_sha256
            ):
                raise ValueError("Controlled resume initial-state hash mismatch")
            if pairing_metadata.get("pair_order_sha256") != pair_order_sha256:
                raise ValueError("Controlled resume pair-order hash mismatch")
            resume_optimizer_step = int(pairing_metadata["optimizer_step"])
        trainer.set_epoch(start_epoch)
    if start_epoch >= epochs:
        raise ValueError("Resume checkpoint already reached the requested epoch count")
    startup_seconds = time.perf_counter() - main_started

    startup.update(
        {
            "parameter_count": parameter_count,
            "initial_state_hash": initial_state_hash,
            "train_pairs": len(train_dataset),
            "validation_pairs": len(validation_dataset),
            "data_setup_seconds": data_setup_seconds,
            "pin_memory_effective": effective_pin_memory,
        }
    )
    print(json.dumps({"resolved_startup": startup}, indent=2, sort_keys=True), flush=True)

    train_rows: list[dict[str, Any]] = []
    optimizer_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []
    epoch_summaries: list[dict[str, Any]] = []
    phase_seconds = {
        "data_load": 0.0,
        "forward_loss": 0.0,
        "backward": 0.0,
        "optimizer": 0.0,
        "validation": 0.0,
    }
    best_value = math.inf
    best_epoch = None
    if controlled_paired_run and start_epoch:
        for validation_path in sorted(output_dir.glob("validation_epoch_*.json")):
            saved = json.loads(validation_path.read_text(encoding="utf-8"))
            saved_epoch = int(saved["training"]["epoch"]) + 1
            if saved_epoch > start_epoch:
                continue
            value = float(
                saved["training"][
                    "validation_normalized_arithmetic_average_relative_l2"
                ]
            )
            if value < best_value:
                best_value = value
                best_epoch = saved_epoch
    nonfinite_count = 0
    optimizer_updates = resume_optimizer_step
    clipping_count = 0
    full_h1_ratios: list[float] = []
    full_h1_gradient_ratios: list[float] = []
    diagnostic_upstream_h1_values: list[float] = []
    diagnostic_unit_index_h1_values: list[float] = []
    current_to_selected_h1_ratios: list[float] = []
    unit_index_to_selected_h1_ratios: list[float] = []
    training_started = time.perf_counter()
    model_hash_before_training = model_tensor_state_sha256(model)
    epoch_peak_allocated_mib: list[float] = []
    epoch_peak_reserved_mib: list[float] = []

    for epoch in range(start_epoch, epochs):
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        trainer.set_epoch(epoch)
        model.train()
        processor.train()
        train_loader = make_train_loader(epoch)
        iterator = iter(train_loader)
        epoch_batch_count = min(max_train_batches, len(train_loader))
        if controlled_paired_run and epoch_batch_count != 79:
            raise ValueError("Controlled run must consume exactly 79 train pairs per epoch")
        epoch_train_row_start = len(train_rows)
        epoch_optimizer_row_start = len(optimizer_rows)
        epoch_clipping_start = clipping_count
        component_base_accumulator: list[torch.Tensor | None] | None = None
        component_h1_accumulator: list[torch.Tensor | None] | None = None
        component_other_accumulator: list[torch.Tensor | None] | None = None
        observed_pair_order: list[int] = []
        final_accumulation_count = 0
        for batch_index in range(epoch_batch_count):
            data_t0 = time.perf_counter()
            batch = next(iterator)
            synchronize(device)
            phase_seconds["data_load"] += time.perf_counter() - data_t0
            clear_gradients = batch_index % gradient_accumulation == 0
            accumulation_count = accumulation_count_for_batch(
                batch_index,
                total_batches=epoch_batch_count,
                accumulation=gradient_accumulation,
            )
            forward_t0 = time.perf_counter()
            result = trainer.compute_batch(
                batch,
                batch_index=batch_index,
                clear_gradients=clear_gradients,
            )
            synchronize(device)
            phase_seconds["forward_loss"] += time.perf_counter() - forward_t0
            if not torch.isfinite(result.loss):
                nonfinite_count += 1
                raise FloatingPointError("Nonfinite Stage F training loss")
            parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            h1_grad_norm = None
            base_grad_norm = None
            h1_gradient_ratio = None
            diagnostic_upstream = None
            diagnostic_unit_index = None
            current_to_selected_h1_ratio = None
            unit_index_to_selected_h1_ratio = None
            if config.mode == "full":
                full_result = loss_adapter.last_result
                if full_result is None:
                    raise RuntimeError("Full loss did not expose component tensors")
                base_gradients = component_gradient_tensors(
                    full_result.base_fidelity_weighted, parameters
                )
                h1_gradients = component_gradient_tensors(
                    full_result.h1_weighted, parameters
                )
                other_weighted = (
                    full_result.roi_weighted
                    + full_result.bounds_weighted
                    + full_result.envelope_weighted
                    + full_result.dissipation_weighted
                )
                other_gradients = component_gradient_tensors(
                    other_weighted, parameters
                )
                base_grad_norm = gradient_tensor_norm(base_gradients)
                h1_grad_norm = gradient_tensor_norm(h1_gradients)
                component_base_accumulator = add_scaled_gradients(
                    component_base_accumulator,
                    base_gradients,
                    scale=1.0 / accumulation_count,
                )
                component_h1_accumulator = add_scaled_gradients(
                    component_h1_accumulator,
                    h1_gradients,
                    scale=1.0 / accumulation_count,
                )
                component_other_accumulator = add_scaled_gradients(
                    component_other_accumulator,
                    other_gradients,
                    scale=1.0 / accumulation_count,
                )
                if diagnostic_h1_mode is not None:
                    diagnostic_upstream = full_result.diagnostics.get(
                        "diagnostic_current_upstream_h1_raw"
                    )
                    if not torch.is_tensor(diagnostic_upstream):
                        raise RuntimeError(
                            "Stage I diagnostic omitted detached current-upstream H1"
                        )
                    diagnostic_upstream_value = float(
                        diagnostic_upstream.detach().cpu()
                    )
                    diagnostic_upstream_h1_values.append(
                        diagnostic_upstream_value
                    )
                    diagnostic_unit_index = full_result.diagnostics.get(
                        "diagnostic_unit_index_h1_raw"
                    )
                    if not torch.is_tensor(diagnostic_unit_index):
                        raise RuntimeError(
                            "Stage I diagnostic omitted detached unit-index H1"
                        )
                    diagnostic_unit_index_value = float(
                        diagnostic_unit_index.detach().cpu()
                    )
                    diagnostic_unit_index_h1_values.append(
                        diagnostic_unit_index_value
                    )
                    if not no_h1_diagnostic:
                        selected_raw_value = float(
                            full_result.h1_raw.detach().cpu()
                        )
                        current_to_selected_h1_ratio = (
                            diagnostic_upstream_value
                            / max(selected_raw_value, 1e-30)
                        )
                        current_to_selected_h1_ratios.append(
                            current_to_selected_h1_ratio
                        )
                        if diagnostic_h1_mode == (
                            "stored_coordinate_volume_proxy"
                        ):
                            unit_index_to_selected_h1_ratio = (
                                diagnostic_unit_index_value
                                / max(selected_raw_value, 1e-30)
                            )
                            unit_index_to_selected_h1_ratios.append(
                                unit_index_to_selected_h1_ratio
                            )
                if no_h1_diagnostic:
                    h1_value_ratio = None
                else:
                    h1_value_ratio = float(
                        full_result.h1_weighted.detach().cpu()
                        / max(
                            float(
                                full_result.base_fidelity_weighted.detach().cpu()
                            ),
                            1e-12,
                        )
                    )
                    full_h1_ratios.append(h1_value_ratio)
            else:
                h1_value_ratio = None

            backward_t0 = time.perf_counter()
            (result.loss / accumulation_count).backward()
            synchronize(device)
            phase_seconds["backward"] += time.perf_counter() - backward_t0
            gradient_before = parameter_gradient_norm(parameters)
            if not np.isfinite(gradient_before):
                nonfinite_count += 1
                raise FloatingPointError("Nonfinite Stage F gradient")
            update_now = is_accumulation_step(
                batch_index,
                total_batches=epoch_batch_count,
                accumulation=gradient_accumulation,
            )
            clipping_triggered = False
            gradient_after = gradient_before
            optimizer_seconds = 0.0
            update_norm = None
            step_base_gradient_norm = None
            step_h1_gradient_norm = None
            step_other_gradient_norm = None
            base_h1_gradient_cosine = None
            effective_base_gradient_projection = None
            effective_h1_gradient_projection = None
            effective_base_gradient_norm = None
            effective_h1_gradient_norm = None
            clip_scale = None
            if update_now:
                final_accumulation_count = accumulation_count
                optimizer_t0 = time.perf_counter()
                threshold = float(config.values["runtime"]["gradient_clip_norm"])
                if config.mode == "full":
                    assert component_base_accumulator is not None
                    assert component_h1_accumulator is not None
                    assert component_other_accumulator is not None
                    step_base_gradient_norm = gradient_tensor_norm(
                        component_base_accumulator
                    )
                    step_h1_gradient_norm = gradient_tensor_norm(
                        component_h1_accumulator
                    )
                    step_other_gradient_norm = gradient_tensor_norm(
                        component_other_accumulator
                    )
                    base_h1_gradient_cosine = gradient_tensor_cosine(
                        component_base_accumulator,
                        component_h1_accumulator,
                    )
                    h1_gradient_ratio = selected_h1_gradient_ratio(
                        base_gradient_norm=step_base_gradient_norm,
                        selected_h1_gradient_norm=step_h1_gradient_norm,
                        diagnostic_h1_mode=diagnostic_h1_mode,
                    )
                    if not no_h1_diagnostic:
                        assert h1_gradient_ratio is not None
                        full_h1_gradient_ratios.append(h1_gradient_ratio)
                clip_scale = min(
                    1.0,
                    threshold / max(float(gradient_before), 1e-30),
                )
                if step_base_gradient_norm is not None:
                    effective_base_gradient_norm = (
                        clip_scale * step_base_gradient_norm
                    )
                if step_h1_gradient_norm is not None:
                    effective_h1_gradient_norm = (
                        clip_scale * step_h1_gradient_norm
                    )
                if config.mode == "full":
                    total_gradients = [
                        (
                            None
                            if parameter.grad is None
                            else parameter.grad.detach()
                        )
                        for parameter in parameters
                    ]
                    base_projection = gradient_tensor_projection(
                        component_base_accumulator,
                        total_gradients,
                    )
                    h1_projection = gradient_tensor_projection(
                        component_h1_accumulator,
                        total_gradients,
                    )
                    effective_base_gradient_projection = (
                        None
                        if base_projection is None
                        else clip_scale * base_projection
                    )
                    effective_h1_gradient_projection = (
                        None
                        if h1_projection is None
                        else clip_scale * h1_projection
                    )
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=threshold)
                gradient_after = parameter_gradient_norm(parameters)
                clipping_triggered = gradient_before > threshold
                parameters_before = [
                    parameter.detach().clone() for parameter in parameters
                ]
                optimizer.step()
                synchronize(device)
                update_norm = parameter_update_norm(parameters_before, parameters)
                optimizer_seconds = time.perf_counter() - optimizer_t0
                phase_seconds["optimizer"] += optimizer_seconds
                optimizer_updates += 1
                clipping_count += int(clipping_triggered)
                if not all(torch.isfinite(parameter).all() for parameter in parameters):
                    nonfinite_count += 1
                    raise FloatingPointError(
                        "Optimizer produced nonfinite model parameters"
                    )
                optimizer_row = {
                    "epoch": epoch,
                    "optimizer_step_in_epoch": len(optimizer_rows)
                    - epoch_optimizer_row_start,
                    "global_optimizer_step": optimizer_updates,
                    "accumulation_count": accumulation_count,
                    "base_only_gradient_norm": step_base_gradient_norm,
                    "h1_only_gradient_norm": step_h1_gradient_norm,
                    "selected_h1_gradient_norm": step_h1_gradient_norm,
                    "other_prior_gradient_norm": step_other_gradient_norm,
                    "h1_to_base_gradient_ratio": h1_gradient_ratio,
                    "base_h1_gradient_cosine": base_h1_gradient_cosine,
                    "total_gradient_norm_before_clip": gradient_before,
                    "total_gradient_norm_after_clip": gradient_after,
                    "clipping_triggered": clipping_triggered,
                    "clip_scale": clip_scale,
                    "effective_base_gradient_projection": (
                        effective_base_gradient_projection
                    ),
                    "effective_h1_gradient_projection": (
                        effective_h1_gradient_projection
                    ),
                    "effective_base_gradient_norm": (
                        effective_base_gradient_norm
                    ),
                    "effective_h1_gradient_norm": (
                        effective_h1_gradient_norm
                    ),
                    "parameter_update_norm": update_norm,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "nonfinite": False,
                }
                optimizer_rows.append(optimizer_row)
                print(
                    json.dumps(
                        _json_value({"optimizer": optimizer_row}), sort_keys=True
                    ),
                    flush=True,
                )
                component_base_accumulator = None
                component_h1_accumulator = None
                component_other_accumulator = None
            fields = processor.last_batch
            if fields is None:
                raise RuntimeError("PaperDataProcessor lost the current batch")
            source_snapshot = int(fields["source_snapshot_index"][0])
            pair_index = source_snapshot - 11
            observed_pair_order.append(pair_index)
            normalized_prediction = result.normalized_prediction.detach()
            predicted_residual = (
                None
                if result.predicted_residual is None
                else result.predicted_residual.detach()
            )
            true_residual = fields["normalized_residual_target"].detach()
            if stage_r and predicted_residual is None:
                raise RuntimeError("Stage R batch lost the raw predicted residual")
            decoded_prediction = processor.preprocessor.decode(
                normalized_prediction, channel_axis=1
            )
            prediction_nonfinite_count = int(
                (~torch.isfinite(normalized_prediction)).sum().detach().cpu()
                + (~torch.isfinite(decoded_prediction)).sum().detach().cpu()
            )
            if prediction_nonfinite_count:
                nonfinite_count += prediction_nonfinite_count
                raise FloatingPointError("Training prediction or decoded state is nonfinite")
            row = {
                "epoch": epoch,
                "microbatch": batch_index,
                "pair_index": pair_index,
                "source_snapshot": source_snapshot,
                "target_snapshot": int(fields["target_snapshot_index"][0]),
                "mode": config.mode,
                "diagnostic_h1_mode": diagnostic_h1_mode,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "accumulation_count": accumulation_count,
                "total_gradient_norm_unclipped": gradient_before,
                "total_gradient_norm_clipped": gradient_after,
                "clipping_triggered": clipping_triggered,
                "optimizer_update": update_now,
                "parameter_update_norm": update_norm,
                "optimizer_seconds": optimizer_seconds,
                "loss_finite": True,
                "nonfinite": False,
                "nonfinite_count": prediction_nonfinite_count,
                "total_loss": float(result.loss.detach().cpu()),
                "input_norm": float(fields["model_input"].detach().float().norm().cpu()),
                "transformed_input_norm": float(
                    fields["normalized_input"].detach().float().norm().cpu()
                ),
                "target_norm": float(
                    fields["normalized_target"].detach().float().norm().cpu()
                ),
                "prediction_norm": float(normalized_prediction.float().norm().cpu()),
                "prediction_mode": config.prediction_mode,
                "true_residual_norm": float(true_residual.float().norm().cpu()),
                "predicted_residual_norm": (
                    None
                    if predicted_residual is None
                    else float(predicted_residual.float().norm().cpu())
                ),
                "residual_over_true_residual": (
                    None
                    if predicted_residual is None
                    else float(
                        predicted_residual.float().norm().cpu()
                        / max(float(true_residual.float().norm().cpu()), 1.0e-30)
                    )
                ),
                "residual_cosine": (
                    None
                    if predicted_residual is None
                    else float(
                        torch.nn.functional.cosine_similarity(
                            predicted_residual.float().reshape(1, -1),
                            true_residual.float().reshape(1, -1),
                        ).cpu()
                    )
                ),
                "residual_sign_agreement": (
                    None
                    if predicted_residual is None
                    else float(
                        (
                            torch.signbit(predicted_residual)
                            == torch.signbit(true_residual)
                        )
                        .float()
                        .mean()
                        .cpu()
                    )
                ),
                "decoded_rho_minimum": float(decoded_prediction[:, 3].min().cpu()),
                "decoded_press_minimum": float(decoded_prediction[:, 4].min().cpu()),
                **flatten_record(result.loss_log, "loss"),
            }
            if config.uses_p3:
                saturation_limit = (
                    processor.preprocessor.gamma
                    * processor.preprocessor.inverse_clamp_fraction
                )
                for channel_index, channel_name in enumerate(CHANNELS):
                    target_channel = fields["normalized_target"][:, channel_index]
                    prediction_channel = normalized_prediction[:, channel_index]
                    row[f"plain_squared_loss_{channel_name}"] = float(
                        (prediction_channel - target_channel)
                        .square()
                        .mean()
                        .cpu()
                    )
                    row[f"target_transformed_min_{channel_name}"] = float(
                        target_channel.min().cpu()
                    )
                    row[f"target_transformed_max_{channel_name}"] = float(
                        target_channel.max().cpu()
                    )
                    row[f"prediction_transformed_min_{channel_name}"] = float(
                        prediction_channel.min().cpu()
                    )
                    row[f"prediction_transformed_max_{channel_name}"] = float(
                        prediction_channel.max().cpu()
                    )
                    row[f"target_saturation_{channel_name}"] = float(
                        (torch.abs(target_channel) > saturation_limit)
                        .float()
                        .mean()
                        .cpu()
                    )
                    row[f"model_saturation_{channel_name}"] = float(
                        (torch.abs(prediction_channel) > saturation_limit)
                        .float()
                        .mean()
                        .cpu()
                    )
                    row[f"decoded_prediction_min_{channel_name}"] = float(
                        decoded_prediction[:, channel_index].min().cpu()
                    )
                    row[f"decoded_prediction_max_{channel_name}"] = float(
                        decoded_prediction[:, channel_index].max().cpu()
                    )
                    if p3_train_envelope is not None:
                        ood = ood_diagnostics(
                            prediction_channel.detach().cpu().numpy(),
                            p3_train_envelope["channels"][channel_name]["normalized"],
                        )
                        row[f"prediction_q_ood_{channel_name}"] = ood[
                            "fraction_outside_train_q001_q999"
                        ]
                        row[f"prediction_range_ood_{channel_name}"] = ood[
                            "fraction_outside_train_min_max"
                        ]
                    if predicted_residual is not None:
                        predicted_channel_residual = predicted_residual[:, channel_index]
                        target_channel_residual = true_residual[:, channel_index]
                        row[f"residual_squared_loss_{channel_name}"] = float(
                            (predicted_channel_residual - target_channel_residual)
                            .square()
                            .mean()
                            .cpu()
                        )
                row["normalized_q_ood_fraction"] = float(
                    np.mean(
                        [row[f"prediction_q_ood_{name}"] for name in CHANNELS]
                    )
                )
                row["normalized_train_range_exceedance_fraction"] = float(
                    np.mean(
                        [row[f"prediction_range_ood_{name}"] for name in CHANNELS]
                    )
                )
            if config.mode == "full":
                assert full_result is not None
                selected_per_channel = full_result.diagnostics.get(
                    "selected_h1_per_channel"
                )
                selected_per_direction = full_result.diagnostics.get(
                    "selected_h1_per_direction"
                )
                selected_per_shell = full_result.diagnostics.get(
                    "selected_h1_per_shell"
                )
                if diagnostic_h1_mode is not None and (
                    not torch.is_tensor(selected_per_channel)
                    or not isinstance(selected_per_direction, Mapping)
                    or not torch.is_tensor(selected_per_shell)
                ):
                    raise RuntimeError(
                        "Stage I selected H1 decomposition is missing"
                    )
                row.update(
                    {
                        "base_fidelity_raw": float(
                            full_result.base_fidelity_raw.detach().cpu()
                        ),
                        "base_fidelity_weighted": float(
                            full_result.base_fidelity_weighted.detach().cpu()
                        ),
                        "selected_h1_raw": float(
                            full_result.h1_raw.detach().cpu()
                        ),
                        "selected_h1_weighted": float(
                            full_result.h1_weighted.detach().cpu()
                        ),
                        "diagnostic_current_upstream_h1_raw": (
                            float(diagnostic_upstream.detach().cpu())
                            if torch.is_tensor(diagnostic_upstream)
                            else None
                        ),
                        "diagnostic_unit_index_h1_raw": (
                            float(diagnostic_unit_index.detach().cpu())
                            if torch.is_tensor(diagnostic_unit_index)
                            else None
                        ),
                        "current_to_selected_h1_raw_ratio": (
                            current_to_selected_h1_ratio
                        ),
                        "unit_index_to_selected_h1_raw_ratio": (
                            unit_index_to_selected_h1_ratio
                        ),
                        "other_prior_weighted": float(
                            (
                                full_result.roi_weighted
                                + full_result.bounds_weighted
                                + full_result.envelope_weighted
                                + full_result.dissipation_weighted
                            )
                            .detach()
                            .cpu()
                        ),
                        "roi_ramp": float(result.context.epoch) / 375
                        if result.context.epoch < 375
                        else 1.0,
                        "h1_to_base_value_ratio": h1_value_ratio,
                        "h1_gradient_norm": h1_grad_norm,
                        "selected_h1_gradient_norm": h1_grad_norm,
                        "base_gradient_norm": base_grad_norm,
                        "h1_to_base_gradient_ratio_at_optimizer_step": (
                            h1_gradient_ratio if update_now else None
                        ),
                    }
                )
                if torch.is_tensor(selected_per_channel):
                    for channel_index, channel_name in enumerate(CHANNELS):
                        row[f"selected_h1_channel_{channel_name}"] = float(
                            selected_per_channel[channel_index].detach().cpu()
                        )
                if isinstance(selected_per_direction, Mapping):
                    for direction in ("phi", "theta", "r"):
                        value = selected_per_direction[direction]
                        row[f"selected_h1_direction_{direction}"] = float(
                            value.detach().cpu()
                        )
                if torch.is_tensor(selected_per_shell):
                    shell_values = [
                        float(value.detach().cpu())
                        for value in selected_per_shell
                    ]
                    for shell_index, value in enumerate(shell_values):
                        row[f"selected_h1_shell_{shell_index}"] = value
                    row["selected_h1_shell_inner_0_1"] = sum(
                        shell_values[0:2]
                    )
                    row["selected_h1_shell_middle_2_5"] = sum(
                        shell_values[2:6]
                    )
                    row["selected_h1_shell_outer_6_7"] = sum(
                        shell_values[6:8]
                    )
            train_rows.append(row)
        if controlled_paired_run:
            assert epoch_pair_orders is not None
            if observed_pair_order != epoch_pair_orders[epoch]:
                raise ValueError(f"Controlled epoch {epoch} pair order changed")
            if len(optimizer_rows) - epoch_optimizer_row_start != 20:
                raise ValueError(f"Controlled epoch {epoch} did not produce 20 optimizer steps")
            if final_accumulation_count != 3:
                raise ValueError(f"Controlled epoch {epoch} final accumulation was not three")
        scheduler.step()

        validation_t0 = time.perf_counter()
        validation = evaluate_one_step(
            model=model,
            processor=processor,
            loss_adapter=loss_adapter,
            loader=validation_loader,
            max_batches=min(max_validation_batches, len(validation_loader)),
            epoch=epoch,
        )
        synchronize(device)
        phase_seconds["validation"] += time.perf_counter() - validation_t0
        normalized_metrics = validation["oracle_aware"]["metrics"]["E_norm"]
        normalized_value = normalized_metrics["arithmetic_average"]
        validation_row = {
            "epoch": epoch,
            "mode": config.mode,
            "diagnostic_h1_mode": diagnostic_h1_mode,
            "loss": validation["loss"],
            "normalized_arithmetic_average_relative_l2": normalized_value,
            "normalized_global_relative_l2": normalized_metrics[
                "global_relative_l2"
            ],
            "normalized_per_channel_relative_l2": normalized_metrics[
                "per_channel"
            ],
            "model_to_oracle_global_relative_l2": validation["oracle_aware"]["metrics"][
                "E_model_oracle"
            ]["global_relative_l2"],
            "model_to_raw_global_relative_l2": validation["oracle_aware"]["metrics"][
                "E_model_raw"
            ]["global_relative_l2"],
            "oracle_floor_global_relative_l2": validation["oracle_aware"]["metrics"][
                "E_oracle_raw"
            ]["global_relative_l2"],
            "evaluation_bound_clamp_fraction": validation[
                "evaluation_bound_clamp_fraction"
            ],
            "rho_press_positive": validation["rho_press_positive"],
            "finite": validation["finite"],
            "seconds": validation["seconds"],
            "prediction_mode": config.prediction_mode,
            "residual_metrics": validation["residual_metrics"],
        }
        validation_rows.append(validation_row)
        print(json.dumps({"validation": validation_row}, sort_keys=True), flush=True)
        epoch_train_rows = train_rows[epoch_train_row_start:]
        epoch_optimizer_rows = optimizer_rows[epoch_optimizer_row_start:]
        epoch_peak_allocated = (
            torch.cuda.max_memory_allocated(device) / 2**20
            if device.type == "cuda"
            else 0.0
        )
        epoch_peak_reserved = (
            torch.cuda.max_memory_reserved(device) / 2**20
            if device.type == "cuda"
            else 0.0
        )
        epoch_peak_allocated_mib.append(epoch_peak_allocated)
        epoch_peak_reserved_mib.append(epoch_peak_reserved)
        epoch_summary = {
            "epoch": epoch,
            "microbatches_seen": len(epoch_train_rows),
            "optimizer_steps": len(epoch_optimizer_rows),
            "final_accumulation_count": final_accumulation_count,
            "train_total": numeric_summary(
                row["total_loss"] for row in epoch_train_rows
            ),
            "train_predicted_residual_norm": numeric_summary(
                row["predicted_residual_norm"]
                for row in epoch_train_rows
                if row.get("predicted_residual_norm") is not None
            ),
            "train_true_residual_norm": numeric_summary(
                row["true_residual_norm"] for row in epoch_train_rows
            ),
            "train_residual_over_true_residual": numeric_summary(
                row["residual_over_true_residual"]
                for row in epoch_train_rows
                if row.get("residual_over_true_residual") is not None
            ),
            "train_residual_cosine": numeric_summary(
                row["residual_cosine"]
                for row in epoch_train_rows
                if row.get("residual_cosine") is not None
            ),
            "train_base_weighted": numeric_summary(
                row["base_fidelity_weighted"]
                for row in epoch_train_rows
                if row.get("base_fidelity_weighted") is not None
            ),
            "train_other_prior_weighted": numeric_summary(
                row["other_prior_weighted"]
                for row in epoch_train_rows
                if row.get("other_prior_weighted") is not None
            ),
            "train_selected_h1_raw": numeric_summary(
                row["selected_h1_raw"]
                for row in epoch_train_rows
                if row.get("selected_h1_raw") is not None
            ),
            "train_selected_h1_weighted": numeric_summary(
                row["selected_h1_weighted"]
                for row in epoch_train_rows
                if row.get("selected_h1_weighted") is not None
            ),
            "diagnostic_current_upstream_h1_raw": numeric_summary(
                row["diagnostic_current_upstream_h1_raw"]
                for row in epoch_train_rows
                if row.get("diagnostic_current_upstream_h1_raw") is not None
            ),
            "diagnostic_unit_index_h1_raw": numeric_summary(
                row["diagnostic_unit_index_h1_raw"]
                for row in epoch_train_rows
                if row.get("diagnostic_unit_index_h1_raw") is not None
            ),
            "current_to_selected_h1_raw_ratio": numeric_summary(
                row["current_to_selected_h1_raw_ratio"]
                for row in epoch_train_rows
                if row.get("current_to_selected_h1_raw_ratio") is not None
            ),
            "unit_index_to_selected_h1_raw_ratio": numeric_summary(
                row["unit_index_to_selected_h1_raw_ratio"]
                for row in epoch_train_rows
                if row.get("unit_index_to_selected_h1_raw_ratio") is not None
            ),
            "selected_h1_per_direction": {
                direction: numeric_summary(
                    row[f"selected_h1_direction_{direction}"]
                    for row in epoch_train_rows
                    if row.get(f"selected_h1_direction_{direction}")
                    is not None
                )
                for direction in ("phi", "theta", "r")
            },
            "selected_h1_per_channel": {
                channel: numeric_summary(
                    row[f"selected_h1_channel_{channel}"]
                    for row in epoch_train_rows
                    if row.get(f"selected_h1_channel_{channel}") is not None
                )
                for channel in CHANNELS
            },
            "selected_h1_per_shell": {
                str(shell): numeric_summary(
                    row[f"selected_h1_shell_{shell}"]
                    for row in epoch_train_rows
                    if row.get(f"selected_h1_shell_{shell}") is not None
                )
                for shell in range(8)
            },
            "selected_h1_shell_regions": {
                region: numeric_summary(
                    row[f"selected_h1_shell_{region}"]
                    for row in epoch_train_rows
                    if row.get(f"selected_h1_shell_{region}") is not None
                )
                for region in ("inner_0_1", "middle_2_5", "outer_6_7")
            },
            "h1_to_base_value_ratio": (
                None
                if no_h1_diagnostic
                else numeric_summary(
                    row["h1_to_base_value_ratio"]
                    for row in epoch_train_rows
                    if row.get("h1_to_base_value_ratio") is not None
                )
            ),
            "h1_to_base_gradient_ratio": (
                None
                if no_h1_diagnostic
                else numeric_summary(
                    row["h1_to_base_gradient_ratio"]
                    for row in epoch_optimizer_rows
                    if row.get("h1_to_base_gradient_ratio") is not None
                )
            ),
            "h1_to_base_gradient_ratio_semantics": (
                "not_applicable"
                if no_h1_diagnostic
                else "selected_h1_to_base"
            ),
            "base_h1_gradient_cosine": numeric_summary(
                row["base_h1_gradient_cosine"]
                for row in epoch_optimizer_rows
                if row.get("base_h1_gradient_cosine") is not None
            ),
            "clip_scale": numeric_summary(
                row["clip_scale"]
                for row in epoch_optimizer_rows
                if row.get("clip_scale") is not None
            ),
            "effective_base_gradient_projection": numeric_summary(
                row["effective_base_gradient_projection"]
                for row in epoch_optimizer_rows
                if row.get("effective_base_gradient_projection") is not None
            ),
            "effective_h1_gradient_projection": numeric_summary(
                row["effective_h1_gradient_projection"]
                for row in epoch_optimizer_rows
                if row.get("effective_h1_gradient_projection") is not None
            ),
            "effective_base_gradient_norm": numeric_summary(
                row["effective_base_gradient_norm"]
                for row in epoch_optimizer_rows
                if row.get("effective_base_gradient_norm") is not None
            ),
            "effective_h1_gradient_norm": numeric_summary(
                row["effective_h1_gradient_norm"]
                for row in epoch_optimizer_rows
                if row.get("effective_h1_gradient_norm") is not None
            ),
            "parameter_update_norm": numeric_summary(
                row["parameter_update_norm"]
                for row in epoch_optimizer_rows
                if row.get("parameter_update_norm") is not None
            ),
            "clipping_fraction": (
                (clipping_count - epoch_clipping_start)
                / max(len(epoch_optimizer_rows), 1)
            ),
            "nonfinite_count": 0,
            "roi_ramp": min(epoch / 375.0, 1.0),
            "wall_seconds": time.perf_counter() - epoch_started,
            "peak_allocated_mib": epoch_peak_allocated,
            "peak_reserved_mib": epoch_peak_reserved,
            "parameter_displacement_from_initial": (
                model_displacement_norm(model, shared_initial_state)
                if shared_initial_state is not None
                else None
            ),
            "validation_normalized_arithmetic_average_relative_l2": normalized_value,
            "validation_normalized_global_relative_l2": normalized_metrics[
                "global_relative_l2"
            ],
            "validation_residual_metrics": validation["residual_metrics"],
        }
        epoch_summaries.append(epoch_summary)
        if controlled_paired_run:
            (output_dir / f"validation_epoch_{epoch + 1:03d}.json").write_text(
                json.dumps(
                    _json_value(
                        {
                            "schema_version": (
                                "paper-stage-r-validation-epoch-v1"
                                if stage_r
                                else "paper-stage-o-validation-epoch-v1"
                                if stage_o
                                else "paper-stage-k-validation-epoch-v1"
                                if stage_k
                                else
                                "paper-stage-i-validation-epoch-v1"
                                if diagnostic_h1_mode is not None
                                else "paper-stage-g-validation-epoch-v1"
                            ),
                            "training": epoch_summary,
                            "validation": validation,
                        }
                    ),
                    indent=2,
                    sort_keys=True,
                )
                + "\n",
                encoding="utf-8",
            )
        metadata = _checkpoint_metadata(
            config,
            environment=environment,
            config_checksum=config_checksum,
            epoch=epoch + 1,
            experiment_name=args.experiment_name,
            gradient_accumulation=gradient_accumulation,
            probe_sha256=validation["probe_prediction_sha256"],
            stage_g=(
                {
                    "shared_initial_state_sha256": shared_initial_state_sha256,
                    "pair_order_sha256": pair_order_sha256,
                    "optimizer_step": optimizer_updates,
                    "h1_weight": (
                        0.0
                        if no_h1_diagnostic
                        else 0.05
                        if config.mode == "full"
                        else None
                    ),
                    "gradient_clip_norm": 1.0,
                    "resource_scaled_training": True,
                }
                if stage_g
                else None
            ),
            stage_i=(
                {
                    "diagnostic_h1_mode": diagnostic_h1_mode,
                    "reproduction_level": "diagnostic_extension",
                    "paper_faithful_full": False,
                    "extension_reason": (
                        "isolate_H1_contribution_under_spherical_grid_adaptation"
                        if no_h1_diagnostic
                        else config.values["reproduction_metadata"][
                            "extension_reason"
                        ]
                    ),
                    "comparison_parent": "stage_g_paper_adapted_full",
                    "stage_g_parent_config": (
                        "configs/paper_reduced100/full_fno_proxy.yaml"
                    ),
                    "selected_h1_training_coefficient": (
                        0.0 if no_h1_diagnostic else 0.05
                    ),
                    "diagnostic_current_upstream_h1_trained": False,
                    "non_h1_loss_contract": {
                        name: config.values["loss"][name]
                        for name in (
                            "base",
                            "roi",
                            "bounds",
                            "envelope",
                            "dissipation",
                        )
                    },
                    **(
                        {
                            "selected_h1_definition": (
                                stage_i_selected_h1_definition(config)
                            )
                        }
                        if stage_i_selected_h1_definition(config) is not None
                        else {}
                    ),
                }
                if diagnostic_h1_mode is not None
                else None
            ),
            stage_k=(
                {
                    "classification": config.values["reproduction_metadata"][
                        "classification"
                    ],
                    "architecture": config.architecture,
                    "n_dim": int(config.values["model"]["n_dim"]),
                    "differential_enabled": True,
                    "disco_enabled": False,
                    "parameter_count": parameter_count,
                    "input_channels": int(config.values["model"]["in_channels"]),
                    "output_channels": int(config.values["model"]["out_channels"]),
                    "prediction_mode": config.prediction_mode,
                    "plain_loss_contract": dict(config.values["loss"]),
                    "initial_state_sha256": shared_initial_state_sha256,
                    "pair_order_sha256": pair_order_sha256,
                    "optimizer_step": optimizer_updates,
                    "gradient_clip_norm": 1.0,
                    "resource_scaled_training": True,
                    "run_kind": (
                        "engineering_smoke" if localno_smoke else "pilot30"
                    ),
                }
                if stage_k
                else None
            ),
            stage_o=(
                {
                    "classification": "adapted_transform_model_pilot",
                    "p3_config_sha256": config.values["preprocessing"][
                        "prototype_config_checksum"
                    ],
                    "p3_statistics_sha256": config.values["preprocessing"][
                        "stats_checksum"
                    ],
                    "p3_train_indices": list(range(11, 91)),
                    "architecture_config_sha256": config.values["provenance"][
                        "stage_o_frozen"
                    ]["stage_k_config"]["sha256"],
                    "stage_k_initial_state_file_sha256": config.values[
                        "provenance"
                    ]["stage_o_frozen"]["stage_k_initial_state_file"]["sha256"],
                    "initial_state_sha256": shared_initial_state_sha256,
                    "pair_order_sha256": pair_order_sha256,
                    "optimizer_step": optimizer_updates,
                    "run_kind": (
                        "engineering_smoke" if stage_o_smoke else "pilot30"
                    ),
                    "oracle_conditioned_gate_version": "stage_m_v1",
                    "canonical_replacement": False,
                }
                if stage_o
                else None
            ),
            stage_r=(
                {
                    "classification": "adapted_residual_contract_model_pilot",
                    "prediction_mode": "normalized_residual",
                    "state_skip": "identity",
                    "residual_scale": 1.0,
                    "learnable_scale": False,
                    "clipping": False,
                    "loss_target": "normalized_residual",
                    "loss_equivalence_contract": (
                        "PlainL2(r_theta,delta_z_true)=="
                        "PlainL2(z_t+r_theta,z_t1)"
                    ),
                    "p3_config_sha256": config.values["preprocessing"][
                        "prototype_config_checksum"
                    ],
                    "p3_statistics_sha256": config.values["preprocessing"][
                        "stats_checksum"
                    ],
                    "architecture_config_sha256": config.values["provenance"][
                        "stage_o_frozen"
                    ]["stage_k_config"]["sha256"],
                    "initial_state_sha256": shared_initial_state_sha256,
                    "pair_order_sha256": pair_order_sha256,
                    "optimizer_step": optimizer_updates,
                    "run_kind": (
                        "engineering_smoke" if stage_r_smoke else "pilot30"
                    ),
                    "oracle_conditioned_gate_version": "stage_m_v1",
                }
                if stage_r
                else None
            ),
        )
        best_checkpoint_dir = (
            output_dir / "best_validation_l2"
            if controlled_paired_run
            else output_dir / "best.pt"
        )
        last_checkpoint_dir = (
            output_dir / "last" if controlled_paired_run else output_dir / "last.pt"
        )
        if normalized_value < best_value:
            best_value = normalized_value
            best_epoch = epoch + 1
            save_paper_checkpoint(
                best_checkpoint_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                metadata=metadata,
            )
        if controlled_paired_run or epoch + 1 == epochs:
            save_paper_checkpoint(
                last_checkpoint_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                metadata=metadata,
            )
        write_csv(output_dir / "train_log.csv", train_rows)
        write_csv(output_dir / "optimizer_log.csv", optimizer_rows)
        write_csv(output_dir / "validation_log.csv", validation_rows)

    wall_seconds = time.perf_counter() - training_started
    model_hash_after_training = model_tensor_state_sha256(model)
    optimizer_updated_model = model_hash_after_training != model_hash_before_training
    if not optimizer_updated_model or optimizer_updates == 0:
        raise RuntimeError("Paper training did not produce a finite optimizer update")
    expected_controlled_updates = epochs * 20
    if controlled_paired_run and optimizer_updates != expected_controlled_updates:
        raise RuntimeError(
            f"Controlled run expected {expected_controlled_updates} optimizer steps, "
            f"observed {optimizer_updates}"
        )
    write_csv(output_dir / "train_log.csv", train_rows)
    write_csv(output_dir / "optimizer_log.csv", optimizer_rows)
    write_csv(output_dir / "validation_log.csv", validation_rows)

    best_reload, best_model, best_processor = verify_checkpoint(
        output_dir / ("best_validation_l2" if controlled_paired_run else "best.pt"),
        config=config,
        config_checksum=config_checksum,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        validation_batch=validation_probe_batch,
        device=device,
        expected_shared_initial_state_sha256=shared_initial_state_sha256,
        expected_pair_order_sha256=pair_order_sha256,
    )
    last_reload, _, _ = verify_checkpoint(
        output_dir / ("last" if controlled_paired_run else "last.pt"),
        config=config,
        config_checksum=config_checksum,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        validation_batch=validation_probe_batch,
        device=device,
        expected_shared_initial_state_sha256=shared_initial_state_sha256,
        expected_pair_order_sha256=pair_order_sha256,
    )
    best_loss_adapter = PaperTrainerLossAdapter(build_paper_training_loss(config)).to(device)
    best_one_step = evaluate_one_step(
        model=best_model,
        processor=best_processor,
        loss_adapter=best_loss_adapter,
        loader=validation_loader,
        max_batches=min(max_validation_batches, len(validation_loader)),
        epoch=best_reload["epoch"],
    )
    rollout3 = evaluate_rollout3(
        model=best_model,
        processor=best_processor,
        validation_dataset=datasets["validation"],
        epoch=best_reload["epoch"],
    )
    total_wall_seconds = time.perf_counter() - main_started

    clipping_fraction = clipping_count / max(optimizer_updates, 1)
    h1_warning = config.mode == "full" and diagnostic_h1_mode is None and (
        (full_h1_ratios and max(full_h1_ratios) > 10.0)
        or (full_h1_gradient_ratios and max(full_h1_gradient_ratios) > 10.0)
        or clipping_fraction > 0.5
    )
    status = "passed_with_h1_warning" if h1_warning else "passed"
    peak_allocated = max(epoch_peak_allocated_mib, default=0.0)
    peak_reserved = max(epoch_peak_reserved_mib, default=0.0)
    sample_count = len(train_rows)
    runtime = {
        "schema_version": (
            "paper-stage-r-runtime-v1"
            if stage_r
            else "paper-stage-o-runtime-v1"
            if stage_o
            else "paper-stage-k-runtime-v1"
            if stage_k
            else "paper-stage-i-runtime-v1"
            if diagnostic_h1_mode is not None
            else "paper-stage-g-runtime-v1"
            if stage_g
            else "paper-stage-f-runtime-v1"
        ),
        "status": status,
        "wall_seconds": wall_seconds,
        "total_wall_seconds": total_wall_seconds,
        "startup_seconds": startup_seconds,
        "phase_seconds": phase_seconds,
        "data_setup_seconds": data_setup_seconds,
        "samples": sample_count,
        "samples_per_second": sample_count / max(wall_seconds, 1e-12),
        "device": str(device),
        "cuda_peak_memory_reset": device.type == "cuda",
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
        "gpu_memory_available": device.type == "cuda",
        "optimizer_updates": optimizer_updates,
        "expected_optimizer_updates": (
            expected_controlled_updates if controlled_paired_run else optimizer_updates
        ),
        "optimizer_updated_model": optimizer_updated_model,
        "nonfinite_count": nonfinite_count,
        "clipping_fraction": clipping_fraction,
    }
    metrics = {
        "schema_version": (
            (
                "paper-stage-r-engineering-smoke-metrics-v1"
                if stage_r_smoke
                else "paper-stage-r-pilot-metrics-v1"
            )
            if stage_r
            else (
                "paper-stage-o-engineering-smoke-metrics-v1"
                if stage_o_smoke
                else "paper-stage-o-pilot-metrics-v1"
            )
            if stage_o
            else (
                "paper-stage-k-engineering-smoke-metrics-v1"
                if stage_k_smoke
                else "paper-stage-k-pilot-metrics-v1"
            )
            if stage_k
            else "paper-stage-i-diagnostic-pilot-metrics-v1"
            if diagnostic_h1_mode is not None
            else "paper-stage-g-pilot-metrics-v1"
            if stage_g
            else "paper-stage-f-smoke-metrics-v1"
        ),
        "status": status,
        "engineering_smoke_only": args.smoke or localno_smoke,
        "resource_scaled_pilot": formal_pilot,
        "scientific_result": False,
        "mode": config.mode,
        "model_architecture": config.architecture,
        "prediction_mode": config.prediction_mode,
        "classification": (
            config.values["reproduction_metadata"]["classification"]
            if stage_k
            else None
        ),
        "diagnostic_h1_mode": diagnostic_h1_mode,
        "reproduction_level": (
            "diagnostic_extension"
            if diagnostic_h1_mode is not None
            else config.values["reproduction_metadata"]["reproduction_level"]
        ),
        "paper_faithful_full": (
            False if diagnostic_h1_mode is not None else None
        ),
        "comparison_parent": (
            "stage_g_paper_adapted_full"
            if diagnostic_h1_mode is not None
            else None
        ),
        "experiment_name": args.experiment_name,
        "parameter_count": parameter_count,
        "initial_state_hash": initial_state_hash,
        "model_hash_after_training": model_hash_after_training,
        "protocol_checksum": config.values["provenance"]["manifest"]["sha256"],
        "shared_initial_state_sha256": shared_initial_state_sha256,
        "pair_order_sha256": pair_order_sha256,
        "epochs": epochs,
        "train_batches": len(train_rows),
        "best_epoch": best_epoch,
        "best_validation_normalized_arithmetic_average_relative_l2": best_value,
        "runtime": runtime,
        "epoch_summaries": epoch_summaries,
        "h1_monitoring": {
            "selected_h1_training_coefficient": (
                0.0 if no_h1_diagnostic else 0.05 if config.mode == "full" else None
            ),
            "selected_h1_weighted_exact_zero": (
                True if no_h1_diagnostic else None
            ),
            "selected_h1_gradient_norm": (
                0.0 if no_h1_diagnostic else None
            ),
            "value_ratios": (
                None if no_h1_diagnostic else full_h1_ratios
                if config.mode == "full"
                else None
            ),
            "gradient_ratios": (
                None if no_h1_diagnostic else full_h1_gradient_ratios
                if config.mode == "full"
                else None
            ),
            "gradient_ratio_semantics": (
                "not_applicable" if no_h1_diagnostic else "selected_h1_to_base"
            ),
            "diagnostic_current_upstream_h1_raw": (
                diagnostic_upstream_h1_values
                if diagnostic_h1_mode is not None
                else None
            ),
            "diagnostic_unit_index_h1_raw": (
                diagnostic_unit_index_h1_values
                if diagnostic_h1_mode is not None
                else None
            ),
            "current_to_selected_h1_raw_ratios": (
                current_to_selected_h1_ratios
                if diagnostic_h1_mode not in (None, "no_h1")
                else None
            ),
            "unit_index_to_selected_h1_raw_ratios": (
                unit_index_to_selected_h1_ratios
                if diagnostic_h1_mode
                == "stored_coordinate_volume_proxy"
                else None
            ),
            "selected_h1_definition": (
                stage_i_selected_h1_definition(config)
                if stage_i_selected_h1_definition(config) is not None
                else None
            ),
            "diagnostic_current_upstream_h1_trained": (
                False if diagnostic_h1_mode is not None else None
            ),
            "paper_reference_weight": 0.05 if config.mode == "full" else None,
            "warning": h1_warning if config.mode == "full" else None,
        },
        "disabled_components": (
            loss_adapter.last_log.get("disabled_components")
            if config.mode == "plain" and loss_adapter.last_log
            else []
        ),
        "checkpoint_reload": {"best": best_reload, "last": last_reload},
        "one_step": best_one_step,
        "rollout3": rollout3,
        "stage_d_e_artifacts_unchanged": True,
        "upstream_worktree_clean": True,
    }
    (output_dir / "runtime.json").write_text(
        json.dumps(_json_value(runtime), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(_json_value(metrics), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(_json_value(metrics), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Paper-reduced training failed: {error}", file=sys.stderr, flush=True)
        raise
