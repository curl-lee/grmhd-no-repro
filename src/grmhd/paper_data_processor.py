"""Stage F paper-reduced preprocessing for the pinned upstream Trainer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import torch
from neuralop.data.transforms.data_processors import DataProcessor

from .paper_bounds import BoundsClampResult, PaperPhysicalBounds
from .paper_config import ResolvedPaperConfig
from .paper_losses import PaperLossContext
from .paper_preprocessing import PaperPreprocessor
from .paper_priors import PAPER_COORDINATE_SYSTEM, read_prior_json
from .paper_radial import PaperRadialBaseline
from .paper_stage_n import NO_SOFTCLIP, PrototypePreprocessor, prototype_specs
from .paper_stage_r import normalized_residual_target, reconstruct_normalized_state
from .paper_velocity_roi import (
    raw_physical_velocity_roi_diagnostic_mask,
    stored_component_speed_proxy,
    top_fraction_mask,
)
from .shells import radial_shells_tensor


@dataclass(frozen=True)
class PaperPredictionDecode:
    """Evaluation-only decode result; never used by the training loss."""

    normalized_unclamped: torch.Tensor
    normalized_clamped: torch.Tensor
    physical_prediction: torch.Tensor
    clamp_mask: torch.Tensor
    clamp_fraction: Mapping[str, float]


def _batch_indices(value: Any, *, name: str, batch_size: int) -> tuple[int, ...]:
    if torch.is_tensor(value):
        indices = tuple(int(item) for item in value.detach().cpu().reshape(-1).tolist())
    elif isinstance(value, (tuple, list)):
        indices = tuple(int(item) for item in value)
    else:
        indices = (int(value),)
    if len(indices) != batch_size:
        raise ValueError(f"{name} length does not match batch size")
    return indices


def _batch_times(value: Any, *, name: str, batch_size: int, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float64, device=device).reshape(-1)
    if tensor.numel() != batch_size:
        raise ValueError(f"{name} length does not match batch size")
    return tensor


class PaperDataProcessor(DataProcessor):
    """Create one canonical model/loss batch from explicit physical-space keys.

    The selected radial baseline is an envelope reference only. It is not
    subtracted from the input or target. Direct-state and Stage R residual-target
    models receive the same normalized state and the same eight shell channels;
    Stage R reconstruction is applied only after the raw eight-channel output.
    """

    representation_semantics = "direct_canonical_state_radial_envelope_reference_only"

    def __init__(
        self,
        *,
        config: ResolvedPaperConfig,
        preprocessor: PaperPreprocessor | PrototypePreprocessor,
        radial: PaperRadialBaseline,
        bounds: PaperPhysicalBounds,
        shells: torch.Tensor,
        shell_metadata: Mapping[str, Any],
    ) -> None:
        super().__init__()
        if shells.ndim != 4 or shells.shape[0] != 8:
            raise ValueError("Paper shells must have shape (8,Nphi,Ntheta,Nr)")
        if radial.mode != config.values["representation"]["radial"]["selected_mode"]:
            raise ValueError("DataProcessor radial mode differs from resolved config")
        if int(shell_metadata["n_shells"]) != 8:
            raise ValueError("Paper DataProcessor requires exactly eight shells")
        self.config = config
        self.preprocessor = preprocessor
        self.radial = radial
        self.bounds = bounds
        self.shell_metadata = dict(shell_metadata)
        self.register_buffer("shells", shells.to(dtype=torch.float32).unsqueeze(0))
        spatial_shape = tuple(int(value) for value in shells.shape[1:])
        self.register_buffer(
            "radial_baseline_normalized",
            radial.state(spatial_shape, normalized=True).unsqueeze(0),
        )
        self.device = torch.device("cpu")
        self.model: torch.nn.Module | None = None
        self.epoch = 0
        self.last_batch: dict[str, Any] | None = None
        self.last_normalized_prediction: torch.Tensor | None = None
        self.last_predicted_residual: torch.Tensor | None = None
        self.transform_counts = {
            "input_encode": 0,
            "target_encode": 0,
            "oracle_decode": 0,
            "prediction_decode": 0,
        }

    @classmethod
    def from_config(cls, config: ResolvedPaperConfig) -> "PaperDataProcessor":
        values = config.values
        protocol = values["protocol"]
        provenance = values["provenance"]
        h5_path = config.resolve_path(str(protocol["dataset"]))
        stats_path = config.resolve_path(str(values["preprocessing"]["stats_path"]))
        train_indices = tuple(range(*protocol["train_snapshots"]))
        if config.uses_p3:
            p3_spec = prototype_specs(NO_SOFTCLIP)["P3"]
            preprocessor = PrototypePreprocessor.load(
                config.resolve_path(str(values["preprocessing"]["artifact"])),
                spec=p3_spec,
            )
            preprocessor.validate_compatibility(
                h5_path=h5_path,
                expected_source_hdf5_checksum=provenance["dataset"]["sha256"],
                expected_training_indices=train_indices,
                expected_protocol_name=f"{protocol['name']}__stage_n__p3",
            )
        else:
            preprocessor = PaperPreprocessor.load(
                stats_path,
                h5_path=h5_path,
                expected_source_hdf5_checksum=provenance["dataset"]["sha256"],
                expected_training_indices=train_indices,
                expected_protocol_name=protocol["name"],
            )
        expected = {
            "source_hdf5_checksum": provenance["dataset"]["sha256"],
            "preprocessing_stats_checksum": config.prior_preprocessing_checksum,
            "training_indices": train_indices,
            "protocol_name": protocol["name"],
            "thermal_channel": values["thermal"]["channel"],
        }
        artifacts = provenance["artifacts"]
        radial = PaperRadialBaseline.load(
            config.resolve_path(str(artifacts["radial"]["path"])), **expected
        )
        bounds = PaperPhysicalBounds.load(
            config.resolve_path(str(artifacts["bounds"]["path"])), **expected
        )
        shell_payload = read_prior_json(
            config.resolve_path(str(artifacts["shells"]["path"])),
            expected_schema="paper-shell-metadata-v1",
            **expected,
        )
        with h5py.File(h5_path, "r") as handle:
            r = np.asarray(handle["coords/r"][...], dtype=np.float64)
            shape = tuple(int(value) for value in handle["snapshots"].shape[2:])
        shells, generated = radial_shells_tensor(
            r,
            shape[0],
            shape[1],
            n_shells=int(values["representation"]["shells"]["count"]),
        )
        frozen_shells = shell_payload["shells"]
        if generated.as_dict() != frozen_shells:
            raise ValueError("Generated shells differ from the frozen Stage D artifact")
        return cls(
            config=config,
            preprocessor=preprocessor,
            radial=radial,
            bounds=bounds,
            shells=shells,
            shell_metadata=frozen_shells,
        )

    def to(self, device: torch.device | str):
        self.device = torch.device(device)
        # Upstream DataProcessor declares an abstract ``to`` method whose
        # implementation is a no-op, shadowing ``nn.Module.to``.  Call the
        # module implementation directly so registered paper buffers follow
        # the model onto CUDA.
        torch.nn.Module.to(self, self.device)
        return self

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("Paper DataProcessor epoch must be nonnegative")
        self.epoch = epoch

    def _validate_raw_state(self, value: Any, name: str) -> torch.Tensor:
        if not torch.is_tensor(value):
            raise TypeError(f"{name} must be a tensor")
        state = value.to(device=self.device, dtype=torch.float32, non_blocking=True)
        if state.ndim != 5 or state.shape[1] != 8:
            raise ValueError(f"{name} must have shape (B,8,Nphi,Ntheta,Nr)")
        if tuple(state.shape[2:]) != tuple(self.shells.shape[2:]):
            raise ValueError(f"{name} spatial shape differs from the frozen shells")
        if not torch.isfinite(state).all():
            raise FloatingPointError(f"{name} contains NaN/Inf")
        return state

    def preprocess(self, data_dict: Mapping[str, Any], batched: bool = True, **_: Any):
        if not batched:
            raise ValueError("PaperDataProcessor requires explicit batch-first data")
        required = {
            "physical_input",
            "physical_target",
            "source_index",
            "target_index",
            "time",
            "target_time",
        }
        missing = sorted(required - set(data_dict))
        if missing:
            raise KeyError(f"Paper batch is missing explicit fields: {missing}")
        raw_input = self._validate_raw_state(data_dict["physical_input"], "physical_input")
        raw_target = self._validate_raw_state(data_dict["physical_target"], "physical_target")
        if raw_input.shape != raw_target.shape:
            raise ValueError("Physical input and target shapes differ")
        batch_size = raw_input.shape[0]
        source_indices = _batch_indices(
            data_dict["source_index"], name="source_index", batch_size=batch_size
        )
        target_indices = _batch_indices(
            data_dict["target_index"], name="target_index", batch_size=batch_size
        )
        if any(target != source + 1 for source, target in zip(source_indices, target_indices)):
            raise ValueError("Paper training batch is not a one-step transition")
        input_time = _batch_times(
            data_dict["time"], name="time", batch_size=batch_size, device=self.device
        )
        target_time = _batch_times(
            data_dict["target_time"],
            name="target_time",
            batch_size=batch_size,
            device=self.device,
        )
        if torch.any(target_time <= input_time):
            raise ValueError("Paper batch target time must follow input time")

        normalized_input = self.preprocessor.encode(raw_input, channel_axis=1)
        self.transform_counts["input_encode"] += 1
        normalized_target = self.preprocessor.encode(raw_target, channel_axis=1)
        self.transform_counts["target_encode"] += 1
        oracle_target = self.preprocessor.decode(normalized_target, channel_axis=1)
        self.transform_counts["oracle_decode"] += 1
        canonical_roi = top_fraction_mask(
            stored_component_speed_proxy(oracle_target), top_fraction=0.20
        )
        raw_roi = raw_physical_velocity_roi_diagnostic_mask(
            raw_target, top_fraction=0.20
        )
        shells = self.shells.expand(batch_size, -1, -1, -1, -1).to(
            dtype=normalized_input.dtype
        )
        model_input = torch.cat((normalized_input, shells), dim=1)
        baseline = self.radial_baseline_normalized.expand(
            batch_size, -1, -1, -1, -1
        ).to(dtype=normalized_input.dtype)
        protocol_metadata = {
            "protocol_name": self.config.values["protocol"]["name"],
            "thermal_channel": self.config.values["thermal"]["channel"],
            "paper_adaptation": True,
            "validation_not_used_for_fit": True,
            "selected_radial_mode": self.radial.mode,
            "coordinate_system": PAPER_COORDINATE_SYSTEM,
            "canonical_roi_source": "oracle_physical_target",
            "representation_semantics": (
                "p3_normalized_residual_target_identity_state_skip"
                if self.config.stage_r
                else self.representation_semantics
            ),
            "shell_channels": 8,
            "preprocessing_mode": self.config.values["preprocessing"]["mode"],
            "canonical_replacement": bool(
                self.config.values["preprocessing"].get(
                    "canonical_replacement", False
                )
            ),
        }
        context = PaperLossContext(
            normalized_input=normalized_input,
            normalized_target=normalized_target,
            raw_physical_target=raw_target,
            oracle_physical_target=oracle_target,
            canonical_roi_mask=canonical_roi,
            radial_baseline_normalized=baseline,
            normalized_bounds=self.bounds.normalized_bounds,
            epoch=self.epoch,
            snapshot_indices=target_indices,
            protocol_metadata=protocol_metadata,
            # The raw mask remains a separate diagnostic and cannot enter loss context.
            raw_roi_diagnostic_mask=None,
        )
        fields: dict[str, Any] = {
            "model_input": model_input,
            "normalized_input": normalized_input,
            "normalized_target": normalized_target,
            "normalized_residual_target": normalized_residual_target(
                normalized_input, normalized_target
            ),
            "raw_physical_target": raw_target,
            "oracle_physical_target": oracle_target,
            "canonical_roi_mask": canonical_roi,
            "raw_roi_diagnostic_mask": raw_roi,
            "radial_baseline_normalized": baseline,
            "normalized_bounds": dict(self.bounds.normalized_bounds),
            "source_snapshot_index": source_indices,
            "target_snapshot_index": target_indices,
            "input_time": input_time,
            "target_time": target_time,
            "protocol_metadata": protocol_metadata,
            "prediction_mode": self.config.prediction_mode,
            "context": context,
        }
        self.last_batch = fields
        self.last_normalized_prediction = None
        self.last_predicted_residual = None
        # Only x/y are exposed to LocalNO. Stage R makes the raw training target
        # explicitly residual-valued; postprocess reconstructs the state prediction.
        training_target = (
            fields["normalized_residual_target"]
            if self.config.stage_r
            else normalized_target
        )
        return {"x": model_input, "y": training_target}

    def encode_rollout_input(self, physical_state: Any) -> dict[str, torch.Tensor]:
        """Encode one physical autoregressive state without constructing a target."""

        raw_input = self._validate_raw_state(physical_state, "physical_input")
        normalized_input = self.preprocessor.encode(raw_input, channel_axis=1)
        self.transform_counts["input_encode"] += 1
        shells = self.shells.expand(raw_input.shape[0], -1, -1, -1, -1).to(
            dtype=normalized_input.dtype
        )
        return {
            "model_input": torch.cat((normalized_input, shells), dim=1),
            "normalized_input": normalized_input,
            "physical_input": raw_input,
        }

    def postprocess(self, output: torch.Tensor, data_dict: Mapping[str, Any]):
        if self.last_batch is None:
            raise RuntimeError("PaperDataProcessor.postprocess called before preprocess")
        if output.ndim != 5 or output.shape[1] != 8:
            raise ValueError("Paper model output must have shape (B,8,Nphi,Ntheta,Nr)")
        if output.shape != self.last_batch["normalized_target"].shape:
            raise ValueError("Paper model output and normalized target shapes differ")
        fields = dict(self.last_batch)
        if self.config.stage_r:
            predicted_residual = output
            normalized_prediction = reconstruct_normalized_state(
                fields["normalized_input"], predicted_residual
            )
            fields["predicted_residual"] = predicted_residual
            self.last_predicted_residual = predicted_residual
        else:
            normalized_prediction = output
            fields["predicted_residual"] = None
            self.last_predicted_residual = None
        self.last_normalized_prediction = normalized_prediction
        return normalized_prediction, fields

    def decode_prediction(
        self, normalized_prediction: torch.Tensor, *, apply_evaluation_clamp: bool = True
    ) -> PaperPredictionDecode:
        """Decode exactly once for evaluation, optionally after rho/press bounds clamp."""

        if apply_evaluation_clamp:
            clamped: BoundsClampResult = self.bounds.clamp_normalized(normalized_prediction)
        else:
            mask = torch.zeros_like(normalized_prediction, dtype=torch.bool)
            clamped = BoundsClampResult(
                unclamped=normalized_prediction.clone(),
                clamped=normalized_prediction.clone(),
                mask=mask,
                hit_fraction={"rho": 0.0, "press": 0.0, "combined": 0.0},
            )
        physical = self.preprocessor.decode(clamped.clamped, channel_axis=1)
        self.transform_counts["prediction_decode"] += 1
        return PaperPredictionDecode(
            normalized_unclamped=clamped.unclamped,
            normalized_clamped=clamped.clamped,
            physical_prediction=physical,
            clamp_mask=clamped.mask,
            clamp_fraction=clamped.hit_fraction,
        )

    def forward(self, **data_dict: Any):
        if self.model is None:
            raise RuntimeError("PaperDataProcessor must wrap a model before forward")
        processed = self.preprocess(data_dict)
        output = self.model(x=processed["x"])
        return self.postprocess(output, processed)
