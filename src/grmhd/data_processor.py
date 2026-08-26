"""Upstream-compatible physical-state processing for GRMHD trajectories."""

from __future__ import annotations

from typing import Any

import torch

from neuralop.data.transforms.data_processors import DataProcessor

from . import CHANNELS
from .hybrid import HybridTargetStats
from .normalizer import GRMHDNormalizer


class GRMHDDataProcessor(DataProcessor):
    """Translate explicit physical samples to/from neuraloperator tensors.

    The trajectory state is always physical.  Encoded tensors use only ``x``
    and, during training, ``y``.  An autoregressive postprocess stores the
    physical prediction in ``trajectory_state`` so the following preprocess
    necessarily encodes it again.
    """

    def __init__(
        self,
        *,
        normalizer: GRMHDNormalizer,
        shells: torch.Tensor | None = None,
        target_mode: str = "state",
        hybrid_stats: HybridTargetStats | None = None,
        downsample: int = 1,
        device: str | torch.device = "cpu",
        record_rollout_trace: bool = False,
    ) -> None:
        super().__init__()
        if target_mode not in {"state", "hybrid"}:
            raise ValueError("target_mode must be 'state' or 'hybrid'")
        if target_mode == "hybrid" and hybrid_stats is None:
            raise ValueError("hybrid target_mode requires HybridTargetStats")
        if downsample <= 0:
            raise ValueError("downsample must be positive")
        if shells is not None and shells.ndim != 4:
            raise ValueError("shells must have shape (channels,phi,theta,r)")
        self.normalizer = normalizer
        self.shells = None if shells is None else shells.to(device)
        self.target_mode = target_mode
        self.hybrid_stats = hybrid_stats
        self.downsample = int(downsample)
        self.device = torch.device(device)
        self.record_rollout_trace = bool(record_rollout_trace)
        self.rollout_trace: list[dict[str, Any]] = []
        self.model = None
        self.reset_transform_counts()

    def reset_transform_counts(self) -> None:
        self.transform_counts = {
            "physical_input_encode": 0,
            "physical_target_encode": 0,
            "state_decode": 0,
            "hybrid_reconstruct": 0,
        }

    def reset_rollout_trace(self) -> None:
        self.rollout_trace = []

    def to(self, device: str | torch.device) -> "GRMHDDataProcessor":
        self.device = torch.device(device)
        if self.shells is not None:
            self.shells = self.shells.to(self.device)
        return self

    def _downsample(self, state: torch.Tensor) -> torch.Tensor:
        if self.downsample == 1:
            return state
        return state[..., :: self.downsample, :: self.downsample, :: self.downsample]

    @staticmethod
    def _validate_physical(name: str, state: torch.Tensor) -> None:
        if state.ndim != 5 or state.shape[1] != len(CHANNELS):
            raise ValueError(f"{name} must have shape (batch,8,phi,theta,r), got {state.shape}")
        if not torch.isfinite(state).all():
            raise FloatingPointError(f"{name} contains NaN/Inf")
        if torch.any(state[:, 3:5] <= 0):
            raise ValueError(f"{name} rho/press must be strictly positive")

    def _resolve_physical_pair(
        self, sample: dict[str, Any], step: int | None
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if step is None:
            physical_input = sample["physical_input"].to(self.device)
            physical_target = sample["physical_target"].to(self.device)
            return self._downsample(physical_input), self._downsample(physical_target)

        trajectory = sample["physical_trajectory"].to(self.device)
        if trajectory.ndim != 6 or trajectory.shape[2] != len(CHANNELS):
            raise ValueError(
                "physical_trajectory must have shape (batch,time,8,phi,theta,r)"
            )
        target_position = step + 1
        if target_position >= trajectory.shape[1]:
            return None
        if step == 0:
            physical_input = self._downsample(trajectory[:, 0])
        else:
            physical_input = sample["trajectory_state"].to(self.device)
        physical_target = self._downsample(trajectory[:, target_position])
        return physical_input, physical_target

    def preprocess(self, sample: dict[str, Any], step: int | None = None) -> dict[str, Any] | None:
        output = dict(sample)
        pair = self._resolve_physical_pair(output, step)
        if pair is None:
            return None
        physical_input, physical_target = pair
        self._validate_physical("physical_input", physical_input)
        self._validate_physical("physical_target", physical_target)

        encoded_state = self.normalizer.encode_tensor(physical_input, channel_axis=1)
        self.transform_counts["physical_input_encode"] += 1
        if self.shells is not None:
            if tuple(self.shells.shape[1:]) != tuple(encoded_state.shape[2:]):
                raise ValueError(
                    f"shell spatial shape {self.shells.shape[1:]} != state {encoded_state.shape[2:]}"
                )
            expanded_shells = self.shells.unsqueeze(0).expand(
                encoded_state.shape[0], -1, -1, -1, -1
            )
            model_input = torch.cat((encoded_state, expanded_shells), dim=1)
            output["shells"] = expanded_shells
        else:
            model_input = encoded_state
            output["shells"] = None

        if self.training:
            if self.target_mode == "state":
                target_for_loss = self.normalizer.encode_tensor(
                    physical_target, channel_axis=1
                )
            else:
                assert self.hybrid_stats is not None
                target_for_loss = self.hybrid_stats.encode_target(
                    physical_input, physical_target
                )
            self.transform_counts["physical_target_encode"] += 1
        else:
            target_for_loss = physical_target

        output.update(
            {
                "physical_input": physical_input,
                "physical_target": physical_target,
                "trajectory_state": physical_input,
                "x": model_input,
                "y": target_for_loss,
                "encoded_state": encoded_state,
                "rollout_step": step,
            }
        )
        if step is not None and self.record_rollout_trace:
            trace = {
                "step": int(step),
                "physical_input": physical_input.detach().cpu().clone(),
                "encoded_input": encoded_state.detach().cpu().clone(),
                "physical_target": physical_target.detach().cpu().clone(),
            }
            self.rollout_trace.append(trace)
            output["_rollout_trace_index"] = len(self.rollout_trace) - 1
        return output

    def reconstruct_physical(
        self, raw_output: torch.Tensor, sample: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the target-space output and one physical reconstruction."""
        if raw_output.shape[1] != len(CHANNELS):
            raise ValueError("Model output must contain exactly eight state channels")
        if self.target_mode == "state":
            target_output = raw_output
            physical_prediction = self.normalizer.decode_tensor(raw_output, channel_axis=1)
            self.transform_counts["state_decode"] += 1
        else:
            assert self.hybrid_stats is not None
            target_output = self.hybrid_stats.bounded_target_prediction(raw_output)
            physical_prediction = self.hybrid_stats.reconstruct(
                sample["physical_input"], raw_output
            )
            self.transform_counts["hybrid_reconstruct"] += 1
        if not torch.isfinite(physical_prediction).all():
            raise FloatingPointError("Physical reconstruction contains NaN/Inf")
        if torch.any(physical_prediction[:, 3:5] <= 0):
            raise FloatingPointError("Physical reconstruction violates rho/press positivity")
        return target_output, physical_prediction

    def postprocess(
        self,
        raw_output: torch.Tensor,
        sample: dict[str, Any],
        step: int | None = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        output = dict(sample)
        needs_physical = (not self.training) or self.target_mode == "hybrid" or step is not None
        if needs_physical:
            target_output, physical_prediction = self.reconstruct_physical(raw_output, output)
            output["physical_prediction"] = physical_prediction
        else:
            target_output = raw_output
            physical_prediction = None

        if step is not None:
            assert physical_prediction is not None
            output["trajectory_state"] = physical_prediction
            if self.record_rollout_trace:
                trace_index = int(output["_rollout_trace_index"])
                self.rollout_trace[trace_index]["physical_output"] = (
                    physical_prediction.detach().cpu().clone()
                )

        if self.training:
            prediction_for_loss = target_output
        else:
            assert physical_prediction is not None
            prediction_for_loss = physical_prediction
            output["y"] = output["physical_target"]
        return prediction_for_loss, output

    def forward(self, **sample: Any):
        if self.model is None:
            raise RuntimeError("Call processor.wrap(model) before forward")
        processed = self.preprocess(sample)
        if processed is None:
            return None
        raw_output = self.model(**processed)
        return self.postprocess(raw_output, processed)
