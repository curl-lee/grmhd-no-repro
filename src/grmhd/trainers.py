"""Minimal GRMHD-specific extensions of the pinned upstream Trainer."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_

from neuralop import LpLoss
from neuralop.training import Trainer

from .data_processor import GRMHDDataProcessor
from .dataset import GRMHDPairedDataset


class GRMHDRolloutTrainer(Trainer):
    """Add hybrid-target and differentiable rollout loss to upstream Trainer.

    Device/mode ownership and the public training/evaluation interface remain
    inherited.  The batch and epoch methods are overridden only because pinned
    Trainer has no multi-step BPTT, gradient accumulation, or component logging.
    """

    def __init__(
        self,
        *,
        rollout_dataset: GRMHDPairedDataset,
        loss_weights: dict[str, float],
        gradient_accumulation: int = 4,
        gradient_clip: float = 1.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if not isinstance(self.data_processor, GRMHDDataProcessor):
            raise TypeError("GRMHDRolloutTrainer requires GRMHDDataProcessor")
        if self.data_processor.target_mode != "hybrid":
            raise ValueError("GRMHDRolloutTrainer requires target_mode='hybrid'")
        if rollout_dataset.stride != 1:
            raise ValueError("Differentiable rollout currently requires stride=1")
        if gradient_accumulation <= 0:
            raise ValueError("gradient_accumulation must be positive")
        self.rollout_dataset = rollout_dataset
        self.loss_weights = {
            name: float(loss_weights.get(name, 0.0))
            for name in ("target", "decoded", "rollout", "range")
        }
        self.gradient_accumulation = int(gradient_accumulation)
        self.gradient_clip = float(gradient_clip)
        self.target_loss = torch.nn.SmoothL1Loss(reduction="mean")
        self.physical_loss = LpLoss(d=3, p=2, reduction="sum")
        self.rollout_k = 1
        self.last_components: dict[str, torch.Tensor] = {}

    def on_epoch_start(self, epoch: int):
        super().on_epoch_start(epoch)
        epoch_number = epoch + 1
        if epoch_number <= 3:
            self.rollout_k = 1
        elif epoch_number <= 8:
            self.rollout_k = 2
        else:
            self.rollout_k = 3

    def _model_input(self, physical_state: torch.Tensor) -> torch.Tensor:
        encoded = self.data_processor.normalizer.encode_tensor(
            physical_state, channel_axis=1
        )
        shells = self.data_processor.shells
        if shells is None:
            return encoded
        return torch.cat(
            (
                encoded,
                shells.unsqueeze(0).expand(encoded.shape[0], -1, -1, -1, -1),
            ),
            dim=1,
        )

    def rollout_from_physical(
        self, physical_input: torch.Tensor, steps: int
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Free-run without detach; return ``(raw, physical)`` at every step."""
        if steps <= 0:
            raise ValueError("steps must be positive")
        assert self.data_processor.hybrid_stats is not None
        current = physical_input
        predictions = []
        for _ in range(steps):
            raw = self.model(x=self._model_input(current))
            current = self.data_processor.hybrid_stats.reconstruct(current, raw)
            predictions.append((raw, current))
        return predictions

    @staticmethod
    def _range_penalty(
        prediction: torch.Tensor, truth: torch.Tensor
    ) -> torch.Tensor:
        prediction_range = prediction[:, :3].amax(dim=(-3, -2, -1)) - prediction[
            :, :3
        ].amin(dim=(-3, -2, -1))
        truth_range = truth[:, :3].amax(dim=(-3, -2, -1)) - truth[:, :3].amin(
            dim=(-3, -2, -1)
        )
        return torch.relu(
            (prediction_range - truth_range) / truth_range.clamp_min(1.0e-12)
        ).mean()

    def _truth_after(self, source_indices: torch.Tensor, step: int) -> torch.Tensor:
        truths = []
        for source_index in source_indices.detach().cpu().tolist():
            target_index = int(source_index) + step
            if target_index >= self.rollout_dataset.split.stop:
                raise IndexError("rollout target crosses the training split")
            truth = self.rollout_dataset.load_snapshot(target_index).to(self.device)
            truth = self.data_processor._downsample(truth.unsqueeze(0))[0]
            truths.append(truth)
        return torch.stack(truths, dim=0)

    def loss_components(self, sample: dict[str, Any]) -> dict[str, torch.Tensor]:
        processed = self.data_processor.preprocess(sample)
        assert processed is not None
        assert self.data_processor.hybrid_stats is not None
        raw = self.model(x=processed["x"])
        target_prediction, one_step_physical = self.data_processor.reconstruct_physical(
            raw, processed
        )
        target = processed["y"]
        target_loss = self.target_loss(target_prediction, target)
        decoded_loss = self.physical_loss(one_step_physical, processed["physical_target"])

        source_indices = processed["source_index"]
        if not torch.is_tensor(source_indices):
            source_indices = torch.as_tensor(source_indices, device=self.device)
        available = min(
            self.rollout_dataset.split.stop - 1 - int(index)
            for index in source_indices.detach().cpu().tolist()
        )
        actual_k = max(1, min(self.rollout_k, available))
        physical_predictions = [one_step_physical]
        current = one_step_physical
        for _ in range(2, actual_k + 1):
            raw_step = self.model(x=self._model_input(current))
            current = self.data_processor.hybrid_stats.reconstruct(current, raw_step)
            physical_predictions.append(current)
        rollout_losses = []
        range_losses = []
        for step, prediction in enumerate(physical_predictions, start=1):
            truth = self._truth_after(source_indices, step)
            rollout_losses.append(self.physical_loss(prediction, truth))
            range_losses.append(self._range_penalty(prediction, truth))
        rollout_loss = torch.stack(rollout_losses).mean()
        range_loss = torch.stack(range_losses).mean()
        components = {
            "target": target_loss,
            "decoded": decoded_loss,
            "rollout": rollout_loss,
            "range": range_loss,
        }
        components["loss"] = sum(
            self.loss_weights[name] * value for name, value in components.items()
        )
        components["rollout_k"] = target_loss.new_tensor(float(actual_k))
        self.last_components = components
        return components

    def train_one_batch(self, idx, sample, training_loss=None):
        del idx, training_loss
        return self.loss_components(sample)["loss"]

    def train_one_epoch(self, epoch, train_loader, training_loss=None):
        del training_loss
        self.on_epoch_start(epoch)
        self.model.train()
        self.data_processor.train()
        self.optimizer.zero_grad(set_to_none=True)
        totals = {name: 0.0 for name in ("loss", "target", "decoded", "rollout", "range")}
        gradient_norm_total = 0.0
        optimizer_steps = 0
        self.n_samples = 0
        for batch_index, sample in enumerate(train_loader):
            components = self.loss_components(sample)
            (components["loss"] / self.gradient_accumulation).backward()
            batch_size = int(sample["physical_input"].shape[0])
            self.n_samples += batch_size
            for name in totals:
                totals[name] += float(components[name].detach().cpu())
            last_batch = batch_index + 1 == len(train_loader)
            if (batch_index + 1) % self.gradient_accumulation == 0 or last_batch:
                gradient_norm = clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError("Non-finite rollout gradient norm")
                gradient_norm_total += float(gradient_norm.detach().cpu())
                optimizer_steps += 1
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
        if self.scheduler is not None:
            self.scheduler.step()
        batches = len(train_loader)
        self.epoch_component_metrics = {
            name: value / batches for name, value in totals.items()
        }
        self.epoch_component_metrics.update(
            {
                "gradient_norm": gradient_norm_total / max(optimizer_steps, 1),
                "rollout_k": self.rollout_k,
                "optimizer_steps": optimizer_steps,
            }
        )
        # Preserve Trainer's return contract for callers that use train().
        return (
            totals["loss"] / batches,
            totals["loss"] / max(self.n_samples, 1),
            None,
            math.nan,
        )
