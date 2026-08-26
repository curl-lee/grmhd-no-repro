"""Minimal Stage F adapters around the pinned upstream Trainer batch path."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch
from neuralop.training import Trainer
from torch import nn

from . import CHANNELS
from .paper_data_processor import PaperDataProcessor
from .paper_losses import PaperCompositeLoss, PaperLossContext, PaperLossResult, PlainL2Loss
from .paper_stage_r import PREDICTION_MODE, loss_equivalence


DISABLED_PLAIN_COMPONENTS = (
    "h1",
    "roi",
    "bounds_training_penalty",
    "radial_envelope",
    "dissipation",
)


def _fraction_outside_bounds(
    state: torch.Tensor, bounds: Mapping[str, tuple[float, float]]
) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for channel, name in ((3, "rho"), (4, "press")):
        lower, upper = bounds[name]
        output[name] = ((state[:, channel] < lower) | (state[:, channel] > upper)).float().mean()
    output["combined"] = torch.stack((output["rho"], output["press"])).mean()
    return output


class PaperTrainerLossAdapter(nn.Module):
    """Expose Full/Plain Stage E losses through Trainer's keyword call contract."""

    def __init__(self, loss: PaperCompositeLoss | PlainL2Loss) -> None:
        super().__init__()
        self.loss = loss
        self.epoch = 0
        self.last_result: PaperLossResult | None = None
        self.last_log: dict[str, Any] | None = None

    @property
    def mode(self) -> str:
        return "full" if isinstance(self.loss, PaperCompositeLoss) else "plain"

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("Paper loss epoch must be nonnegative")
        self.epoch = epoch

    def forward(
        self,
        normalized_prediction: torch.Tensor,
        *,
        context: PaperLossContext,
        normalized_target: torch.Tensor,
        normalized_input: torch.Tensor,
        normalized_residual_target: torch.Tensor,
        predicted_residual: torch.Tensor | None = None,
        prediction_mode: str = "direct",
        raw_roi_diagnostic_mask: torch.Tensor | None = None,
        **_: Any,
    ) -> torch.Tensor:
        if int(context.epoch) != self.epoch:
            raise ValueError("PaperLossContext epoch differs from Trainer adapter epoch")
        context.validate(normalized_prediction)
        target_bound_clamp = _fraction_outside_bounds(
            normalized_target, context.normalized_bounds
        )
        model_bound_clamp = _fraction_outside_bounds(
            normalized_prediction, context.normalized_bounds
        )
        inverse_limit = 6.0 * 0.99
        target_inverse_clamp = torch.abs(normalized_target) > inverse_limit
        model_inverse_clamp = torch.abs(normalized_prediction) > inverse_limit
        if self.mode == "full":
            assert isinstance(self.loss, PaperCompositeLoss)
            result = self.loss.components(normalized_prediction, context=context)
            self.last_result = result
            self.last_log = result.detached_log()
            self.last_log["diagnostics"]["target_bound_clamp_fraction"] = {
                key: float(value.detach().cpu()) for key, value in target_bound_clamp.items()
            }
            self.last_log["diagnostics"]["model_bound_clamp_fraction"] = {
                key: float(value.detach().cpu()) for key, value in model_bound_clamp.items()
            }
            self.last_log["diagnostics"]["model_clamp_fraction_by_channel"] = {
                name: float(model_inverse_clamp[:, index].float().mean().detach().cpu())
                for index, name in enumerate(CHANNELS)
            }
            if raw_roi_diagnostic_mask is not None:
                canonical = context.canonical_roi_mask
                raw = raw_roi_diagnostic_mask
                intersection = int((canonical & raw).sum().detach().cpu())
                union = int((canonical | raw).sum().detach().cpu())
                self.last_log["diagnostics"]["canonical_raw_roi_jaccard"] = (
                    None if union == 0 else intersection / union
                )
            else:
                self.last_log["diagnostics"].pop("canonical_raw_roi_jaccard", None)
            return result.total

        assert isinstance(self.loss, PlainL2Loss)
        if prediction_mode == PREDICTION_MODE:
            if predicted_residual is None:
                raise ValueError("Stage R loss requires the raw predicted residual")
            if predicted_residual.shape != normalized_residual_target.shape:
                raise ValueError("Stage R predicted/target residual shapes differ")
            equivalence = loss_equivalence(
                predicted_residual,
                normalized_input,
                normalized_target,
                loss=self.loss,
            )
            if not equivalence["passed"]:
                raise FloatingPointError("Stage R residual/state Plain L2 equivalence failed")
            total = equivalence["residual_loss"]
            channels = self.loss.fidelity.components(
                predicted_residual, normalized_residual_target
            )["channel_raw"]
            reconstructed_channels = self.loss.fidelity.components(
                normalized_prediction, normalized_target
            )["channel_raw"]
        elif prediction_mode == "direct":
            if predicted_residual is not None:
                raise ValueError("Direct-state loss received a residual output")
            total = self.loss(normalized_prediction, normalized_target)
            channels = self.loss.fidelity.components(
                normalized_prediction, normalized_target
            )["channel_raw"]
            reconstructed_channels = channels
            equivalence = None
        else:
            raise ValueError(f"Unsupported paper prediction mode {prediction_mode!r}")
        self.last_result = None
        self.last_log = {
            "total": float(total.detach().cpu()),
            "normalized_per_channel_squared_error": {
                name: float(channels[index].detach().cpu())
                for index, name in enumerate(CHANNELS)
            },
            "reconstructed_state_per_channel_squared_error": {
                name: float(reconstructed_channels[index].detach().cpu())
                for index, name in enumerate(CHANNELS)
            },
            "prediction_mode": prediction_mode,
            "loss_target": (
                "normalized_residual"
                if prediction_mode == PREDICTION_MODE
                else "normalized_state"
            ),
            "loss_equivalence": (
                None
                if equivalence is None
                else {
                    "residual_loss": float(
                        equivalence["residual_loss"].detach().cpu()
                    ),
                    "reconstructed_state_loss": float(
                        equivalence["reconstructed_state_loss"].detach().cpu()
                    ),
                    "absolute_difference": float(
                        equivalence["absolute_difference"].detach().cpu()
                    ),
                    "tolerance": float(equivalence["tolerance"].detach().cpu()),
                    "passed": True,
                }
            ),
            "target_bound_clamp_fraction": {
                key: float(value.detach().cpu()) for key, value in target_bound_clamp.items()
            },
            "model_bound_clamp_fraction": {
                key: float(value.detach().cpu()) for key, value in model_bound_clamp.items()
            },
            "target_inverse_clamp_fraction_by_channel": {
                name: float(target_inverse_clamp[:, index].float().mean().detach().cpu())
                for index, name in enumerate(CHANNELS)
            },
            "model_inverse_clamp_fraction_by_channel": {
                name: float(model_inverse_clamp[:, index].float().mean().detach().cpu())
                for index, name in enumerate(CHANNELS)
            },
            "disabled_components": list(DISABLED_PLAIN_COMPONENTS),
            "metadata": self.loss.metadata,
        }
        return total


class _OptimizerZeroGradGate:
    """Let upstream Trainer retain gradients within a local accumulation group."""

    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer
        self.clear_on_next_batch = True

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        if self.clear_on_next_batch:
            self.optimizer.zero_grad(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.optimizer, name)


@dataclass(frozen=True)
class PaperBatchResult:
    loss: torch.Tensor
    normalized_prediction: torch.Tensor
    predicted_residual: torch.Tensor | None
    loss_log: Mapping[str, Any]
    context: PaperLossContext


class PaperTrainerAdapter:
    """Use upstream ``Trainer.train_one_batch`` with explicit epoch propagation."""

    def __init__(
        self,
        *,
        model: nn.Module,
        data_processor: PaperDataProcessor,
        loss: PaperTrainerLossAdapter,
        optimizer: torch.optim.Optimizer,
        n_epochs: int,
        device: torch.device | str,
        mixed_precision: bool = False,
        verbose: bool = False,
    ) -> None:
        self.model = model
        self.data_processor = data_processor
        self.loss = loss
        self.optimizer = optimizer
        self.optimizer_gate = _OptimizerZeroGradGate(optimizer)
        self.upstream = Trainer(
            model=model,
            n_epochs=int(n_epochs),
            device=device,
            mixed_precision=mixed_precision,
            data_processor=data_processor,
            verbose=verbose,
            wandb_log=False,
        )
        self.upstream.optimizer = self.optimizer_gate
        self.upstream.regularizer = None
        self.upstream.n_samples = 0
        self.current_epoch = 0
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        epoch = int(epoch)
        if epoch < 0:
            raise ValueError("Paper Trainer epoch must be nonnegative")
        self.current_epoch = epoch
        self.data_processor.set_epoch(epoch)
        self.loss.set_epoch(epoch)
        self.upstream.on_epoch_start(epoch)

    def compute_batch(
        self,
        sample: Mapping[str, Any],
        *,
        batch_index: int = 0,
        clear_gradients: bool = True,
    ) -> PaperBatchResult:
        self.optimizer_gate.clear_on_next_batch = bool(clear_gradients)
        value = self.upstream.train_one_batch(batch_index, sample, self.loss)
        prediction = self.data_processor.last_normalized_prediction
        fields = self.data_processor.last_batch
        if prediction is None or fields is None or self.loss.last_log is None:
            raise RuntimeError("Upstream Trainer did not complete the paper batch contract")
        return PaperBatchResult(
            loss=value,
            normalized_prediction=prediction,
            predicted_residual=self.data_processor.last_predicted_residual,
            loss_log=self.loss.last_log,
            context=fields["context"],
        )


def parameter_gradient_norm(parameters: Any) -> float:
    squares = []
    for parameter in parameters:
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                return float("nan")
            squares.append(torch.sum(torch.abs(parameter.grad.detach()).float().square()))
    if not squares:
        return 0.0
    return float(torch.sqrt(torch.stack(squares).sum()).cpu())


def component_gradient_norm(
    component: torch.Tensor, parameters: list[nn.Parameter]
) -> float:
    gradients = torch.autograd.grad(
        component,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    squares = [
        torch.sum(torch.abs(gradient.detach()).float().square())
        for gradient in gradients
        if gradient is not None
    ]
    if not squares:
        return 0.0
    value = torch.sqrt(torch.stack(squares).sum())
    return float(value.cpu()) if torch.isfinite(value) else math.nan


def build_paper_optimizer(
    model: nn.Module, *, learning_rate: float, weight_decay: float
) -> torch.optim.Adam:
    """The paired Stage F contract requires Adam, not the Round 1--3 AdamW path."""

    return torch.optim.Adam(
        model.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )


def build_warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    total_epochs: int,
    warmup_epochs: int,
    min_learning_rate: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    total_epochs = int(total_epochs)
    warmup_epochs = int(warmup_epochs)
    if total_epochs <= 0 or not 0 <= warmup_epochs <= total_epochs:
        raise ValueError("Warmup/cosine epoch configuration is invalid")
    base_learning_rates = [float(group["lr"]) for group in optimizer.param_groups]
    if any(rate <= 0 or min_learning_rate < 0 or min_learning_rate > rate for rate in base_learning_rates):
        raise ValueError("Warmup/cosine learning-rate bounds are invalid")

    def schedule(epoch: int) -> float:
        base = base_learning_rates[0]
        minimum_ratio = float(min_learning_rate) / base
        if warmup_epochs and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        cosine_epochs = max(1, total_epochs - warmup_epochs)
        progress = min(1.0, max(0.0, (epoch - warmup_epochs + 1) / cosine_epochs))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=schedule)
