"""Thin adapters around the pinned neuraloperator public interfaces.

The neural-operator model, optimizer, losses, trainer, and training-state
serialization remain upstream objects.  This module only translates the local
GRMHD configuration and dataset schema into those interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import Dataset

from neuralop import FNO, H1Loss, LpLoss
from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.training import AdamW
from neuralop.training.training_state import load_training_state, save_training_state

from .dataset import GRMHDPairedDataset


PINNED_NEURALOP_COMMIT = "86a8bc7812a31b42c4f7895693cf4ac11521c066"


@dataclass(frozen=True)
class UpstreamFNOConfig:
    """The deliberately small 3-D FNO used by this reproduction."""

    in_channels: int
    out_channels: int = 8
    n_modes: tuple[int, int, int] = (8, 8, 8)
    hidden_channels: int = 16
    n_layers: int = 4
    positional_embedding: None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_arch": "fno",
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "n_modes": list(self.n_modes),
            "hidden_channels": self.hidden_channels,
            "n_layers": self.n_layers,
            "positional_embedding": self.positional_embedding,
        }


def build_upstream_fno(config: UpstreamFNOConfig, **kwargs: Any) -> FNO:
    """Instantiate the pinned upstream FNO without copying its implementation."""
    return FNO(
        n_modes=config.n_modes,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        hidden_channels=config.hidden_channels,
        n_layers=config.n_layers,
        positional_embedding=config.positional_embedding,
        **kwargs,
    )


def build_upstream_optimizer(
    model: nn.Module,
    *,
    learning_rate: float = 3.0e-4,
    weight_decay: float = 1.0e-4,
) -> AdamW:
    """Return neuraloperator's AdamW, not torch.optim.Adam/AdamW."""
    return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


def build_upstream_loss(name: str = "l2") -> LpLoss | H1Loss:
    """Construct an upstream 3-D loss with Trainer-compatible batch summation."""
    normalized = name.lower()
    if normalized in {"l2", "lp", "lp2"}:
        return LpLoss(d=3, p=2, reduction="sum")
    if normalized == "h1":
        return H1Loss(d=3, reduction="sum")
    raise ValueError(f"Unknown upstream loss {name!r}")


class GRMHDNextStepDataset(Dataset[dict[str, Any]]):
    """Expose a paired dataset with unambiguous physical-space keys."""

    def __init__(self, dataset: GRMHDPairedDataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int) -> dict[str, Any]:
        sample = self.dataset[item]
        return {
            "physical_input": sample["x"],
            "physical_target": sample["y"],
            "source_index": sample["index"],
            "target_index": sample["target_index"],
            "time": sample["time"],
            "target_time": sample["target_time"],
            "dt": sample["dt"],
        }


class GRMHDAutoregressiveDataset(Dataset[dict[str, Any]]):
    """One physical trajectory sample for upstream autoregressive evaluation."""

    def __init__(self, dataset: GRMHDPairedDataset) -> None:
        if dataset.stride != 1:
            raise ValueError("Round-3 autoregression requires stride=1")
        self.dataset = dataset
        self.indices = dataset.trajectory_indices()
        if len(self.indices) < 2:
            raise ValueError("Autoregressive trajectory requires at least two states")

    def __len__(self) -> int:
        return 1

    def __getitem__(self, item: int) -> dict[str, Any]:
        if item != 0:
            raise IndexError(item)
        trajectory = torch.stack(
            [self.dataset.load_snapshot(index) for index in self.indices], dim=0
        )
        return {
            "physical_trajectory": trajectory,
            "trajectory_state": trajectory[0],
            "source_index": torch.tensor(self.indices, dtype=torch.int64),
            "time": torch.as_tensor(
                [self.dataset.times[index] for index in self.indices], dtype=torch.float64
            ),
        }


def save_upstream_training_bundle(
    save_dir: str | Path,
    save_name: str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    epoch: int,
    metadata: dict[str, Any],
) -> None:
    """Use upstream serialization and add only a GRMHD metadata sidecar."""
    save_dir = Path(save_dir)
    save_training_state(
        save_dir=save_dir,
        save_name=save_name,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        regularizer=None,
        epoch=epoch,
    )
    payload = dict(metadata)
    payload["neuraloperator_commit"] = PINNED_NEURALOP_COMMIT
    payload["save_name"] = save_name
    payload["epoch"] = int(epoch)
    (save_dir / f"{save_name}_grmhd_metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def load_upstream_training_bundle(
    save_dir: str | Path,
    save_name: str,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
) -> tuple[nn.Module, torch.optim.Optimizer | None, Any | None, int | None, dict[str, Any]]:
    """Load upstream state and verify the pinned-commit metadata sidecar."""
    save_dir = Path(save_dir)
    metadata = json.loads(
        (save_dir / f"{save_name}_grmhd_metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("neuraloperator_commit") != PINNED_NEURALOP_COMMIT:
        raise ValueError("Checkpoint neuraloperator commit mismatch")
    # The pinned BaseModel checkpoint embeds its activation callable in
    # ``_extra_state``.  PyTorch >=2.6 defaults torch.load to weights-only, so
    # narrowly allowlist the upstream FNO's GELU callable while delegating the
    # actual state restoration to the pinned training_state utility.
    with torch.serialization.safe_globals([torch._C._nn.gelu, SpectralConv]):
        model, optimizer, scheduler, _, epoch = load_training_state(
            save_dir=save_dir,
            save_name=save_name,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            regularizer=None,
            map_location="cpu",
        )
    return model, optimizer, scheduler, epoch, metadata
