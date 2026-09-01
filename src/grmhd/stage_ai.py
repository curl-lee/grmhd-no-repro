"""Frozen model and comparison contracts for the Stage AI benchmark."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from grmhd.models import trainable_parameter_count
from grmhd.paper_stage_g import tensor_state_sha256
from grmhd.stage_u_training import build_stage_u_model, load_coords


CANONICAL_MODELS = (
    "Persistence",
    "FNO",
    "3D CNN/U-Net",
    "Differential LocalNO",
    "Index-space DISCO3D LocalNO",
    "Anisotropic spherical DISCO3D LocalNO",
)


class SphericalTensorPad3d(nn.Module):
    """Periodic phi padding and zero theta/r padding for (N,C,phi,theta,r)."""

    def __init__(self, width: int = 1) -> None:
        super().__init__()
        if width < 0:
            raise ValueError("padding width must be nonnegative")
        self.width = int(width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.width == 0:
            return value
        width = self.width
        value = F.pad(value, (0, 0, 0, 0, width, width), mode="circular")
        return F.pad(value, (width, width, width, width, 0, 0), mode="constant")


class BoundaryConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("Stage AI boundary convolution requires an odd kernel")
        self.pad = SphericalTensorPad3d(kernel_size // 2)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, padding=0)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(value))


class UNetBlock3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = 4 if out_channels % 4 == 0 else 1
        self.layers = nn.Sequential(
            BoundaryConv3d(in_channels, out_channels),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            BoundaryConv3d(out_channels, out_channels),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layers(value)


class ParameterMatchedUNet3d(nn.Module):
    """Two-level tensor-grid U-Net frozen by parameter-count arithmetic only."""

    def __init__(self, in_channels: int = 16, out_channels: int = 8, base: int = 16) -> None:
        super().__init__()
        self.encoder_0 = UNetBlock3d(in_channels, base)
        self.encoder_1 = UNetBlock3d(base, 2 * base)
        self.bottleneck = UNetBlock3d(2 * base, 4 * base)
        self.pool = nn.MaxPool3d(2)
        self.up_1 = nn.ConvTranspose3d(4 * base, 2 * base, kernel_size=2, stride=2)
        self.decoder_1 = UNetBlock3d(4 * base, 2 * base)
        self.up_0 = nn.ConvTranspose3d(2 * base, base, kernel_size=2, stride=2)
        self.decoder_0 = UNetBlock3d(2 * base, base)
        self.projection = nn.Conv3d(base, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, **_: Any) -> torch.Tensor:
        level_0 = self.encoder_0(x)
        level_1 = self.encoder_1(self.pool(level_0))
        center = self.bottleneck(self.pool(level_1))
        decoded_1 = self.decoder_1(torch.cat((self.up_1(center), level_1), dim=1))
        decoded_0 = self.decoder_0(torch.cat((self.up_0(decoded_1), level_0), dim=1))
        return self.projection(decoded_0)


def build_canonical_fno(
    stage_s: Mapping[str, Any], *, dataset: str, initial_state: Mapping[str, torch.Tensor]
) -> tuple[nn.Module, dict[str, Any]]:
    """Build the frozen spectral-only LocalNO backbone used as canonical FNO."""

    coords = load_coords(dataset)
    model, initialization = build_stage_u_model(
        stage_s,
        "spectral_only",
        coords=coords,
        initial_state=initial_state,
        padding={"phi": "periodic", "theta": "zero", "r": "zero"},
    )
    return model, initialization


def build_canonical_cnn(seed: int = 42) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = ParameterMatchedUNet3d(in_channels=16, out_channels=8, base=16)
    count = trainable_parameter_count(model)
    target = 363_480
    return model, {
        "method": "single_seed_default_PyTorch_initialization",
        "seed": int(seed),
        "base_channels": 16,
        "channel_widths": [16, 32, 64],
        "kernel_size": 3,
        "downsampling": "MaxPool3d(2)",
        "upsampling": "ConvTranspose3d(kernel=2,stride=2)",
        "normalization": "GroupNorm(4)",
        "activation": "SiLU",
        "phi_boundary": "circular",
        "theta_boundary": "zero",
        "r_boundary": "zero",
        "parameter_count": count,
        "target_parameter_count": target,
        "relative_parameter_difference": (count - target) / target,
        "trainable_state_sha256": tensor_state_sha256(dict(model.named_parameters())),
    }


def tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def parameter_match_pass(parameter_count: int, target: int = 363_480) -> bool:
    return abs(int(parameter_count) - int(target)) / int(target) <= 0.10


def first_landmark(values: Sequence[bool]) -> int | str:
    for index, triggered in enumerate(values, start=1):
        if bool(triggered):
            return index
    return "NOT_REACHED"
