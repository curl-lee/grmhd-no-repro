"""Attach Stage AF anisotropic spherical DISCO branches to frozen LocalNO."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from grmhd.operators.anisotropic_spherical_disco3d import AnisotropicSphericalDISCO3d


def attach_anisotropic_spherical_disco3d(
    model: nn.Module,
    *,
    r: Sequence[float],
    theta: Sequence[float],
    phi: Sequence[float],
    groups: int = 1,
    bias: bool = True,
    seed: int = 42,
) -> nn.Module:
    blocks = getattr(model, "local_no_blocks", None)
    if blocks is None:
        raise TypeError("model does not expose LocalNO blocks")
    if len(blocks.local_convs) or any(index != -1 for index in blocks.disco_idx_list):
        raise ValueError("LocalNO already contains local-integral branches")
    if blocks.n_dim != 3 or int(blocks.n_layers) != 4:
        raise ValueError("Stage AF freezes four volumetric LocalNO layers")
    if int(blocks.in_channels) != int(blocks.out_channels):
        raise ValueError("Stage AF requires equal hidden widths")
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    branches = [
        AnisotropicSphericalDISCO3d(
            int(blocks.in_channels), int(blocks.out_channels),
            r=r, theta=theta, phi=phi, groups=groups, bias=bias,
        )
        for _ in range(4)
    ]
    blocks.local_convs = nn.ModuleList(branches)
    blocks.disco_layers = [True] * 4
    blocks.disco_idx_list = list(range(4))
    blocks.disco_kernel_shape = [5]
    blocks.disco_groups = int(groups)
    blocks.disco_bias = bool(bias)
    blocks.periodic = False
    return model


def anisotropic_disco_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, _ in model.named_parameters() if ".local_convs." in name]
