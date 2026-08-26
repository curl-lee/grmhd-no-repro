"""Attach the Stage AD local-integral branch to a frozen upstream LocalNO."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

from grmhd.operators.disco3d import AdaptedRadialDISCO3d


def attach_adapted_disco3d(
    model: nn.Module,
    *,
    grid_shape: Sequence[int] = (64, 64, 64),
    domain_length: Sequence[float] = (2.0, 2.0, 2.0),
    radius_cells: int = 3,
    basis_count: int = 5,
    groups: int = 1,
    bias: bool = True,
    seed: int = 42,
) -> nn.Module:
    """Enable four adapted 3-D DISCO branches without touching upstream code.

    The input model must be the frozen Stage-T differential LocalNO with its
    common state already loaded.  Seeding happens immediately before construction
    so only the newly authorized DISCO parameters consume this RNG stream.
    """

    blocks = getattr(model, "local_no_blocks", None)
    if blocks is None:
        raise TypeError("model does not expose upstream LocalNO blocks")
    if len(blocks.local_convs) or any(index != -1 for index in blocks.disco_idx_list):
        raise ValueError("LocalNO already contains a local-integral branch")
    if blocks.n_dim != 3 or int(blocks.n_layers) != 4:
        raise ValueError("Stage AD requires four volumetric LocalNO layers")
    if int(blocks.in_channels) != int(blocks.out_channels):
        raise ValueError("Stage AD requires equal hidden input/output widths")

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    branches = [
        AdaptedRadialDISCO3d(
            int(blocks.in_channels),
            int(blocks.out_channels),
            grid_shape=grid_shape,
            domain_length=domain_length,
            radius_cells=radius_cells,
            basis_count=basis_count,
            groups=groups,
            bias=bias,
        )
        for _ in range(int(blocks.n_layers))
    ]
    blocks.local_convs = nn.ModuleList(branches)
    blocks.disco_layers = [True] * int(blocks.n_layers)
    blocks.disco_idx_list = list(range(int(blocks.n_layers)))
    blocks.disco_kernel_shape = [basis_count]
    blocks.radius_cutoff = branches[0].radius_cutoff
    blocks.domain_length = list(float(value) for value in domain_length)
    blocks.disco_groups = int(groups)
    blocks.disco_bias = bool(bias)
    blocks.periodic = True
    return model


def disco_parameter_names(model: nn.Module) -> list[str]:
    """Return the exact trainable parameter names introduced by Stage AD."""

    return [name for name, _ in model.named_parameters() if ".local_convs." in name]
