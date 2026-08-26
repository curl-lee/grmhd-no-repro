"""Radial one-hot positional channels built from physical r coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class ShellMetadata:
    n_shells: int
    edges: tuple[float, ...]
    r_min: float
    r_max: float
    spacing: str = "log"
    coordinate: str = "spherical Kerr-Schild r"

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_shells": self.n_shells,
            "edges": list(self.edges),
            "r_min": self.r_min,
            "r_max": self.r_max,
            "spacing": self.spacing,
            "coordinate": self.coordinate,
        }


def radial_shells(
    r: np.ndarray,
    nphi: int,
    ntheta: int,
    *,
    n_shells: int = 8,
    dtype: np.dtype = np.float32,
) -> tuple[np.ndarray, ShellMetadata]:
    r = np.asarray(r, dtype=np.float64)
    if r.ndim != 1 or len(r) == 0 or not np.all(np.isfinite(r)):
        raise ValueError("r must be a non-empty finite one-dimensional array")
    if np.any(r <= 0) or np.any(np.diff(r) <= 0):
        raise ValueError("r must be positive and strictly increasing")
    if min(nphi, ntheta, n_shells) <= 0:
        raise ValueError("nphi, ntheta, and n_shells must be positive")
    edges = np.geomspace(float(r[0]), float(r[-1]), n_shells + 1)
    shell_index = np.digitize(r, edges[1:-1], right=False)
    radial = np.eye(n_shells, dtype=dtype)[shell_index].T
    shells = np.broadcast_to(radial[:, None, None, :], (n_shells, nphi, ntheta, len(r))).copy()
    if not np.all(shells.sum(axis=0) == 1):
        raise RuntimeError("Each voxel must belong to exactly one radial shell")
    metadata = ShellMetadata(
        n_shells=n_shells,
        edges=tuple(float(value) for value in edges),
        r_min=float(r[0]),
        r_max=float(r[-1]),
    )
    return shells, metadata


def radial_shells_tensor(
    r: np.ndarray,
    nphi: int,
    ntheta: int,
    *,
    n_shells: int = 8,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, ShellMetadata]:
    values, metadata = radial_shells(r, nphi, ntheta, n_shells=n_shells)
    return torch.from_numpy(values).to(device=device), metadata
