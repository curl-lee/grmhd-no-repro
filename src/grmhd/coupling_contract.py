"""Minimal, testable coarse/fine coupling contract for Stage J.

This module is an engineering contract, not a claim that the paper specifies
these numerical operators.  It intentionally contains no trainer, optimizer,
checkpoint, or model dependency.  The arithmetic restriction and trilinear
prolongation are toy reference operators for interface testing only; they are
not conservative on the native nonuniform Kerr--Schild mesh and they do not
preserve a constrained-transport magnetic field.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F


SPATIAL_AXES = ("z", "y", "x")
VALID_FACES = tuple(
    f"{axis}_{side}" for axis in SPATIAL_AXES for side in ("low", "high")
)


def _triple(values: Sequence[Any], *, name: str, cast: type) -> tuple[Any, Any, Any]:
    result = tuple(cast(value) for value in values)
    if len(result) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    return result  # type: ignore[return-value]


@dataclass(frozen=True)
class GridMetadata:
    """Metadata required to interpret a ``[B, C, Z, Y, X]`` state tensor."""

    shape: tuple[int, int, int]
    channel_names: tuple[str, ...]
    coordinates: str
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    component_basis: str = "unspecified"
    magnetic_staggering: str = "unspecified"

    def __post_init__(self) -> None:
        shape = _triple(self.shape, name="shape", cast=int)
        spacing = _triple(self.spacing, name="spacing", cast=float)
        origin = _triple(self.origin, name="origin", cast=float)
        channels = tuple(str(name) for name in self.channel_names)
        if any(size <= 0 for size in shape):
            raise ValueError("shape entries must be positive")
        if any(not math.isfinite(step) or step <= 0.0 for step in spacing):
            raise ValueError("spacing entries must be finite and positive")
        if any(not math.isfinite(value) for value in origin):
            raise ValueError("origin entries must be finite")
        if not channels or len(set(channels)) != len(channels):
            raise ValueError("channel_names must be nonempty and unique")
        if not self.coordinates:
            raise ValueError("coordinates must be explicit")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "channel_names", channels)

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "channel_names": list(self.channel_names),
            "coordinates": self.coordinates,
            "spacing": list(self.spacing),
            "origin": list(self.origin),
            "component_basis": self.component_basis,
            "magnetic_staggering": self.magnetic_staggering,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "GridMetadata":
        return cls(
            shape=tuple(values["shape"]),
            channel_names=tuple(values["channel_names"]),
            coordinates=str(values["coordinates"]),
            spacing=tuple(values["spacing"]),
            origin=tuple(values.get("origin", (0.0, 0.0, 0.0))),
            component_basis=str(values.get("component_basis", "unspecified")),
            magnetic_staggering=str(values.get("magnetic_staggering", "unspecified")),
        )


def _validate_state_tensor(tensor: Tensor, grid: GridMetadata) -> None:
    if tensor.ndim != 5:
        raise ValueError("state tensor must have shape [B, C, Z, Y, X]")
    if tensor.shape[1] != len(grid.channel_names):
        raise ValueError("tensor channel count does not match channel_names")
    if tuple(tensor.shape[-3:]) != grid.shape:
        raise ValueError("tensor spatial shape does not match grid metadata")
    if not tensor.is_floating_point():
        raise TypeError("state tensor must use a floating-point dtype")


def _validate_time(time_index: int, physical_time: float) -> None:
    if time_index < 0:
        raise ValueError("time_index must be nonnegative")
    if not math.isfinite(physical_time):
        raise ValueError("physical_time must be finite")


@dataclass(frozen=True)
class CoarseState:
    tensor: Tensor
    grid: GridMetadata
    time_index: int
    physical_time: float
    autoregressive_owner: str = "coarse_solver"

    def __post_init__(self) -> None:
        _validate_state_tensor(self.tensor, self.grid)
        _validate_time(self.time_index, self.physical_time)
        if not self.autoregressive_owner:
            raise ValueError("autoregressive_owner must be explicit")

    def descriptor(self) -> dict[str, Any]:
        return _state_descriptor(self)


@dataclass(frozen=True)
class FineState:
    tensor: Tensor
    grid: GridMetadata
    time_index: int
    physical_time: float
    autoregressive_owner: str = "fine_solver"

    def __post_init__(self) -> None:
        _validate_state_tensor(self.tensor, self.grid)
        _validate_time(self.time_index, self.physical_time)
        if not self.autoregressive_owner:
            raise ValueError("autoregressive_owner must be explicit")

    def descriptor(self) -> dict[str, Any]:
        return _state_descriptor(self)


@dataclass(frozen=True)
class BoundaryState:
    """Full-grid boundary buffer; only declared ghost faces may be consumed."""

    tensor: Tensor
    grid: GridMetadata
    time_index: int
    physical_time: float
    ghost_width: int
    faces: tuple[str, ...] = VALID_FACES
    autoregressive_owner: str = "boundary_exchange"

    def __post_init__(self) -> None:
        _validate_state_tensor(self.tensor, self.grid)
        _validate_time(self.time_index, self.physical_time)
        faces = tuple(str(face) for face in self.faces)
        invalid = set(faces).difference(VALID_FACES)
        if invalid:
            raise ValueError(f"unknown boundary faces: {sorted(invalid)}")
        if not faces or len(set(faces)) != len(faces):
            raise ValueError("faces must be nonempty and unique")
        if self.ghost_width <= 0:
            raise ValueError("ghost_width must be positive")
        if any(2 * self.ghost_width > size for size in self.grid.shape):
            raise ValueError("ghost_width leaves no unambiguous interior")
        if not self.autoregressive_owner:
            raise ValueError("autoregressive_owner must be explicit")
        object.__setattr__(self, "faces", faces)

    def descriptor(self) -> dict[str, Any]:
        values = _state_descriptor(self)
        values.update({"ghost_width": self.ghost_width, "faces": list(self.faces)})
        return values


def _state_descriptor(state: CoarseState | FineState | BoundaryState) -> dict[str, Any]:
    return {
        "grid": state.grid.to_dict(),
        "time_index": state.time_index,
        "physical_time": state.physical_time,
        "autoregressive_owner": state.autoregressive_owner,
        "tensor_shape": list(state.tensor.shape),
        "tensor_dtype": str(state.tensor.dtype),
    }


class RestrictionOperator(nn.Module):
    """Arithmetic cell-average restriction for interface tests.

    This preserves constants and the global arithmetic mean when each axis is
    divisible by ``factor``.  It is not a physical-volume conservative mapping
    for nonuniform grids.
    """

    def __init__(self, factor: Sequence[int] = (2, 2, 2)) -> None:
        super().__init__()
        self.factor = _triple(factor, name="factor", cast=int)
        if any(value <= 0 for value in self.factor):
            raise ValueError("factor entries must be positive")

    def forward(self, fine: Tensor) -> Tensor:
        if fine.ndim != 5 or not fine.is_floating_point():
            raise ValueError("fine tensor must be floating [B, C, Z, Y, X]")
        if any(size % factor for size, factor in zip(fine.shape[-3:], self.factor)):
            raise ValueError("fine spatial shape must be divisible by factor")
        return F.avg_pool3d(fine, kernel_size=self.factor, stride=self.factor)

    def extra_repr(self) -> str:
        return f"factor={self.factor}, conservation=arithmetic_mean_only"


class ProlongationOperator(nn.Module):
    """Trilinear cell-center interpolation for interface tests."""

    def __init__(self, target_shape: Sequence[int]) -> None:
        super().__init__()
        self.target_shape = _triple(target_shape, name="target_shape", cast=int)
        if any(value <= 0 for value in self.target_shape):
            raise ValueError("target_shape entries must be positive")

    def forward(self, coarse: Tensor) -> Tensor:
        if coarse.ndim != 5 or not coarse.is_floating_point():
            raise ValueError("coarse tensor must be floating [B, C, Z, Y, X]")
        return F.interpolate(
            coarse,
            size=self.target_shape,
            mode="trilinear",
            align_corners=True,
        )

    def extra_repr(self) -> str:
        return f"target_shape={self.target_shape}, mode=trilinear"


class BoundaryExchange(nn.Module):
    """Overwrite authorized channels in ghost zones and leave interior untouched.

    Magnetic channels are rejected by default because this cell-centered toy
    exchange is not a constrained-transport/EMF boundary update.
    """

    def __init__(
        self,
        ghost_width: int,
        *,
        faces: Sequence[str] = VALID_FACES,
        channel_indices: Sequence[int] | None = None,
        magnetic_channel_indices: Sequence[int] = (),
        allow_unverified_magnetic_exchange: bool = False,
    ) -> None:
        super().__init__()
        self.ghost_width = int(ghost_width)
        self.faces = tuple(str(face) for face in faces)
        self.channel_indices = (
            None if channel_indices is None else tuple(int(index) for index in channel_indices)
        )
        self.magnetic_channel_indices = tuple(int(index) for index in magnetic_channel_indices)
        self.allow_unverified_magnetic_exchange = bool(allow_unverified_magnetic_exchange)
        if self.ghost_width <= 0:
            raise ValueError("ghost_width must be positive")
        invalid = set(self.faces).difference(VALID_FACES)
        if invalid or not self.faces or len(set(self.faces)) != len(self.faces):
            raise ValueError(f"faces must be unique members of {VALID_FACES}")
        if self.channel_indices is not None and (
            len(set(self.channel_indices)) != len(self.channel_indices)
            or any(index < 0 for index in self.channel_indices)
        ):
            raise ValueError("channel_indices must be unique and nonnegative")
        if any(index < 0 for index in self.magnetic_channel_indices):
            raise ValueError("magnetic_channel_indices must be nonnegative")
        selected = set(self.channel_indices or ())
        magnetic = selected.intersection(self.magnetic_channel_indices)
        if magnetic and not self.allow_unverified_magnetic_exchange:
            raise ValueError(
                "cell-centered magnetic overwrite is unverified; use a CT/EMF contract"
            )

    def _mask(self, reference: Tensor) -> Tensor:
        if reference.ndim != 5:
            raise ValueError("state tensors must have shape [B, C, Z, Y, X]")
        if any(2 * self.ghost_width > size for size in reference.shape[-3:]):
            raise ValueError("ghost_width leaves no unambiguous interior")
        spatial = torch.zeros(
            (1, 1, *reference.shape[-3:]), dtype=torch.bool, device=reference.device
        )
        axis_to_dim = {"z": 2, "y": 3, "x": 4}
        for face in self.faces:
            axis, side = face.split("_")
            index = [slice(None)] * 5
            dimension = axis_to_dim[axis]
            index[dimension] = (
                slice(0, self.ghost_width)
                if side == "low"
                else slice(-self.ghost_width, None)
            )
            spatial[tuple(index)] = True

        if self.channel_indices is None:
            return spatial
        channel = torch.zeros(
            (1, reference.shape[1], 1, 1, 1), dtype=torch.bool, device=reference.device
        )
        for index in self.channel_indices:
            if index >= reference.shape[1]:
                raise ValueError("channel index is outside the state tensor")
            channel[:, index] = True
        return spatial & channel

    def forward(self, interior: Tensor, boundary: Tensor) -> Tensor:
        if interior.shape != boundary.shape:
            raise ValueError("interior and boundary tensors must have identical shape")
        if interior.device != boundary.device or interior.dtype != boundary.dtype:
            raise ValueError("interior and boundary tensors must share device and dtype")
        if not interior.is_floating_point():
            raise TypeError("state tensors must use a floating-point dtype")
        return torch.where(self._mask(interior), boundary, interior)


@dataclass(frozen=True)
class TimeStepRatio:
    """Number of fine solver steps per one coarse solver step."""

    fine_steps_per_coarse: int

    def __post_init__(self) -> None:
        if self.fine_steps_per_coarse <= 0:
            raise ValueError("fine_steps_per_coarse must be positive")

    def to_dict(self) -> dict[str, int]:
        return {"fine_steps_per_coarse": self.fine_steps_per_coarse}

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "TimeStepRatio":
        return cls(fine_steps_per_coarse=int(values["fine_steps_per_coarse"]))


@dataclass(frozen=True)
class CouplingSchedule:
    """Integer-index synchronization schedule with explicit physical cadence."""

    ratio: TimeStepRatio
    coarse_start_index: int = 0
    fine_start_index: int = 0
    physical_start_time: float = 0.0
    coarse_dt: float = 1.0

    def __post_init__(self) -> None:
        if self.coarse_start_index < 0 or self.fine_start_index < 0:
            raise ValueError("start indices must be nonnegative")
        if not math.isfinite(self.physical_start_time):
            raise ValueError("physical_start_time must be finite")
        if not math.isfinite(self.coarse_dt) or self.coarse_dt <= 0.0:
            raise ValueError("coarse_dt must be finite and positive")

    @property
    def fine_dt(self) -> float:
        return self.coarse_dt / self.ratio.fine_steps_per_coarse

    def is_synchronization_step(self, fine_index: int) -> bool:
        offset = self._fine_offset(fine_index)
        return offset % self.ratio.fine_steps_per_coarse == 0

    def coarse_bracket(self, fine_index: int) -> tuple[int, int, float]:
        offset = self._fine_offset(fine_index)
        ratio = self.ratio.fine_steps_per_coarse
        quotient, remainder = divmod(offset, ratio)
        left = self.coarse_start_index + quotient
        if remainder == 0:
            return left, left, 0.0
        return left, left + 1, remainder / ratio

    def physical_time(self, fine_index: int) -> float:
        return self.physical_start_time + self._fine_offset(fine_index) * self.fine_dt

    def _fine_offset(self, fine_index: int) -> int:
        offset = int(fine_index) - self.fine_start_index
        if offset < 0:
            raise ValueError("fine_index precedes the coupling schedule")
        return offset

    def to_dict(self) -> dict[str, Any]:
        return {
            "ratio": self.ratio.to_dict(),
            "coarse_start_index": self.coarse_start_index,
            "fine_start_index": self.fine_start_index,
            "physical_start_time": self.physical_start_time,
            "coarse_dt": self.coarse_dt,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "CouplingSchedule":
        return cls(
            ratio=TimeStepRatio.from_dict(values["ratio"]),
            coarse_start_index=int(values.get("coarse_start_index", 0)),
            fine_start_index=int(values.get("fine_start_index", 0)),
            physical_start_time=float(values.get("physical_start_time", 0.0)),
            coarse_dt=float(values.get("coarse_dt", 1.0)),
        )


def interpolate_boundary(
    left: BoundaryState,
    right: BoundaryState,
    alpha: float,
    *,
    time_index: int,
    physical_time: float,
) -> BoundaryState:
    """Linearly interpolate compatible boundary buffers in physical state space."""

    if left.grid != right.grid:
        raise ValueError("boundary grids must match")
    if left.ghost_width != right.ghost_width or left.faces != right.faces:
        raise ValueError("boundary ghost geometry must match")
    if not 0.0 <= alpha <= 1.0 or not math.isfinite(alpha):
        raise ValueError("alpha must be finite and in [0, 1]")
    tensor = torch.lerp(left.tensor, right.tensor, float(alpha))
    return BoundaryState(
        tensor=tensor,
        grid=left.grid,
        time_index=time_index,
        physical_time=physical_time,
        ghost_width=left.ghost_width,
        faces=left.faces,
        autoregressive_owner=left.autoregressive_owner,
    )


__all__ = [
    "BoundaryExchange",
    "BoundaryState",
    "CoarseState",
    "CouplingSchedule",
    "FineState",
    "GridMetadata",
    "ProlongationOperator",
    "RestrictionOperator",
    "TimeStepRatio",
    "VALID_FACES",
    "interpolate_boundary",
]
