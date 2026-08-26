"""Lazy, leakage-safe temporal datasets for regridded GRMHD trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
from typing import Callable, Iterator, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from . import CHANNELS


def _decode(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


@dataclass(frozen=True)
class TemporalSplit:
    name: str
    start: int
    stop: int

    @property
    def snapshot_indices(self) -> tuple[int, ...]:
        return tuple(range(self.start, self.stop))

    @property
    def snapshot_count(self) -> int:
        return self.stop - self.start


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def contiguous_snapshot_splits(
    snapshot_count: int,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    *,
    snapshot_start: int = 0,
    snapshot_end: int | None = None,
    train_snapshot_count: int | None = None,
    val_snapshot_count: int | None = None,
    test_snapshot_count: int | None = None,
) -> dict[str, TemporalSplit]:
    if snapshot_count < 3:
        raise ValueError("At least three snapshots are needed for temporal splits")
    resolved_start = snapshot_start + snapshot_count if snapshot_start < 0 else snapshot_start
    requested_end = snapshot_count if snapshot_end is None else snapshot_end
    resolved_end = requested_end + snapshot_count if requested_end < 0 else requested_end
    if not 0 <= resolved_start < resolved_end <= snapshot_count:
        raise ValueError(
            f"Snapshot window [{snapshot_start}, {snapshot_end}) resolves to "
            f"[{resolved_start}, {resolved_end}) outside trajectory length {snapshot_count}"
        )
    window_count = resolved_end - resolved_start
    explicit = (train_snapshot_count, val_snapshot_count, test_snapshot_count)
    if any(value is not None for value in explicit):
        if not all(value is not None for value in explicit):
            raise ValueError("train/val/test snapshot counts must be specified together")
        train_count, val_count, test_count = (int(value) for value in explicit)
        if min(train_count, val_count, test_count) < 2:
            raise ValueError("Each explicit split needs at least two snapshots")
        if train_count + val_count + test_count != window_count:
            raise ValueError(
                f"Explicit split counts sum to {train_count + val_count + test_count}, "
                f"but selected window has {window_count} snapshots"
            )
        train_stop = resolved_start + train_count
        val_stop = train_stop + val_count
        return {
            "train": TemporalSplit("train", resolved_start, train_stop),
            "val": TemporalSplit("val", train_stop, val_stop),
            "test": TemporalSplit("test", val_stop, resolved_end),
        }
    if not (0 < train_fraction < 1 and 0 <= val_fraction < 1):
        raise ValueError("Invalid train/validation fractions")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be < 1")
    train_stop = resolved_start + max(1, int(np.floor(window_count * train_fraction)))
    val_stop = resolved_start + max(
        train_stop - resolved_start,
        int(np.floor(window_count * (train_fraction + val_fraction))),
    )
    # Keep every non-empty requested split usable whenever the trajectory permits.
    if window_count >= 6:
        train_stop = min(max(train_stop, resolved_start + 2), resolved_end - 4)
        val_stop = min(max(val_stop, train_stop + 2), resolved_end - 2)
    return {
        "train": TemporalSplit("train", resolved_start, train_stop),
        "val": TemporalSplit("val", train_stop, val_stop),
        "test": TemporalSplit("test", val_stop, resolved_end),
    }


class GRMHDPairedDataset(Dataset[dict[str, object]]):
    """Return ``snapshot[i] -> snapshot[i + stride]`` pairs lazily.

    A split owns a contiguous, non-overlapping range of snapshot indices.  Pairs
    that would cross a split boundary are omitted, preventing the same physical
    snapshot from appearing in two splits.
    """

    def __init__(
        self,
        h5_path: str | Path,
        split: TemporalSplit,
        *,
        stride: int = 1,
        time_range: tuple[float | None, float | None] | None = None,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        validate_finite: bool = True,
    ) -> None:
        super().__init__()
        if stride <= 0:
            raise ValueError("stride must be positive")
        self.h5_path = Path(h5_path).resolve()
        self.split = split
        self.stride = int(stride)
        self.transform = transform
        self.validate_finite = validate_finite
        self._handle: h5py.File | None = None
        self._handle_pid: int | None = None

        with h5py.File(self.h5_path, "r") as handle:
            if "snapshots" not in handle or "times" not in handle:
                raise KeyError("HDF5 file must contain snapshots and times")
            snapshots = handle["snapshots"]
            times = np.asarray(handle["times"][...], dtype=np.float64)
            channels = _decode(handle["channels"][...])
            if channels != list(CHANNELS):
                raise ValueError(f"Expected channel order {CHANNELS}, found {channels}")
            if snapshots.ndim != 5 or snapshots.shape[1] != len(CHANNELS):
                raise ValueError(f"Expected snapshots (N,8,Nphi,Ntheta,Nr), found {snapshots.shape}")
            if len(times) != snapshots.shape[0]:
                raise ValueError("times and snapshots length differ")
            if not np.all(np.diff(times) > 0):
                raise ValueError("Snapshot times must be strictly increasing")
            if not (0 <= split.start <= split.stop <= len(times)):
                raise ValueError(f"Split {split} lies outside trajectory of length {len(times)}")
            self.snapshot_shape = tuple(int(value) for value in snapshots.shape[1:])
            self.times = times
            self.coords = {
                axis: np.asarray(handle[f"coords/{axis}"][...], dtype=np.float64)
                for axis in ("r", "theta", "phi")
            }
            self.source_files = (
                _decode(handle["source_files"][...])
                if "source_files" in handle
                else [str(index) for index in range(len(times))]
            )
            self.hdf5_metadata = {
                "format_version": str(handle.attrs.get("format_version", "unknown")),
                "metadata_json": str(handle.attrs.get("metadata_json", "{}")),
            }

        starts = np.arange(split.start, max(split.stop - self.stride, split.start), dtype=np.int64)
        if time_range is not None:
            lower, upper = time_range
            keep = np.ones(starts.shape, dtype=bool)
            if lower is not None:
                keep &= self.times[starts] >= lower
            if upper is not None:
                keep &= self.times[starts + self.stride] <= upper
            starts = starts[keep]
        self.pair_starts = starts

    def _get_handle(self) -> h5py.File:
        pid = os.getpid()
        if self._handle is None or self._handle_pid != pid:
            if self._handle is not None:
                self._handle.close()
            self._handle = h5py.File(self.h5_path, "r")
            self._handle_pid = pid
        return self._handle

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_handle"] = None
        state["_handle_pid"] = None
        return state

    def __del__(self) -> None:
        self.close()

    def close(self) -> None:
        """Close this process's lazily opened HDF5 handle, if any."""
        handle = getattr(self, "_handle", None)
        if handle is not None:
            try:
                handle.close()
            except Exception:
                pass
        self._handle = None
        self._handle_pid = None

    def __len__(self) -> int:
        return int(len(self.pair_starts))

    def load_snapshot(self, index: int) -> torch.Tensor:
        if index not in range(self.split.start, self.split.stop):
            raise IndexError(f"snapshot {index} not owned by split {self.split}")
        values = np.asarray(self._get_handle()["snapshots"][index], dtype=np.float32)
        tensor = torch.from_numpy(values)
        if self.validate_finite and not torch.isfinite(tensor).all():
            raise FloatingPointError(f"snapshot {index} contains NaN/Inf")
        return tensor

    def __getitem__(self, item: int) -> dict[str, object]:
        index = int(self.pair_starts[item])
        target_index = index + self.stride
        x = self.load_snapshot(index)
        y = self.load_snapshot(target_index)
        if self.transform is not None:
            x, y = self.transform(x), self.transform(y)
        return {
            "x": x,
            "y": y,
            "index": index,
            "target_index": target_index,
            "time": float(self.times[index]),
            "target_time": float(self.times[target_index]),
            "dt": float(self.times[target_index] - self.times[index]),
            "source_file": self.source_files[index],
            "target_source_file": self.source_files[target_index],
        }

    @property
    def owned_snapshot_indices(self) -> tuple[int, ...]:
        return self.split.snapshot_indices

    def trajectory_indices(self) -> tuple[int, ...]:
        """Indices for an ordered ground-truth rollout trajectory."""
        return tuple(range(self.split.start, self.split.stop, self.stride))

    def iter_trajectory(self) -> Iterator[tuple[int, float, torch.Tensor]]:
        for index in self.trajectory_indices():
            yield index, float(self.times[index]), self.load_snapshot(index)

    def manifest(self) -> dict[str, object]:
        return {
            "split": self.split.name,
            "snapshot_range": [self.split.start, self.split.stop],
            "snapshot_count": self.split.snapshot_count,
            "pair_count": len(self),
            "stride": self.stride,
            "snapshot_indices": list(self.owned_snapshot_indices),
            "source_files": [self.source_files[index] for index in self.owned_snapshot_indices],
            "times": [float(self.times[index]) for index in self.owned_snapshot_indices],
        }


def make_temporal_datasets(
    h5_path: str | Path,
    *,
    stride: int = 1,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    snapshot_start: int = 0,
    snapshot_end: int | None = None,
    train_snapshot_count: int | None = None,
    val_snapshot_count: int | None = None,
    test_snapshot_count: int | None = None,
    train_time_range: tuple[float | None, float | None] | None = None,
    transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> dict[str, GRMHDPairedDataset]:
    with h5py.File(h5_path, "r") as handle:
        snapshot_count = int(handle["snapshots"].shape[0])
    splits = contiguous_snapshot_splits(
        snapshot_count,
        train_fraction,
        val_fraction,
        snapshot_start=snapshot_start,
        snapshot_end=snapshot_end,
        train_snapshot_count=train_snapshot_count,
        val_snapshot_count=val_snapshot_count,
        test_snapshot_count=test_snapshot_count,
    )
    return {
        name: GRMHDPairedDataset(
            h5_path,
            split,
            stride=stride,
            time_range=train_time_range if name == "train" else None,
            transform=transform,
        )
        for name, split in splits.items()
    }


def discover_regrid_files(pattern: str = "data_proc/*.h5") -> Sequence[Path]:
    from glob import glob

    return tuple(Path(path) for path in sorted(glob(pattern)))
