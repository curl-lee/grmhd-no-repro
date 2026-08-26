from __future__ import annotations

import h5py
import numpy as np
import torch

from grmhd import CHANNELS
from grmhd.dataset import make_temporal_datasets, sha256_file


def make_file(path, count=12, shape=(4, 3, 2)):
    with h5py.File(path, "w") as handle:
        snapshots = np.empty((count, 8, *shape), dtype=np.float32)
        for index in range(count):
            snapshots[index] = index
        handle.create_dataset("snapshots", data=snapshots)
        handle.create_dataset("times", data=np.arange(count, dtype=np.float64) * 2.5)
        handle.create_dataset("channels", data=np.asarray(CHANNELS, dtype=h5py.string_dtype()))
        handle.create_dataset(
            "source_files",
            data=np.asarray([f"snapshot_{i}.athdf" for i in range(count)], dtype=h5py.string_dtype()),
        )
        coords = handle.create_group("coords")
        coords.create_dataset("r", data=np.geomspace(1.0, 10.0, shape[2]))
        coords.create_dataset("theta", data=np.linspace(0.1, 3.0, shape[1]))
        coords.create_dataset("phi", data=np.linspace(0.1, 6.0, shape[0]))


def test_pairs_order_shape_dtype_and_finite(tmp_path):
    path = tmp_path / "trajectory.h5"
    make_file(path)
    datasets = make_temporal_datasets(path, train_fraction=0.5, val_fraction=0.25)
    sample = datasets["train"][2]
    assert sample["index"] == 2
    assert sample["target_index"] == 3
    assert sample["time"] == 5.0
    assert sample["dt"] == 2.5
    assert sample["x"].shape == (8, 4, 3, 2)
    assert sample["x"].dtype == torch.float32
    assert torch.all(sample["x"] == 2)
    assert torch.all(sample["y"] == 3)
    assert torch.isfinite(sample["x"]).all() and torch.isfinite(sample["y"]).all()
    assert list(datasets["val"].pair_starts) == [6, 7]
    assert list(datasets["test"].pair_starts) == [9, 10]


def test_splits_have_no_snapshot_overlap(tmp_path):
    path = tmp_path / "trajectory.h5"
    make_file(path)
    datasets = make_temporal_datasets(path, train_fraction=0.5, val_fraction=0.25)
    owned = [set(dataset.owned_snapshot_indices) for dataset in datasets.values()]
    assert owned[0].isdisjoint(owned[1])
    assert owned[0].isdisjoint(owned[2])
    assert owned[1].isdisjoint(owned[2])
    assert datasets["val"].trajectory_indices() == (6, 7, 8)


def test_stride_and_training_time_range(tmp_path):
    path = tmp_path / "trajectory.h5"
    make_file(path)
    datasets = make_temporal_datasets(
        path,
        stride=2,
        train_fraction=0.75,
        val_fraction=0.10,
        train_time_range=(5.0, 17.5),
    )
    assert list(datasets["train"].pair_starts) == [2, 3, 4, 5]
    first = datasets["train"][0]
    assert first["index"] == 2 and first["target_index"] == 4
    assert first["dt"] == 5.0


def test_all111_explicit_split_counts_manifest_and_no_leakage(tmp_path):
    path = tmp_path / "trajectory_111.h5"
    make_file(path, count=111, shape=(1, 1, 1))
    datasets = make_temporal_datasets(
        path,
        snapshot_start=0,
        snapshot_end=111,
        train_snapshot_count=80,
        val_snapshot_count=10,
        test_snapshot_count=21,
    )
    assert {name: len(dataset) for name, dataset in datasets.items()} == {
        "train": 79,
        "val": 9,
        "test": 20,
    }
    assert datasets["train"].owned_snapshot_indices == tuple(range(80))
    assert datasets["val"].owned_snapshot_indices == tuple(range(80, 90))
    assert datasets["test"].owned_snapshot_indices == tuple(range(90, 111))
    manifest = datasets["test"].manifest()
    assert manifest["source_files"] == [f"snapshot_{i}.athdf" for i in range(90, 111)]
    assert manifest["times"] == [i * 2.5 for i in range(90, 111)]
    owned = [set(dataset.owned_snapshot_indices) for dataset in datasets.values()]
    assert all(owned[left].isdisjoint(owned[right]) for left, right in ((0, 1), (0, 2), (1, 2)))


def test_late100_explicit_window(tmp_path):
    path = tmp_path / "trajectory_111.h5"
    make_file(path, count=111, shape=(1, 1, 1))
    datasets = make_temporal_datasets(
        path,
        snapshot_start=-100,
        train_snapshot_count=70,
        val_snapshot_count=10,
        test_snapshot_count=20,
    )
    assert datasets["train"].split.start == 11
    assert datasets["test"].split.stop == 111
    assert [len(datasets[name]) for name in ("train", "val", "test")] == [69, 9, 19]


def test_sha256_changes_with_data(tmp_path):
    path = tmp_path / "trajectory.h5"
    make_file(path)
    before = sha256_file(path)
    with h5py.File(path, "r+") as handle:
        handle["snapshots"][0, 0, 0, 0, 0] = 123.0
    assert sha256_file(path) != before
