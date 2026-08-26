from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

from grmhd import CHANNELS
from grmhd.dataset import sha256_file
from grmhd.paper_protocol import (
    EXPECTED_UPSTREAM_COMMIT,
    PROTOCOL_NAME,
    PaperReduced100Protocol,
    write_protocol_manifest,
)


def make_trajectory(path: Path) -> None:
    shape = (1, 1, 1)
    snapshots = np.ones((111, 8, *shape), dtype=np.float32)
    snapshots[:, 0:3] *= np.arange(111, dtype=np.float32)[:, None, None, None, None]
    snapshots[:, 3] *= 2.0
    snapshots[:, 4] *= 0.25
    with h5py.File(path, "w") as handle:
        handle.create_dataset("snapshots", data=snapshots)
        handle.create_dataset("times", data=np.arange(111, dtype=np.float64) * 10.0)
        handle.create_dataset("source_times", data=np.arange(111, dtype=np.float64) * 10.0 + 0.5)
        handle.create_dataset("channels", data=np.asarray(CHANNELS, dtype=h5py.string_dtype()))
        handle.create_dataset(
            "source_files",
            data=np.asarray(
                [f"mad98.prim.{index:05d}.athdf" for index in range(111)],
                dtype=h5py.string_dtype(),
            ),
        )
        coords = handle.create_group("coords")
        coords.create_dataset("r", data=np.asarray([2.0]))
        coords.create_dataset("theta", data=np.asarray([1.0]))
        coords.create_dataset("phi", data=np.asarray([0.0]))
        metadata = handle.create_group("metadata")
        metadata.attrs["coordinates"] = "spherical Kerr-Schild: phi,theta,r"
        metadata.attrs["axis_order"] = "N,C,Nphi,Ntheta,Nr"


def write_config(path: Path, dataset_path: Path) -> None:
    values = {
        "dataset_path": str(dataset_path),
        "protocol_name": PROTOCOL_NAME,
        "source_snapshot_start": 11,
        "source_snapshot_end": 111,
        "train_snapshot_start": 11,
        "train_snapshot_end": 91,
        "validation_snapshot_start": 91,
        "validation_snapshot_end": 111,
        "stride": 1,
        "thermal_channel": "press",
        "paper_adaptation": True,
        "eos_conversion": "disabled_unverified_gamma",
        "spherical_ks_adaptation": True,
        "coordinate_system": "spherical Kerr-Schild: phi,theta,r",
        "channel_order": list(CHANNELS),
        "expected_upstream_commit": EXPECTED_UPSTREAM_COMMIT,
        "preprocessing": {"fit_split": "train"},
        "outputs": {},
    }
    path.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")


def test_canonical_config_locks_indices_pairs_and_press_adaptation():
    protocol = PaperReduced100Protocol.from_yaml("configs/data/paper_reduced100.yaml")
    assert protocol.source_indices == tuple(range(11, 111))
    assert protocol.train_indices == tuple(range(11, 91))
    assert protocol.validation_indices == tuple(range(91, 111))
    assert protocol.dropped_transition == (90, 91)
    assert protocol.thermal_channel == "press"
    assert protocol.paper_adaptation is True
    assert protocol.eos_conversion == "disabled_unverified_gamma"
    assert protocol.spherical_ks_adaptation is True
    assert protocol.channel_order == CHANNELS


def test_protocol_datasets_manifest_csv_and_no_cross_boundary_leakage(tmp_path):
    h5_path = tmp_path / "trajectory.h5"
    config_path = tmp_path / "paper.yaml"
    make_trajectory(h5_path)
    write_config(config_path, h5_path)
    protocol = PaperReduced100Protocol.from_yaml(config_path, project_root=tmp_path)
    datasets = protocol.make_datasets()

    assert set(datasets) == {"train", "validation"}
    assert len(datasets["train"]) == 79
    assert len(datasets["validation"]) == 19
    assert datasets["train"].owned_snapshot_indices == tuple(range(11, 91))
    assert datasets["validation"].owned_snapshot_indices == tuple(range(91, 111))
    assert list(datasets["train"].pair_starts) == list(range(11, 90))
    assert list(datasets["validation"].pair_starts) == list(range(91, 110))
    pairs = {
        (int(index), int(index) + 1)
        for dataset in datasets.values()
        for index in dataset.pair_starts
    }
    assert (90, 91) not in pairs
    assert set(datasets["train"].owned_snapshot_indices).isdisjoint(
        datasets["validation"].owned_snapshot_indices
    )

    manifest = protocol.build_manifest(
        project_git_commit="abc123",
        project_git_dirty=True,
        upstream_commit=EXPECTED_UPSTREAM_COMMIT,
    )
    assert manifest["dataset"]["sha256"] == sha256_file(h5_path)
    assert manifest["source"]["snapshot_count"] == 100
    assert len(manifest["source"]["records"]) == 100
    assert manifest["source"]["records"][0] == {
        "index": 11,
        "source_file": "mad98.prim.00011.athdf",
        "source_time": 110.5,
        "regridded_time": 110.0,
        "split": "train",
    }
    assert manifest["splits"]["train"]["pair_count"] == 79
    assert manifest["splits"]["validation"]["pair_count"] == 19
    assert manifest["test_split"] is None
    assert manifest["dropped_transitions"] == [
        {
            "input_index": 90,
            "target_index": 91,
            "input_source_file": "mad98.prim.00090.athdf",
            "target_source_file": "mad98.prim.00091.athdf",
            "reason": "cross_train_validation_boundary",
        }
    ]
    assert manifest["fit_policy"] == {
        "fit_split": "train",
        "fit_snapshot_indices": list(range(11, 91)),
        "validation_excluded_from_all_fits": True,
    }
    assert manifest["thermal_channel"] == "press"
    assert manifest["paper_adaptation"] is True
    assert manifest["upstream_commit"] == EXPECTED_UPSTREAM_COMMIT
    assert manifest["project_git_commit"] == "abc123"

    json_path = tmp_path / "manifest.json"
    csv_path = tmp_path / "manifest.csv"
    write_protocol_manifest(manifest, json_path, csv_path)
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["source"]["snapshot_indices"] == list(range(11, 111))
    with csv_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 100
    row_90 = next(row for row in rows if row["index"] == "90")
    assert row_90["split"] == "train"
    assert row_90["is_pair_input"] == "False"
    assert row_90["is_pair_target"] == "True"
    assert row_90["dropped_outgoing_transition"] == "True"
    row_91 = next(row for row in rows if row["index"] == "91")
    assert row_91["split"] == "validation"
    assert row_91["is_pair_input"] == "True"
    assert row_91["is_pair_target"] == "False"


def test_protocol_rejects_missing_adaptation_or_wrong_upstream(tmp_path):
    h5_path = tmp_path / "trajectory.h5"
    config_path = tmp_path / "paper.yaml"
    make_trajectory(h5_path)
    write_config(config_path, h5_path)
    protocol = PaperReduced100Protocol.from_yaml(config_path, project_root=tmp_path)
    with pytest.raises(ValueError, match="thermal_channel='press'"):
        replace(protocol, thermal_channel="eint").validate_static()
    with pytest.raises(ValueError, match="paper_adaptation"):
        replace(protocol, paper_adaptation=False).validate_static()
    with pytest.raises(ValueError, match="EOS conversion"):
        replace(protocol, eos_conversion="gamma_4_3").validate_static()
    with pytest.raises(ValueError, match="Upstream commit"):
        protocol.build_manifest(
            project_git_commit="abc123",
            project_git_dirty=False,
            upstream_commit="wrong",
        )
