"""Leakage-safe data protocol for the paper-adapted reduced100 reproduction."""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np
import yaml

from . import CHANNELS
from .dataset import GRMHDPairedDataset, TemporalSplit, sha256_file


PROTOCOL_NAME = "paper_reduced100_press_spherical_ks"
EXPECTED_UPSTREAM_COMMIT = "86a8bc7812a31b42c4f7895693cf4ac11521c066"


def _decode_strings(values: np.ndarray) -> list[str]:
    return [value.decode() if isinstance(value, bytes) else str(value) for value in values]


@dataclass(frozen=True)
class PaperReduced100Protocol:
    """Resolved two-way split and mandatory adaptation metadata.

    All ranges use Python's half-open convention.  For the canonical file this
    owns source snapshots 11..110, train snapshots 11..90, and validation
    snapshots 91..110.  Constructing this object validates the fixed scientific
    protocol; validating an HDF5 file is a separate explicit operation.
    """

    config_path: Path
    dataset_path: Path
    protocol_name: str
    source_snapshot_start: int
    source_snapshot_end: int
    train_snapshot_start: int
    train_snapshot_end: int
    validation_snapshot_start: int
    validation_snapshot_end: int
    stride: int
    thermal_channel: str
    paper_adaptation: bool
    eos_conversion: str
    spherical_ks_adaptation: bool
    coordinate_system: str
    channel_order: tuple[str, ...]
    expected_upstream_commit: str
    preprocessing: Mapping[str, Any]
    outputs: Mapping[str, str]

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        project_root: str | Path | None = None,
    ) -> "PaperReduced100Protocol":
        config_path = Path(path).resolve()
        values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("Paper protocol config must contain a YAML mapping")
        root = (
            Path(project_root).resolve()
            if project_root is not None
            else config_path.parents[2]
        )
        dataset_path = Path(values["dataset_path"])
        if not dataset_path.is_absolute():
            dataset_path = root / dataset_path
        outputs = {
            str(key): str(value)
            for key, value in dict(values.get("outputs", {})).items()
        }
        protocol = cls(
            config_path=config_path,
            dataset_path=dataset_path.resolve(),
            protocol_name=str(values["protocol_name"]),
            source_snapshot_start=int(values["source_snapshot_start"]),
            source_snapshot_end=int(values["source_snapshot_end"]),
            train_snapshot_start=int(values["train_snapshot_start"]),
            train_snapshot_end=int(values["train_snapshot_end"]),
            validation_snapshot_start=int(values["validation_snapshot_start"]),
            validation_snapshot_end=int(values["validation_snapshot_end"]),
            stride=int(values["stride"]),
            thermal_channel=str(values["thermal_channel"]),
            paper_adaptation=bool(values["paper_adaptation"]),
            eos_conversion=str(values["eos_conversion"]),
            spherical_ks_adaptation=bool(values["spherical_ks_adaptation"]),
            coordinate_system=str(values["coordinate_system"]),
            channel_order=tuple(str(item) for item in values["channel_order"]),
            expected_upstream_commit=str(values["expected_upstream_commit"]),
            preprocessing=dict(values.get("preprocessing", {})),
            outputs=outputs,
        )
        protocol.validate_static()
        return protocol

    def validate_static(self) -> None:
        if self.protocol_name != PROTOCOL_NAME:
            raise ValueError(
                f"Expected protocol_name={PROTOCOL_NAME!r}, found {self.protocol_name!r}"
            )
        expected_ranges = (11, 111, 11, 91, 91, 111, 1)
        actual_ranges = (
            self.source_snapshot_start,
            self.source_snapshot_end,
            self.train_snapshot_start,
            self.train_snapshot_end,
            self.validation_snapshot_start,
            self.validation_snapshot_end,
            self.stride,
        )
        if actual_ranges != expected_ranges:
            raise ValueError(
                "paper_reduced100 ranges/stride changed: "
                f"expected={expected_ranges}, actual={actual_ranges}"
            )
        if self.thermal_channel != "press":
            raise ValueError("paper_reduced100 must retain thermal_channel='press'")
        if not self.paper_adaptation:
            raise ValueError("paper_adaptation must remain true")
        if self.eos_conversion != "disabled_unverified_gamma":
            raise ValueError("EOS conversion must remain disabled while Gamma is unverified")
        if not self.spherical_ks_adaptation:
            raise ValueError("spherical_ks_adaptation must remain true")
        if self.channel_order != CHANNELS:
            raise ValueError(
                f"Expected channel order {CHANNELS}, found {self.channel_order}"
            )
        if self.expected_upstream_commit != EXPECTED_UPSTREAM_COMMIT:
            raise ValueError("Fixed upstream commit changed")
        if self.train_snapshot_end != self.validation_snapshot_start:
            raise ValueError("Train and validation snapshot ranges must be adjacent")

    @property
    def source_indices(self) -> tuple[int, ...]:
        return tuple(range(self.source_snapshot_start, self.source_snapshot_end))

    @property
    def train_indices(self) -> tuple[int, ...]:
        return tuple(range(self.train_snapshot_start, self.train_snapshot_end))

    @property
    def validation_indices(self) -> tuple[int, ...]:
        return tuple(range(self.validation_snapshot_start, self.validation_snapshot_end))

    @property
    def dropped_transition(self) -> tuple[int, int]:
        return self.train_snapshot_end - self.stride, self.validation_snapshot_start

    def validate_hdf5(self) -> dict[str, Any]:
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Paper protocol dataset is missing: {self.dataset_path}")
        with h5py.File(self.dataset_path, "r") as handle:
            required = {"snapshots", "times", "channels", "coords"}
            missing = sorted(required.difference(handle.keys()))
            if missing:
                raise KeyError(f"Paper protocol HDF5 is missing datasets/groups: {missing}")
            snapshots = handle["snapshots"]
            if snapshots.ndim != 5 or snapshots.shape[1] != len(CHANNELS):
                raise ValueError(f"Unexpected snapshot shape {snapshots.shape}")
            snapshot_shape = tuple(int(item) for item in snapshots.shape)
            if snapshots.shape[0] < self.source_snapshot_end:
                raise ValueError(
                    f"Dataset has {snapshots.shape[0]} snapshots; protocol needs index "
                    f"{self.source_snapshot_end - 1}"
                )
            channels = tuple(_decode_strings(handle["channels"][...]))
            if channels != self.channel_order:
                raise ValueError(
                    f"HDF5 channel order {channels} != protocol {self.channel_order}"
                )
            times = np.asarray(handle["times"][...], dtype=np.float64)
            if len(times) != snapshots.shape[0] or not np.all(np.diff(times) > 0):
                raise ValueError("HDF5 times must match snapshots and be strictly increasing")
            metadata_coordinates = str(
                handle["metadata"].attrs.get("coordinates", "")
            ) if "metadata" in handle else ""
            if metadata_coordinates and metadata_coordinates != self.coordinate_system:
                raise ValueError(
                    f"Coordinate metadata {metadata_coordinates!r} != "
                    f"protocol {self.coordinate_system!r}"
                )
        return {
            "snapshot_count": snapshot_shape[0],
            "snapshot_shape": list(snapshot_shape),
            "channel_order": list(channels),
            "coordinate_system": metadata_coordinates or self.coordinate_system,
        }

    def make_datasets(self) -> dict[str, GRMHDPairedDataset]:
        self.validate_hdf5()
        datasets = {
            "train": GRMHDPairedDataset(
                self.dataset_path,
                TemporalSplit("train", self.train_snapshot_start, self.train_snapshot_end),
                stride=self.stride,
            ),
            "validation": GRMHDPairedDataset(
                self.dataset_path,
                TemporalSplit(
                    "validation",
                    self.validation_snapshot_start,
                    self.validation_snapshot_end,
                ),
                stride=self.stride,
            ),
        }
        if len(datasets["train"]) != 79 or len(datasets["validation"]) != 19:
            raise RuntimeError("paper_reduced100 pair counts are not 79 train / 19 validation")
        train_pairs = {
            (int(index), int(index) + self.stride)
            for index in datasets["train"].pair_starts
        }
        validation_pairs = {
            (int(index), int(index) + self.stride)
            for index in datasets["validation"].pair_starts
        }
        if self.dropped_transition in train_pairs | validation_pairs:
            raise RuntimeError("Cross-boundary transition leaked into a dataset")
        return datasets

    def build_manifest(
        self,
        *,
        project_git_commit: str,
        project_git_dirty: bool,
        upstream_commit: str,
    ) -> dict[str, Any]:
        validation = self.validate_hdf5()
        if upstream_commit != self.expected_upstream_commit:
            raise ValueError(
                f"Upstream commit {upstream_commit} != fixed {self.expected_upstream_commit}"
            )
        datasets = self.make_datasets()
        with h5py.File(self.dataset_path, "r") as handle:
            times = np.asarray(handle["times"][...], dtype=np.float64)
            if "source_times" in handle:
                source_times = np.asarray(handle["source_times"][...], dtype=np.float64)
                source_time_dataset = "source_times"
            else:
                source_times = times
                source_time_dataset = "times (fallback; source_times absent)"
            source_files = (
                _decode_strings(handle["source_files"][...])
                if "source_files" in handle
                else [str(index) for index in range(len(times))]
            )
            metadata = (
                {
                    str(key): (
                        value.decode()
                        if isinstance(value, bytes)
                        else value.item()
                        if isinstance(value, np.generic)
                        else value
                    )
                    for key, value in handle["metadata"].attrs.items()
                }
                if "metadata" in handle
                else {}
            )
        source_records = [
            {
                "index": index,
                "source_file": source_files[index],
                "source_time": float(source_times[index]),
                "regridded_time": float(times[index]),
                "split": "train" if index in self.train_indices else "validation",
            }
            for index in self.source_indices
        ]
        dropped_start, dropped_target = self.dropped_transition
        return {
            "schema_version": "paper-reduced100-manifest-v1",
            "protocol_name": self.protocol_name,
            "reproduction_level": "paper_method_adapted",
            "dataset": {
                "configured_path": "data_proc/grmhd_regrid_inner_r200_64.h5",
                "resolved_path": str(self.dataset_path),
                "sha256": sha256_file(self.dataset_path),
                **validation,
                "hdf5_metadata": metadata,
            },
            "source": {
                "snapshot_range": [self.source_snapshot_start, self.source_snapshot_end],
                "snapshot_count": len(self.source_indices),
                "snapshot_indices": list(self.source_indices),
                "source_time_dataset": source_time_dataset,
                "records": source_records,
            },
            "splits": {
                "train": datasets["train"].manifest(),
                "validation": datasets["validation"].manifest(),
            },
            "test_split": None,
            "dropped_transitions": [
                {
                    "input_index": dropped_start,
                    "target_index": dropped_target,
                    "input_source_file": source_files[dropped_start],
                    "target_source_file": source_files[dropped_target],
                    "reason": "cross_train_validation_boundary",
                }
            ],
            "channel_order": list(self.channel_order),
            "coordinate_system": self.coordinate_system,
            "thermal_channel": self.thermal_channel,
            "paper_adaptation": self.paper_adaptation,
            "eos_conversion": self.eos_conversion,
            "spherical_ks_adaptation": self.spherical_ks_adaptation,
            "fit_policy": {
                "fit_split": "train",
                "fit_snapshot_indices": list(self.train_indices),
                "validation_excluded_from_all_fits": True,
            },
            "upstream_commit": upstream_commit,
            "project_git_commit": project_git_commit,
            "project_git_dirty": bool(project_git_dirty),
        }


def write_protocol_manifest(
    manifest: Mapping[str, Any],
    json_path: str | Path,
    csv_path: str | Path,
) -> None:
    """Write the complete JSON manifest and a one-row-per-snapshot CSV view."""

    json_path = Path(json_path)
    csv_path = Path(csv_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    dropped_inputs = {
        int(record["input_index"])
        for record in manifest["dropped_transitions"]
    }
    stride = int(manifest["splits"]["train"]["stride"])
    fieldnames = (
        "protocol_name",
        "index",
        "split",
        "source_file",
        "source_time",
        "regridded_time",
        "is_pair_input",
        "is_pair_target",
        "dropped_outgoing_transition",
        "thermal_channel",
        "paper_adaptation",
    )
    split_manifests = manifest["splits"]
    pair_inputs = {
        int(index)
        for split in split_manifests.values()
        for index in range(
            int(split["snapshot_range"][0]),
            int(split["snapshot_range"][1]) - stride,
        )
    }
    pair_targets = {index + stride for index in pair_inputs}
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in manifest["source"]["records"]:
            index = int(record["index"])
            writer.writerow(
                {
                    "protocol_name": manifest["protocol_name"],
                    **record,
                    "is_pair_input": index in pair_inputs,
                    "is_pair_target": index in pair_targets,
                    "dropped_outgoing_transition": index in dropped_inputs,
                    "thermal_channel": manifest["thermal_channel"],
                    "paper_adaptation": manifest["paper_adaptation"],
                }
            )
