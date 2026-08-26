#!/usr/bin/env python3
"""Fit and freeze the expanded-train-only Stage S P3 normalizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

import h5py
import numpy as np

from grmhd import CHANNELS
from grmhd.dataset import sha256_file
from grmhd.paper_preprocessing import (
    PAPER_GAMMA,
    PAPER_INVERSE_CLAMP_FRACTION,
    PaperPreprocessor,
)
from grmhd.paper_stage_n import NO_SOFTCLIP, PrototypePreprocessor, prototype_specs


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=Path,
        default=Path("data_proc/grmhd_regrid_inner_r200_64_expanded.h5"),
    )
    parser.add_argument(
        "--split", type=Path, default=Path("artifacts/stage_s/train_val_split.json")
    )
    parser.add_argument(
        "--audit", type=Path, default=Path("artifacts/stage_s/expanded_data_audit.json")
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("artifacts/stage_s/p3_normalizer_expanded"),
    )
    args = parser.parse_args()

    dataset = args.dataset.resolve()
    split = read_json(args.split)
    audit = read_json(args.audit)
    expected_sha = audit["processed_dataset"]["sha256"]
    observed_sha = sha256_file(dataset)
    if observed_sha != expected_sha:
        raise ValueError("Expanded HDF5 checksum differs from the completed data audit")
    if not audit["compatibility"]["operationally_compatible"]:
        raise ValueError("Expanded data compatibility gate did not pass")
    if not audit["summary"]["time_series_contiguous"]:
        raise ValueError("Expanded time-series continuity gate did not pass")
    if not audit["summary"]["grid_layout_static"]:
        raise ValueError("Expanded grid-layout gate did not pass")
    if not split["validation_held_out"] or split["validation_used_for_fit"]:
        raise ValueError("Stage S validation holdout contract changed")

    train_indices = tuple(int(value) for value in split["train_snapshot_indices"])
    validation_indices = tuple(int(value) for value in split["validation_snapshot_indices"])
    if train_indices != tuple(range(0, 169)):
        raise ValueError("Unexpected Stage S chronological training split")
    if validation_indices != tuple(range(169, 212)):
        raise ValueError("Unexpected Stage S chronological validation split")
    if set(train_indices) & set(validation_indices):
        raise ValueError("Training and validation snapshots overlap")

    base = PaperPreprocessor.fit_hdf5(
        dataset,
        training_indices=train_indices,
        protocol_name="stage_s_expanded_train_only_p3",
        expected_source_hdf5_checksum=observed_sha,
        gamma=PAPER_GAMMA,
        inverse_clamp_fraction=PAPER_INVERSE_CLAMP_FRACTION,
        thermal_channel="press",
        paper_adaptation=True,
        eos_conversion="disabled_unverified_gamma",
    )
    spec = prototype_specs(NO_SOFTCLIP)["P3"]
    processor = PrototypePreprocessor(base, spec)
    artifacts = processor.save(args.output_dir)

    checks: list[dict[str, Any]] = []
    with h5py.File(dataset, "r") as handle:
        snapshots = handle["snapshots"]
        for split_name, indices in (
            ("train", (train_indices[0], train_indices[len(train_indices) // 2], train_indices[-1])),
            ("validation_check_only", (validation_indices[0], validation_indices[-1])),
        ):
            for index in indices:
                raw = np.asarray(snapshots[index], dtype=np.float32)
                encoded, decoded = processor.round_trip(raw, channel_axis=0)
                checks.append(
                    {
                        "split": split_name,
                        "snapshot_index": index,
                        "raw_finite": bool(np.isfinite(raw).all()),
                        "encoded_finite": bool(np.isfinite(encoded).all()),
                        "decoded_finite": bool(np.isfinite(decoded).all()),
                        "rho_positive": bool(np.all(decoded[3] > 0)),
                        "press_positive": bool(np.all(decoded[4] > 0)),
                    }
                )
    if not all(
        row[key]
        for row in checks
        for key in ("raw_finite", "encoded_finite", "decoded_finite", "rho_positive", "press_positive")
    ):
        raise FloatingPointError("Expanded P3 finite/positivity check failed")

    # Required convenient root-level artifacts; the conventional directory is
    # retained so PrototypePreprocessor.load can enforce its metadata contract.
    prefixed_npz = args.output_dir.parent / "p3_normalizer_expanded.npz"
    shutil.copyfile(args.output_dir / "normalizer.npz", prefixed_npz)
    prefixed_json = args.output_dir.parent / "p3_normalizer_expanded.json"
    prefixed_payload = {
        "schema_version": "stage-s-expanded-p3-normalizer-v1",
        "directory": str(args.output_dir),
        "prototype": spec.as_dict(),
        "train_snapshot_range": [train_indices[0], train_indices[-1]],
        "train_snapshot_count": len(train_indices),
        "validation_indices_used_for_fit": [],
        "normalizer": processor.normalizer_metadata(),
        "artifacts": artifacts,
        "prefixed_npz_sha256": sha256_file(prefixed_npz),
        "checks": checks,
    }
    write_json(prefixed_json, prefixed_payload)

    provenance = {
        "schema_version": "stage-s-preprocessing-provenance-v1",
        "status": "passed",
        "project_commit": git_value("rev-parse", "HEAD"),
        "project_worktree_dirty_during_generation": bool(git_value("status", "--short")),
        "upstream_commit": git_value("-C", "external/neuraloperator", "rev-parse", "HEAD"),
        "dataset": {"path": str(dataset), "sha256": observed_sha},
        "split": {
            "path": str(args.split),
            "sha256": sha256_file(args.split),
            "training_indices": list(train_indices),
            "validation_indices_used_for_fit": [],
        },
        "mapping": dict(zip(CHANNELS, spec.channel_policies, strict=True)),
        "gamma": processor.gamma,
        "inverse_clamp_fraction": processor.inverse_clamp_fraction,
        "fit_scope": processor.base.fit_scope,
        "sample_count_per_channel": processor.base.sample_count_per_channel,
        "artifacts": {
            **artifacts,
            "prefixed_npz": str(prefixed_npz),
            "prefixed_npz_sha256": sha256_file(prefixed_npz),
            "prefixed_json": str(prefixed_json),
            "prefixed_json_sha256": sha256_file(prefixed_json),
        },
        "source_code": {
            "paper_preprocessing.py": sha256_file(Path("src/grmhd/paper_preprocessing.py")),
            "paper_stage_n.py": sha256_file(Path("src/grmhd/paper_stage_n.py")),
            "prepare_script": sha256_file(Path(__file__)),
        },
        "round_trip_checks": checks,
    }
    write_json(args.output_dir.parent / "preprocessing_provenance.json", provenance)
    print(json.dumps(provenance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
