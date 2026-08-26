#!/usr/bin/env python
"""Regenerate the canonical/raw ROI mask audit from frozen Stage D inputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

from grmhd.dataset import sha256_file
from grmhd.paper_preprocessing import PaperPreprocessor
from grmhd.paper_prior_audit import audit_roi_split
from grmhd.paper_priors import read_prior_json, write_prior_json
from grmhd.paper_protocol import PaperReduced100Protocol
from grmhd.paper_velocity_roi import PaperVelocityROI


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    protocol = PaperReduced100Protocol.from_yaml(
        root / "configs/data/paper_reduced100.yaml", project_root=root
    )
    normalizer_path = root / "outputs/paper_reduced100/stats/normalizer.npz"
    prior_dir = root / "outputs/paper_reduced100/priors"
    hdf5_checksum = sha256_file(protocol.dataset_path)
    preprocessing_checksum = sha256_file(normalizer_path)
    preprocessor = PaperPreprocessor.load(
        normalizer_path,
        h5_path=protocol.dataset_path,
        expected_training_indices=protocol.train_indices,
        expected_protocol_name=protocol.protocol_name,
    )
    expected = {
        "source_hdf5_checksum": hdf5_checksum,
        "preprocessing_stats_checksum": preprocessing_checksum,
        "training_indices": protocol.train_indices,
        "protocol_name": protocol.protocol_name,
        "thermal_channel": protocol.thermal_channel,
    }
    roi = PaperVelocityROI.load(prior_dir / "velocity_roi.json", **expected)
    shell = read_prior_json(
        prior_dir / "shell_metadata.json",
        expected_schema="paper-shell-metadata-v1",
        **expected,
    )
    # Shell membership must use the 64 physical cell centres, not the eight edges.
    with h5py.File(protocol.dataset_path, "r") as handle:
        r = np.asarray(handle["coords/r"][...], dtype=np.float64)
    shell_index = np.digitize(r, np.asarray(shell["shells"]["edges"])[1:-1])
    audits = {}
    rows = []
    for split, indices in (
        ("train", protocol.train_indices),
        ("validation", protocol.validation_indices),
    ):
        audits[split] = audit_roi_split(
            protocol.dataset_path,
            indices,
            preprocessor=preprocessor,
            roi=roi,
            shell_index=shell_index,
        )
        rows.extend({"split": split, **record} for record in audits[split]["snapshot_records"])
    write_prior_json(
        prior_dir / "roi_stats.json",
        {
            "schema_version": "paper-roi-stats-v1",
            "provenance": roi.provenance.as_dict(),
            "parameters": roi.as_dict(),
            "splits": audits,
            "validation_used_for_fit": False,
        },
    )
    fields = (
        "split",
        "snapshot_index",
        "canonical_fraction",
        "raw_fraction",
        "jaccard",
        "disagreement_fraction",
        "canonical_clamp_overlap",
        "raw_clamp_overlap",
    )
    with (prior_dir / "roi_mask_audit.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# paper_reduced100 canonical/raw velocity ROI audit",
        "",
        "The main mask uses the canonical paper-decoded stored-component speed proxy.",
        "The raw physical mask is a diagnostic extension only; neither is a metric-correct",
        "relativistic speed in spherical Kerr--Schild coordinates.",
        "",
        "| split | canonical fraction | raw fraction | Jaccard | disagreement | canonical/clamp | raw/clamp |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for split, audit in audits.items():
        lines.append(
            f"| {split} | {audit['canonical_fraction']:.9g} | {audit['raw_fraction']:.9g} | "
            f"{audit['jaccard']:.9g} | {audit['disagreement_fraction']:.9g} | "
            f"{audit['canonical_clamp_overlap']:.9g} | {audit['raw_clamp_overlap']:.9g} |"
        )
    (prior_dir / "roi_mask_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "splits": list(audits)}, indent=2))


if __name__ == "__main__":
    main()
