#!/usr/bin/env python
"""Reload and independently audit Stage D artifacts without a neural network."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grmhd.dataset import sha256_file
from grmhd.paper_bounds import PaperPhysicalBounds
from grmhd.paper_dissipation import PaperDissipativeReference
from grmhd.paper_preprocessing import PaperPreprocessor
from grmhd.paper_prior_audit import audit_stage_d_split, write_stage_d_audit
from grmhd.paper_priors import PaperResidualEnvelope, read_prior_json
from grmhd.paper_protocol import PaperReduced100Protocol
from grmhd.paper_radial import PaperRadialBaseline
from grmhd.paper_velocity_roi import PaperVelocityROI


EXPECTED_HDF5_SHA256 = "cd3966bed1f617f49d775151cff87ae92eb2432a560bb82ae9355605e7aea70a"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config", type=Path, default=Path("configs/data/paper_reduced100.yaml")
    )
    parser.add_argument(
        "--normalizer", type=Path, default=Path("outputs/paper_reduced100/stats/normalizer.npz")
    )
    parser.add_argument(
        "--prior-dir", type=Path, default=Path("outputs/paper_reduced100/priors")
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    normalizer_path = args.normalizer if args.normalizer.is_absolute() else root / args.normalizer
    prior_dir = args.prior_dir if args.prior_dir.is_absolute() else root / args.prior_dir
    protocol = PaperReduced100Protocol.from_yaml(config, project_root=root)
    hdf5_checksum = sha256_file(protocol.dataset_path)
    if hdf5_checksum != EXPECTED_HDF5_SHA256:
        raise ValueError("Stage D canonical HDF5 checksum mismatch")
    preprocessing_checksum = sha256_file(normalizer_path)
    preprocessor = PaperPreprocessor.load(
        normalizer_path,
        h5_path=protocol.dataset_path,
        expected_source_hdf5_checksum=hdf5_checksum,
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
    shell = read_prior_json(
        prior_dir / "shell_metadata.json",
        expected_schema="paper-shell-metadata-v1",
        **expected,
    )
    radial = PaperRadialBaseline.load(prior_dir / "radial_selected.json", **expected)
    bounds = PaperPhysicalBounds.load(prior_dir / "physical_bounds.json", **expected)
    envelope = PaperResidualEnvelope.load(prior_dir / "residual_envelope.json", **expected)
    roi = PaperVelocityROI.load(prior_dir / "velocity_roi.json", **expected)
    dissipation = PaperDissipativeReference.load(
        prior_dir / "dissipative_reference.json", **expected
    )
    if envelope.radial_mode != radial.mode:
        raise ValueError("Residual-envelope and selected radial modes differ")
    split_audits = {}
    for split, indices in (
        ("train", protocol.train_indices),
        ("validation", protocol.validation_indices),
    ):
        print(f"auditing Stage D artifacts on {split}", flush=True)
        split_audits[split] = audit_stage_d_split(
            protocol.dataset_path,
            split=split,
            snapshot_indices=indices,
            preprocessor=preprocessor,
            radial=radial,
            bounds=bounds,
            roi=roi,
            dissipation=dissipation,
            shell_edges=shell["shells"]["edges"],
        )
    payload = {
        "schema_version": "paper-stage-d-prior-audit-v1",
        "status": "passed",
        "provenance": radial.provenance.as_dict(),
        "selected_radial_mode": radial.mode,
        "selection_reason": read_prior_json(
            prior_dir / "radial_selected.json",
            expected_schema=PaperRadialBaseline.schema_version,
            **expected,
        )["selection"]["reason"],
        "validation_not_used_for_fit": True,
        "validation_role": "post-fit distribution-shift diagnostics only",
        "splits": split_audits,
    }
    write_stage_d_audit(
        payload,
        json_path=prior_dir / "prior_audit.json",
        csv_path=prior_dir / "prior_audit.csv",
        markdown_path=prior_dir / "prior_audit.md",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selected_radial_mode": radial.mode,
                "validation_not_used_for_fit": True,
                "train_norm_max": split_audits["train"]["dissipation"]["norm_max"],
                "validation_norm_max": split_audits["validation"]["dissipation"][
                    "norm_max"
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
