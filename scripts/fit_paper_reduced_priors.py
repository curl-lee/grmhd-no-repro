#!/usr/bin/env python
"""Fit all Stage D priors from paper_reduced100 train snapshots only."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any, Mapping

import h5py
import numpy as np

from grmhd.dataset import sha256_file
from grmhd.paper_bounds import fit_paper_physical_bounds
from grmhd.paper_dissipation import fit_dissipative_reference
from grmhd.paper_preprocessing import PaperPreprocessor
from grmhd.paper_prior_audit import (
    audit_bounds_split,
    audit_dissipation_split,
    audit_preprocessing_target_clamp_overlap,
    audit_roi_split,
    write_stage_d_audit,
)
from grmhd.paper_priors import (
    PaperResidualEnvelope,
    PriorProvenance,
    read_prior_json,
    write_prior_json,
)
from grmhd.paper_protocol import EXPECTED_UPSTREAM_COMMIT, PaperReduced100Protocol
from grmhd.paper_radial import (
    evaluate_radial_baseline,
    fit_appendix_literal_press_proxy,
    fit_spherical_logr_channelwise,
    select_radial_mode,
)
from grmhd.paper_references import REFERENCE_SEMANTICS_VERSION
from grmhd.paper_velocity_roi import PaperVelocityROI
from grmhd.shells import radial_shells


EXPECTED_HDF5_SHA256 = "cd3966bed1f617f49d775151cff87ae92eb2432a560bb82ae9355605e7aea70a"


def git_output(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_radial_comparison(
    path_csv: Path,
    path_md: Path,
    audits: Mapping[str, Mapping[str, Mapping[str, Any]]],
    *,
    selected_mode: str,
    selection_reason: str,
) -> None:
    fieldnames = (
        "split",
        "mode",
        "channel",
        "residual_mean",
        "residual_std",
        "residual_median",
        "residual_mad",
        "q0.001",
        "q0.999",
        "q_span",
        "inner_two_shell_bias",
        "outer_two_shell_bias",
        "dynamic_range",
        "reconstruction_relative_l2",
        "envelope_violation_fraction",
        "finite",
    )
    rows: list[dict[str, Any]] = []
    for split, split_audits in audits.items():
        for mode, audit in split_audits.items():
            for channel, record in audit["channels"].items():
                rows.append(
                    {
                        "split": split,
                        "mode": mode,
                        "channel": channel,
                        "residual_mean": record["residual_mean"],
                        "residual_std": record["residual_std"],
                        "residual_median": record["residual_median"],
                        "residual_mad": record["residual_mad"],
                        "q0.001": record["residual_q0.001"],
                        "q0.999": record["residual_q0.999"],
                        "q_span": record["residual_q_span"],
                        "inner_two_shell_bias": record["inner_two_shell_bias"],
                        "outer_two_shell_bias": record["outer_two_shell_bias"],
                        "dynamic_range": record["residual_dynamic_range"],
                        "reconstruction_relative_l2": record[
                            "physical_reconstruction_relative_l2"
                        ],
                        "envelope_violation_fraction": record[
                            "envelope_violation_fraction"
                        ],
                        "finite": record["finite"],
                    }
                )
    with path_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# paper_reduced100 radial candidate comparison",
        "",
        f"- Selected mode: `{selected_mode}`",
        f"- Selection reason: {selection_reason}.",
        "- Selection used train diagnostics only; validation is report-only.",
        "- A smaller adapted residual is not sufficient to replace the literal candidate.",
        "",
        "| split | mode | channel | mean | std | q span | reconstruction L2 | envelope violation |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['split']} | {row['mode']} | {row['channel']} | "
            f"{row['residual_mean']:.9g} | {row['residual_std']:.9g} | "
            f"{row['q_span']:.9g} | {row['reconstruction_relative_l2']:.9g} | "
            f"{row['envelope_violation_fraction']:.9g} |"
        )
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_roi_audit(
    path_csv: Path,
    path_md: Path,
    audits: Mapping[str, Mapping[str, Any]],
) -> None:
    fieldnames = (
        "split",
        "snapshot_index",
        "canonical_fraction",
        "raw_fraction",
        "jaccard",
        "disagreement_fraction",
        "canonical_clamp_overlap",
        "raw_clamp_overlap",
    )
    rows: list[dict[str, Any]] = []
    for split, audit in audits.items():
        for record in audit["snapshot_records"]:
            rows.append({"split": split, **record})
    with path_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
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
    path_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
        "--output-dir", type=Path, default=Path("outputs/paper_reduced100/priors")
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    normalizer_path = (
        args.normalizer if args.normalizer.is_absolute() else root / args.normalizer
    )
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol = PaperReduced100Protocol.from_yaml(config, project_root=root)
    validation = protocol.validate_hdf5()
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
    provenance = PriorProvenance(
        source_hdf5_checksum=hdf5_checksum,
        preprocessing_stats_checksum=preprocessing_checksum,
        training_indices=protocol.train_indices,
        protocol_name=protocol.protocol_name,
        thermal_channel=protocol.thermal_channel,
        paper_adaptation=protocol.paper_adaptation,
        eos_conversion=protocol.eos_conversion,
        coordinate_system=protocol.coordinate_system,
    )

    upstream_root = root / "external/neuraloperator"
    upstream_commit = git_output(upstream_root, "rev-parse", "HEAD")
    if upstream_commit != EXPECTED_UPSTREAM_COMMIT:
        raise ValueError("Fixed neuraloperator commit changed")
    if git_output(upstream_root, "status", "--short"):
        raise RuntimeError("Fixed neuraloperator worktree must remain clean")

    with h5py.File(protocol.dataset_path, "r") as handle:
        r = np.asarray(handle["coords/r"][...], dtype=np.float64)
    nphi, ntheta, _ = validation["snapshot_shape"][2:]
    _, shell_values = radial_shells(r, nphi, ntheta, n_shells=8)
    shell_payload = {
        "schema_version": "paper-shell-metadata-v1",
        "provenance": provenance.as_dict(),
        "shells": shell_values.as_dict(),
        "physical_coordinate": True,
        "fit_parameters_from_validation": False,
    }
    shell_path = output_dir / "shell_metadata.json"
    if shell_path.exists():
        existing = read_prior_json(
            shell_path,
            expected_schema="paper-shell-metadata-v1",
            source_hdf5_checksum=hdf5_checksum,
            preprocessing_stats_checksum=preprocessing_checksum,
            training_indices=protocol.train_indices,
            protocol_name=protocol.protocol_name,
            thermal_channel=protocol.thermal_channel,
        )
        if existing["shells"] != shell_payload["shells"]:
            raise ValueError("Existing shell metadata conflicts with canonical grid")
        shell_payload = existing
    else:
        write_prior_json(shell_path, shell_payload)
    shell_edges = shell_payload["shells"]["edges"]

    print("fitting literal radial candidate", flush=True)
    literal = fit_appendix_literal_press_proxy(
        str(protocol.dataset_path),
        training_indices=protocol.train_indices,
        r=r,
        preprocessor=preprocessor,
        provenance=provenance,
    )
    print("fitting spherical-logr radial candidate", flush=True)
    adapted = fit_spherical_logr_channelwise(
        str(protocol.dataset_path),
        training_indices=protocol.train_indices,
        r=r,
        preprocessor=preprocessor,
        provenance=provenance,
    )
    radial_audits: dict[str, dict[str, Any]] = {"train": {}, "validation": {}}
    for split, indices in (
        ("train", protocol.train_indices),
        ("validation", protocol.validation_indices),
    ):
        for candidate in (literal, adapted):
            print(f"auditing {candidate.mode} on {split}", flush=True)
            radial_audits[split][candidate.mode] = evaluate_radial_baseline(
                candidate,
                str(protocol.dataset_path),
                snapshot_indices=indices,
                preprocessor=preprocessor,
                shell_edges=shell_edges,
            )
    selected_mode, selection_reason = select_radial_mode(
        radial_audits["train"][literal.mode], radial_audits["train"][adapted.mode]
    )
    selected = literal if selected_mode == literal.mode else adapted
    for candidate, filename in (
        (literal, "radial_literal.json"),
        (adapted, "radial_spherical_adapted.json"),
    ):
        payload = candidate.as_dict()
        payload["audit"] = {
            split: radial_audits[split][candidate.mode]
            for split in ("train", "validation")
        }
        write_prior_json(output_dir / filename, payload)
    selected_payload = selected.as_dict()
    selected_payload["selection"] = {
        "selected_mode": selected_mode,
        "reason": selection_reason,
        "selection_split": "train",
        "validation_used_for_selection": False,
    }
    selected_payload["audit"] = {
        split: radial_audits[split][selected.mode] for split in ("train", "validation")
    }
    write_prior_json(output_dir / "radial_selected.json", selected_payload)
    _write_radial_comparison(
        output_dir / "radial_comparison.csv",
        output_dir / "radial_comparison.md",
        radial_audits,
        selected_mode=selected_mode,
        selection_reason=selection_reason,
    )

    print("fitting physical bounds", flush=True)
    bounds = fit_paper_physical_bounds(
        protocol.dataset_path,
        training_indices=protocol.train_indices,
        preprocessor=preprocessor,
        provenance=provenance,
    )
    bounds.save(output_dir / "physical_bounds.json")
    envelope = PaperResidualEnvelope(provenance=provenance, radial_mode=selected_mode)
    envelope_payload = envelope.as_dict()
    envelope_payload["diagnostics"] = {
        split: {
            channel: radial_audits[split][selected_mode]["channels"][channel][
                "envelope_violation_fraction"
            ]
            for channel in ("rho", "press")
        }
        for split in ("train", "validation")
    }
    write_prior_json(output_dir / "residual_envelope.json", envelope_payload)

    roi = PaperVelocityROI(provenance)
    write_prior_json(output_dir / "velocity_roi.json", roi.as_dict())
    shell_index = np.digitize(r, np.asarray(shell_edges)[1:-1], right=False)
    roi_audits: dict[str, Any] = {}
    for split, indices in (
        ("train", protocol.train_indices),
        ("validation", protocol.validation_indices),
    ):
        print(f"auditing ROI masks on {split}", flush=True)
        roi_audits[split] = audit_roi_split(
            protocol.dataset_path,
            indices,
            preprocessor=preprocessor,
            roi=roi,
            shell_index=shell_index,
        )
    write_prior_json(
        output_dir / "roi_stats.json",
        {
            "schema_version": "paper-roi-stats-v1",
            "provenance": provenance.as_dict(),
            "parameters": roi.as_dict(),
            "splits": roi_audits,
            "validation_used_for_fit": False,
        },
    )
    _write_roi_audit(
        output_dir / "roi_mask_audit.csv",
        output_dir / "roi_mask_audit.md",
        roi_audits,
    )

    print("fitting dissipative global-norm reference", flush=True)
    dissipation = fit_dissipative_reference(
        protocol.dataset_path,
        training_indices=protocol.train_indices,
        preprocessor=preprocessor,
        provenance=provenance,
    )
    dissipation.save(output_dir / "dissipative_reference.json")

    print("assembling Stage D train/validation audit", flush=True)
    split_audits: dict[str, Any] = {}
    for split, indices in (
        ("train", protocol.train_indices),
        ("validation", protocol.validation_indices),
    ):
        split_audits[split] = {
            "split": split,
            "snapshot_indices": list(indices),
            "radial": radial_audits[split][selected_mode],
            "bounds": audit_bounds_split(
                protocol.dataset_path, indices, preprocessor=preprocessor, bounds=bounds
            ),
            "roi": roi_audits[split],
            "dissipation": audit_dissipation_split(
                protocol.dataset_path,
                indices,
                preprocessor=preprocessor,
                reference=dissipation,
            ),
            "preprocessing_target_clamp_overlap": audit_preprocessing_target_clamp_overlap(
                protocol.dataset_path, indices, preprocessor=preprocessor
            ),
        }
    audit_payload = {
        "schema_version": "paper-stage-d-prior-audit-v1",
        "status": "passed",
        "provenance": provenance.as_dict(),
        "selected_radial_mode": selected_mode,
        "selection_reason": selection_reason,
        "validation_not_used_for_fit": True,
        "validation_role": "post-fit distribution-shift diagnostics only",
        "splits": split_audits,
    }
    write_stage_d_audit(
        audit_payload,
        json_path=output_dir / "prior_audit.json",
        csv_path=output_dir / "prior_audit.csv",
        markdown_path=output_dir / "prior_audit.md",
    )

    project_commit = git_output(root, "rev-parse", "HEAD")
    project_branch = git_output(root, "branch", "--show-current")
    project_dirty = bool(git_output(root, "status", "--short"))
    artifacts = {}
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "fit_manifest.json":
            artifacts[path.name] = {"path": str(path), "sha256": sha256_file(path)}
    fit_manifest = {
        "schema_version": "paper-stage-d-fit-manifest-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join([sys.executable, *sys.argv]),
        "protocol_name": protocol.protocol_name,
        "hdf5": {"path": str(protocol.dataset_path), "sha256": hdf5_checksum},
        "source_indices": list(protocol.source_indices),
        "train_indices": list(protocol.train_indices),
        "validation_indices": list(protocol.validation_indices),
        "dropped_transition": list(protocol.dropped_transition),
        "preprocessing_stats": {
            "path": str(normalizer_path),
            "sha256": preprocessing_checksum,
            "gamma": preprocessor.gamma,
            "inverse_clamp_fraction": preprocessor.inverse_clamp_fraction,
        },
        "oracle_reference_semantics_version": REFERENCE_SEMANTICS_VERSION,
        "thermal_channel": protocol.thermal_channel,
        "paper_adaptation": protocol.paper_adaptation,
        "eos_conversion": protocol.eos_conversion,
        "coordinate_system": protocol.coordinate_system,
        "stored_components_not_cartesian": True,
        "selected_radial_mode": selected_mode,
        "radial_selection_split": "train",
        "validation_not_used_for_fit": True,
        "project": {
            "branch": project_branch,
            "commit": project_commit,
            "dirty_at_fit": project_dirty,
        },
        "upstream": {"commit": upstream_commit, "dirty": False},
        "artifacts": artifacts,
        "training_or_model_execution": False,
    }
    write_prior_json(output_dir / "fit_manifest.json", fit_manifest)
    print(
        json.dumps(
            {
                "status": "passed",
                "output_dir": str(output_dir),
                "selected_radial_mode": selected_mode,
                "literal_k": literal.fit["k"],
                "Rmax": dissipation.rmax,
                "artifact_count": len(artifacts) + 1,
                "validation_not_used_for_fit": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
