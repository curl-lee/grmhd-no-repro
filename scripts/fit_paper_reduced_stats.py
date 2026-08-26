#!/usr/bin/env python
"""Fit and audit the independent paper_reduced100 preprocessing statistics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

from grmhd.dataset import sha256_file
from grmhd.paper_preprocessing import PaperPreprocessor, write_preprocessing_audit
from grmhd.paper_protocol import PaperReduced100Protocol, write_protocol_manifest


def git_output(args: list[str], cwd: Path) -> str:
    return subprocess.run(
        args,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def resolve_output(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/data/paper_reduced100.yaml"),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    config_path = args.config if args.config.is_absolute() else project_root / args.config
    protocol = PaperReduced100Protocol.from_yaml(config_path, project_root=project_root)
    project_commit = git_output(["git", "rev-parse", "HEAD"], project_root)
    project_dirty = bool(git_output(["git", "status", "--short"], project_root))
    upstream_root = project_root / "external/neuraloperator"
    upstream_commit = git_output(["git", "rev-parse", "HEAD"], upstream_root)
    upstream_status = git_output(["git", "status", "--short"], upstream_root)
    if upstream_status:
        raise RuntimeError("Refusing paper stats fit with a dirty upstream worktree")

    outputs = protocol.outputs
    manifest_json = resolve_output(project_root, outputs["manifest_json"])
    manifest_csv = resolve_output(project_root, outputs["manifest_csv"])
    manifest = protocol.build_manifest(
        project_git_commit=project_commit,
        project_git_dirty=project_dirty,
        upstream_commit=upstream_commit,
    )
    write_protocol_manifest(manifest, manifest_json, manifest_csv)

    preprocessing = protocol.preprocessing
    preprocessor = PaperPreprocessor.fit_hdf5(
        protocol.dataset_path,
        training_indices=protocol.train_indices,
        protocol_name=protocol.protocol_name,
        expected_source_hdf5_checksum=manifest["dataset"]["sha256"],
        gamma=float(preprocessing["gamma"]),
        inverse_clamp_fraction=float(preprocessing["inverse_clamp_fraction"]),
        mad_multiplier=float(preprocessing["mad_multiplier"]),
        min_scale=float(preprocessing["min_scale"]),
        invalid_epsilon_fallback=float(preprocessing["invalid_epsilon_fallback"]),
        thermal_channel=protocol.thermal_channel,
        paper_adaptation=protocol.paper_adaptation,
        eos_conversion=protocol.eos_conversion,
        fit_scope=str(preprocessing["fit_scope"]),
    )
    stats_dir = resolve_output(project_root, outputs["stats_dir"])
    normalizer_json = stats_dir / "normalizer.json"
    normalizer_npz = stats_dir / "normalizer.npz"
    preprocessor.save(normalizer_json, normalizer_npz)

    audit = preprocessor.audit_hdf5(protocol.dataset_path)
    audit_json = resolve_output(project_root, outputs["preprocessing_audit_json"])
    audit_markdown = resolve_output(project_root, outputs["preprocessing_audit_markdown"])
    write_preprocessing_audit(audit, audit_json, audit_markdown)

    fit_manifest = {
        "schema_version": "paper-reduced100-fit-manifest-v1",
        "protocol_name": protocol.protocol_name,
        "command": shlex.join([sys.executable, *sys.argv]),
        "source_hdf5": str(protocol.dataset_path),
        "source_hdf5_checksum": manifest["dataset"]["sha256"],
        "fit_snapshot_indices": list(protocol.train_indices),
        "validation_snapshot_indices": list(protocol.validation_indices),
        "validation_excluded_from_fit": True,
        "thermal_channel": protocol.thermal_channel,
        "paper_adaptation": protocol.paper_adaptation,
        "eos_conversion": protocol.eos_conversion,
        "upstream_commit": upstream_commit,
        "upstream_worktree_clean": True,
        "project_git_commit": project_commit,
        "project_git_dirty_at_fit": project_dirty,
        "artifacts": {
            "protocol_manifest_json": {
                "path": str(manifest_json),
                "sha256": sha256_file(manifest_json),
            },
            "protocol_manifest_csv": {
                "path": str(manifest_csv),
                "sha256": sha256_file(manifest_csv),
            },
            "normalizer_json": {
                "path": str(normalizer_json),
                "sha256": sha256_file(normalizer_json),
            },
            "normalizer_npz": {
                "path": str(normalizer_npz),
                "sha256": sha256_file(normalizer_npz),
            },
            "preprocessing_audit_json": {
                "path": str(audit_json),
                "sha256": sha256_file(audit_json),
            },
            "preprocessing_audit_markdown": {
                "path": str(audit_markdown),
                "sha256": sha256_file(audit_markdown),
            },
        },
        "deferred_until_later_gates": [
            "shell_metadata.json",
            "radial_baseline.json",
            "physical_bounds.json",
            "residual_envelope.json",
            "velocity_roi.json",
            "dissipative_reference.json",
        ],
    }
    fit_manifest_path = stats_dir / "fit_manifest.json"
    fit_manifest_path.write_text(json.dumps(fit_manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "protocol_name": protocol.protocol_name,
                "source_hdf5_checksum": manifest["dataset"]["sha256"],
                "train_snapshots": len(protocol.train_indices),
                "validation_snapshots": len(protocol.validation_indices),
                "train_pairs": manifest["splits"]["train"]["pair_count"],
                "validation_pairs": manifest["splits"]["validation"]["pair_count"],
                "normalizer_json": str(normalizer_json),
                "normalizer_npz": str(normalizer_npz),
                "fit_manifest": str(fit_manifest_path),
                "preprocessing_audit": str(audit_json),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
