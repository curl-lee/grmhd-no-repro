#!/usr/bin/env python3
"""Build the machine-readable Stage-Y provenance inventories.

This script is deliberately audit-only.  It reads existing manifests and hashes
existing files; it never opens raw/processed scientific data for writing and it
does not import or execute any model/training code.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "stage_y"
LOCAL = OUT / "local_provenance"
PAPER = OUT / "paper_assets"
TRANSFORM = OUT / "transform"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL
    ).strip()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def build_simulation_manifest(source: Path) -> list[dict[str, str]]:
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))

    columns = [
        "file_path",
        "file",
        "snapshot_index",
        "physical_time",
        "num_cycles",
        "file_size",
        "physical_content_sha256",
        "coordinates",
        "root_grid_size_x1_x2_x3",
        "mesh_block_size_x1_x2_x3",
        "num_meshblocks",
        "max_level",
        "level_distribution",
        "variable_names",
        "dataset_shapes",
        "component_basis_metadata",
        "physical_units_metadata",
        "code_version_metadata",
    ]
    destination = LOCAL / "simulation_file_manifest.csv"
    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row[column] for column in columns})
    return rows


def build_stage_x_manifest(rows: list[dict[str, str]]) -> None:
    source_manifest = ROOT / "artifacts" / "stage_s" / "data_manifest.csv"
    processed = ROOT / "data_proc" / "grmhd_regrid_inner_r200_64_expanded.h5"
    stage_x_files = {
        "report": ROOT / "artifacts" / "stage_x" / "STAGE_X_REPORT.md",
        "decision": ROOT / "artifacts" / "stage_x" / "STAGE_X_DECISION.md",
        "gap_matrix": ROOT / "artifacts" / "stage_x" / "gap_matrix.csv",
    }
    raw_target = (ROOT / "data_raw").resolve()
    raw_stat = raw_target.stat()
    processed_stat = processed.stat()
    paper_tmp = Path("/tmp/stage_y_2512.01576v1.pdf")
    paper_sha = "fe84a42289da7e86dd8460d223e57627a86aa0b2114a64a13b83f84388441808"
    paper_verified = paper_tmp.is_file() and sha256(paper_tmp) == paper_sha

    payload = {
        "schema_version": 1,
        "stage": "Y",
        "audit_date": "2026-08-17",
        "project": {
            "root": str(ROOT),
            "git_commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "worktree_clean": not bool(git("status", "--short")),
            "note": "Stage-S-through-X changes pre-existed this Stage-Y audit.",
        },
        "pinned_neuraloperator": {
            "path": str(ROOT / "external" / "neuraloperator"),
            "commit": git("rev-parse", "HEAD", cwd=ROOT / "external" / "neuraloperator"),
            "worktree_clean": not bool(
                git("status", "--short", cwd=ROOT / "external" / "neuraloperator")
            ),
        },
        "paper": {
            "identifier": "arXiv:2512.01576v1",
            "official_url": "https://arxiv.org/abs/2512.01576v1",
            "pdf_sha256": paper_sha,
            "audit_copy": str(paper_tmp),
            "audit_copy_hash_verified": paper_verified,
        },
        "stage_x": {
            "primary_decision": "E",
            "primary_decision_label": "MULTIPLE_FUNDAMENTAL_METHOD_GAPS",
            "coordinate_information_gap": "HIGH",
            "vector_basis_gap": "FUNDAMENTAL",
            "vector_transform_provenance": "INCOMPLETE",
            "paper_operator_gap": "HIGH",
            "temporal_increment_regrid_fidelity": "POOR",
            "thermal_variable_gap": "HIGH",
            "current_target_not_scientifically_comparable_to_paper": True,
            "authorize_next_stage": "data_or_code_acquisition",
            "artifacts": {
                name: {"path": str(path), "sha256": sha256(path)}
                for name, path in stage_x_files.items()
            },
        },
        "raw_data": {
            "project_symlink": str(ROOT / "data_raw"),
            "resolved_directory": str(raw_target),
            "directory_size_metadata": {
                "mtime_ns": raw_stat.st_mtime_ns,
                "device": raw_stat.st_dev,
                "inode": raw_stat.st_ino,
            },
            "source_manifest": str(source_manifest),
            "source_manifest_sha256": sha256(source_manifest),
            "snapshot_count": len(rows),
            "indices": [int(row["snapshot_index"]) for row in rows],
            "times": [float(row["physical_time"]) for row in rows],
            "cycles": [int(row["num_cycles"]) for row in rows],
            "first_path": rows[0]["file_path"],
            "last_path": rows[-1]["file_path"],
            "per_file_hash_kind": "physical_content_sha256 from frozen Stage-S manifest",
            "stage_y_manifest": str(LOCAL / "simulation_file_manifest.csv"),
        },
        "processed_data": {
            "path": str(processed),
            "size_bytes": processed_stat.st_size,
            "mtime_ns": processed_stat.st_mtime_ns,
            "sha256": sha256(processed),
            "shape": [212, 8, 64, 64, 64],
            "axis_order": ["snapshot", "channel", "phi", "theta", "r"],
        },
        "safety_assertions": {
            "NO_RAW_MUTATION": True,
            "NO_PROCESSED_MUTATION": True,
            "NO_MODEL_TRAINING": True,
            "transform_tests_skipped": True,
            "transform_tests_skip_reason": "VECTOR_TRANSFORM_PROVENANCE=INCOMPLETE",
        },
    }
    write_json(OUT / "stage_x_input_manifest.json", payload)


def build_public_asset_manifest() -> None:
    columns = [
        "asset",
        "url",
        "repository",
        "commit_or_tag",
        "release_date",
        "license",
        "downloaded_file_or_audited_path",
        "sha256_or_provider_checksum",
        "relation_to_paper",
        "provenance_class",
        "audit_result",
    ]
    rows = [
        [
            "target paper arXiv abstract",
            "https://arxiv.org/abs/2512.01576v1",
            "arXiv",
            "2512.01576v1",
            "2025-12-01",
            "not stated in inspected source package",
            "web page",
            "not downloaded",
            "paper record",
            "OFFICIAL_PAPER_ASSET",
            "FOUND_VERIFIED",
        ],
        [
            "target paper PDF",
            "https://arxiv.org/pdf/2512.01576v1",
            "arXiv",
            "2512.01576v1",
            "2025-12-01",
            "not stated in inspected source package",
            "/tmp/stage_y_2512.01576v1.pdf",
            "sha256:fe84a42289da7e86dd8460d223e57627a86aa0b2114a64a13b83f84388441808",
            "paper text",
            "OFFICIAL_PAPER_ASSET",
            "FOUND_VERIFIED",
        ],
        [
            "target paper TeX source",
            "https://arxiv.org/e-print/2512.01576v1",
            "arXiv",
            "2512.01576v1",
            "2025-12-01",
            "not stated in inspected source package",
            "/tmp/stage_y_2512.01576v1_source.tar",
            "sha256:6e758b1fdd74f27f332a0bfa21dc9a44afd1a458e7459c2048539f0c98c861bf",
            "paper source; TeX/bibliography/figures only",
            "OFFICIAL_PAPER_ASSET",
            "FOUND_VERIFIED_NO_CODE_OR_DATA",
        ],
        [
            "NeurIPS workshop page",
            "https://neurips.cc/virtual/2025/loc/san-diego/122904",
            "NeurIPS",
            "2025 workshop record",
            "2025",
            "not available",
            "web page",
            "not downloaded",
            "official venue record linking only OpenReview",
            "OFFICIAL_PAPER_ASSET",
            "FOUND_VERIFIED_NO_CODE_OR_DATA",
        ],
        [
            "OpenReview project page",
            "https://openreview.net/forum?id=mL9sQvIQ8U",
            "OpenReview",
            "forum mL9sQvIQ8U",
            "2025",
            "not available",
            "web page; API access denied during audit",
            "not downloaded",
            "official venue-linked project page",
            "OFFICIAL_PAPER_ASSET",
            "FOUND_PARTIAL_NO_ASSET_LINK_OBSERVED",
        ],
        [
            "Chuwei Wang author homepage",
            "https://cwwangcal.github.io/",
            "author homepage",
            "site updated 2026-01-09",
            "2026-01-09",
            "not available",
            "web page",
            "not downloaded",
            "lists paper with arXiv link only",
            "AUTHOR_RELATED_ASSET",
            "FOUND_VERIFIED_NO_CODE_OR_DATA",
        ],
        [
            "Chuwei Wang neuraloperator fork",
            "https://github.com/cwwangcal/neuraloperator",
            "cwwangcal/neuraloperator",
            "8719ad20733d4b58414a65706c62dda634a642bc",
            "2025-08-12",
            "MIT",
            "/tmp/stage_y_cwwang_neuraloperator",
            "git commit:8719ad20733d4b58414a65706c62dda634a642bc",
            "author fork predating paper submission; one branch; no paper/GRMHD/3D DISCO files",
            "AUTHOR_RELATED_ASSET",
            "FOUND_VERIFIED_NOT_PAPER_CODE",
        ],
        [
            "pinned neuraloperator dependency",
            "https://github.com/neuraloperator/neuraloperator",
            "neuraloperator/neuraloperator",
            "86a8bc7812a31b42c4f7895693cf4ac11521c066",
            "commit date available in Git",
            "MIT",
            "external/neuraloperator",
            "git commit:86a8bc7812a31b42c4f7895693cf4ac11521c066",
            "project-pinned dependency; not the paper repository",
            "UPSTREAM_DEPENDENCY",
            "FOUND_VERIFIED_2D_DISCO_ONLY",
        ],
        [
            "LocalNO primary reference PDF",
            "https://arxiv.org/pdf/2402.16845",
            "arXiv",
            "2402.16845v2",
            "2024-06-08",
            "arXiv record",
            "/tmp/stage_y_2402.16845.pdf",
            "sha256:485956551f9b683c2ba621bb1d02973238df5d90732c2d090403525401287125",
            "primary operator reference; general/2D contract, not target volumetric implementation",
            "UPSTREAM_DEPENDENCY",
            "FOUND_VERIFIED_PARTIAL_SPECIFICATION",
        ],
        [
            "public Athena++ source",
            "https://github.com/PrincetonUniversity/athena-public-version",
            "PrincetonUniversity/athena-public-version",
            "9c266692b9423743d8e23509b3ab266a232a92d2",
            "local clone commit",
            "BSD-3-Clause",
            "/home/curl/athena-public-version",
            "git commit:9c266692b9423743d8e23509b3ab266a232a92d2",
            "software convention reference; no content link to local raw run",
            "UPSTREAM_DEPENDENCY",
            "FOUND_VERIFIED_NOT_RUN_PROVENANCE",
        ],
        [
            "M87 MAD98 data/code archive",
            "https://doi.org/10.5281/zenodo.10570521",
            "Zenodo record 10570521",
            "record version 1",
            "2024-01-26",
            "CC-BY-4.0",
            "metadata and ZIP central directory/ranged small scripts only",
            "provider md5:5c08f6db83437f5177cac5473ae328d6; 2,011,328,260 bytes",
            "different M87 paper; similar MAD98 family, not target-paper data and not cryptographically bound to local raw files",
            "THIRD_PARTY",
            "FOUND_VERIFIED_EXCLUDED_AS_EXACT_EVIDENCE",
        ],
    ]
    with (PAPER / "public_asset_manifest.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def build_acquisition_matrix() -> None:
    columns = ["required item", "status", "source", "exact/inferred", "blocks reproduction"]
    rows = [
        ["paper training data", "NOT_FOUND", "arXiv source; NeurIPS/OpenReview; author pages/repositories; repository/data-host searches", "exact search result", "true"],
        ["paper preprocessing code", "NOT_FOUND", "official paper assets contain no code locator", "exact search result", "true"],
        ["paper model code", "NOT_FOUND", "official paper assets and author public repositories", "exact search result", "true"],
        ["3D DISCO implementation", "NOT_FOUND", "paper; author fork; pinned upstream and its local history", "exact search result", "true"],
        ["coarse/fine coupling", "FOUND_PARTIAL", "paper Appendix E, main.tex:755-815", "paper-explicit but implementation-incomplete", "true"],
        ["paper BH spin", "FOUND_VERIFIED", "paper Appendix A.2, main.tex:357", "exact paper statement: a=0.9", "false"],
        ["current-raw BH spin", "AMBIGUOUS", "ATHDF has no spin; filename is not evidence; similar public MAD98 archive lacks checksum binding", "unknown", "true"],
        ["coordinate mapping", "FOUND_PARTIAL", "ATHDF Coordinates=kerr-schild; numeric axes; public Athena source not bound to run", "x1/x2/x3 axis mapping exact; metric implementation provenance incomplete", "true"],
        ["magnetic vector convention", "FOUND_PARTIAL", "ATHDF Bcc names; public Athena outputs/KS source; unrelated MAD98 conversion script", "software-family convention only", "true"],
        ["velocity convention", "FOUND_PARTIAL", "ATHDF vel names; public Athena primitive/KS source; unrelated MAD98 conversion script", "software-family convention only", "true"],
        ["current-raw EOS", "NOT_FOUND", "212 ATHDF headers; local configs/history; no matching input deck", "unknown", "true"],
        ["current-raw Gamma", "NOT_FOUND", "212 ATHDF headers; no matching input/build record", "unknown", "true"],
        ["paper press/eint definition", "AMBIGUOUS", "paper Appendix A.2/B says P; C/model figures say eint", "contradictory paper statements", "true"],
        ["Cartesian vector transform", "NOT_FOUND", "requires current spin/source/basis and verified target convention", "not authorized", "true"],
        ["paper spatial domain", "NOT_FOUND", "paper gives L^3 and 64^3 but no numeric Cartesian cube bounds", "unknown", "true"],
        ["paper remap method", "NOT_FOUND", "paper/source do not state AMR-to-Cartesian remap", "unknown", "true"],
        ["paper time cadence", "FOUND_PARTIAL", "paper main.tex:179 gives DeltaT=T/Ndata; no numeric T/cadence", "symbolic only", "true"],
        ["snapshot protocol", "FOUND_VERIFIED", "paper main.tex:179,226", "300 generated; last 250; 80/20", "false"],
        ["training configuration", "FOUND_PARTIAL", "paper Appendix C, main.tex:373-605", "loss/schedule explicit; exact 3D backbone shape/config absent", "true"],
        ["current raw spatial information", "FOUND_VERIFIED", "212 AMR ATHDF; Stage-S manifest and Stage-X raw AMR audit", "leaf-cell fields available; only Bcc, no face-field export", "false"],
    ]
    with (OUT / "acquisition_matrix.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(rows)


def build_readiness() -> None:
    payload = {
        "schema_version": 1,
        "stage": "Y",
        "audit_date": "2026-08-17",
        "gates": {
            "R1_DATA": {
                "status": "FAIL",
                "reason": "No official Cartesian-KS/eint paper data; current raw lacks a complete spin/vector/EOS provenance chain.",
            },
            "R2_OPERATOR": {
                "status": "FAIL",
                "reason": "No exact volumetric 3D DISCO code and critical 3D basis/support/boundary/per-layer choices remain unknown.",
            },
            "R3_PREPROCESSING": {
                "status": "FAIL",
                "reason": "Paper Cartesian domain bounds and AMR/remap method are not published.",
            },
            "R4_TRAINING": {
                "status": "FAIL",
                "reason": "Optimization/loss are detailed, but exact volumetric LocalNO architecture parameters and executable configuration are absent.",
            },
            "R5_COUPLING": {
                "status": "FAIL",
                "reason": "Appendix E is partial and does not define an executable B-to-EMF/CT interface or complete boundary/data contract.",
            },
        },
        "raw_salvage_conditions": {
            "S1_coordinate_mapping_complete": False,
            "S2_vector_basis_transform_complete": False,
            "S3_thermal_conversion_complete": False,
            "S4_sufficient_raw_spatial_information": True,
            "CURRENT_RAW_DATA_SALVAGEABLE": False,
        },
        "provenance": {
            "BH_SPIN_VALUE_CURRENT_RAW": "UNKNOWN",
            "BH_SPIN_VALUE_PAPER": 0.9,
            "BH_SPIN_PROVENANCE": "INCOMPLETE",
            "COORDINATE_MAPPING_PROVENANCE": "INCOMPLETE",
            "VECTOR_TRANSFORM_PROVENANCE": "INCOMPLETE",
            "CARTESIAN_VECTOR_CONVERSION_AUTHORIZED": False,
            "EOS_PROVENANCE": "INCOMPLETE",
            "PRESS_TO_EINT_CONVERSION_AUTHORIZED": False,
            "OFFICIAL_PAPER_DATA_FOUND": False,
            "OFFICIAL_PAPER_CODE_FOUND": False,
            "EXACT_3D_DISCO_IMPLEMENTATION_FOUND": False,
            "PAPER_COARSE_FINE_CONTRACT": "PARTIAL",
        },
        "decision": {
            "PRIMARY_DECISION": "F",
            "PRIMARY_DECISION_LABEL": "EXACT_REPRODUCTION_BLOCKED_BY_MISSING_ASSETS",
            "EXACT_REPRODUCTION_BLOCKED": True,
            "AUTHORIZE_NEXT_STAGE": "adapted_workflow_reproduction_only",
        },
        "restrictions_observed": {
            "NO_RAW_MUTATION": True,
            "NO_PROCESSED_MUTATION": True,
            "NO_MODEL_TRAINING": True,
            "NO_SPECULATIVE_VECTOR_TRANSFORM": True,
            "NO_SPECULATIVE_EOS_CONVERSION": True,
            "NO_SPECULATIVE_3D_DISCO": True,
        },
    }
    write_json(OUT / "reproduction_readiness.json", payload)


def main() -> None:
    for directory in (OUT, LOCAL, PAPER, TRANSFORM):
        directory.mkdir(parents=True, exist_ok=True)
    source = ROOT / "artifacts" / "stage_s" / "data_manifest.csv"
    rows = build_simulation_manifest(source)
    if len(rows) != 212 or [int(row["snapshot_index"]) for row in rows] != list(range(212)):
        raise RuntimeError("Frozen Stage-S raw manifest is not the expected contiguous 0..211 sequence")
    build_stage_x_manifest(rows)
    build_public_asset_manifest()
    build_acquisition_matrix()
    build_readiness()


if __name__ == "__main__":
    main()
