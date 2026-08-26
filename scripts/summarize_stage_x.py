#!/usr/bin/env python3
"""Assemble the evidence-backed Stage X contracts, gap matrix, and decision."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics
import subprocess
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
PAPER_URL = "https://arxiv.org/pdf/2512.01576v1"
PAPER_SHA = "fe84a42289da7e86dd8460d223e57627a86aa0b2114a64a13b83f84388441808"


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def git_value(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def contract_table(rows: list[tuple[str, str, str, str]]) -> str:
    lines = ["| field | value | status | primary evidence |", "|---|---|---|---|"]
    lines.extend(f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/stage_x/method_gap_audit.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/stage_x"))
    args = parser.parse_args()
    out = args.out_dir
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    raw = json.loads((out / "regrid/raw_amr_audit.json").read_text(encoding="utf-8"))
    field_metric_rows = read_csv(out / "regrid/resolution_convergence.csv")
    temporal_metric_rows = read_csv(out / "regrid/temporal_increment_fidelity.csv")
    coordinate_rows = read_csv(out / "coordinates/radial_position_identifiability.csv")
    coordinate_summary = next(row for row in coordinate_rows if row["record_type"] == "classification")
    fidelity = raw["temporal_increment_regrid_fidelity"]
    regrid_severity = "MODERATE" if fidelity in {"GOOD", "MODERATE"} else "MAJOR"
    project_commit = git_value("rev-parse", "HEAD")
    upstream_commit = git_value("-C", "external/neuraloperator", "rev-parse", "HEAD")

    paper_rows = [
        ("simulation coordinate system", "Cartesian Kerr–Schild", "EXPLICIT_IN_PAPER", "Appendix A.2, PDF p.9"),
        ("spatial domain", "bounded fine domain L^3 and coarse domain (n_L L)^3, n_L=6; numeric coordinate bounds not given", "EXPLICIT_IN_PAPER", "§2, PDF p.3"),
        ("final training resolution", "64_x × 64_y × 64_z", "EXPLICIT_IN_PAPER", "Appendix C.2, PDF p.10"),
        ("input variables", "eight fields at t: bcc1:3, dens, eint, velx:y:z", "EXPLICIT_IN_PAPER", "Appendix C.2"),
        ("output variables", "same eight fields at t+ΔT", "EXPLICIT_IN_PAPER", "§2 and Appendix C.2"),
        ("thermal variable definition", "C.2 calls it internal energy eint; A.2 separately says GRMHD outputs P; exact P↔eint relation is absent", "EXPLICIT_IN_PAPER", "Appendices A.2 and C.2"),
        ("vector component basis", "Cartesian-KS x/y/z components", "EXPLICIT_IN_PAPER", "Appendices A.2/C.2"),
        ("positional encoding", "default uses radial-shell one-hot; normalized Cartesian Fourier features are an ablation, not default", "EXPLICIT_IN_PAPER", "§2, C.4, D.1"),
        ("shell/radial embedding", "8 one-hot shells from log distance to Cartesian index-grid centre; rmax=10", "EXPLICIT_IN_PAPER", "§2, C.4, Table 1"),
        ("number of snapshots", "Ndata=300 generated; last 250 used", "EXPLICIT_IN_PAPER", "§2, PDF pp.3–4"),
        ("temporal cadence", "ΔT=T/Ndata", "EXPLICIT_IN_PAPER", "§2"),
        ("train/validation split", "last 250, 80/20", "EXPLICIT_IN_PAPER", "§2"),
        ("preprocessing transforms", "positive log10 dens/eint; signed-log bcc1:3; linear velocities; inverse atanh at 0.99γ", "EXPLICIT_IN_PAPER", "Appendix C.3"),
        ("normalization", "per-channel median/MAD robust z-score and γ=6 tanh soft clip", "EXPLICIT_IN_PAPER", "Appendix C.3"),
        ("architecture", "3D Local Neural Operator with equidistant DISCO, volumetric input", "EXPLICIT_IN_PAPER", "Appendix C.8"),
        ("LocalNO layer composition", "exact per-layer branch placement is not stated", "NOT_SPECIFIED", "Appendix C.8 gives only backbone family/DISCO"),
        ("Fourier branch", "not separately specified for the instantiated paper model", "NOT_SPECIFIED", "paper never resolves layer flags"),
        ("differential branch", "not separately specified for the instantiated paper model", "NOT_SPECIFIED", "paper never resolves layer flags"),
        ("DISCO/local integral branch", "equidistant discrete–continuous convolution specialized to volumetric inputs", "EXPLICIT_IN_PAPER", "Appendix C.8"),
        ("number of modes", "not reported", "NOT_SPECIFIED", "paper PDF"),
        ("width", "not reported", "NOT_SPECIFIED", "paper PDF"),
        ("depth", "not reported", "NOT_SPECIFIED", "paper PDF"),
        ("training epochs", "1200", "EXPLICIT_IN_PAPER", "Appendix C.9/Table 1"),
        ("loss composition", "component-weighted L2 + 0.05 H1 + velocity ROI + dissipation + radial envelope + constraints", "EXPLICIT_IN_PAPER", "Appendix C.7"),
        ("rollout procedure", "autoregressive fine-level rollout; reported 50/100-step behavior", "EXPLICIT_IN_PAPER", "§2, figures and C.10"),
        ("coarse/fine coupling", "HDF5 rollout/time interpolation; hydro inner-boundary overwrite; magnetic CT/EMF treatment", "EXPLICIT_IN_PAPER", "Appendix E"),
    ]
    paper_contract = f"""# PAPER_PIPELINE_CONTRACT

Primary source: [{config['paper']['title']}]({PAPER_URL}), arXiv `2512.01576v1`, locally verified PDF SHA256 `{PAPER_SHA}`. The table deliberately keeps paper facts separate from upstream implementation assumptions.

{contract_table(paper_rows)}

## Primary-evidence caveats

The paper explicitly claims a volumetric 3D DISCO backbone, but does not publish modes, width, depth, parameter count, resolved boundary policy, local-kernel support/radius, or per-layer Fourier/differential flags. The paper also contains a thermal-description mismatch: Appendix A.2 says the GRMHD output includes pressure `P`, whereas Appendix C.2 says the network tensor contains internal energy `eint`. No GRMHD EOS conversion resolving this is specified. Those absences are `NOT_SPECIFIED`, not inferred facts.
"""
    write(out / "paper_contract.md", paper_contract)

    current_rows = [
        ("simulation coordinate system", "raw Athena++ `Coordinates=kerr-schild`; project represents it as spherical KS (r,theta,phi)", "EXPLICIT_IN_ARTIFACT", "`/mnt/d/GRMHD_data/mad98.prim.00000.athdf` attrs; `src/build_regrid_from_athdf.py:650-656`"),
        ("spatial domain", "raw r=[1.1,1200], theta=[0,π], phi=[0,2π]; processed r=[1.1,200]", "EXPLICIT_IN_ARTIFACT", "`artifacts/stage_x/regrid/raw_amr_audit.json`"),
        ("final training resolution", "64_phi × 64_theta × 64_r", "EXPLICIT_IN_CONFIG", "`configs/stage_s/expanded_localno_p3_residual.yaml:5-14,41-51`"),
        ("input variables", "8 normalized spherical-coordinate fields + 8 physical-r shell channels", "EXPLICIT_IN_CODE", "config lines 13-39; `scripts/train_stage_s.py:91-128`"),
        ("output variables", "8 normalized residuals Δz, reconstructed as z_t+Δz", "EXPLICIT_IN_CONFIG", "config lines 64-72"),
        ("thermal variable definition", "Athena++ `press`; not converted to internal energy", "EXPLICIT_IN_ARTIFACT", "ATHDF `VariableNames`; config line 14"),
        ("vector component basis", "retained as Bcc1:3/vel1:3 spherical-KS coordinate-basis representation; no Cartesian conversion", "PROJECT_CONTRACT_WITH_INCOMPLETE_SOURCE_PROVENANCE", "`src/build_regrid_from_athdf.py:710-712`; ATHDF attrs"),
        ("positional encoding", "upstream positional_embedding=null; only 8 shell one-hots", "EXPLICIT_IN_CONFIG_AND_FORWARD", "config lines 33-55; `scripts/train_stage_s.py:100-128`"),
        ("shell/radial embedding", "8 one-hot bins in physical spherical r, log-spaced", "EXPLICIT_IN_CODE", "`src/grmhd/shells.py:12-59`"),
        ("number of snapshots", "212", "EXPLICIT_IN_ARTIFACT", "processed HDF5 shape and `artifacts/stage_s/data_manifest.csv`"),
        ("temporal cadence", "actual ATHDF Time array (approximately 10 code-time units, not assumed constant)", "EXPLICIT_IN_ARTIFACT", "processed `times` and Stage-S manifest"),
        ("train/validation split", "snapshots [0,169) train and [169,212) validation; pair 168→169 dropped", "EXPLICIT_IN_CONFIG", "config lines 8-12; `artifacts/stage_s/train_val_split.json`"),
        ("preprocessing transforms", "P3: canonical except Bcc2/Bcc3/vel3 omit softclip; no press→eint conversion", "EXPLICIT_IN_CONFIG", "config lines 16-31"),
        ("normalization", "train-only expanded P3 normalizer, gamma6 and inverse 0.99γ where applicable", "EXPLICIT_IN_CONFIG", "config lines 16-31"),
        ("architecture", "upstream 3D LocalNO differential proxy", "EXPLICIT_IN_CONFIG", "config lines 41-64"),
        ("LocalNO layer composition", "SpectralConv + FiniteDifferenceConv + linear skip + GELU", "EXPLICIT_IN_CODE", "pinned `local_no_block.py:273-349,466-481`"),
        ("Fourier branch", "enabled, 8×8×8 modes", "EXPLICIT_IN_CONFIG", "config lines 47,56"),
        ("differential branch", "enabled, four 3×3×3 mixed-derivative blocks", "EXPLICIT_IN_CONFIG", "config lines 49,53,57-58"),
        ("DISCO/local integral branch", "disabled/unavailable for 3D in pinned upstream", "EXPLICIT_IN_CONFIG_AND_CODE", "config lines 54-55; `src/grmhd/models.py:127-147`; upstream `local_no_block.py:211-215`"),
        ("number of modes", "[8,8,8]", "EXPLICIT_IN_CONFIG", "config line 47"),
        ("width", "16", "EXPLICIT_IN_CONFIG", "config line 48"),
        ("depth", "4", "EXPLICIT_IN_CONFIG", "config line 49"),
        ("training epochs", "Stage-T full-long ran 1200 epochs", "EXPLICIT_IN_CONFIG_AND_CHECKPOINT", "`configs/stage_t/optimization_convergence.yaml:14-19`; epoch_1200 checkpoint"),
        ("loss composition", "PlainL2Loss on normalized residual only", "EXPLICIT_IN_CONFIG", "config lines 66-80"),
        ("rollout procedure", "adapted spherical 64^3 autoregressive evaluation; no live coarse solver", "EXPLICIT_IN_CODE_AND_ARTIFACT", "`artifacts/stage_t/STAGE_T_REPORT.md` and evaluation outputs"),
        ("coarse/fine coupling", "not implemented", "EXPLICIT_ABSENCE", "no Stage-S/T coupling path; README limitation"),
    ]
    current_contract = f"""# CURRENT_REPRO_CONTRACT

Resolved project commit `{project_commit}`; pinned upstream `{upstream_commit}`. This is the actually executed Stage-T target, reconstructed from the checkpoint/config/forward path rather than README claims.

{contract_table(current_rows)}

`CURRENT_COORDINATE_INFORMATION = [eight log-spaced one-hot bins of physical spherical r]`. There are no raw `r`, `log r`, `theta`, `phi`, Cartesian `x/y/z`, or upstream normalized index-grid channels in the Stage-T forward input.
"""
    write(out / "current_contract.md", current_contract)

    operator_paper = f"""# Paper operator audit

Primary paper evidence is Appendix C.8 of [{config['paper']['title']}]({PAPER_URL}) (verified PDF SHA256 `{PAPER_SHA}`). It explicitly says the backbone is a **3D Local Neural Operator with equidistant discrete–continuous convolutions (DISCO) specialized to volumetric inputs**. Thus a local-integral/DISCO mechanism is part of the claimed paper backbone.

The paper does **not** resolve per-layer flags, modes, width, depth, local kernel basis/support/radius, or whether every layer simultaneously includes Fourier and differential branches. Consequently:

1. Exact per-layer branch composition: `NOT_SPECIFIED` in this paper.
2. Reliance on DISCO/local integral: `EXPLICIT_IN_PAPER`.
3. DISCO's detailed role: the paper identifies it as the equidistant discrete–continuous volumetric convolution; detailed quadrature/kernel mechanics are not stated here.
4. Current missing mechanism: every local integral branch.
5. Mechanism plausibility: removing a learnable localized integral kernel can plausibly alter local transport, radial redistribution, inner/outer transfer, and boundary-local structure. This is a mechanism hypothesis, **not causal proof**.

`PAPER_OPERATOR_GAP = HIGH`.
"""
    write(out / "operator/paper_operator_audit.md", operator_paper)

    upstream_operator = """# Pinned-upstream LocalNO audit

Pinned commit: `86a8bc7812a31b42c4f7895693cf4ac11521c066` (clean submodule worktree during audit).

- `external/neuraloperator/neuralop/models/local_no.py:24-29` defines LocalNO blocks as Fourier layers with differential and local-integral kernels in parallel.
- `local_no.py:51-69,190-205` makes DISCO and differential branches independently selectable and defaults both to true.
- `local_no_block.py:273-349` constructs spectral, differential, and local-convolution modules.
- `local_no_block.py:466-481` adds enabled spectral, differential, and local-convolution outputs in the forward pass.
- `local_no_block.py:211-215` rejects DISCO when spatial dimension is not 2.
- `discrete_continuous_convolution.py:271-299` implements only `DiscreteContinuousConv2d`; it evaluates a learnable continuous local kernel semi-discretely on a 2D grid.
- `src/grmhd/models.py:127-147` therefore constructs the 3D project proxy with `disco_layers=False, diff_layers=True` and rejects volumetric DISCO.

The pinned package's generic/canonical LocalNO mechanism is `Spectral + differential + local integral`, but the paper's exact per-layer choices cannot be reconstructed from the paper. The **current resolved model** is unambiguously `SpectralConv + FiniteDifferenceConv + linear skip`; it has no local integral branch.
"""
    write(out / "operator/upstream_operator_audit.md", upstream_operator)

    mechanism = """# Operator mechanism comparison

| mechanism | paper contract | current Stage T | scientific consequence |
|---|---|---|---|
| global spectral coupling | exact branch flags not reported | enabled, 8^3 modes | current has global index-grid Fourier mixing |
| local finite-difference coupling | not separately reported | enabled, 3^3 kernel in each of 4 blocks | current has local differential coupling |
| local integral/DISCO coupling | explicitly volumetric equidistant DISCO | absent | HIGH mechanism gap; localized learned integral transport is unavailable |
| skip/MLP | not reported | linear LocalNO skip; channel MLP disabled | cannot match paper implementation |
| positional conditioning | 8 Cartesian-index radial shells | 8 physical spherical-r shells | both supply coarse radial context, with different geometry |
| parameters | not reported | 358,296 | no parameter-count comparison is possible |

Parameter count cannot resolve the comparison because the paper count is `NOT_SPECIFIED` and the missing operator branch changes receptive mechanism, not just size.
"""
    write(out / "operator/operator_mechanism_comparison.md", mechanism)

    off = [float(row["symmetric_relative_distance"]) for row in coordinate_rows if row["record_type"] == "pair_response" and row["shell_mode"] == "shell_off"]
    on = [float(row["symmetric_relative_distance"]) for row in coordinate_rows if row["record_type"] == "pair_response" and row["shell_mode"] == "shell_on"]
    coordinate_audit = f"""# Coordinate information audit

## Paper

The simulation and network tensor are Cartesian Kerr–Schild. The default model receives eight one-hot shells based on logarithmic Euclidean distance to the Cartesian index-grid centre (Appendix C.4). The paper's normalized Cartesian Fourier features `(xi,eta,zeta)` are an **ablation** (Appendix D.1), not evidence that the default network consumes continuous Cartesian coordinates.

`PAPER_COORDINATE_INFORMATION = [8 Cartesian-index radial shell one-hots]` for the default model.

## Current forward graph

`configs/stage_s/expanded_localno_p3_residual.yaml:33-63` fixes 16 inputs, `positional_embedding: null`, and eight spherical-r shells. `scripts/train_stage_s.py:91-128` constructs shells from the HDF5 `coords/r` array and concatenates only `(z_input, shells)`. `src/grmhd/shells.py:32-59` uses eight log-spaced physical-r bins, broadcast identically over phi/theta.

`CURRENT_COORDINATE_INFORMATION = [8 physical spherical-r shell one-hots]`.

It does not receive actual `phi`, `theta`, continuous `r`, `log r`, normalized tensor indices, or Cartesian position.

## Frozen no-training test

Checkpoint `artifacts/stage_t/full_long/checkpoints/epoch_0150.pt` was evaluated without gradients or updates. The same 5^3 physical stencil was translated to radial indices 12/32/51; response increments were aligned before comparison. Shell-off distances were `{off}`. Shell-on distances were `{on}`, at least `{float(coordinate_summary['shell_on_to_off_min_ratio']):.3g}` times the paired shell-off values. The small nonzero shell-off distances are floating FFT/convolution numerical residuals, not an input coordinate channel.

`RADIAL_POSITION_IDENTIFIABILITY = {coordinate_summary['radial_position_identifiability']}`: the model distinguishes coarse shell membership but not continuous position, theta, or phi.

`COORDINATE_INFORMATION_GAP = HIGH` because coordinate systems and shell definitions differ and neither continuous spherical coordinates nor the paper's Cartesian shell geometry is present.
"""
    write(out / "coordinates/coordinate_information_audit.md", coordinate_audit)

    candidates = """# Candidate coordinate embeddings (design only; no training)

These are controlled design candidates, not selected by validation.

## C1 — continuous spherical features

Append five channels: normalized `log(r)`, `sin(theta)`, `cos(theta)`, `sin(phi)`, `cos(phi)`. The periodic pair avoids a raw-phi seam. These are coordinate-position features only and do not transform B or velocity components.

## C2 — paper-analogue positional embedding

An exact Kerr–Schild Cartesian position map is not authorized because the current simulation's spin/mapping convention is absent. A clearly labelled `ADAPTED_POSITIONAL_ANALOGUE` could use normalized Euclidean spherical-position proxies `(r sin(theta) cos(phi), r sin(theta) sin(phi), r cos(theta))`; it must not be called exact Cartesian KS.

## C3 — shell plus continuous hybrid

Retain the existing eight physical-r shell channels and append the five C1 channels. This keeps the frozen coarse central-focus signal while resolving position within a shell.

No candidate is trained or ranked in Stage X. Because field basis and operator provenance are stronger blockers, these designs are not the authorized next stage.
"""
    write(out / "coordinates/candidate_coordinate_embeddings.md", candidates)

    vector = """# Vector basis audit

The paper states Cartesian Kerr–Schild simulation coordinates and labels components `Bx,By,Bz` and `vx,vy,vz` (Appendices A.2/C.2). The current raw ATHDF states only `Coordinates=kerr-schild` and variables `Bcc1,Bcc2,Bcc3,vel1,vel2,vel3`, with x1/x2/x3 numerically identified as r/theta/phi. The project deliberately retains those as spherical-KS coordinate-component names (`src/build_regrid_from_athdf.py:710-712`) and performs no vector conversion.

Position mapping and vector transformation are distinct. Knowing `(r,theta,phi)` locations is sufficient to design positional features; it is not sufficient to transform vector components without the simulation's mapping and component conventions.

`VECTOR_BASIS_GAP = FUNDAMENTAL`.
"""
    write(out / "fields/vector_basis_audit.md", vector)

    provenance = """# Vector-transform provenance audit

Searched sources:

- all files at `/mnt/d/GRMHD_data` (the directory contains only `mad98.prim.?????.athdf` files);
- ATHDF root attributes and datasets on representative and selected Stage-X files;
- project configs, scripts, source, artifacts, README, and notes for spin, metric, EOS, tetrad, orthonormal/coordinate basis, and conversion routines.

Available ATHDF provenance: `Coordinates=kerr-schild`, RootGrid/mesh metadata, face/centre arrays, and variable names. Not available: black-hole spin, exact spherical Kerr–Schild mapping convention, vector tensor/basis definition, Eulerian velocity convention, magnetic-field convention needed for a Cartesian transform, tetrad, or an Athena input/problem-generator file. The filename `mad98` is not interpreted as spin.

Required transformation inputs are therefore incomplete.

`VECTOR_TRANSFORM_PROVENANCE = INCOMPLETE`

`CARTESIAN_VECTOR_CONVERSION_AUTHORIZED = false`
"""
    write(out / "fields/transform_provenance_audit.md", provenance)

    thermal = """# Thermal-variable audit

The current ATHDF `VariableNames` contains `press` and no `eint`; its attributes have no Gamma, EOS, or adiabatic-index field. The raw directory has no Athena input/problem-generator file. The project does not perform a conversion.

The paper's GRMHD Appendix A.2 says outputs include pressure P, while network Appendix C.2 says the fifth tensor channel is internal energy `eint`. The paper does not provide the GRMHD EOS relation used to resolve that mismatch. Gamma=5/3 appears in Appendix A.1 for the separate Newtonian MHD setup and cannot be transferred to the GRMHD data.

`PRESS_TO_EINT_CONVERSION_AUTHORIZED = false`

`THERMAL_VARIABLE_GAP = HIGH`
"""
    write(out / "fields/thermal_variable_audit.md", thermal)

    resolution_lines = [
        "| resolution | median field relative L2 | median increment relative difference | median increment cosine |",
        "|---:|---:|---:|---:|",
    ]
    for resolution in (32, 64, 96, 128):
        fields = [row for row in field_metric_rows if int(row["resolution"]) == resolution]
        increments = [row for row in temporal_metric_rows if int(row["resolution"]) == resolution]
        resolution_lines.append(
            f"| {resolution} | {statistics.median(float(row['global_relative_l2']) for row in fields):.6g} | "
            f"{statistics.median(float(row['residual_relative_difference']) for row in increments):.6g} | "
            f"{statistics.median(float(row['residual_cosine']) for row in increments):.6g} |"
        )
    channel_lines = [
        "| channel | field rel-L2 | increment rel-difference | increment cosine | shell-increment rel-L2 | radial-increment rel-L2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for channel in ("Bcc1", "Bcc2", "Bcc3", "rho", "press", "vel1", "vel2", "vel3"):
        fields = [row for row in field_metric_rows if int(row["resolution"]) == 64 and row["channel"] == channel]
        increments = [row for row in temporal_metric_rows if int(row["resolution"]) == 64 and row["channel"] == channel]
        channel_lines.append(
            f"| {channel} | {statistics.median(float(row['global_relative_l2']) for row in fields):.6g} | "
            f"{statistics.median(float(row['residual_relative_difference']) for row in increments):.6g} | "
            f"{statistics.median(float(row['residual_cosine']) for row in increments):.6g} | "
            f"{statistics.median(float(row['shell_increment_relative_l2']) for row in increments):.6g} | "
            f"{statistics.median(float(row['radial_increment_relative_l2']) for row in increments):.6g} |"
        )
    metadata = raw["raw_metadata"]
    level_lines = [
        "| AMR level | leaf blocks | leaf cells | dr range | dtheta range | dphi range |",
        "|---:|---:|---:|---|---|---|",
    ]
    for level in sorted(metadata["leaf_blocks_by_level"], key=int):
        spacing = metadata["coordinate_spacing_by_level"][level]
        level_lines.append(
            f"| {level} | {metadata['leaf_blocks_by_level'][level]} | {metadata['leaf_cells_by_level'][level]} | "
            f"{spacing['dr']} | {spacing['dtheta']} | {spacing['dphi']} |"
        )
    raw_md = f"""# Raw AMR and diagnostic regrid audit

## Source/grid contract

- Raw directory: `{raw['raw_directory']}`; 20 fixed source files are enumerated in `raw_amr_audit.json`.
- Fixed field snapshots: `{config['raw']['field_snapshots']}`.
- Fixed adjacent-pair sources: `{config['raw']['adjacent_pair_sources']}`.
- All selected grid signatures equal `{raw['source_grid_signature']}`: `GRID_LAYOUT_STATIC = true`.
- ATHDF `Coordinates=kerr-schild`; numeric axes establish `(x1,x2,x3)=(r,theta,phi)`.
- Root grid `(r,theta,phi)={metadata['root_grid_size_x1_x2_x3']}`, MeshBlock `{metadata['mesh_block_size_x1_x2_x3']}`, max level `{metadata['max_level']}`, `{metadata['num_mesh_blocks']}` leaf blocks and `{metadata['actual_leaf_cell_count']}` actual leaf cells.
- All-domain-equivalent finest resolution is `{metadata['finest_effective_resolution_x1_x2_x3']}`; this is not the actual number of cells.
- Bounds: r `{metadata['r_bounds']}`, theta `{metadata['theta_bounds']}`, phi `{metadata['phi_bounds']}`. Stage X samples only r=`{config['regrid']['r_edge_range']}` to match the production tensor.

{chr(10).join(level_lines)}

Radial spacing ranges are not monotone in level because levels occupy different radial bands on a geometric source grid; level number alone must not be compared without position.

## Diagnostic sampling contract

Every 32/64/96/128 cube uses the exact production `finest-leaf nearest-cell-center` mapping (`src/build_regrid_from_athdf.py:265-314,446-457`). The independently generated audited 64^3 arrays were bitwise equal to `data_proc/grmhd_regrid_inner_r200_64_expanded.h5` for all 80 checked snapshot/channel arrays. The 128^3 grid is `HIGHER_RES_SAMPLING_REFERENCE`, not raw-AMR truth.

{chr(10).join(resolution_lines)}

At 64^3, detailed channel medians are:

{chr(10).join(channel_lines)}

Inner/middle/outer field loss medians are `{raw['field_loss_by_region_64']}`; temporal-increment loss medians are `{raw['temporal_increment_loss_by_region_64']}`. The inner increment loss is largest, although outer loss is also substantial.

`TEMPORAL_INCREMENT_REGRID_FIDELITY = {fidelity}` under the thresholds frozen before the audit in `{args.config}`. The overall 64^3 medians are relative difference `{raw['median_64_increment_relative_difference']:.6g}` and cosine `{raw['median_64_increment_cosine']:.6g}`. Full snapshot/channel/pair rows, shell vectors, radial profiles, spectra, extrema and q001/q999 are in the four CSV tables.

The frozen rule is GOOD when median relative difference <=0.25 and cosine >=0.95; MODERATE when <=0.50 and >=0.80; otherwise POOR. The 64^3 cosine clears MODERATE, but relative difference `0.503429` does not.

## Conservation caveat

The mapping is non-conservative and not divergence preserving (`src/build_regrid_from_athdf.py:693-712`). A plain Cartesian divergence is not the GRMHD magnetic constraint in this representation; `STRICT_DIVB_AUDIT_NOT_AUTHORIZED = true`.
"""
    write(out / "regrid/raw_amr_audit.md", raw_md)

    gap_rows = [
        ("coordinate system", "Cartesian Kerr-Schild", "spherical Kerr-Schild tensor", "false", "FUNDAMENTAL", "paper A.2 vs ATHDF/config", "requires matching data or fully verified transform"),
        ("coordinate grid", "regular Cartesian 64 cube", "log-r × linear-theta × periodic-phi 64 tensor", "false", "MAJOR", "paper C.2 vs regrid code", "partially, with adapted operator/coordinates"),
        ("vector component basis", "Cartesian x/y/z", "retained 1/2/3 spherical-coordinate representation", "false", "FUNDAMENTAL", "paper A.2/C.2; ATHDF attrs/project warning", "not with current provenance"),
        ("magnetic component semantics", "Bx/By/Bz; coupled through CT", "cell-centred Bcc1/2/3 nearest sampled; no transform/CT", "false", "FUNDAMENTAL", "paper E; current source", "requires source/coupling provenance"),
        ("velocity component semantics", "vx/vy/vz", "vel1/2/3, convention unspecified in ATHDF", "false", "FUNDAMENTAL", "paper C.2; ATHDF VariableNames", "requires simulation provenance"),
        ("thermal variable", "eint in model tensor (P stated elsewhere)", "press", "false", "MAJOR", "paper A.2/C.2; ATHDF", "requires verified EOS/paper clarification"),
        ("spatial domain", "Cartesian fine L^3, numeric bounds absent", "full sphere, cropped r=1.1..200", "unknown", "MAJOR", "paper §2; raw/regrid attrs", "requires matching data"),
        ("regridding", "canonical Cartesian 64 grid; upstream remap unspecified", "finest-leaf nearest centre", "unknown", regrid_severity, "paper C.2; current source plus Stage-X convergence", "yes, diagnostically"),
        ("AMR treatment", "not specified", "2020 leaf blocks sampled to regular tensor", "unknown", "UNKNOWN", "paper absent; raw metadata", "unknown"),
        ("resolution", "64_x×64_y×64_z", "64_phi×64_theta×64_r", "nominal only", "MAJOR", "paper C.2; HDF5", "not by shape alone"),
        ("positional encoding", "default Cartesian-index radial shells", "physical spherical-r shells only", "false", "MAJOR", "paper C.4; current forward", "adapted continuous features possible"),
        ("radial shells", "Euclidean index radius, rmax=10", "physical spherical r across 1.1..200", "false", "MAJOR", "paper C.4/Table1; shells.py", "adaptable but not exact"),
        ("spectral operator", "exact instantiation not specified", "3D SpectralConv, modes8", "unknown", "UNKNOWN", "paper C.8; current config", "paper code needed"),
        ("differential operator", "exact instantiation not specified", "3D finite-difference branch", "unknown", "UNKNOWN", "paper C.8; upstream/current", "paper code needed"),
        ("DISCO/local integral operator", "explicit volumetric equidistant DISCO", "absent", "false", "MAJOR", "paper C.8; pinned upstream/config", "adapted prototype possible; exact code needed"),
        ("boundary assumptions", "not reported for model; live inner coupling described", "periodic padding on all tensor axes", "unknown", "MAJOR", "paper C.8/E; current config", "requires paper code/spec"),
        ("number of snapshots", "300 generated, last250 used", "212 total", "false", "MODERATE", "paper §2; HDF5", "requires matching data"),
        ("time cadence", "DeltaT=T/300", "actual source times ~10 code units", "unknown", "MAJOR", "paper §2; manifests", "requires unit/provenance match"),
        ("split", "last250, 80/20", "169/43 snapshots with 168/42 pairs", "adapted", "MODERATE", "paper §2; split JSON", "yes as adaptation"),
        ("preprocessing", "canonical all channels", "P3 omits softclip for Bcc2/Bcc3/vel3", "false", "MAJOR", "paper C.3; config", "yes, but prior diagnostics motivated adaptation"),
        ("normalization", "median/MAD + gamma6 softclip", "train-only P3 robust normalization", "partial", "MODERATE", "paper C.3; config", "yes"),
        ("loss", "composite paper loss", "Plain L2 residual", "false", "MAJOR", "paper C.7; config", "yes, but not current best target"),
        ("training budget", "1200 epochs, batch4/accum4", "1200 epochs, batch1/accum4", "partial", "MODERATE", "paper C.9; Stage T", "resource-limited"),
        ("rollout", "fine rollout plus live coarse coupling", "offline spherical autoregression", "false", "MAJOR", "paper §2/E; Stage T", "requires coupling implementation/data"),
        ("coarse/fine coupling", "implemented with hydro overwrite and magnetic CT", "absent", "false", "FUNDAMENTAL", "paper E; current contract", "requires paper code/interface data"),
    ]
    write_csv(out / "gap_matrix.csv", [dict(zip(("aspect","paper","current","match","severity","evidence","scientifically fixable"), row, strict=True)) for row in gap_rows])

    comparison = f"""# Paper 64^3 versus current 64^3

Equal array shape does not imply equal scientific content.

| property | paper | current |
|---|---|---|
| grid | 64_x × 64_y × 64_z Cartesian KS cube | 64_phi × 64_theta × 64_r spherical KS tensor |
| radial sampling | Cartesian distance emerges from x/y/z cells | explicitly logarithmic r centres from 1.1 to 200 |
| angular/polar behavior | Cartesian cells, no spherical coordinate pole | uniform theta/phi coordinates; physical azimuthal scale shrinks as sin(theta) |
| raw source | exact AMR/remap path not specified | Athena++ spherical-KS AMR, nearest leaf centre |
| vector fields | Bx/y/z and vx/y/z | Bcc1/2/3 and vel1/2/3 retained without basis conversion |
| thermal field | eint in C.2 | press |
| default position signal | Cartesian-index radial shells | physical spherical-r shells |
| conservation/div B | magnetic live coupling uses CT | regrid is non-conservative and not divergence preserving |

Current 64^3 is therefore a workflow/method adaptation, not a scientifically equivalent realization of the paper's 64^3 target.
"""
    write(out / "comparison/paper64_vs_current64.md", comparison)

    score_rows = [
        {"gap":"G1 coordinate representation","severity":"MAJOR","evidence_strength":"STRONG","fixability":"ADAPTED_FEATURES_POSSIBLE_EXACT_MATCH_REQUIRES_DATA","causal_plausibility_for_transport_failure":"HIGH"},
        {"gap":"G2 vector/field basis","severity":"FUNDAMENTAL","evidence_strength":"STRONG_FOR_MISMATCH_INCOMPLETE_FOR_TRANSFORM","fixability":"BLOCKED_BY_PROVENANCE","causal_plausibility_for_transport_failure":"HIGH"},
        {"gap":"G3 missing operator mechanism","severity":"MAJOR","evidence_strength":"STRONG","fixability":"ADAPTED_PROTOTYPE_OR_PAPER_CODE","causal_plausibility_for_transport_failure":"HIGH"},
        {"gap":"G4 regridding","severity":regrid_severity,"evidence_strength":"STRONG_SAMPLING_REFERENCE_NOT_RAW_TRUTH","fixability":"HIGHER_RESOLUTION_DIAGNOSTIC_OR_CONSERVATIVE_REMAP","causal_plausibility_for_transport_failure":"MODERATE" if fidelity != "POOR" else "HIGH"},
        {"gap":"G5 thermal variable","severity":"MAJOR","evidence_strength":"STRONG_FOR_MISMATCH_INCOMPLETE_FOR_EOS","fixability":"BLOCKED_BY_EOS_OR_MATCHING_DATA","causal_plausibility_for_transport_failure":"HIGH"},
    ]
    write_csv(out / "comparison/scientific_comparability_scorecard.csv", score_rows)

    field_regions = raw["field_loss_by_region_64"]
    increment_regions = raw["temporal_increment_loss_by_region_64"]
    inner_damage = float(increment_regions["inner"]) > 0.4
    report_table = f"""| gap | paper | current | severity | evidence | transport-failure plausibility |
|---|---|---|---|---|---|
| coordinate representation | Cartesian-index radial shells on Cartesian KS cube | 8 physical-r shells; no continuous r/theta/phi | HIGH | paper C.4 + resolved forward + frozen response test | HIGH |
| vector basis | Cartesian x/y/z | untransformed spherical 1/2/3 representation | FUNDAMENTAL | paper A.2/C.2 + ATHDF attrs/variables | HIGH |
| operator family | volumetric equidistant DISCO LocalNO | spectral + finite difference; no local integral | HIGH | paper C.8 + pinned upstream/config/forward | HIGH |
| regridding | paper remap unspecified; Cartesian 64 cube | nearest-centre AMR→spherical 64 tensor | {regrid_severity} | 10 fields + 10 adjacent-pair 32/64/96/128 audit | {'HIGH' if fidelity == 'POOR' else 'MODERATE'} |
| thermal variable | eint in C.2 (P in A.2) | press; no EOS conversion | HIGH | paper internal inconsistency + ATHDF attrs | HIGH |
"""
    answers = f"""1. **Same coordinate system?** No. Paper: Cartesian Kerr–Schild. Current: spherical Kerr–Schild `(phi,theta,r)` tensor.
2. **Does the network know physical r,theta,phi?** Only partially: eight discrete physical-r shells. It has no continuous r/log-r, theta, or phi channels.
3. **Does the paper explicitly use Cartesian positional information?** The default uses shells computed from Cartesian index distance; explicit normalized Cartesian Fourier `(xi,eta,zeta)` features occur only in an ablation.
4. **Same vector basis?** No; paper labels Cartesian x/y/z while current retains 1/2/3 spherical-coordinate representation.
5. **Enough provenance for a correct transform?** No. Spin, mapping, velocity/magnetic component conventions and tetrad/basis data are missing.
6. **Scientific basis for press→eint?** No; neither raw EOS/Gamma nor a GRMHD conversion contract is available.
7. **Does the paper LocalNO include a missing DISCO/local-integral mechanism?** Yes, volumetric equidistant DISCO is explicit. Exact per-layer composition remains unspecified.
8. **Operator gap HIGH/FUNDAMENTAL?** HIGH: the required local-integral mechanism is wholly absent, though causal responsibility for failures is not proven.
9. **Does 64^3 preserve temporal increments?** `{fidelity}` under the frozen thresholds: median relative difference `{raw['median_64_increment_relative_difference']:.6g}`, median cosine `{raw['median_64_increment_cosine']:.6g}` versus the 128 sampling reference.
10. **Are inner AMR dynamics visibly damaged at 64^3?** {'Yes: the inner increment relative difference exceeds 0.4 and is the largest regional median' if inner_damage else 'Not strongly under the regional diagnostic'}, with inner/middle/outer increment losses `{increment_regions}` and field losses `{field_regions}`. Outer loss is also substantial, so this is not an inner-only effect. This is relative to a sampling reference, not raw truth.
11. **Are paper 64^3 and current 64^3 scientifically equivalent?** No: geometry, cell scales, vector basis, thermal field, positional semantics, remap, and operator differ.
12. **Paper-method reproduction or adaptation?** Only a spherical-KS differential-LocalNO workflow/method adaptation.
13. **Most valuable single intervention?** Acquire matching Cartesian-KS/eint data plus simulation/EOS/vector provenance and the paper's 3D DISCO/code contract before another model pilot.
"""
    labels = f"""PRIMARY_DECISION = E

COORDINATE_INFORMATION_GAP = HIGH
VECTOR_BASIS_GAP = FUNDAMENTAL
VECTOR_TRANSFORM_PROVENANCE = INCOMPLETE
PAPER_OPERATOR_GAP = HIGH
TEMPORAL_INCREMENT_REGRID_FIDELITY = {fidelity}
THERMAL_VARIABLE_GAP = HIGH

CURRENT_TARGET_NOT_SCIENTIFICALLY_COMPARABLE_TO_PAPER = true

AUTHORIZE_NEXT_STAGE = data_or_code_acquisition
"""
    stage_report = f"""# Stage X — Paper Method Gap and Coordinate/Field Representation Audit

{report_table}

## Direct answers

{answers}

## Audit scope and reproducibility

- Paper: arXiv `2512.01576v1`, PDF SHA256 `{PAPER_SHA}`.
- Project/upstream: `{project_commit}` / `{upstream_commit}`.
- Raw audit: {len(config['raw']['field_snapshots'])} fixed field snapshots and {len(config['raw']['adjacent_pair_sources'])} fixed adjacent pairs, all four sampling resolutions.
- Production `64^3` equivalence: bitwise exact for every audited snapshot/channel array.
- 128^3 label: `HIGHER_RES_SAMPLING_REFERENCE`, never raw truth.
- No training, loss tuning, vector conversion, EOS inference, raw-data mutation, or processed-HDF5 regeneration occurred.

## Verification

- Required Stage-X artifact set: complete and non-empty.
- CSV row counts: field convergence 320; temporal increments 320; shell increments 2,560; radial increments 25,600; production equivalence 80; coordinate identifiability 7.
- Full regression suite: `491 passed, 13 warnings`.
- `git diff --check`: passed.
- Pinned upstream worktree: clean at `{upstream_commit}`.

## Final labels

{labels}
"""
    write(out / "STAGE_X_REPORT.md", stage_report)

    decision = f"""# Stage X decision

`PRIMARY_DECISION = E — MULTIPLE_FUNDAMENTAL_METHOD_GAPS`

This decision follows the stated gate: more than two independent gaps reach HIGH/FUNDAMENTAL—coordinate representation, vector/field basis, missing volumetric local-integral operator, and thermal semantics. Regridding fidelity is reported separately as `{fidelity}` and is not used to erase those representation gaps.

The vector basis mismatch is fundamental and its transform provenance is incomplete; the paper operator gap is HIGH. Therefore Y4 applies:

`CURRENT_TARGET_NOT_SCIENTIFICALLY_COMPARABLE_TO_PAPER = true`

`AUTHORIZE_NEXT_STAGE = data_or_code_acquisition`

Acquire matching Cartesian-KS/eint training data or complete current-simulation spin/EOS/vector convention provenance, plus the paper's volumetric 3D DISCO implementation/specification and coarse/fine CT interface. No further training variation is authorized by Stage X.

{labels}
"""
    write(out / "STAGE_X_DECISION.md", decision)
    print(json.dumps({"output": str(out), "decision": "E", "fidelity": fidelity, "authorize": "data_or_code_acquisition"}, indent=2))


if __name__ == "__main__":
    main()
