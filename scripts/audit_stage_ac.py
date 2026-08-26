#!/usr/bin/env python3
"""Materialize the Stage AC pre-implementation feasibility decision."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import yaml

from grmhd.dataset import sha256_file
from grmhd.stage_ac import audit_radial_hat_feasibility


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/stage_ac"
UPSTREAM = "86a8bc7812a31b42c4f7895693cf4ac11521c066"
SCOPE = "ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION"


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True))


def main() -> None:
    stage_s_path = ROOT / "configs/stage_s/expanded_localno_p3_residual.yaml"
    stage_t_path = ROOT / "configs/stage_t/optimization_convergence.yaml"
    stage_s = yaml.safe_load(stage_s_path.read_text(encoding="utf-8"))
    stage_t = yaml.safe_load(stage_t_path.read_text(encoding="utf-8"))
    upstream = subprocess.check_output(
        ["git", "-C", str(ROOT / "external/neuraloperator"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    upstream_status = subprocess.check_output(
        ["git", "-C", str(ROOT / "external/neuraloperator"), "status", "--short"],
        text=True,
    ).strip()
    if upstream != UPSTREAM or upstream_status:
        raise ValueError("pinned upstream provenance changed")
    dataset = ROOT / stage_s["data"]["dataset"]
    normalizer = ROOT / stage_s["preprocessing"]["artifact"] / "normalizer.npz"
    if sha256_file(dataset) != stage_s["data"]["dataset_sha256"]:
        raise ValueError("frozen Stage AC dataset changed")
    if sha256_file(normalizer) != stage_s["preprocessing"]["normalizer_sha256"]:
        raise ValueError("frozen Stage AC P3 normalizer changed")

    basis = audit_radial_hat_feasibility(
        in_shape=(64, 64, 64),
        out_shape=(64, 64, 64),
        domain_length=(2.0, 2.0, 2.0),
        kernel_size=5,
        eps=1.0e-12,
    )
    write_json(OUT / "implementation/disco3d_basis.json", basis)

    contract = f"""# Stage AC Contract

- `REPRODUCTION_SCOPE = {SCOPE}`
- `MODEL_NAME = ADAPTED_VOLUMETRIC_3D_DISCO_LOCALNO`
- `DISCO3D_IMPLEMENTATION_CLASS = ADAPTED_ISOTROPIC_RADIAL_PIECEWISE_LINEAR`
- `DISCO3D_DISTANCE = NORMALIZED_COMPUTATIONAL_INDEX_DISTANCE`
- `DISCO3D_BOUNDARY = COMPUTATIONAL_PERIODIC_ADAPTATION`
- resolution: 64 x 64 x 64
- dataset: `{stage_s['data']['dataset']}`
- dataset SHA256: `{stage_s['data']['dataset_sha256']}`
- P3 normalizer SHA256: `{stage_s['preprocessing']['normalizer_sha256']}`
- split: 168 train pairs, dropped 168->169, 42 validation pairs
- target: plain normalized residual
- frozen Stage-T scheduler: {stage_t['full_long']['warmup_epochs']} warmup epochs in the 1200-epoch schedule
- upstream commit: `{upstream}`; worktree clean

No data, preprocessing, split, loss, boundary, width, modes, or pinned upstream source was changed.
"""
    write_text(OUT / "scope/stage_ac_contract.md", contract)
    write_text(OUT / "scope/scientific_limitations.md", """# Stage AC Scientific Limitations

The requested operator is an adapted computational-grid mechanism, not an exact paper
implementation. Official paper data/code and the exact volumetric 3D DISCO basis remain
unavailable. Computational index distance and circular padding are not Kerr-Schild proper
distance or physical spherical boundary conditions.

The frozen Stage AC radial-hat discretization is internally infeasible at 64 cubed: a one-cell
cutoff produces only the sampled radii 0 and R, while K=5 requires nonzero discrete support at
five radial centers. No scientific training may begin until that contract is explicitly revised.
""")

    write_text(OUT / "source_audit/upstream_disco2d_contract.md", """# Upstream 2D DISCO Contract Audit

Classification: `UPSTREAM_EXPLICIT` unless otherwise marked.

- Base weight shape is `(out_channels, in_channels/groups, kernel_size)` with scale
  `sqrt(1/groupsize)` and Gaussian initialization; optional bias is zero-initialized.
- For non-Morlet `[2,4]`, `kernel_size=(2-1)*4+1=5`.
- Equidistant 2D cutoff defaults to `max(domain_length[i]/out_shape[i])`.
- Local stencil sizes use `floor(2*R*in_shape[i]/domain_length[i])+1`.
- Quadrature is `L0*L1/(N0*N1)` and the basis is discretely normalized.
- The dense local buffer has shape `(K, psi_local_h, psi_local_w)` and is nonpersistent.
- Kernel synthesis is `einsum('kxy,ogk->ogxy', basis, weight)` after the upstream
  orientation permutation/flip.
- Efficient evaluation maps to grouped `conv2d`; integer stride implements resolution scaling.
- `periodic` sets a padding-mode attribute, but this pinned forward passes integer padding
  directly to `conv2d` and does not itself call circular `pad`. This implementation detail is
  explicit in the pinned source.
- Groups and bias follow PyTorch grouped-convolution semantics.

Evidence: `external/neuraloperator/neuralop/layers/discrete_continuous_convolution.py`,
class `DiscreteContinuousConv` and class `EquidistantDiscreteContinuousConv2d`.
""")
    write_text(OUT / "source_audit/upstream_localno_contract.md", """# Upstream LocalNO Contract Audit

Classification: `UPSTREAM_EXPLICIT`.

- Pinned `LocalNOBlocks` expands scalar `disco_layers` and `diff_layers` across all layers.
- Enabled differential kernels support up to three dimensions.
- Enabled local-integral DISCO is rejected unless `len(n_modes)==2`.
- The pinned implementation constructs `EquidistantDiscreteContinuousConv2d` branches.
- Post-activation ordering is spectral, differential, local integral, branch sum, optional norm,
  local skip addition, activation, then optional channel MLP/normalization.
- The branch equation is `x_spectral + x_differential + x_local_integral` before skip.

Evidence: `external/neuraloperator/neuralop/layers/local_no_block.py`, constructor checks,
local-convolution construction, and `forward_with_postactivation`.
""")
    write_text(OUT / "source_audit/paper_operator_gap.md", """# Paper Operator Gap

- `UPSTREAM_EXPLICIT`: general LocalNO uses parallel spectral, differential, and DISCO local-integral branches.
- `UPSTREAM_EXPLICIT`: the pinned executable local-integral implementation is 2D only.
- `UNKNOWN_PAPER_DETAIL`: exact volumetric basis, quadrature, support radius, boundary semantics,
  and per-layer placement for the target paper are unavailable.
- `ADAPTED_3D_EXTENSION`: Stage AC proposed isotropic radial piecewise-linear hats on normalized
  computational indices, not a physical or exact paper operator.

`EXACT_3D_DISCO_IMPLEMENTATION_FOUND = false`
""")

    zero = basis["zero_support_basis_indices"]
    write_text(OUT / "implementation/disco3d_design.md", f"""# DISCO3D Design Feasibility

The frozen formulas resolve to:

- `radius_cutoff = {basis['radius_cutoff']}`
- `psi_local_phi/theta/r = {basis['local_stencil_shape']}`
- `q = {basis['quadrature_weight']}`
- sampled radii within support: `{basis['unique_supported_radii_within_cutoff']}`
- zero-support basis indices: `{zero}`

For K=5 the centers are 0, R/4, R/2, 3R/4, R. The exact 3x3x3 cell-centre stencil has
supported radii only 0 and R. Consequently bases 1, 2, and 3 have `Z_k=0`; division by
`Z_k+eps` leaves their quadrature integral at zero, not one. Their trainable coefficients would
also have identically zero forward contribution and gradient.

Implementing a trainable operator would require an unapproved scientific choice: increase the
cutoff/stencil, reduce K, or define a subcell/voxel-integrated projection rule. Stage AC forbids
changing radius or basis, and supplies no subcell quadrature contract, so implementation stops
before LocalNO integration and training.
""")
    blocked = {
        "schema_version": "stage-ac-mandatory-tests-v1",
        "status": "BLOCKED_MATHEMATICAL_CONTRACT_INFEASIBLE",
        "quadrature_normalization": "FAIL",
        "zero_support_basis_indices": zero,
        "shape": "NOT_RUN",
        "finite": "NOT_RUN",
        "backward": "NOT_RUN",
        "compact_support": "NOT_RUN",
        "translation_equivariance": "NOT_RUN",
        "constant_field": "NOT_RUN",
        "groups": "NOT_RUN",
        "bias": "NOT_RUN",
        "cpu_cuda_agreement": "NOT_RUN",
    }
    write_json(OUT / "implementation/disco3d_unit_tests.json", blocked)
    write_json(OUT / "implementation/dense_reference_test.json", {
        "status": "NOT_RUN_IMPLEMENTATION_GATE_FAILED", "match": False,
    })
    write_json(OUT / "implementation/gradcheck.json", {
        "status": "NOT_RUN_IMPLEMENTATION_GATE_FAILED", "pass": False,
    })
    write_json(OUT / "preflight/cuda_preflight.json", {
        "status": "NOT_RUN_IMPLEMENTATION_GATE_FAILED", "training_authorized": False,
        "cpu_fallback": False,
    })
    write_json(OUT / "preflight/branch_norms_initial.json", {
        "status": "NOT_RUN_IMPLEMENTATION_GATE_FAILED", "branch_active": False,
    })
    write_json(OUT / "training/disco3d_localno/resolved_config.json", {
        "status": "BLOCKED_BEFORE_MODEL_CONSTRUCTION",
        "reproduction_scope": SCOPE,
        "dataset": stage_s["data"],
        "preprocessing": stage_s["preprocessing"],
        "model_common": stage_s["model"],
        "training_started": False,
        "optimizer_updates": 0,
        "reason": "K=5 one-cell-cutoff point-sampled radial basis has three zero-support hats",
    })

    report = f"""# Stage AC — Adapted Volumetric 3D DISCO LocalNO

| model | params | local integral | state L2 | residual L2 | cosine | shell skill | radial skill | first10x |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Persistence | 0 | no | 0.310282 | 1 | 0 | 0 | 0 | stable |
| Stage-T differential LocalNO | 358296 | no | 0.266711 | 0.891851 | 0.740602 | -1.679472 | -0.348089 | 1 |
| Adapted 3D-DISCO LocalNO | not constructed | intended | not run | not run | not run | not run | not run | not run |

## Mandatory implementation-gate result

The upstream audit completed and the pinned source remains clean. The exact frozen Stage AC
formulas produce a 3x3x3 stencil with only radii 0 and R inside support. K=5 radial hats require
centers at 0, R/4, R/2, 3R/4, and R, so basis indices {zero} are identically zero. Their
normalizations fail (`Z_k=0`), and the associated weights cannot receive gradients.

This is not a GPU, I/O, or training failure and is not a scientific negative result about DISCO.
It is an internally inconsistent adapted-discretization contract discovered before model
construction. No radius, K, basis, boundary, loss, data, or upstream code was changed; no CUDA
preflight or training was run.

## Required contract decision

Exactly one of these scientific choices must be newly authorized and frozen before repair:

1. increase cutoff to at least three computational cell spacings (changing the frozen radius),
2. reduce the radial basis count to the two radii represented by the stencil (changing K), or
3. define a precise subcell/voxel-integrated basis projection and quadrature rule.

Until then, dense-reference, gradcheck, branch activity, controlled training, attribution, and
rollout cannot be meaningfully evaluated.
"""
    write_text(OUT / "STAGE_AC_REPORT.md", report)
    write_text(OUT / "STAGE_AC_DECISION.md", f"""# Stage AC Decision

PRIMARY_DECISION = F
PRIMARY_DECISION_LABEL = DISCO3D_IMPLEMENTATION_INVALID

REPRODUCTION_SCOPE = {SCOPE}
DISCO3D_IMPLEMENTATION = ADAPTED_ISOTROPIC_RADIAL_PIECEWISE_LINEAR
EXACT_3D_DISCO_IMPLEMENTATION_FOUND = false
EXACT_REPRODUCTION_BLOCKED = true

DISCO3D_UNIT_TESTS_PASS = false
DENSE_REFERENCE_MATCH = false
GRADCHECK_PASS = false
DISCO_BRANCH_ACTIVE = false

STATE_RETENTION_GATE = FAIL
RESIDUAL_GATE = FAIL
DIRECTION_GATE = FAIL
SHELL_IMPROVEMENT_GATE = FAIL
RADIAL_IMPROVEMENT_GATE = FAIL
ROLLOUT_IMPROVEMENT_GATE = FAIL

AUTHORIZE_NEXT_STAGE = implementation_repair
UPSTREAM_NEURALOPERATOR_MODIFIED = false
TRAINING_STARTED = false
""")
    print(json.dumps({
        "PRIMARY_DECISION": "F",
        "zero_support_basis_indices": zero,
        "quadrature_normalization_contract_pass": False,
        "training_started": False,
        "AUTHORIZE_NEXT_STAGE": "implementation_repair",
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
