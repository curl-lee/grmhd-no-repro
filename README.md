# GRMHD neural-operator adapted reproduction

This repository contains a WSL2/Linux reproduction workflow for the standalone
next-state surrogate studied in **From Black Hole to Galaxy: Neural Operator
Framework for Accretion and Feedback Dynamics**:

```text
u(t) -> u(t + delta_t)
u = [Bcc1, Bcc2, Bcc3, rho, press, vel1, vel2, vel3]
```

The published snapshot contains the confirmed WSL work from Stage F through
Stage AH. It is an **adapted reproduction**, not an exact reproduction of the
paper. The working data use spherical Kerr--Schild coordinates, `press` as the
thermal channel, reduced/resampled grids, and proxy FNO/LocalNO operators. The
paper's exact Cartesian-KS conversion, `eint` contract, coarse/fine coupling,
official data, and exact volumetric 3-D DISCO implementation were not available.

## Environment

The recorded execution environment was Ubuntu under WSL2 with Python 3.11.15,
PyTorch 2.12.1+cu130, CUDA 13.0, and an NVIDIA GeForce RTX 5070. The exact
machine-readable capture is in
[`outputs/environment_wsl.json`](outputs/environment_wsl.json). The upstream
`neuraloperator` source is a submodule pinned at:

```text
86a8bc7812a31b42c4f7895693cf4ac11521c066
```

Clone and install with:

```bash
git clone --recurse-submodules https://github.com/curl-lee/grmhd-no-repro.git
cd grmhd-no-repro
git -C external/neuraloperator rev-parse HEAD
python -m pip install -e external/neuraloperator
python -m pip install -e '.[test,plot,train]'
python -m pytest -q
```

## Dataset contract

Raw Athena++ `.athdf` snapshots and processed HDF5 files are not committed.
The principal expanded dataset has shape `(212, 8, 64, 64, 64)` with axes
`(N,C,phi,theta,r)` and channel order shown above. It is produced by
nearest-cell-center sampling from the finest containing AMR leaf block onto a
linear-periodic phi grid, linear theta grid, and logarithmic radial grid. This
operation is neither conservative nor divergence preserving and performs no
interpolation.

Expected filenames, sizes, SHA256 values, coordinate metadata, and the 64/96/128
contracts are in [`docs/data_manifest.md`](docs/data_manifest.md). Put local data
under `data_proc/` or update a config path; never commit the datasets.

## Operators and models

- Persistence and oracle-aware baselines.
- Upstream FNO and differential LocalNO proxies.
- Mixed-basis spectral and spherical finite-difference diagnostics.
- Project-local adapted volumetric DISCO3D and LocalNO attachment.
- Isotropic spherical DISCO3D geometry audit.
- Anisotropic local-spherical-tangent DISCO3D and LocalNO attachment.
- Canonical/P3 preprocessing, shell channels, train-only priors, paper-adapted
  losses, training/checkpoint utilities, rollout, and morphology metrics.

See [`docs/operator_inventory.md`](docs/operator_inventory.md) for implementation
paths, stage usage, geometry, tests, and validity labels.

## WSL stage index

| Stage | Purpose | Operator/model | Main result | Artifacts |
|---|---|---|---|---|
| F | Paired integration smoke | FNO Full / Plain | engineering path passed; Full H1 warning | `outputs/paper_reduced100/stage_f/` |
| G | Matched 30-epoch pilots | FNO Full / Plain | finite pilots; pause for H1 diagnosis | `outputs/paper_reduced100/stage_g/` |
| H | H1 definition/gradient audit | diagnostic | index-grid scaling mismatch supported | `outputs/paper_reduced100/stage_h/` |
| I | H1 diagnostic extensions | FNO variants | no extension rescued Full | `outputs/paper_reduced100/stage_i/` |
| J | Method/data/coupling audit | source audit | exact paper path remained blocked | `outputs/paper_reduced100/stage_j/` |
| K | Differential LocalNO pilot | LocalNO | training complete; rollout unstable | `outputs/paper_reduced100/stage_k/` |
| L | Collapse attribution | post-hoc metrics | mixed preprocessing/model failure | `outputs/paper_reduced100/stage_l/` |
| M | Transform/oracle transport | post-hoc metrics | mixed transport/preprocessing failure | `outputs/paper_reduced100/stage_m/` |
| N | P3 and operator response | differential LocalNO | P3 ready; mixed operator-response failure | `outputs/paper_reduced100/stage_n/` |
| O | P3 LocalNO pilot | differential LocalNO | pilot complete; rollout unstable | `outputs/paper_reduced100/stage_o/` |
| P | Closed-loop attribution | counterfactual audit | mixed closed-loop failure | `outputs/paper_reduced100/stage_p/` |
| Q | Persistence anchor audit | residual wrapper | no valid anchor candidate | `outputs/paper_reduced100/stage_q/` |
| R | Residual P3 pilot | differential LocalNO | one-step gain; rollout unstable | `outputs/paper_reduced100/stage_r/` |
| S | Expanded-data audit | differential LocalNO | more optimization, not more data | `artifacts/stage_s/` |
| T | Paper-budget convergence | differential LocalNO | state improved, dynamics did not | `artifacts/stage_t/` |
| U | Grid/operator geometry | geometry variants | spectral branch suspected | `artifacts/stage_u/` |
| V | Objective alignment | objective variants | objective mismatch not supported | `artifacts/stage_v/` |
| W | Mixed spectral basis | mixed-basis LocalNO | mixed basis not supported | `artifacts/stage_w/` |
| X | Paper-method gap | audit | multiple fundamental gaps | `artifacts/stage_x/` |
| Y | Provenance recovery | asset audit | exact reproduction blocked by missing assets | `artifacts/stage_y/` |
| Z | Higher resolution | Z64/Z96/Z128 audit | information gain; model comparison incomplete | `artifacts/stage_z/` |
| AA | GPU recovery gate | environment audit | WSL GPU bridge unresolved | `artifacts/stage_aa/` |
| AB | GPU bridge recovery | Z96 LocalNO | bridge restored; Z96 model worse | `artifacts/stage_ab/` |
| AC | Volumetric DISCO contract | adapted DISCO3D | one-cell/K=5 contract invalid | `artifacts/stage_ac/` |
| AD | Repaired volumetric DISCO | adapted DISCO3D LocalNO | one-step-only improvement | `artifacts/stage_ad/` |
| AE | Isotropic spherical geometry | spherical DISCO3D | normalization invalid at four corners | `artifacts/stage_ae/` |
| AF | Anisotropic spherical geometry | anisotropic DISCO3D | implementation gates passed; no training | `artifacts/stage_af/` |
| AG | Controlled spherical training | anisotropic spherical DISCO LocalNO | 300 epochs / 12,600 updates completed; best/last reload passed | `artifacts/stage_ag/` |
| AH | Post-training scientific analysis | Stage AD vs Stage AG | partial transport improvement; one-step direction and rollout gates failed | `artifacts/stage_ah/` |

The table reports each stage's frozen decision rather than reinterpreting its
metrics. Detailed status and provenance are in
[`docs/reproduction_status_through_stage_ah.md`](docs/reproduction_status_through_stage_ah.md)
and [`docs/wsl_stage_provenance.md`](docs/wsl_stage_provenance.md).

## Repository layout

- `src/grmhd/`: data, preprocessing, operators, models, loss, training, and evaluation code.
- `scripts/`: reproducible audit, preparation, training, and evaluation entry points.
- `configs/`: frozen experiment contracts through Stage AH.
- `tests/`: unit, dense-reference, gradient, boundary, equivariance, and CPU/CUDA tests.
- `outputs/paper_reduced100/`: Stage F--R lightweight results.
- `artifacts/stage_*/`: Stage S--AH reports, metrics, figures, and provenance.
- `docs/`: scientific contracts and publication manifests.

Most checkpoint binaries are intentionally omitted to keep this source-and-results
release reviewable. Stage AG's formal `best.pt` and `last.pt` are retained; all
other omitted paths, sizes, SHA256 values, and regeneration routes are indexed
in [`docs/excluded_checkpoint_sha256.csv`](docs/excluded_checkpoint_sha256.csv)
and [`docs/stage_ag_excluded_artifacts_sha256.csv`](docs/stage_ag_excluded_artifacts_sha256.csv).

## Scientific limitations

This work does not establish an exact paper reproduction. The released results
are limited by adapted spherical geometry, coordinate-basis vector components,
the `press` thermal proxy, nearest-neighbor nonconservative regridding, a strong
preprocessing oracle floor for selected channels, and proxy operator choices.
Stage AG/AH test that anisotropic spherical-coordinate geometry proxy under a
controlled 300-epoch budget. It retains state accuracy and improves median shell
transport error, but does not improve residual L2/direction or the first-10x
rollout landmark; this is a partial adapted-workflow result, not exact DISCO.
