# Reproduction status through Stage AF

## Verified / reproduced within the adapted contract

- WSL2/Linux data audit and deterministic AMR-to-grid construction for 111- and
  212-snapshot contracts at Z64, plus Z96/Z128 information audits.
- Canonical and P3 preprocessing, train-only fitting, oracle-aware metrics,
  paper-adapted losses/priors, shell channels, paired initialization/order,
  checkpoint strict reload, and physical autoregression accounting.
- Matched FNO Full/Plain pilots, H1 extensions, differential LocalNO pilots,
  expanded-data and paper-budget experiments, geometry/objective/spectral
  diagnostics, and the Stage-AD adapted volumetric DISCO3D run.
- Stage-AF anisotropic spherical DISCO3D implementation checks: dense reference,
  constant field, gradcheck, boundary behavior, phi equivariance, CPU/CUDA
  agreement, branch activity, and production forward/backward feasibility.

These claims reproduce the committed code and artifact contracts. They do not
elevate an adapted proxy to the paper's exact method.

## Adapted

- Geometry is spherical Kerr--Schild with axes `(phi,theta,r)` after regridding.
- The thermal channel is `press`; the paper's exact `eint` conversion was not
  verified.
- Neural inputs use reduced regular tensors (principally 64 cubed), while the
  Athena++ source is an AMR mesh.
- Regridding uses finest-containing-leaf nearest-cell-center sampling and is
  neither conservative nor divergence preserving.
- FNO and differential LocalNO are proxies. Project-local volumetric/spherical
  DISCO3D branches are explicit adaptations.
- Stage-AF geometry is a Euclidean spherical-coordinate tangent proxy, not a
  Kerr--Schild covariant operator, and vector components are not transformed.

## Blocked / unavailable

- Official paper training/evaluation data and complete acquisition provenance.
- Exact Cartesian Kerr--Schild coordinate and vector-component transforms.
- Verified `press` to paper-thermal-variable equivalence.
- Exact paper coarse/fine coupling and closed-loop simulation interface.
- Exact volumetric 3-D DISCO implementation and its paper-scale training setup.

Therefore `EXACT_REPRODUCTION_BLOCKED = true` and the repository scope is
`ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION`.

## Current scientific endpoint

Stage AD's repaired index-space DISCO3D produced only a one-step improvement;
paired confidence intervals crossed zero and transport/rollout dynamics were
not rescued. Stage AE exposed a scalar-support normalization failure at four
corners. Stage AF repaired that geometry with directional local scales and
passed all no-update implementation gates, but did not train a model. Its result
is readiness for a separately authorized controlled experiment, not evidence of
forecast improvement.

The frozen numerical reports and decisions remain under `outputs/` and
`artifacts/`; no scientific metric was rewritten for publication.
