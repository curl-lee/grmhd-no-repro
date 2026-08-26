# Stage U Decision

`PRIMARY_DECISION = D`

`D = SPECTRAL_AND_DIFFERENTIAL_GEOMETRY_BOTH_MISMATCH`

The no-training audits find strong differential, boundary, and spectral index-space mismatches. Spectral-only is worse than Stage T on aggregate transport and fails state/residual/rollout gates. Coordinate-aware FD does not rescue transport, and the naive spherical proxy is numerically pathological near the poles. Therefore repairing only the differential path is insufficient.

```text
PRIMARY_DECISION = D
DIFFERENTIAL_BRANCH_SUSPECTED = false
SPECTRAL_BRANCH_SUSPECTED = true
BOUNDARY_MISMATCH = THETA_AND_R
INDEX_FD_GEOMETRY_ERROR = STRONG
AUTHORIZE_NEXT_STAGE = loss_objective_audit_with_distribution_shift_reporting
CARTESIAN_REMAP_NOT_AUTHORIZED
```
