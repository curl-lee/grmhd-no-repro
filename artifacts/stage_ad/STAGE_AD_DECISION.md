# Stage AD Decision

`PRIMARY_DECISION = C` — `ADAPTED_3D_DISCO_IMPROVES_ONE_STEP_ONLY`

The implementation gates, CUDA preflight, controlled training, strict reload, branch-activity audit, and 100-step rollout all completed. Aggregate state L2 is nominally 0.993% better, residual L2 is better by only 0.000211481, and cosine improves. The state/residual paired confidence intervals nevertheless cross zero, both aggregate transport skills remain negative and are worse than Stage T at the formal-best checkpoint, and FIRST_10X remains 1. The frozen hierarchy therefore classifies this as a one-step-only improvement, not a dynamics rescue.

The tested adapted isotropic radial 3D DISCO extension does not rescue the spherical-KS workflow. This result does **not** establish that DISCO generally fails and does not test the paper's unavailable exact 3D DISCO implementation.

```text
PRIMARY_DECISION = C

REPRODUCTION_SCOPE =
ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION

DISCO3D_IMPLEMENTATION =
ADAPTED_ISOTROPIC_RADIAL_PIECEWISE_LINEAR

DISCO3D_RADIUS_CELLS = 3
DISCO3D_BASIS_COUNT = 5
DISCO3D_STENCIL_SHAPE = [7,7,7]

ALL_BASES_ACTIVE = true
BASIS_PARTITION_OF_UNITY_PASS = true
DENSE_REFERENCE_MATCH = true
KERNEL_ORIENTATION_PASS = true
GRADCHECK_PASS = true
DISCO3D_UNIT_TESTS_PASS = true

DISCO_BRANCH_ACTIVE = true
TRAINING_STARTED = true
TRAINING_COMPLETED = true

STATE_RETENTION_GATE = PASS
RESIDUAL_GATE = PASS
DIRECTION_GATE = PASS
SHELL_IMPROVEMENT_GATE = FAIL
RADIAL_IMPROVEMENT_GATE = FAIL
ROLLOUT_IMPROVEMENT_GATE = FAIL

EXACT_3D_DISCO_IMPLEMENTATION_FOUND = false
EXACT_REPRODUCTION_BLOCKED = true

AUTHORIZE_NEXT_STAGE = adapted_baseline_suite_with_disco3d
```
