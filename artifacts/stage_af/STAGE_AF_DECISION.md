# Stage AF Decision

`PRIMARY_DECISION = A` — `ANISOTROPIC_SPHERICAL_DISCO_READY_FOR_TRAINING`

All five readiness gates pass. The full production grid has no zero directional scale or zero normalization; all five bases are active at every target. Dense-reference, constant-field, phi-equivariance, gradcheck, boundary, CPU/CUDA, and repository tests pass. The AF geometry is nontrivially distinct from Stage AD, while all 363,480 trainable tensors at initialization are bitwise identical. The RTX 5070 production forward/backward pass is finite and its DISCO output and gradient are nonzero.

This decision authorizes only a separately approved next-stage controlled training run. Stage AF did not train or update a parameter.

```text
PRIMARY_DECISION = A

REPRODUCTION_SCOPE =
ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION

DISCO_GEOMETRY =
ANISOTROPIC_LOCAL_SPHERICAL_TANGENT_PROXY

KERR_SCHILD_COVARIANT_GEOMETRY = false
VECTOR_COMPONENT_TRANSFORMATION = false

NO_MODEL_TRAINING = true
TRAINING_STARTED = false
TRAINING_COMPLETED = false

DISCO3D_BASIS_COUNT = 5
CANDIDATE_STENCIL_SHAPE = [7,7,7]
DIRECTIONAL_RADIUS_MULTIPLIER = 3

ZERO_DIRECTIONAL_SCALE_COUNT = 0
ZERO_Z_COUNT = 0

ALL_BASES_ACTIVE = true
CONSTANT_FIELD_PASS = true
ANISOTROPIC_DENSE_REFERENCE_MATCH = true
PHI_EQUIVARIANCE_PASS = true
GRADCHECK_PASS = true
ANISOTROPIC_DISCO_UNIT_TESTS_PASS = true

ANISOTROPIC_GEOMETRY_DISTINCT_FROM_STAGE_AD = true

TRAINABLE_PARAMETER_COUNT_MATCH = true
TRAINABLE_INITIALIZATION_HASH_MATCH = true

PRODUCTION_FORWARD_PASS = true
PRODUCTION_BACKWARD_PASS = true
DISCO_BRANCH_FORWARD_ACTIVE = true
DISCO_BRANCH_GRADIENT_ACTIVE = true

ESTIMATED_300_EPOCH_RUNTIME = 16.003981921864007 hours

EXACT_3D_DISCO_IMPLEMENTATION_FOUND = false
EXACT_REPRODUCTION_BLOCKED = true

AUTHORIZE_NEXT_STAGE =
anisotropic_spherical_disco_controlled_training
```
