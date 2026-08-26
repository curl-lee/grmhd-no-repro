# Stage Y decision

`PRIMARY_DECISION = F — EXACT_REPRODUCTION_BLOCKED_BY_MISSING_ASSETS`

The official target-paper data and training code were not found.  The exact
volumetric 3D DISCO implementation/specification is missing; the paper's
coarse/fine CT interface is only partial; and current raw ATHDF cannot be
scientifically transformed because its run-specific spin, exact source/vector
contract, EOS/Gamma, and paper-compatible thermal mapping are not provenance
complete.  Raw AMR spatial information itself is sufficient, but that cannot
repair semantic gaps.

Do not continue tuning or relabel the existing differential/FNO proxies as the
paper method.  Do not generate Cartesian/eint data, invent 3D DISCO, or rerun a
simulation from guessed settings.  Exact reproduction can reopen only when an
authoritative paper dataset/code release or matching run provenance and
operator/coupling contracts are acquired.

```text
PRIMARY_DECISION = F

BH_SPIN_PROVENANCE = INCOMPLETE
COORDINATE_MAPPING_PROVENANCE = INCOMPLETE
VECTOR_TRANSFORM_PROVENANCE = INCOMPLETE
CARTESIAN_VECTOR_CONVERSION_AUTHORIZED = false

EOS_PROVENANCE = INCOMPLETE
PRESS_TO_EINT_CONVERSION_AUTHORIZED = false

OFFICIAL_PAPER_DATA_FOUND = false
OFFICIAL_PAPER_CODE_FOUND = false
EXACT_3D_DISCO_IMPLEMENTATION_FOUND = false
PAPER_COARSE_FINE_CONTRACT = PARTIAL

CURRENT_RAW_DATA_SALVAGEABLE = false

R1_DATA = FAIL
R2_OPERATOR = FAIL
R3_PREPROCESSING = FAIL
R4_TRAINING = FAIL
R5_COUPLING = FAIL

EXACT_REPRODUCTION_BLOCKED = true

AUTHORIZE_NEXT_STAGE = adapted_workflow_reproduction_only
```
