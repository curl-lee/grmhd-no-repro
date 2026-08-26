# Stage AA decision

`PRIMARY_DECISION = F — GPU_ENVIRONMENT_UNRESOLVED`

The mandatory GPU scientific gate failed before Z96 model preflight because
the WSL GPU bridge exposes no `/dev/dxg`. No training, optimizer update, CPU
fallback, driver/package change, or scientific-configuration change occurred.

```text
GPU_FAILURE_LAYER = WSL_GPU_BRIDGE
GPU_SCIENTIFIC_GATE = FAIL
HIGHER_RES_DATA_INFORMATION_GAIN = true
Z96_TRAINING_RESOURCE_FEASIBLE = false
SELECTED_ADAPTED_RESOLUTION = UNRESOLVED
AUTHORIZE_Z128_PILOT = false
EXACT_REPRODUCTION_BLOCKED = true
REPRODUCTION_SCOPE = ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION
AUTHORIZE_NEXT_STAGE = resource_recovery
```
