# Stage Z decision

`PRIMARY_DECISION = E — HIGH_RES_MODEL_RESOURCE_LIMITED`

Z96 passes all four predeclared information-gain conditions relative to Z64,
but WSL currently exposes no CUDA device, so scientifically comparable Z96 and
Z128 model training/evaluation cannot run. No CPU fallback was used. Therefore
Stage Z establishes that 64^3 loses temporal information, but cannot determine
whether that loss is the dominant learned-dynamics bottleneck.

```text
REPRODUCTION_SCOPE = ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION
Z64_REGRID_REGRESSION_PASS = true
HIGHER_RES_DATA_INFORMATION_GAIN = true
SELECTED_ADAPTED_RESOLUTION = UNRESOLVED_PENDING_GPU_MODEL_COMPARISON
EXACT_REPRODUCTION_BLOCKED = true
AUTHORIZE_NEXT_STAGE = resource_feasible_adapted_baseline_suite
```
