# Stage AA — GPU Recovery and Z96 Controlled Model Completion

`REPRODUCTION_SCOPE = ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION`

| quantity | Z64 | Z96 |
|---|---:|---:|
| increment rel diff | 0.503429 | 0.378347 |
| increment cosine | 0.871549 | 0.926872 |
| persistence L2 | 0.310282 | 0.310069 |
| model state L2 | 0.266711 | not available |
| residual L2 | 0.891851 | not available |
| residual cosine | 0.740602 | not available |
| shell skill | -1.679472 | not available |
| radial skill | -0.348089 | not available |
| first10x | 1 | not available |

## GPU recovery result

The GPU scientific gate did not recover. WSL `nvidia-smi` returns `GPU access
blocked by the operating system`; `/dev/dxg` is absent. CUDA-enabled PyTorch
`2.12.1+cu130` is installed with compiled CUDA
`13.0`, but reports no CUDA device. WSL CUDA/NVML
stub libraries are present. Windows-side `nvidia-smi.exe` could not be queried
because WSL interop itself failed with `UtilBindVsockAnyPort`, so host driver
visibility remains unknown rather than assumed.

This locates the directly observed failure at `GPU_FAILURE_LAYER =
WSL_GPU_BRIDGE`. No package reinstall, driver change, project/model change, or
CPU fallback was attempted.

## Frozen artifact verification

Z64/Z96/Z128 full SHA256 values match the Stage-Z manifests. Z96 remains
`(212,8,96,96,96)` float32 with axis order `(N,C,Nphi,Ntheta,Nr)`, all finite,
and strictly positive rho/press. The Z96 P3 normalizer checksum, train-only fit
indices 0..168, and validation oracle floors exactly reproduce Stage Z.

The Z64 reference was read from the frozen Stage-T epoch-150 metrics and
rollout. Same-resolution persistence was read from the Stage-Z artifact. No
numbers in the comparison table were substituted for missing Z96 model output.

## Direct answers

1. **Failure layer?** Direct evidence identifies the WSL GPU bridge; host driver
   state is independently unknown because Windows interop also fails.
2. **GPU scientific gate recovered?** No, FAIL.
3. **Can Z96 train with the frozen architecture?** Not testable in this WSL
   instance; model preflight was not entered.
4. **Does Z96 beat same-resolution persistence better than Z64?** Unavailable.
5. **Residual L2 improved?** Unavailable.
6. **Residual cosine improved?** Unavailable.
7. **Shell skill improved?** Unavailable.
8. **Radial skill improved?** Unavailable.
9. **Was rollout step-1 failure delayed?** Unavailable.
10. **Do fidelity-sensitive channels gain most?** Association/Spearman metrics
    are not computable without Z96 model results.
11. **Did middle/inner sampling gain become dynamics gain?** Unavailable.
12. **Does the P3 rho/press physical-tail problem remain independent?** The
    preprocessing oracle remains unchanged; model-tail behavior cannot be tested.
13. **Is 64^3 the dominant learned-dynamics bottleneck?** Still unresolved.
14. **Select 64^3 or 96^3?** UNRESOLVED; selection is forbidden before model comparison.
15. **Authorize Z128?** No.
16. **Enter unified benchmark?** No; first restore WSL CUDA and complete Z96.

## Final labels

```text
PRIMARY_DECISION = F
PRIMARY_DECISION_LABEL = GPU_ENVIRONMENT_UNRESOLVED

GPU_FAILURE_LAYER = WSL_GPU_BRIDGE
GPU_SCIENTIFIC_GATE = FAIL

HIGHER_RES_DATA_INFORMATION_GAIN = true
Z96_TRAINING_RESOURCE_FEASIBLE = false

SELECTED_ADAPTED_RESOLUTION = UNRESOLVED
AUTHORIZE_Z128_PILOT = false

EXACT_REPRODUCTION_BLOCKED = true
REPRODUCTION_SCOPE =
ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION

AUTHORIZE_NEXT_STAGE = resource_recovery
```
