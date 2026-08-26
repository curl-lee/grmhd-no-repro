# Stage AB — WSL GPU Bridge Recovery and Z96 Controlled Experiment Resume

`REPRODUCTION_SCOPE = ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION`

| quantity | Z64 | Z96 |
|---|---:|---:|
| increment rel diff | 0.503429 | 0.378347 |
| increment cosine | 0.871549 | 0.926872 |
| persistence L2 | 0.310282 | 0.310069 |
| model state L2 | 0.266711 | 0.283215 |
| residual L2 | 0.891851 | 0.983736 |
| residual cosine | 0.740602 | 0.724798 |
| shell skill | -1.679472 | -105.108833 |
| radial skill | -0.348089 | -1.698125 |
| first10x | 1 | 1 |

## GPU resource chain

The real WSL instance passes interop, exposes `/dev/dxg`, and reports an RTX 5070 through
both `nvidia-smi` and PyTorch CUDA. The earlier Stage-AA failure was observed in a restricted
device sandbox; Stage AB separated that sandbox-scoped false negative from the underlying
host-to-WSL resource chain. No driver, CUDA package, PyTorch package, or scientific setting
was changed.

## Controlled training

Z96 completed 150 epochs, 25200 microbatches,
and 6300 updates in 5746.43 seconds. All
values remained finite. The formal best checkpoint is epoch 75; best and epoch-150 checkpoints
strictly reload, reproduce their saved validation metrics, and produce deterministic probes.

## Scientific comparison

Z96 beats its same-resolution persistence baseline (ratio 0.913395),
but it is worse than the frozen Z64 model on the formal state L2 (0.283215
vs 0.266711), residual L2 (0.983736 vs
0.891851), residual cosine (0.724798 vs
0.740602), shell skill, and radial skill. Both rollouts hit the 10x
physical-range landmark at step 1. Z96 remains finite and rho/press-positive through step 100,
but the P3 decoded physical tail remains catastrophic and scientifically separate from the
normalized metrics.

The channel-level Spearman associations are {'fidelity_vs_residual_gain_spearman': 0.07142857142857144, 'fidelity_vs_shell_skill_change_spearman': -0.38095238095238104, 'fidelity_vs_radial_skill_change_spearman': -0.5476190476190477}. These are rank associations only,
not causal evidence. The regional table shows that the known inner/middle sampling improvement
does not translate into a consistent model-dynamics gain.

## Answers to the Stage-AA/AB questions

1. No unresolved host, WSL, or PyTorch fault remains; the prior negative evidence was sandbox-scoped.
2. GPU scientific gate: PASS.
3. The identical 358,296-parameter architecture trains at Z96 without resource modification.
4. Z96 exceeds Z96 persistence, but less strongly than Z64 exceeds Z64 persistence.
5. Residual L2 does not improve relative to Z64.
6. Residual cosine does not improve relative to Z64.
7. Shell skill is substantially worse.
8. Radial skill is worse.
9. The step-1 physical failure is not delayed.
10. Fidelity-sensitive channels do not show a consistent matching learned-dynamics gain.
11. Inner/middle sampling gain does not consistently transfer to regional dynamics gain.
12. The P3 rho/press inverse-tail amplification remains present.
13. The 64-cubed sampling loss is real but is not the dominant learned-dynamics bottleneck.
14. Selected adapted resolution: 64.
15. Z128 training is not authorized.
16. The unified adapted baseline suite is authorized at the selected 64 resolution.

## Decision

`PRIMARY_DECISION = D` (`Z96_MODEL_WORSE`). The extra sampling information did not rescue the
adapted LocalNO dynamics and the degradation is not attributable to OOM, nonfinite training,
checkpoint failure, or a changed scientific contract.
