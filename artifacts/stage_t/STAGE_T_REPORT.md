# Stage T — Optimization Convergence and Paper-Budget Audit

| checkpoint | updates | norm L2 | persistence ratio | residual L2 | residual cosine | shell skill | radial skill | first 10x step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | not evaluated | 0.310282 | 1 | 1 | 0 | 0 | 0 | not evaluated |
| small-600 | 600 | 0.333462 | 1.0747 | 1.1668 | 0.319241 | -0.00620851 | -0.038014 | 1 |
| full-600 | 600 | 0.343186 | 1.10604 | 1.22404 | 0.0773009 | -0.0290467 | -0.0584656 | 1 |
| small-1260 | 1260 | 0.29687 | 0.956772 | 1.13881 | 0.737977 | -5.04903 | -0.411599 | not evaluated |
| full-1260 | 1260 | 0.297125 | 0.957594 | 1.15347 | 0.741839 | -3.53789 | -0.374952 | 1 |
| full-75epoch | 3150 | 0.267541 | 0.862251 | 0.900865 | 0.735673 | -1.42945 | -0.335417 | 1 |
| full-150epoch | 6300 | 0.266711 | 0.859576 | 0.891851 | 0.740602 | -1.67947 | -0.348089 | 1 |
| full-300epoch | 12600 | 0.277623 | 0.894742 | 0.919076 | 0.738565 | -1.42804 | -0.242897 | 1 |
| full-600epoch | 25200 | 0.268323 | 0.864772 | 0.91007 | 0.738041 | -2.69949 | -0.381047 | 1 |
| full-1200epoch | 50400 | 0.267753 | 0.862933 | 0.903383 | 0.737221 | -2.85734 | -0.544547 | 1 |

## Frozen contract and executed budget

`DATASET_FROZEN=true`, `SPLIT_FROZEN=true`, `PREPROCESSING_FROZEN=true`, and `MODEL_FROZEN=true`. The expanded HDF5, 0..168/169..211 split, P3 normalizer, eight radial shells, 358,296-parameter LocalNO, Plain L2 residual contract, Adam settings, and shared initial tensor hash were checksum-verified before training.

The long run used 168 microbatches and 42 optimizer updates per epoch. It completed 1,200 epochs / 50,400 updates; warmup was 75 epochs / 3,150 updates, followed by cosine decay to 1e-6. No validation metric was used to tune or stop training.

## Causal control

T-small-1260 scored 0.29687; S-full-1260 scored 0.297125. The full-minus-small benefit is -0.086% under the sign convention `(small-full)/small`. With the predeclared ±2% neutral band, `DATA_EFFECT_AT_1260=NEUTRAL`.

## Checkpoint selection and convergence

Formal best-state-L2: epoch 150 (0.266711). Diagnostic best-residual-L2: epoch 150 (0.891851); best-shell: epoch 2 (-0.00492227); best-radial: epoch 10 (-0.097327). These diagnostic selectors do not replace the formal selector.

`OBJECTIVE_METRIC_DECOUPLING=true` using the frozen consecutive-checkpoint rule. Triggering intervals: [[10, 30], [30, 75], [75, 150], [300, 600], [600, 900]].

## Focus-channel dynamics at formal best checkpoint

| channel | state L2 | residual L2 | residual cosine | shell skill | radial skill | physical L2 | decoder-tail fraction | physical-error contribution |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Bcc2 | 0.446834 | 0.796102 | 0.652295 | -28.8714 | -0.413095 | 1.87523 | 0.402112 | 1.40832e-08 |
| Bcc3 | 0.185483 | 0.602315 | 0.801395 | -1.01444 | -0.080661 | 2.52178 | 0.56807 | 4.8835e-06 |
| vel3 | 0.140716 | 0.995332 | 0.340686 | -0.533571 | -0.48678 | 0.136455 | 0.298688 | 6.98728e-11 |
| rho | 0.155337 | 1.0028 | 0.23103 | -919.157 | -17.4028 | 13.9794 | 0 | 0.000129194 |
| press | 0.143862 | 0.994375 | 0.224655 | -224763 | -194.419 | 2362.23 | 0 | 0.999866 |

## Validation-time distribution shift

At the formal best checkpoint, early/middle/late normalized averages are 0.202917, 0.267389, and 0.313665. Late/early=1.54578; `TEMPORAL_SHIFT_SENSITIVITY=STRONG`. This stratification is reporting-only.

## Direct answers

1. **Q1 — independent data-volume effect?** `DATA_EFFECT_AT_1260=NEUTRAL`; the matched small/full difference is -0.086%.
2. **Q2 — were 30 epochs under-trained?** Yes; formal best is epoch 150.
3. **Q3 — did normalized one-step error improve?** Epoch 30=0.389815, best=0.266711, epoch 1200=0.267753.
4. **Q4 — residual rel-L2 below 1?** Yes; minimum=0.891851 at epoch 150.
5. **Q5 — residual cosine sustained improvement?** Epoch 30=0.705081; epoch 1200=0.737221.
6. **Q6 — shell/radial skill positive?** Best shell=-0.00492227; best radial=-0.097327; epoch-1200 values=-2.85734/-0.544547.
7. **Q7 — did rho/press inverse-tail explosion improve?** Physical average: epoch 30=3.33981e+22, best-state checkpoint=297.777, epoch 1200=1.96208e+12, persistence=0.571584. Frozen catastrophic-tail gates are in each checkpoint JSON.
8. **Q8 — was step-1 range failure delayed?** Epoch-1200 first 10x step=1; formal-best rollout proxy uses epoch 150 and first 10x=1.
9. **Q9 — state/transport decoupling?** `OBJECTIVE_METRIC_DECOUPLING=true`.
10. **Q10 — did paper-like optimization rescue this adapted LocalNO?** OPTIMIZATION_IMPROVES_STATE_NOT_DYNAMICS.
11. **Q11 — enter Stage U?** `AUTHORIZE_STAGE_U=true`.

## Scientific scope

This is a spherical-Kerr-Schild, 64^3, expanded212, P3, radial-shell adapted reproduction with a differential LocalNO proxy—not the paper's exact geometry/operator. A small normalized one-step advantage is not treated as operator success unless residual dynamics, transport, physical tails, and closed-loop stability agree.

`PRIMARY_DECISION = B`
`PRIMARY_DECISION_LABEL = OPTIMIZATION_IMPROVES_STATE_NOT_DYNAMICS`
`SECONDARY_DATA_FINDING = NEUTRAL`
`AUTHORIZE_STAGE_U = true`
