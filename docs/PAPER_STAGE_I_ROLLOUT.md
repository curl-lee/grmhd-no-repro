# Stage I Run A and Run B rollout, morphology, and boundary record

## Autoregression contract

Evaluation strictly reloaded the epoch-25 best checkpoint. Starting from real
snapshot 91, every step produced a normalized model output, decoded one
physical state, and re-encoded that physical state once for the next input.
There was no teacher forcing, normalized-tensor feedback, double encode, or
double decode. The 100-step counter audit observed exactly 100 input encodes
and 100 prediction decodes.

## Ground-truth horizon

Values below are arithmetic-average/global relative L2. Artifact flags are
frozen Stage G flags and are diagnostic, not automatic failures.

| step | normalized | model-oracle | model-raw | oracle floor | prediction norm | flags |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.39070 / 0.34204 | 0.61666 / 0.84333 | 0.82759 / 0.87605 | 0.40034 / 0.50439 | 3673.31 | 3 |
| 5 | 0.81781 / 0.88969 | 1.00052 / 0.99354 | 0.95553 / 0.99421 | 0.39247 / 0.39864 | 3534.88 | 6 |
| 10 | 0.90176 / 0.96621 | 1.06825 / 0.99104 | 0.99841 / 0.99329 | 0.39025 / 0.53448 | 3542.87 | 10 |
| 19 | 0.98044 / 1.03319 | 1.09897 / 0.99258 | 1.02665 / 0.99483 | 0.39064 / 0.57285 | 3465.29 | 9 |

All 19 GT steps were finite and rho/press positive. Mean model-only saturation
at steps 1/5/10/19 was 0.00825/0.02318/0.02154/0.01791. Relative to Stage G
Full, normalized average error improved at every selected step:
0.3907 versus 0.4445, 0.8178 versus 0.8849, 0.9018 versus 1.0044, and 0.9804
versus 1.1179.

## No-ground-truth horizon

No GT error was computed after step 19.

| step | finite | rho/press positive | prediction norm | above Rin/Rout | eval bound clamp | flags |
|---:|---|---|---:|---|---:|---:|
| 50 | yes | yes | 3238.57 | false / false | 0.000244 | 3 |
| 100 | yes | yes | 3573.65 | false / false | 0 | 1 |

Stage G Full norms were 4510.89/4798.74 at steps 50/100, so no-H1 was lower at
both horizons. Plain remained lower at step 100 (3444.47). Step 50 no-H1 flags
were Bcc3/rho collapse proxies and vel2 stripe anisotropy; step 100 retained
only the Bcc3 collapse proxy. Full had five combined step-50/100 flags, no-H1
four, and Plain three.

The full JSON retains normalized/physical ranges, evaluation clamp, radial
profile drift, shell means/std/quantiles, outermost two shells, temporal
autocorrelation and PSD, total variation, high-k energy, and all
collapse/ripple/stripe flags for the selected steps.

## Frozen morphology and boundary scores

Lower is better. These are the unchanged Stage G scoring functions.

| category | Stage G Full | Stage G Plain | Stage I no-H1 | best |
|---|---:|---:|---:|---|
| center morphology | 0.91196 | 0.90498 | 0.93864 | Plain |
| polar morphology | 0.88026 | 0.83574 | 0.85606 | Plain |
| magnetic texture | 2.41274 | 0.89983 | 0.85202 | no-H1 |
| radial statistics | 6.56373 | 4.84509 | 5.06848 | Plain |
| outer-shell statistics | 7.18638 | 1.20600 | 1.85820 | Plain |
| model-only saturation | 0.01853 | 0.01071 | 0.01772 | Plain |
| step-50/100 artifact count | 5 | 3 | 4 | Plain |

No-H1 improved six of seven categories relative to Full; center morphology was
the exception. Plain was best in five categories, while no-H1 was best in
magnetic texture.

The rendered views are spherical-coordinate adaptations:
fixed-phi theta-r and equatorial phi-r. They compare raw truth, canonical
preprocessing oracle, persistence, Stage G Full, Stage G Plain, and Stage I
no-H1 at GT steps 1/5/10/19, and omit unavailable raw/oracle panels at steps
50/100. They are not Cartesian central slices. Physical Bcc3/vel3 comparisons
remain constrained by the preprocessing oracle floor.

## Run B unit-index ground-truth horizon

Run B used the same physical autoregression contract. All 19 GT steps were
finite and rho/press positive.

| step | normalized | model-oracle | model-raw | oracle floor | prediction norm | flags |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.38947 / 0.34138 | 0.61521 / 0.84393 | 0.82628 / 0.87649 | 0.40034 / 0.50439 | 3686.47 | 3 |
| 5 | 0.81737 / 0.89052 | 1.00175 / 0.99348 | 0.95305 / 0.99417 | 0.39247 / 0.39864 | 3547.69 | 6 |
| 10 | 0.90006 / 0.96569 | 1.06950 / 0.99094 | 0.99583 / 0.99322 | 0.39025 / 0.53448 | 3558.45 | 10 |
| 19 | 0.97819 / 1.03410 | 1.10578 / 0.99241 | 1.02625 / 0.99472 | 0.39064 / 0.57285 | 3496.95 | 10 |

Mean model-only saturation was 0.00900/0.02411/0.02179/0.01887 at steps
1/5/10/19. Run B improved normalized arithmetic-average error over both Full
and no-H1 at every selected GT step. It improved over Plain only at step 1.

## Run B unit-index no-ground-truth horizon

No GT error was computed after step 19. The transform counter recorded exactly
100 input encodes and 100 prediction decodes.

| step | finite | rho/press positive | prediction norm | above Rin/Rout | eval bound clamp | flags |
|---:|---|---|---:|---|---:|---:|
| 50 | yes | yes | 3301.72 | false / false | 0.000610 | 3 |
| 100 | yes | yes | 3590.06 | false / false | 0 | 1 |

Step 50 retained Bcc3/rho collapse proxies and vel2 stripe anisotropy; step
100 retained only the Bcc3 collapse proxy. Run B norms were below Full at both
horizons, slightly above no-H1 at both horizons, and below Plain only at step
50.

## Four-model frozen morphology and boundary scores

Lower is better. The implementation and thresholds are the unchanged Stage G
scoring functions.

| category | Full | Plain | no-H1 | unit-index | best |
|---|---:|---:|---:|---:|---|
| center morphology | 0.91196 | 0.90498 | 0.93864 | 0.93875 | Plain |
| polar morphology | 0.88026 | 0.83574 | 0.85606 | 0.85996 | Plain |
| magnetic texture | 2.41274 | 0.89983 | 0.85202 | 0.86006 | no-H1 |
| radial statistics | 6.56373 | 4.84509 | 5.06848 | 5.26359 | Plain |
| outer-shell statistics | 7.18638 | 1.20600 | 1.85820 | 1.71580 | Plain |
| model-only saturation | 0.01853 | 0.01071 | 0.01772 | 0.01845 | Plain |
| step-50/100 artifact count | 5 | 3 | 4 | 4 | Plain |

Unit-index improved six categories relative to Full, with center morphology
the exception. Compared with no-H1 it improved only outer-shell statistics;
compared with Plain it improved only magnetic texture. Rendered comparisons
reuse fixed-phi theta-r and equatorial phi-r views and are explicitly
spherical-coordinate adaptations, not Cartesian central slices.

## Run C stored-coordinate rollout

All 19 GT steps and all 100 total steps were finite with positive rho/press.

| step | normalized average | model-oracle | model-raw | oracle floor | norm | flags |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.36597 | 0.60106 | 0.81420 | 0.40034 | 3680.11 | 3 |
| 5 | 0.81080 | 0.99091 | 0.95645 | 0.39247 | 3610.49 | 6 |
| 10 | 0.89666 | 1.04293 | 0.99900 | 0.39025 | 3672.75 | 7 |
| 19 | 0.98735 | 1.07738 | 1.03305 | 0.39064 | 3789.46 | 7 |
| 50 | no GT | no GT | no GT | no GT | 4532.69 | 2 |
| 100 | no GT | no GT | no GT | no GT | 4515.47 | 1 |

Run C improves every selected GT error relative to Full and improves four
aggregate categories relative to unit-index. Against Plain it improves only
step-1 normalized error and the step-100 artifact count, and no aggregate
morphology/boundary category. Final decision is therefore D, not an H1
replacement claim.
