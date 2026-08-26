# Paper-adapted Stage G rollout evaluation

## Protocol

Both best-validation checkpoints were evaluated from validation snapshot 91.
The 19-step ground-truth rollout uses no teacher forcing: each physical
prediction is canonically encoded once, shells are appended, and that tensor
becomes the next input. The 100-step continuation follows the same physical
loop. Ground-truth errors stop at step 19; steps 20--100 report only stability
and morphology diagnostics.

All steps were finite and decoded `rho`/`press` remained positive. Evaluation
clamping is diagnostic-only and does not alter the training objective.

## Ground-truth rollout

The table reports the arithmetic average normalized relative L2.

| Step | Full | Plain | Full model-to-oracle | Plain model-to-oracle | Oracle floor |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.444516 | 0.389972 | 0.644668 | 0.613808 | 0.400338 |
| 5 | 0.884910 | 0.810102 | 0.955976 | 0.961540 | 0.392469 |
| 10 | 1.004437 | 0.886356 | 1.043738 | 1.031965 | 0.390249 |
| 19 | 1.117864 | 0.969457 | 1.113039 | 1.048138 | 0.390640 |

At step 19 Full was above the frozen `Rin` diagnostic threshold but below
`Rout`; Plain was below both. Neither rollout catastrophically diverged under
the Stage G safety definition, but Plain had lower normalized error at every
selected GT horizon.

## No-ground-truth continuation

| Model/step | Finite | rho/press positive | Prediction norm | Above Rin/Rout | Artifact flags |
| --- | --- | --- | ---: | --- | ---: |
| Full 50 | yes | yes | 4510.89 | yes/no | 2 |
| Full 100 | yes | yes | 4798.74 | yes/no | 3 |
| Plain 50 | yes | yes | 3354.49 | no/no | 1 |
| Plain 100 | yes | yes | 3444.47 | no/no | 2 |

Model-only saturation is omitted after ground truth ends because its defining
target mask no longer exists. The saved stepwise outputs include decoded and
normalized ranges, radial profiles, shell and outer-shell quantiles, temporal
autocorrelation and PSD, total variation, high-k energy, prediction norms, and
collapse/ripple/stripe flags.

## Spherical-coordinate morphology

The generated views are:

1. fixed-phi `theta-r`;
2. equatorial `phi-r`.

They are not Cartesian central slices. For GT steps 1, 5, 10, and 19, each
figure compares raw truth, canonical preprocessing oracle, persistence, Full,
and Plain for all eight channels. Steps 50 and 100 compare the two models
without inventing GT errors. Figure annotations identify the preprocessing
floor and target/model clamp semantics.

Machine-readable outputs are in each pilot directory as `gt_rollout.*`,
`no_gt_rollout.*`, `evaluation_summary.json`, and `selected_states.pt`.
Selected-state tensors and PNG figures remain Git ignored.
