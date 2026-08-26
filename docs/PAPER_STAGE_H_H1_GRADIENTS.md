# Stage H unweighted-gradient and clipping audit

## Method

For each frozen state and fixed pair, one forward graph was used to compute
prediction-space and parameter-space gradients for:

- weighted paper base fidelity;
- raw unweighted H1;
- paper-weighted H1 (`0.05 * raw`);
- all other Full priors combined;
- total Full loss;
- Plain unit-weight L2 as an additional reference.

`torch.autograd.grad` was used. Parameter `.grad` remained `None`; no optimizer
or scheduler was constructed and no checkpoint was written. Component
gradients reconstructed total with maximum relative residual `4.02e-4`, within
the recorded float32 reduction-order tolerance.

## Global result

Across all five states and ten pairs:

| Quantity | Mean |
| --- | ---: |
| Prediction gradient: base | 0.01369 |
| Prediction gradient: raw H1 | 16.4449 |
| Prediction gradient: weighted H1 | 0.82224 |
| Prediction gradient: other priors | 0.000381 |
| Prediction gradient: total Full | 0.82878 |
| Parameter weighted-H1/base norm ratio | 12.6434 |
| Parameter weighted-H1/total norm fraction | 0.94281 |
| Base/weighted-H1 cosine | 0.43912 |
| Base/weighted-H1 sign-conflict weight | 0.13336 |
| Total norm before clip | 392.237 |
| Norm-1 clip scale | 0.006080 |

Raw H1 is exactly twenty times weighted H1. The 0.05 coefficient is unchanged.

## Trained-state detail

| State/split | base | raw H1 | weighted H1 | other | total | H1/base | cosine | conflict | clip scale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full best train | 29.80 | 3870.55 | 193.53 | 1.91 | 207.10 | 7.79 | 0.354 | 0.201 | 0.00591 |
| Full best validation | 83.64 | 12299.98 | 615.00 | 2.03 | 662.47 | 7.71 | 0.503 | 0.108 | 0.00165 |
| Plain best under Full loss, train | 10.02 | 5179.11 | 258.95 | 1.06 | 263.62 | 24.37 | 0.427 | 0.188 | 0.00507 |
| Plain best under Full loss, validation | 65.72 | 14452.74 | 722.64 | 1.94 | 762.80 | 14.84 | 0.549 | 0.095 | 0.00144 |

The Plain rows do not mean that Plain was trained with H1. They evaluate the
frozen Plain weights under the same post-hoc Full components to make gradient
geometry comparable.

H1 dominates lifting, all four FNO blocks, spectral convolution, skip paths,
channel MLPs, and projection. Projection and lifting have the largest absolute
weighted-H1 module norms.

## Clipping projection

Global clipping preserves total-gradient direction and multiplies every
component projection by the same scale. The average scale `0.00608` means the
base projection is also reduced by roughly two orders of magnitude when the
combined Full norm is clipped to one. On the trained validation states the
scale is about `0.0014--0.0017`.

The weighted H1 controls total norm, but it is not strictly opposed to base:
the mean cosine is positive. It is only moderately aligned, with meaningful
elementwise sign conflict. The evidence supports direction competition and
severe common compression, not a claim of globally anti-parallel gradients.
