# Paper preprocessing clamp-loss audit

## Decision

The canonical reduced reproduction remains fixed at `gamma=6` and an inverse
clamp of `0.99*gamma`. This audit did not modify the normalizer, choose gamma
from validation, train a model, or write a checkpoint.

The audit nevertheless finds **material, spatially structured information
loss** in the canonical preprocessing oracle
`physical -> encode -> paper-clamped decode`. The all-channel physical relative
L2 of this oracle is 46.63% on train and 52.64% on validation. Therefore later
physical-space Bcc3/vel3 errors and morphology cannot be attributed to a model
without also reporting this preprocessing-only oracle.

The fixed context is:

- protocol: `paper_reduced100_press_spherical_ks`;
- train snapshots: 11..90; validation snapshots: 91..110;
- HDF5 SHA-256:
  `cd3966bed1f617f49d775151cff87ae92eb2432a560bb82ae9355605e7aea70a`;
- thermal channel: `press`, with `paper_adaptation: true`;
- coordinate components: spherical Kerr--Schild stored components;
- fit statistics: train only; validation was not used for normalizer fit or
  gamma selection.

Machine-readable details are in:

- `outputs/paper_reduced100/preprocessing_loss_audit.json`;
- `outputs/paper_reduced100/preprocessing_loss_audit.csv`;
- `outputs/paper_reduced100/preprocessing_loss_audit.md`;
- `outputs/paper_reduced100/preprocessing_loss_figures/`.

## Audit definitions

For each channel and split the script records physical values, transformed
`x_hat`, robust z, and float32 soft-clipped `z_tilde`, with min/max, mean/std,
and q0.001/q0.01/q0.5/q0.99/q0.999. A canonical clamp hit is

```text
z_tilde >  0.99*6    positive hit
z_tilde < -0.99*6    negative hit
```

The round-trip metrics compare the input physical value with the result of the
unchanged paper decoder. Metrics are reported for all, clamped, and unclamped
voxels separately. The all-channel physical relative L2 is

```text
sqrt(sum_channel,voxel((decoded - physical)^2))
------------------------------------------------
sqrt(sum_channel,voxel(physical^2))
```

The per-channel “contribution” is that channel's squared error on clamped
voxels divided by total squared round-trip error across all channels. This is a
decomposition of the relative-L2 numerator, not a channel weight.

Spatial definitions are fixed before examining validation:

- 8 log-spaced shells in physical spherical r;
- equatorial band: `abs(theta-pi/2) <= pi/8`;
- polar caps: `theta <= pi/4 or theta >= 3pi/4`;
- inner region: first two spherical-r shells;
- fixed-phi map: phi index 0, `phi=0.0490874`;
- equatorial map: the stored theta cell nearest pi/2, `theta=1.54625`.

Region enrichment is the share of all clamp hits inside a region divided by
that region's share of spatial voxels. One means uniform, greater than one
means concentration, and less than one means depletion.

## Canonical clamp and round-trip loss

| split | channel | + clamp | - clamp | total clamp | physical round-trip relative L2 (%) |
| --- | --- | ---: | ---: | ---: | ---: |
| train | Bcc1 | 0 | 0 | 0 | 1.35e-5% |
| train | Bcc2 | 0.046997% | 0.062561% | 0.109558% | 7.857% |
| train | Bcc3 | 12.0101% | 11.8619% | 23.8719% | 98.229% |
| train | rho | 0 | 0 | 0 | 2.13e-5% |
| train | press | 0 | 0 | 0 | 1.42e-5% |
| train | vel1 | 0 | 0 | 0 | 1.67e-6% |
| train | vel2 | 1.9338% | 1.8103% | 3.74411% | 40.953% |
| train | vel3 | 22.3144% | 0 | 22.3144% | 81.973% |
| validation | Bcc1 | 0 | 0 | 0 | 2.86e-5% |
| validation | Bcc2 | 6.77874% | 3.66993% | 10.4487% | 91.747% |
| validation | Bcc3 | 28.3865% | 28.1023% | 56.4888% | 99.935% |
| validation | rho | 0 | 0 | 0 | 2.84e-5% |
| validation | press | 0 | 0 | 0 | 3.10e-5% |
| validation | vel1 | 0.010681% | 0 | 0.010681% | 0.228% |
| validation | vel2 | 2.00624% | 1.9256% | 3.93185% | 58.178% |
| validation | vel3 | 39.4723% | 0.001507% | 39.4738% | 85.145% |

Aggregate physical relative L2 is 46.6285% on train and 52.6375% on
validation. In both splits, more than 99.9999999999% of squared round-trip
error comes from voxels that cross the canonical inverse-clamp threshold; the
unclamped path is numerically close to an identity round-trip.

The squared-error contribution is sharply concentrated:

| split | Bcc2 | Bcc3 | vel2 | vel3 |
| --- | ---: | ---: | ---: | ---: |
| train | 0.000691% | 15.4877% | 0.092689% | 84.4189% |
| validation | 0.358839% | 98.3241% | 0.001560% | 1.31545% |

The validation distribution is later in time and is not interchangeable with
the train distribution: Bcc3 clamp frequency increases by 32.62 percentage
points (2.37x), while vel3 increases by 17.16 points (1.77x). Bcc2 also rises
from 0.11% to 10.45%. These are observed distribution-shift diagnostics, not
grounds for retuning gamma on validation.

## Bcc3 and vel3 location

The clamp is not concentrated in the disk midplane. Equatorial enrichment is
below one for both focus channels and splits. Polar enrichment is only modest,
whereas the inner-radius enrichment is large.

| channel/split | equatorial enrichment | polar enrichment | inner-two-shell enrichment | top-5 snapshot hit share |
| --- | ---: | ---: | ---: | ---: |
| Bcc3 train | 0.829 | 1.112 | 3.641 | 9.07% |
| Bcc3 validation | 0.868 | 1.097 | 1.766 | 27.97% |
| vel3 train | 0.815 | 1.120 | 3.727 | 8.10% |
| vel3 validation | 0.706 | 1.243 | 2.427 | 27.25% |

For train, the first two shells contain 91.0% of Bcc3 and 93.2% of vel3 clamp
hits despite containing 25% of spatial voxels. Half the hits require 34/80
Bcc3 snapshots and 35/80 vel3 snapshots; the effect is persistent rather than
being caused by a few snapshots. Validation likewise needs 10/20 snapshots for
half the hits, close to a broad temporal distribution. The validation maps do,
however, show rising Bcc3 clamp rates and a vel3 plateau near 40% over the late
window.

Fixed-phi and equatorial maps confirm that the strongest loss follows the
small-r region over broad angular support. The equatorial phi-r maps are nearly
phi-independent at small r, which argues against a localized azimuthal patch
as the cause. Bcc3 has approximately symmetric positive/negative clipping;
vel3 clipping is almost entirely on the positive side.

## Float32 versus the paper clamp

The diagnostic reference performs float64 `tanh` and then float64 `atanh`
without the 0.99-gamma clamp. It is diagnostic only: it is not used for
training, paper evaluation, or checkpoint reconstruction.

| split/channel | float32 exactly at abs(gamma) | float64 exactly at abs(gamma) | float64 diagnostic nonfinite | diagnostic finite round-trip relative L2 | canonical vs diagnostic relative L2 on finite subset |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bcc3 train | 12.6755% | 2.59338% | 2.59338% | 1.41749% | 94.8018% |
| Bcc3 validation | 42.3983% | 25.7661% | 25.7661% | 1.42751% | 94.9310% |
| vel3 train | 10.2062% | 3.00712% | 3.00712% | 0.255121% | 74.5872% |
| vel3 validation | 19.3017% | 7.53180% | 7.53180% | 0.256624% | 74.5728% |

Float32 therefore makes exact saturation occur earlier, but it is not the sole
cause. Even the float64 `tanh` becomes exactly +/-1 in the extreme tail, where
unclamped `atanh` is nonfinite. On the remaining finite subset, removing the
paper clamp nearly restores the physical values, while canonical versus
diagnostic differences remain large. The 0.99-gamma inverse clamp and the
heavy robust-z tail are the dominant scientific issue; this is not a dtype-only
bug.

## Gamma sensitivity is diagnostic only

| split | gamma=6 canonical | gamma=8 diagnostic | gamma=12 diagnostic | no soft clip diagnostic |
| --- | ---: | ---: | ---: | ---: |
| train | 46.6285% | 44.1281% | 39.5104% | 2.02e-14% |
| validation | 52.6375% | 52.5525% | 52.3982% | 7.88e-14% |

These are all-channel physical round-trip relative L2 values. Higher gamma
reduces train loss, but has little effect on the late validation tail. The
no-soft-clip diagnostic is essentially invertible and shows that the signed-log,
positive-log, and linear channel transforms themselves are not responsible for
the observed loss. Gamma 8/12 and no-clip remain `diagnostic extension, not
paper reproduction`; no validation-based gamma choice is made.

## Consequences for the next phase

1. Normalized-domain Table-2-style metrics remain definable, but physical
   Bcc3/vel3 morphology must always be accompanied by the preprocessing oracle.
2. Future Full and Plain evaluations must report target saturation and model
   output saturation separately. Otherwise target aliasing can be mistaken for
   model collapse or stability.
3. Decoded velocity ROI based on vel3 is exposed to target information loss.
   Its implementation must retain the canonical path for traceability and
   report ROI overlap with clamp masks.
4. Physical plots must not be described as lossless reconstructions. The
   canonical paper-clamped view is an adapted diagnostic with a quantified
   preprocessing floor.
5. Gamma and inverse clamp remain unchanged. Any alternative preprocessing
   would be a separately named diagnostic extension and cannot replace the
   paper-adapted main result.
6. No loss, smoke, or training run is authorized by this document. The next
   implementation stage should carry these oracle and clamp metrics into every
   acceptance gate.
