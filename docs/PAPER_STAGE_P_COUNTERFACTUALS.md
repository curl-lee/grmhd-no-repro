# Stage P counterfactual attribution

All counterfactuals use the frozen Stage O epoch-23 LocalNO and frozen P3
artifacts. They are diagnostic paths, not deployable rollouts and not improved
models.

## Teacher-forced one-step

All 19 independent validation predictions exceed the frozen Rout condition.
Median fractions outside train min/max are 2.81% for Bcc2, 3.41% for Bcc3, and
1.83% for vel3. Bcc2/Bcc3 maximum decoder sensitivities reach 78.1x/39.1x their
train q0.999 derivatives. Thus the range failure cannot be attributed only to
autoregressive accumulation: substantial normalized overshoot and inverse-tail
sensitivity are already present in one-step predictions.

## Normalized-direct feedback

The 19-step path `z_(t+1)=F(z_t,s)` makes no decoder or encoder calls. Its state
norm grows from `41,245.95` at step 1 to `789,368.38` at step 19, a factor of
`19.138`. The corresponding physical-loop model-output norm at step 19 is
`721,782.81`. Because normalized-direct is slightly worse, P3 roundtrip is not
the sole growth source and provides partial projection in this comparison.
Growth without decode nevertheless supports recurrent normalized-space gain.

## Decode/re-encode only

`H(z)=E(D(z))` is applied at most twice and never calls the model. Train and
validation oracle states are fixed to numerical precision for the target
channels. Teacher-forced target outputs also have only roughly `1e-9`--`3e-8`
relative discrepancies. Selected extreme actual outputs develop large Bcc2/Bcc3
roundtrip discrepancies only near representable physical limits; a second
application does not continue accumulating. This is weak evidence for feedback
distortion, not evidence that roundtrip alone creates the instability.

## Train-envelope projection

Projection bounds are the frozen train q0.001/q0.999 values. At step 19:

| diagnostic path | Bcc2 range | Bcc3 range | vel3 range | interpretation |
| --- | ---: | ---: | ---: | --- |
| actual physical loop | `5.39e36` | float32 maximum | `28.84` | unstable reference |
| all-channel projection | `0.0681` | `0.7087` | `1.2038` | removes target explosion; diagnostic only |
| Bcc2+Bcc3 projection | `0.0681` | `0.7087` | `8.9589` | isolates magnetic tails |
| vel3-only projection | `1.32e4` | `3.06e5` | `1.2038` | reduces later cross-channel range but does not cure all channels |

The median all-channel total-range reduction is 100% under the frozen metric,
supporting train-envelope tail involvement. It does not authorize clipping as a
model fix, and the projected paths can still fail Gate 2/3 or control-channel
range checks.

## Diagnostic teacher reset

At each GT step the named predicted channels are replaced with the same-step P3
target oracle. At step 19, resetting all three target channels reduces their
ranges to `2.09`, `30.84`, and `2.07`, removes target Gate 2/3 counts, and lowers
ripple/stripe counts to `2/0`; other channels can still dominate the total
decoded range. This is teacher intervention, not model performance.

The frozen support rule identifies Bcc2, Bcc3, and vel3 as primary drivers across
at least two selected steps and two failure-metric classes. Bcc3 and vel3 have
clear cross-channel support; vel3 has the broadest metric support (8 primary
classes). There is no single uniquely dominant channel, and the finite-difference
off-diagonal evidence is unavailable because no epsilon pair is consistent.

Machine-readable sources are `teacher_forced.json`,
`normalized_direct_feedback.json`, `roundtrip_only.json`,
`envelope_projection.json`, `channel_reset.json`, and
`channel_driver_matrix.json` under `outputs/paper_reduced100/stage_p/`.
