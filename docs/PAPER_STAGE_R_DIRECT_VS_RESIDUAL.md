# Stage O direct state versus Stage R residual target

Stage O and Stage R share P3 statistics, architecture, 358,296-parameter tensor
layout, initial tensor state, 30 epoch pair permutations, split, optimizer,
scheduler, batch/accumulation, learning-rate schedule, and checkpoint selector.
The one training factor is direct-state versus normalized-residual output/target.

| one-step validation | Stage O direct | Stage R residual | P3 persistence |
| --- | ---: | ---: | ---: |
| normalized arithmetic average | 1.047111 | 0.224673 | 0.180266 |
| normalized global | 0.344339 | 0.169252 | 0.168275 |
| model-to-own-oracle average | 1.963e20 | 522.821 | 0.210415 |

Stage R reduces the Stage O normalized average to `0.214565x` and sharply lowers
step-1 overshoot, but it is still `1.246344x` P3 persistence. All eight Stage R
per-channel normalized errors are worse than persistence. Its validation mean
residual norm ratio is 0.3971, cosine 0.1319, sign agreement 0.5242, and residual
relative-L2 average 1.4138 versus the zero-residual baseline 1.0. The gain is
therefore largely persistence-like conservatism, not demonstrated residual
skill.

Both models remain above frozen Rout on every validation pair. Stage R delays
and reduces normalized growth—step-19 average 2.3208 versus Stage O 28.0025 and
step-100 normalized state norm 1.878e6 versus 3.342e8—but does not prevent P3
inverse-tail exposure or physical range explosion.

`stage_m_v1` Gate 2 improves for Bcc2: Stage O failed Bcc2/Bcc3 whereas Stage R
fails Bcc3 only among the three P3 target channels. Gate 3 still fails all three:

| channel | shell skill | radial skill | Gate 3 |
| --- | ---: | ---: | --- |
| Bcc2 | -19.7482 | -13.7216 | fail |
| Bcc3 | -1.88267 | -6.46011 | fail |
| vel3 | -7.34345 | -7.91369 | fail |

The legacy detector reports widespread ripple/stripe behavior in both long
rollouts. `stage_m_v1` is more discriminating: Gate 0 exposes range/Rout failure,
Gate 1 qualifies the P3 floor, Gate 2 shows partial structural improvement, and
Gate 3 shows absent dynamical shell/radial skill. No unique causal operator
defect is established.

Frozen Stage O evaluation used its authorized rho/press evaluation clamp,
whereas Stage R explicitly forbids physical output repair. Normalized and
structural quantities are directly paired; repaired physical ranges are not
claimed as exact like-for-like values.
