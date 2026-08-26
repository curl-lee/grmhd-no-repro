# Stage M transform-floor isolation

## Validation trace

The trace covers snapshot 91 and every validation target 92--110, all eight
channels, and the real T0--T8 stages. It stores only scalar/shell/profile/spectral
statistics, never intermediate tensors.

Median validation occupancy at the canonical normalized stage is:

| channel | `abs(encoded)>=0.90*gamma` | inverse-clamp occupancy |
| --- | ---: | ---: |
| Bcc2 | 0.238552 | 0.103199 |
| Bcc3 | 0.650146 | 0.602295 |
| vel3 | 0.508011 | 0.423523 |

This identifies the lossy boundary between T3 and T5. T0→T1 nonlinear-only and
T1→T2→T6 normalizer-only round trips retain the tested structure to numerical
precision.

## Diagnostic counterfactuals

Selected-step median retentions are shown as variance / shell-radial / high-k /
dynamic span:

| channel | canonical | no final inverse clamp | no softclip | float64 canonical |
| --- | --- | --- | --- | --- |
| Bcc2 | 0.0296 / 0.0191 / 0.00705 / 0.1318 | ~1 / ~1 / ~1 / ~1 | 1 / 1 / 1 / 1 | same as canonical |
| Bcc3 | 3.71e-6 / 1.51e-6 / 8.43e-7 / 6.89e-4 | undefined due exact ±6 | 1 / 1 / 1 / 1 | same as canonical |
| vel3 | 0.0251 / 0.0196 / 0.00581 / 0.0840 | undefined due exact ±6 | 1 / 1 / 1 / 1 | same as canonical |

For Bcc3/vel3, bypassing the final clamp while retaining the already saturated
float32 tanh output leads to `atanh(±1)` and is explicitly reported nonfinite; no
replacement value is fabricated. Bypassing the entire tanh/atanh softclip path is
finite and recovers all four core metrics. Float64 canonical results reproduce the
same loss, so ordinary float32 error is not an explanation.

Bcc2 does not reach exact ±6 in the audited states: both the minimal no-final-clamp
intervention and the broader no-softclip intervention recover all four metrics.
Under the pre-frozen rule these count as two component-related recoveries. Because
the broader intervention contains the clamp-domain removal, this E label should be
read as an interaction/identifiability result, not as proof of two independent
physical processes.

## Frozen source decisions

- Bcc2: **`E. MULTIPLE_COMPONENTS`**
- Bcc3: **`A. FORWARD_SOFTCLIP_DOMINATED`**
- vel3: **`A. FORWARD_SOFTCLIP_DOMINATED`**

These decisions explain the transform floor but do not replace canonical
preprocessing or alter Stage L's `3. MIXED_OVERALL`.
