# Stage Q operator-response audit

Because train-only selection produced no candidate, the frozen response audit
compares only alpha 0 persistence and alpha 1 Stage O direct output. It covers
eight Stage P states: train early/middle/late, validation snapshot 91, GT
step-1 and step-5 inputs, and actual Stage O rollout step-10 and step-19 inputs.

## Confirmatory validation

| control | arithmetic average | global relative L2 | median q-OOD | Rout failures | Gate 2 severe | Gate 3 pass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| persistence, alpha 0 | 0.180266 | 0.168275 | 0.065805 | 19/19 | 6 | 0 |
| Stage O direct, alpha 1 | 1.047111 | 0.344339 | 0.201265 | 19/19 | 36 | 2 |

Validation confirms that direct output is much worse than persistence in
one-step normalized error and OOD occupancy. It does not rescue a candidate:
validation was read only after the train-only selection artifact was frozen.

## Fixed-state and local-gain probes

The fixed-state audit contains 128 per-channel rows. Persistence has exactly
zero output-state delta. Direct output increases median q-OOD from 0.02275 to
0.12070 over these states; every per-state/channel record is associated with a
frozen Rout failure, while direct has more Gate-2 channel failures (6 versus
4). These range flags are diagnostics, not an output repair, and no clipping is
performed.

The finite-difference audit contains 320 finite rows over ten frozen directions
and epsilons `1e-3` and `1e-2`. Alpha 0 has median directional gain 1.000001;
alpha 1 has median 5.211245, with a maximum of 257.978541. For each epsilon the
affine derivative construction and numerical anchored difference agree, with
maximum relative discrepancy 0.01520 (float32 cancellation is largest for the
identity control). The two epsilon sizes agree within 20% for all 160 alpha-0
rows and for none of the 160 alpha-1 rows. The latter is retained as evidence
of strong finite-step nonlinearity on these extreme states, not reported as a
Jacobian spectral radius.

## Two applications and rollout boundary

The two-application probe contains 256 rows and never exceeds two map calls.
Persistence remains fixed. For direct output, median q-OOD grows from 0.12070
after one application to 0.13488 after two; the median second/first update ratio
is 0.5210. This bounded response is not a production rollout.

No 19-step candidate rollout was run. The required artifact explicitly records
`status: not_run_no_candidate`, zero rows, zero steps, and no fabricated
encode/decode counters.
