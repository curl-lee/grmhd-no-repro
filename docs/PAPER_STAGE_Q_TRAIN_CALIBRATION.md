# Stage Q train-only alpha calibration

Alpha calibration uses only the 79 teacher-forced P3 transitions 11->12
through 89->90. The readiness predicates and tie-breaking rule were frozen in
`configs/paper_reduced100/stage_q_residual_anchor_audit.yaml` before the sweep.
Validation snapshots 91--110 were not read for selection; the machine artifact
records `validation_indices_read_before_selection: []` and
`validation_used: false`.

| alpha | normalized average | global L2 | median q001/q999 OOD | Rout failures | residual/true residual | positive cosine channels | Gate 2 severe | Gate 3 pass | shell skill | radial skill |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.132913 | 0.064256 | 0 | 79/79 | 0 | 0/8 | 4 | 10 | -0.00000047 | -0.00000045 |
| 0.125 | 0.143180 | 0.062686 | 0.000732 | 79/79 | 0.220406 | 6/8 | 7 | 22 | -0.012957 | -0.289840 |
| 0.25 | 0.172337 | 0.062346 | 0.001953 | 79/79 | 0.440812 | 6/8 | 7 | 19 | -0.054252 | -0.806098 |
| 0.5 | 0.251643 | 0.065362 | 0.002880 | 79/79 | 0.881625 | 6/8 | 8 | 17 | -0.124025 | -1.715063 |
| 1 | 0.443102 | 0.083207 | 0.009663 | 79/79 | 1.763249 | 6/8 | 11 | 13 | -0.586649 | -3.862942 |

All middle alphas pass engineering, nontrivial-dynamics, and control-channel
groups A, D, and F. None passes all readiness predicates:

- Alpha 0.125 reduces median q-OOD by 92.42% relative to direct, but Rout by
  0%, has candidate/persistence average ratio 1.0772, reduces Gate 2 by only
  36.36%, and is worse than persistence in both transport summaries.
- Alpha 0.25 reduces q-OOD by 79.79%, but has the same zero Rout reduction,
  ratio 1.2966, 36.36% Gate-2 reduction, and worse transport.
- Alpha 0.5 reduces q-OOD by only 70.19%, has zero Rout reduction, ratio
  1.8933, 27.27% Gate-2 reduction, and worse transport.

The frozen Rout diagnostic flags every train prediction, including alpha 0.
This is important: shrinking the model residual reduces normalized overshoot,
but cannot meet the specified Rout-count reduction when the persistence
control already fails that frozen gate. No threshold, alpha grid, or transform
was changed in response.

Therefore `candidate_alpha` is `none`; the selection artifact SHA256 is
`c4b2cd506e9589a26655221aa949c8b5b87bd2d1a2bff775af7bffa4efb118f7`.
The subsequent validation is confirmatory and controls-only, never a tuning
pass.
