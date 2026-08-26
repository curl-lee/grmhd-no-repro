# Stage O P3 differential LocalNO rollout

Both evaluations start from physical snapshot 91. Each prediction is decoded
once, the physical prediction is used as the next state, and that state is
encoded once for the next model input. There is no teacher forcing, direct
normalized-tensor loop, double encode, or double decode.

## Ground-truth rollout

All 19 transitions executed with finite tensors and positive rho/press. That
engineering result is not physical stability: error, clamp occupancy, and
decoded ranges grow persistently.

| step | P3 normalized average | prediction norm | eval clamp | artifact count |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.89574 | 41,245.9 | 0.19562 | 5 |
| 3 | 0.99298 | 41,356.7 | 0.20141 | 8 |
| 5 | 1.27020 | 44,585.1 | 0.28165 | 11 |
| 10 | 3.31425 | 87,119.5 | 0.64317 | 11 |
| 19 | 28.00253 | 721,888.6 | 0.95829 | 10 |

By step 10 Bcc2 spans about `[-135, 66]` and Bcc3 about
`[-8.09e3, 2.44e3]`. At step 19 Bcc3 reaches the float32 finite extrema and
Bcc2 reaches approximately `[-5.39e36, 6.81e34]`. Shell/radial comparison to
the P3 target and P3 persistence is therefore evaluated but fails to show
stable transport skill.

## No-ground-truth rollout

Steps after 19 report no GT error and do not invent a future target oracle or
Gate 3 target transport. The 100-step trajectory remains finite with positive
rho/press, but the decoded-range failure persists.

| step | prediction norm | eval clamp | artifact count | selected range evidence |
| ---: | ---: | ---: | ---: | --- |
| 25 | 2,363,819 | 0.98950 | 8 | Bcc2/Bcc3 at float32 extrema |
| 50 | 40,274,984 | 0.99930 | 7 | vel3 about `[-64.7, 2.20e3]` |
| 75 | 271,780,928 | 0.99987 | 7 | vel3 about `[-453, 1.94e4]` |
| 100 | 334,174,464 | 0.99984 | 7 | vel3 about `[-698, 2.04e4]` |

The final exact transform counter delta is `input_encode=100`,
`target_encode=19`, `oracle_decode=19`, and `prediction_decode=100`. Temporal
statistics use float64 reductions so that diagnostics remain meaningful for
extreme but finite float32 states. This is evaluation precision handling, not
a repair of the trained model.

All morphology products use **spherical-coordinate adapted views** and stored
coordinate shell/radial diagnostics. They are not Cartesian central slices or
Kerr--Schild invariant transport measurements.
