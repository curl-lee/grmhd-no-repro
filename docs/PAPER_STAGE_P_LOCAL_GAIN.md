# Stage P local finite-difference gain

This audit estimates directional responses only. It does not use parameter
backward and does not estimate a complete Jacobian or spectral radius.

Six frozen states are probed: validation initial, GT step 1, GT step 5, actual
steps 10 and 19, and no-GT step 25. Ten fixed directions are used at epsilons
`1e-3` and `1e-2`: Bcc2, Bcc3, vel3, all-channel, inner/outer shell,
radial-low/radial-mid, low-k, and high-k. Both direct operator `F` and physical
feedback operator `G=E o D o F` responses are recorded.

All 120 rows are finite and `used_backward=false`. Raw directional gains span
approximately `1.01`--`606.38` for `J_F` and `0.97`--`422.26` for `J_G`.
However, **0/120 rows** satisfy the predeclared 20% two-epsilon consistency
criterion. Consequently:

- the large raw gains cannot be promoted to stable local-gain evidence;
- median `J_F` and `J_G/J_F` mechanism statistics are intentionally null;
- stable non-diagonal cross-channel gain is unavailable;
- recurrent and cross-channel conclusions must rely on the independent
  normalized-direct, projection, and teacher-reset evidence;
- no claim about a Jacobian spectral radius is made.

This mirrors the caution needed for the Stage N operator-response audit: an
inconsistent finite difference is not evidence of zero gain, but it is also not
reproducible support for a mechanism threshold. See
`outputs/paper_reduced100/stage_p/local_gain.json` and `.csv`.
