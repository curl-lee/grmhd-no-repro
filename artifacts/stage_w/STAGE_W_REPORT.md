# Stage W — Mixed-Basis Non-Euclidean Spectral Operator Audit

| variant | spectral basis | FD boundary | state L2 | residual L2 | cosine | shell skill | radial skill | first10x |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage-T | FFT/FFT/FFT | periodic/all | 0.266711 | 0.891851 | 0.740602 | -1.67947 | -0.348089 | 1 |
| W1 | FFT/DCT/DCT | Stage-T periodic/all | 0.263938 | 0.902452 | 0.748702 | -8.12255 | -1.03988 | 1 |
| W2 | FFT/DCT/DCT | phi periodic, theta/r replicate | 0.260154 | 0.889144 | 0.751458 | -19.1912 | -1.00572 | 1 |

## Frozen controls and training

W1/W2 share initial tensor hash `a31974d5cc76eec0cfc847af7b4f8f8207e3bbccd1d7ae07df46ba77d4e20ac7` and each has 358,296 parameters (delta +0.000%). Both use the exact Stage-T first-150 epoch order, Plain residual L2, Adam, clip=1, and 6,300 updates.

- W1: 2701.2s, mean clipping 0.9952, peak 1049.0/1202.0 MiB, all finite=True.
- W2: 2825.9s, mean clipping 0.9917, peak 1149.5/1316.0 MiB, all finite=True.

## Required scientific answers

1. **Log-r uniformity.** `true`: delta-xi min/max are 0.015873015873/0.015873015873, with std/mean 3.96e-15.
2. **Transform correctness.** All core tests passed=`true`; float32/float64 roundtrip errors are 2.17e-07/4.43e-16, Parseval error 0.
3. **One-step dynamics.** W1 changes state/residual/cosine from 0.266711/0.891851/0.740602 to 0.263938/0.902452/0.748702; W2 gives 0.260154/0.889144/0.751458.
4. **Residual gate.** W1 residual<1 is `true`; W2 is `true`.
5. **Shell transport.** Stage-T/W1/W2 skills are -1.67947/-8.12255/-19.1912; the +0.10 gate is evaluated in the final decision.
6. **Radial transport.** Stage-T/W1/W2 skills are -0.348089/-1.03988/-1.00572.
7. **Largest shell improvements.** W2 shell 7 (outer, +707), W1 shell 7 (outer, +10.4), W1 shell 2 (inner, +1.24). Full shell/r-index attribution is in `comparison/per_shell_metrics.csv` and `radial_profile_metrics.csv`.
   Per-shell skill divides by the true shell increment; isolated values can therefore explode when that increment is nearly zero. The very large outer-shell deltas are diagnostic outliers, not evidence against the aggregate shell-skill failure.
8. **Boundary error.** `NONPERIODIC_BASIS_BOUNDARY_BENEFIT=false` under the predeclared 10% two-axis criterion. W1 theta/r reductions are +4.84%/-5.05%; W2 reductions are +3.22%/-9.41%.
9. **Rollout.** First-10x steps are Stage-T=1, W1=1, W2=1.
10. **Rho/press tails.** Physical L2 (rho/press) is Stage-T=14/2.36e+03, W1=1.53e+21/2.03e+13, W2=5.06e+09/9.06e+12; no clamp or P3 change was made.
11. **State/transport decoupling.** Spectral-geometry hypothesis support is `false` under decision `D`; mixed-basis gains must be judged by transport and rollout, not state L2 alone.
12. **Coordinate conditioning.** Next authorization is `paper_method_gap_and_coordinate_representation_audit`. Explicit coordinate conditioning is authorized only for A/B/C, not retroactively added in Stage W.

## Spectral and temporal attribution

Prediction spectral-reconstruction boundary reductions (theta/r) are W1=-18.94%/+11.07% and W2=-11.48%/+8.20%. These are seam-response diagnostics, not proof of spherical covariance.
The focus-channel high-retained-mode fractions (phi/theta/log-r) are Stage-T=0.727/0.262/0.378, W1=0.720/0.287/0.283, and W2=0.703/0.281/0.283.
Focus-channel low-pass response energy in inner/middle/outer radial thirds is Stage-T=86.26%/3.49%/10.26%, W1=93.92%/4.57%/1.51%, and W2=94.00%/4.46%/1.54%. The mixed basis clearly changes radial response allocation, but the strong loss of outer response and worse aggregate shell/radial skills show that it does not distinguish inner/outer evolution in a transport-beneficial way.
Stage-T does show theta/r seam-sensitive reconstruction error, but mixed basis reduces only the radial reconstruction ratio while worsening theta; learned prediction boundary error also worsens in r. Thus the observed spectral response change is not accompanied by transport improvement.
Late/early state-L2 ratios are Stage-T=1.546, W1=1.404, W2=1.401; the chronological split and Stage-V OOD fit remain unchanged.

## Limitations

The 64^3 data remain nearest-cell resampled spherical Kerr--Schild component maps with P3 and press adaptation. FFT/DCT/DCT is neither spherical harmonics nor a Kerr--Schild covariant vector operator. No coordinate embedding, metric factor, loss change, mode search, or architecture-width/depth search was used.

PRIMARY_DECISION = D
LOG_R_GRID_UNIFORM = true
NONPERIODIC_BASIS_BOUNDARY_BENEFIT = false
SPECTRAL_GEOMETRY_HYPOTHESIS_SUPPORTED = false
AUTHORIZE_NEXT_STAGE = paper_method_gap_and_coordinate_representation_audit
