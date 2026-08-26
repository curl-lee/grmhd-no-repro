# Stage U — Spherical-Grid / Operator Geometry Mismatch Audit

| variant | operator | state L2 | residual L2 | cosine | shell skill | radial skill | first 10x step |
|---|---|---:|---:|---:|---:|---:|---:|
| Stage-T baseline | spectral + index FD | 0.266711 | 0.891851 | 0.740602 | -1.67947 | -0.348089 | 1 |
| spectral-only | spectral | 0.323109 | 1.01913 | 0.690183 | -2.52783 | -0.837303 | 1 |
| coordinate-aware | spectral + coordinate FD | 0.304229 | 1.02973 | 0.690177 | -38.8136 | -2.24104 | 1 |
| spherical-proxy | spectral + spherical-proxy FD | 240.286 | 716.164 | 0.000135965 | -2.92068e+07 | -1733.79 | 1 |

## Frozen contract and audit result

`DATASET_FROZEN = true`; `SPLIT_FROZEN = true`; `PREPROCESSING_FROZEN = true`; `TARGET_CONTRACT_FROZEN = true`; `LOSS_FROZEN = true`.

The frozen expanded HDF5 is `(phi,theta,r)=64^3`; r is geometric with ratio 1.084693 and `dr=0.097027..14.993990`. The pinned upstream FD uses one scalar grid width (1.0 at 64^3) and circular `3x3x3` kernels on all axes. Synthetic results are `INDEX_FD_GEOMETRY_ERROR=STRONG`, `BOUNDARY_MISMATCH=THETA_AND_R`, and `SPECTRAL_INDEX_SPACE_MISMATCH=STRONG`. Synthetic minimax padding selected `{'phi': 'periodic', 'r': 'replicate', 'theta': 'replicate'}`.

## Controlled pilots

All three pilots used seed 42, 168 train pairs, 42 validation pairs, normalized residual targets, PlainL2, Adam, batch 1, accumulation 4, width 16, modes 8^3, four layers, and exactly the first 6,300 updates of Stage T's 50,400-update scheduler. They ran sequentially for 150 epochs. U1 has 330,648 parameters (-7.72%); U3a/U3b have 339,864 (-5.14%); all common non-differential initial tensors match the frozen shared state. U2 was skipped because pinned LocalNO offers no semantics-preserving differential-only switch.

Training runtime / peak allocated MiB / mean clipping fraction were: U1 `1327.1s / 490.3 / 0.9776`, U3a `1884.4s / 715.9 / 0.9970`, U3b `1816.2s / 715.9 / 0.9917`. All training parameters remained finite; U3b's closed-loop physical decode did not.

## Direct answers

1. **Regular-grid assumption?** Yes. Upstream explicitly documents a regular grid and divides its centered convolution by one scalar.
2. **Same effective spacing?** Yes: Stage T passes `grid_width=1.0` to all three directions at 64^3.
3. **Wrong periodic theta/r?** Yes. `conv_padding_mode=periodic` becomes circular for phi, theta, and r.
4. **Log-r FD error?** Yes, strong: the same index derivative cannot represent stored-coordinate radial derivatives across a 154.5x dr range.
5. **Different physical scales?** Yes. The spherical-coordinate proxies vary strongly with r and theta; they are not proper Kerr-Schild distances.
6. **Spectral index-space mismatch?** Strong as an assumption mismatch: FFT is circular-shift equivariant in tensor index although radial physical scale changes with index.
7. **Did disabling FD improve transport?** No (`shell -1.68->-2.53`, `radial -0.348->-0.837`); both worsen, state/residual metrics regress, and first-10x remains step 1. Thus `DIFFERENTIAL_BRANCH_SUSPECTED=false`.
8. **Did coordinate-aware FD improve transport?** No. It worsened shell/radial skill to -38.8/-2.24 and retained step-1 failure.
9. **Did spherical proxy improve further?** No. Near-pole `1/(r sin theta)` conditioning caused catastrophic training/evaluation scales; one-step state L2=240 and closed-loop decode became nonfinite immediately.
10. **Any positive shell/radial skill?** No variant made either aggregate skill positive.
11. **Was step-1 range failure delayed?** No.
12. **Does geometry alone explain state/transport decoupling?** No. Synthetic mismatch is real and strong, but the minimal differential repair does not recover learned transport. The spectral branch remains index-periodic on theta/r, while objective mismatch and temporal shift remain independent plausible causes.
13. **Next direction?** Do not continue this naive spherical proxy. Audit loss/objective alignment next, with distribution-shift/generalization reported in parallel; a future geometry operator would need pole-regular, coordinate-aware spectral/basis treatment rather than only multiplying a learned stencil by singular scale factors.

## Channel-specific result

For Bcc2, Bcc3, vel3, rho, and press, neither U3 variant produces a consistent transport improvement. U3a's aggregate physical L2 is 9.055e+20; U3b's is 2.941e+36. These are decoder-tail failures, not improvements obtained by clipping (no new clipping was added). See `comparison/per_channel_comparison.csv` for exact per-channel state/residual/cosine/shell/radial metrics.

## Temporal distribution shift

Stage-T / U1 / U3a / U3b maximum late-over-early channel ratios and labels are stored in `comparison/temporal_shift_summary.json`: {"stage_t_baseline": [2.1354, "STRONG"], "spectral_only": [2.0159, "STRONG"], "coordinate_fd": [1.9977, "STRONG"], "spherical_proxy_fd": [6.758, "STRONG"]}. Geometry mismatch and chronological distribution shift are not treated as the same causal issue.

## Scientific limitations

This is a reduced, spherical Kerr-Schild coordinate-basis, nearest-leaf 64^3 adaptation. `SPHERICAL_COORDINATE_SCALE_PROXY` is not a strict Kerr-Schild proper distance or covariant derivative. Bcc1/2/3 and vel1/2/3 are not relabeled as Cartesian components. `CARTESIAN_REMAP_NOT_AUTHORIZED`.
