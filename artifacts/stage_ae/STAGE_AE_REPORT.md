# Stage AE — Spherical-Coordinate-Aware DISCO3D LocalNO

| model | geometry | state L2 | residual L2 | cosine | shell skill | radial skill | first10x |
|---|---|---:|---:|---:|---:|---:|---:|
| Stage-T LocalNO | no local integral | 0.266711 | 0.891851 | 0.740602 | -1.67947 | -0.348089 | 1 |
| Stage-AD DISCO3D | index-space | 0.264062 | 0.891640 | 0.757468 | -4.35448 | -0.390352 | 1 |
| Stage-AE spherical DISCO3D | spherical proxy | not run | not run | not run | not run | not run | not run |

## Mandatory normalization-gate result

Actual coordinates were read from `data_proc/grmhd_regrid_inner_r200_64_expanded.h5`. They are 64-point
monotonic arrays with geometric r centres (1.14563457 to 192.033315), uniform
theta centres (0.0245436933 to 3.11704905), and uniform periodic phi
centres (0.0490873866 to 6.2340981). Source faces are absent, so radial
faces were reconstructed with geometric midpoints/ratio extrapolation and
angular faces with verified-uniform midpoint extrapolation.

The prescribed Euclidean spherical embedding, positive spherical-volume proxy,
periodic phi, truncated theta/r boundaries, 7³ candidate box, K=5 hats, local
median scale, and R_i=3h_i were implemented literally. The volume proxy is
finite and positive, generic-grid dense reference/phi equivariance/gradcheck
pass, trainable shapes and seed-42 initialization match Stage AD, and kernel maps
do change with radius and theta.

However, the production Z64 normalization audit finds `ZERO_Z_COUNT=4`:
`[[4, 0, 0], [4, 0, 63], [4, 63, 0], [4, 63, 63]]` in `[basis,theta_index,r_index]` order. They are the outermost
hat at both polar rows combined with both radial boundaries. Near a pole the phi
nearest-neighbor distance becomes very small; with theta/r truncation, the median
nearest-neighbor scale at these corners makes R_i too small for any 7³ candidate
to activate k=4. Consequently the required per-target quadrature integral cannot
equal one.

The frozen instruction requires immediate stop when any Z is zero. No support,
radius, K, boundary, epsilon fallback, architecture, data, or loss was changed.
No CUDA/runtime preflight or training was run. This is an invalid Stage AE
adapted-discretization contract, not a scientific negative result about spherical
DISCO or the paper.

## Required answers

1. Stage AD uses equal normalized tensor-index distances, so it ignores r and sin(theta) scale factors.
2. Stage AE uses `(r sin(theta) cos(phi), r sin(theta) sin(phi), r cos(theta))` Euclidean embedding distances.
3. No unknown Kerr-Schild metric parameter is used.
4. Vector components are not transformed.
5. Phi is handled periodically by modulo source indexing.
6. Theta/r do not wrap; invalid candidates are truncated.
7. The spherical volume proxy is finite and positive, but the complete per-target quadrature audit fails due to zero Z.
8. No: four target/basis normalizations are invalid.
9. The generic valid tiny-grid optimized implementation matches brute force, but the production operator is intentionally non-constructible.
10. Trainable shapes imply the same 363,480 parameters as Stage AD.
11. Audit-only trainable initialization hashes match exactly; no trainable production model was authorized.
12. Yes, valid-location inner/middle/outer kernel maps differ from Stage AD.
13. Yes, near-pole maps change most and reveal the zero-normalization corner defect.
14–21. Not evaluated because training was prohibited before CUDA preflight.
22. No; the next authorized action is implementation repair, not the final adapted benchmark.
