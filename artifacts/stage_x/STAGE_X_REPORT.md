# Stage X — Paper Method Gap and Coordinate/Field Representation Audit

| gap | paper | current | severity | evidence | transport-failure plausibility |
|---|---|---|---|---|---|
| coordinate representation | Cartesian-index radial shells on Cartesian KS cube | 8 physical-r shells; no continuous r/theta/phi | HIGH | paper C.4 + resolved forward + frozen response test | HIGH |
| vector basis | Cartesian x/y/z | untransformed spherical 1/2/3 representation | FUNDAMENTAL | paper A.2/C.2 + ATHDF attrs/variables | HIGH |
| operator family | volumetric equidistant DISCO LocalNO | spectral + finite difference; no local integral | HIGH | paper C.8 + pinned upstream/config/forward | HIGH |
| regridding | paper remap unspecified; Cartesian 64 cube | nearest-centre AMR→spherical 64 tensor | MAJOR | 10 fields + 10 adjacent-pair 32/64/96/128 audit | HIGH |
| thermal variable | eint in C.2 (P in A.2) | press; no EOS conversion | HIGH | paper internal inconsistency + ATHDF attrs | HIGH |


## Direct answers

1. **Same coordinate system?** No. Paper: Cartesian Kerr–Schild. Current: spherical Kerr–Schild `(phi,theta,r)` tensor.
2. **Does the network know physical r,theta,phi?** Only partially: eight discrete physical-r shells. It has no continuous r/log-r, theta, or phi channels.
3. **Does the paper explicitly use Cartesian positional information?** The default uses shells computed from Cartesian index distance; explicit normalized Cartesian Fourier `(xi,eta,zeta)` features occur only in an ablation.
4. **Same vector basis?** No; paper labels Cartesian x/y/z while current retains 1/2/3 spherical-coordinate representation.
5. **Enough provenance for a correct transform?** No. Spin, mapping, velocity/magnetic component conventions and tetrad/basis data are missing.
6. **Scientific basis for press→eint?** No; neither raw EOS/Gamma nor a GRMHD conversion contract is available.
7. **Does the paper LocalNO include a missing DISCO/local-integral mechanism?** Yes, volumetric equidistant DISCO is explicit. Exact per-layer composition remains unspecified.
8. **Operator gap HIGH/FUNDAMENTAL?** HIGH: the required local-integral mechanism is wholly absent, though causal responsibility for failures is not proven.
9. **Does 64^3 preserve temporal increments?** `POOR` under the frozen thresholds: median relative difference `0.503429`, median cosine `0.871549` versus the 128 sampling reference.
10. **Are inner AMR dynamics visibly damaged at 64^3?** Yes: the inner increment relative difference exceeds 0.4 and is the largest regional median, with inner/middle/outer increment losses `{'inner': 0.44360202273396504, 'middle': 0.3644246358208748, 'outer': 0.42273226636403505}` and field losses `{'inner': 0.14971942552350298, 'middle': 0.08266151386917865, 'outer': 0.08918616544353217}`. Outer loss is also substantial, so this is not an inner-only effect. This is relative to a sampling reference, not raw truth.
11. **Are paper 64^3 and current 64^3 scientifically equivalent?** No: geometry, cell scales, vector basis, thermal field, positional semantics, remap, and operator differ.
12. **Paper-method reproduction or adaptation?** Only a spherical-KS differential-LocalNO workflow/method adaptation.
13. **Most valuable single intervention?** Acquire matching Cartesian-KS/eint data plus simulation/EOS/vector provenance and the paper's 3D DISCO/code contract before another model pilot.


## Audit scope and reproducibility

- Paper: arXiv `2512.01576v1`, PDF SHA256 `fe84a42289da7e86dd8460d223e57627a86aa0b2114a64a13b83f84388441808`.
- Project/upstream: `39dde155093436685c20c4f1d97269a02ae1189d` / `86a8bc7812a31b42c4f7895693cf4ac11521c066`.
- Raw audit: 10 fixed field snapshots and 10 fixed adjacent pairs, all four sampling resolutions.
- Production `64^3` equivalence: bitwise exact for every audited snapshot/channel array.
- 128^3 label: `HIGHER_RES_SAMPLING_REFERENCE`, never raw truth.
- No training, loss tuning, vector conversion, EOS inference, raw-data mutation, or processed-HDF5 regeneration occurred.

## Verification

- Required Stage-X artifact set: complete and non-empty.
- CSV row counts: field convergence 320; temporal increments 320; shell increments 2,560; radial increments 25,600; production equivalence 80; coordinate identifiability 7.
- Full regression suite: `491 passed, 13 warnings`.
- `git diff --check`: passed.
- Pinned upstream worktree: clean at `86a8bc7812a31b42c4f7895693cf4ac11521c066`.

## Final labels

PRIMARY_DECISION = E

COORDINATE_INFORMATION_GAP = HIGH
VECTOR_BASIS_GAP = FUNDAMENTAL
VECTOR_TRANSFORM_PROVENANCE = INCOMPLETE
PAPER_OPERATOR_GAP = HIGH
TEMPORAL_INCREMENT_REGRID_FIDELITY = POOR
THERMAL_VARIABLE_GAP = HIGH

CURRENT_TARGET_NOT_SCIENTIFICALLY_COMPARABLE_TO_PAPER = true

AUTHORIZE_NEXT_STAGE = data_or_code_acquisition
