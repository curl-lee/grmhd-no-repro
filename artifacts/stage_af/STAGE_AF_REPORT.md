# Stage AF — Anisotropic Spherical DISCO3D No-Training Audit

| property | Stage AE isotropic spherical | Stage AF anisotropic spherical |
|---|---:|---:|
| scalar local scale | median NN | no |
| directional scales | no | `h_r,h_theta,h_phi` |
| support | spherical | target-local ellipsoid |
| candidate box | 7³ | 7³ |
| K | 5 | 5 |
| zero Z | 4 | 0 |
| trainable params | 363480 | 363480 |
| training | not run | not run |

## Scope and result

Stage AF replaces only the Stage AE scalar support scale with three measured local directional scales in a Euclidean spherical-coordinate orthonormal frame. It is a spherical-coordinate-aware geometry proxy, not Kerr–Schild covariant geometry. It neither transforms Bcc/velocity components nor changes the frozen Z64 data, P3 preprocessing, residual target, Plain L2 contract, split, width, modes, depth, spectral/differential branches, K, stencil, or initialization.

All readiness gates pass. Therefore:

`PRIMARY_DECISION = A — ANISOTROPIC_SPHERICAL_DISCO_READY_FOR_TRAINING`

This authorizes a separately approved next-stage controlled run only. Stage AF created no optimizer or scheduler, executed no epoch loop or parameter update, selected no checkpoint, and computed no trained scientific metric.

## 1. Frozen provenance and real coordinates

Formal geometry values were read from `data_proc/grmhd_regrid_inner_r200_64_expanded.h5`, group `coords`, whose frozen SHA256 is `3582a5c4f52f4fef305578ffd5a63c51423f8583f35d137c288d5c20203b50da`. No approximate coordinate arrays were substituted. The stored axis order is `(phi,theta,r)` and every vector is finite and strictly increasing.

- r centers: 1.1456345721309733 to 192.03331509030292; geometric center ratios are consistent.
- theta centers: 0.02454369328916073 to 3.1170490477234125; uniform spacing.
- phi centers: 0.049087386578559876 to 6.234097921967246; uniform periodic spacing and reconstructed face span `2π`.
- Float64 coordinate byte hashes and the exact hash contract are in `coordinates/coordinate_hashes.json`.
- Pinned `external/neuraloperator` remains at `86a8bc7812a31b42c4f7895693cf4ac11521c066` and was clean when the audit ran.

## 2. Why Stage AE had four zero-Z targets

Stage AE took one median over all valid nearest-neighbor physical distances and used the scalar radius `R_i=3h_i`. Near either polar row, azimuthal nearest-neighbor distance scales as `r sin(theta) Δphi` and is much smaller than radial or polar distance. At the simultaneous theta/r boundaries, theta and radial candidates are truncated. That scalar scale/candidate interaction left the outermost `k=4` hat with no positive-support candidate at `(theta,r)=(0,0),(0,63),(63,0),(63,63)`, so its quadrature normalization was exactly zero.

Stage AF does not add an epsilon, enlarge the radius, alter K, change the hat family, or invent boundary continuation. It repairs the mismatch by measuring the three directions separately.

## 3. Local spherical tangent geometry

For geometry only, centers are embedded as `X=(r sin(theta) cos(phi), r sin(theta) sin(phi), r cos(theta))`. At each target, the rows of the local frame are:

- `e_r=(sin(theta)cos(phi), sin(theta)sin(phi), cos(theta))`
- `e_theta=(cos(theta)cos(phi), cos(theta)sin(phi), -sin(theta))`
- `e_phi=(-sin(phi), cos(phi), 0)`

The maximum float64 orthonormality error over the 64×64 angular targets is `4.440892098500626e-16`, below `1e-12`. A source-target displacement `X_j-X_i` is projected on the target frame to obtain `(delta_r,delta_theta,delta_phi)`.

`h_r`, `h_theta`, and `h_phi` are the median absolute matching component from the valid ±1 neighbor in that coordinate direction. At radial or theta boundaries, the sole inward neighbor is used. Phi remains periodic. All measurements are finite and positive; there is no explicit `1/sin(theta)` or `1/(r sin(theta))`.

| scale | minimum | median | maximum |
|---|---:|---:|---:|
| h_r | 0.0970272155312577 | 1.2081553272922725 | 14.993989545217545 |
| h_theta | 0.0562136256465834 | 0.7283925998930169 | 9.422628426863318 |
| h_phi | 0.002755769668530849 | 0.7452781173952485 | 18.81688793290195 |

`ZERO_DIRECTIONAL_SCALE_COUNT=0`; negative and nonfinite counts are also zero. The local anisotropy ratio `max(h)/min(h)` ranges from 1.5912746280507677 to 35.20875370654068.

## 4. Support, basis, quadrature, and normalization

The fixed normalized displacement is

`s_ij=sqrt[(delta_r/(3h_r))²+(delta_theta/(3h_theta))²+(delta_phi/(3h_phi))²]`,

with support `s_ij<=1`. Thus support is a position-dependent, target-local, anisotropic three-cell ellipsoid within the unchanged 7×7×7 candidate box. K remains five piecewise-linear hats centered at `k/4`, width `1/4`.

The quadrature is the unchanged spherical coordinate-volume proxy `q_j=r_j² sin(theta_j) Δr_j Δtheta_j Δphi_j`; it is finite and positive, but it is not Kerr–Schild proper volume. Each target/basis normalization is computed as `Z_ik=sum_j q_j psi_k(s_ij)` with no epsilon fallback.

The complete 5×64×64 normalization audit gives:

| basis k | minimum Z | maximum Z | minimum positive-support count |
|---:|---:|---:|---:|
| 0 | 1.4460954380366008e-05 | 2774.359750494564 | 1 |
| 1 | 6.428214302015524e-05 | 11392.398125122338 | 7 |
| 2 | 1.5807427312569916e-04 | 46507.86241553386 | 11 |
| 3 | 2.930753720178585e-04 | 90832.94824272694 | 13 |
| 4 | 1.6846214304849956e-04 | 49022.32648836066 | 9 |

All five bases are active at all 4096 `(theta,r)` targets and `ZERO_Z_COUNT=0`. Raw hats sum to one on active candidates with maximum absolute error `2.220446049250313e-16`. Quadrature-normalized constant-field integrals differ from one by at most `5.551115123125783e-16`, well within the `1e-6` production criterion.

## 5. Repaired Stage AE corners

North/south values differ only by last-bit coordinate asymmetry. The exact per-basis values and counts are in `geometry/bad_corner_repair.csv`.

| theta index | r index | h_r | h_theta | h_phi | active candidates | all five Z positive |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.0970272155312577 | 0.05621362564658345 | 0.002755779482437956 | 21 | true |
| 0 | 63 | 14.993989545217545 | 9.422628426863309 | 0.461928683494635 | 31 | true |
| 63 | 0 | 0.09702721553125793 | 0.05621362564658346 | 0.002755769668530849 | 21 | true |
| 63 | 63 | 14.993989545217516 | 9.42262842686331 | 0.46192703846998084 | 31 | true |

At each corner every one of the five positive-support counts is nonzero. The formerly failing `k=4` Z values are approximately `1.68462e-4` at the inner corners and `1349.49` at the outer corners.

## 6. Boundary and rotational semantics

- `PHI_BOUNDARY=PERIODIC`.
- `THETA_BOUNDARY=TRUNCATED_RENORMALIZED`.
- `R_BOUNDARY=TRUNCATED_RENORMALIZED`.
- Theta/r are neither circular nor reflected; no pole-parity continuation is guessed.

Absolute-phi recomputation at shifts 1, 7, 17, and 31 gives displacement-component relative L2 values between `5.56e-16` and `5.78e-16`. The float32 operator shift test is below `1e-5`. Phi equivariance passes.

## 7. Distinctness from Stage AD and comparison to Stage AE

Stage AF is not a numerical restatement of Stage AD. Representative AF-versus-AD weight-map relative L2 values are:

| region | relative L2 | Pearson correlation |
|---|---:|---:|
| inner equator | 0.29830298806126465 | 0.957809046651216 |
| middle equator | 0.14336666023714312 | 0.989640026301352 |
| outer equator | 0.24888126063834823 | 0.9698469999948388 |
| inner near-pole | 0.8264551357671184 | 0.7733790136111252 |
| middle near-pole | 0.5717415679396256 | 0.8660648891943452 |
| outer near-pole | 0.6907885086025706 | 0.8188282190065334 |

The largest difference is at the inner polar corners (`max relative L2=0.8264557795830277`), precisely where spherical anisotropy is strongest and Stage AE degenerated. This is far above the `1e-6` non-distinctness threshold. Full target comparisons against Stage AD and Stage AE, effective local extents, support counts, and nonzero-weight counts are in `comparison/`.

Synthetic impulse and localized-Gaussian geometry probes likewise show target-dependent near-pole and outer-r footprints. They test locality only; no untrained model output is interpreted as a scientific forecast.

## 8. Numerical implementation

The production implementation precomputes `geometry_weight[K,dphi,dtheta,dr,theta,r]`. Its `OFFSET_VECTORIZED_ACCUMULATION` backend vectorizes all voxels and phi offsets and loops only over the 49 theta/r offset planes, avoiding both voxel loops and a full N² pair matrix. Phi rolls are periodic; theta/r use explicit valid source/target slices.

- Tiny `(Nphi,Ntheta,Nr)=(8,6,5)` brute-force relative L2: `1.224647575642275e-16`.
- Tiny brute-force maximum absolute error: `8.881784197001252e-16`.
- Input/weight/bias float64 gradcheck: pass.
- CPU/CUDA float32 relative L2: `6.522432727251726e-08`; maximum absolute error `1.1920928955078125e-07`.
- Stage AF targeted tests with visible CUDA: 9 passed.
- Full repository tests: 532 passed, 2 skipped, 13 warnings. The two skips are CUDA-conditioned tests in the restricted test process; the Stage AF CUDA test passed separately in the GPU-visible process.

## 9. Frozen architecture and initialization

The complete four-layer LocalNO contains 363,480 trainable parameters, identical to Stage AD. Parameter names match, every trainable tensor is bitwise equal, and both trainable-state SHA256 values are `77252855f054a199500fa779f7340b33b6a2ce546b2082f17f93666275f4c588`.

The four trainable DISCO branches retain weight shape `[16,16,5]` and bias shape `[16]`. Only fixed, nonpersistent geometry buffers and forward semantics differ.

## 10. Production CUDA feasibility without training

The full production model ran on `NVIDIA GeForce RTX 5070`, capability 12.0, PyTorch `2.12.1+cu130`, CUDA `13.0`, batch 1, input shape `[1,16,64,64,64]`.

- Full forward: 0.6231939160002185 s.
- Sum of four AF DISCO forward branches: 0.6048121643066406 s.
- Backward: 1.31310383500022 s.
- Peak allocated: 1505.1240234375 MiB.
- Peak reserved: 2000 MiB.
- Output, loss, and all gradients: finite.
- DISCO forward output: nonzero in every layer for both frozen samples.
- DISCO gradient norm: 4.4907379150390625, finite and nonzero.
- Total gradient norm: 10.242422103881836.

After backward, gradients were cleared with `model.zero_grad(set_to_none=True)`. There was no parameter update and no checkpoint.

Relative to the frozen Stage AD no-update preflight, AF used 2.239× peak allocated memory and its measured forward-plus-backward time was 6.492×. Scaling the actual Stage AD 300-epoch runtime by this preflight ratio estimates 57,614 s, or 16.004 h, for a future 300-epoch AF run. This is a feasibility estimate only, not a training result or promise of exact runtime.

## 11. Required answers

1. Stage AE's scalar median was dominated near the pole by tiny phi distance while theta/r candidates were truncated at corners, leaving k=4 with no support.
2. Stage AF uses the Euclidean spherical orthonormal directions `(e_r,e_theta,e_phi)` at each target; maximum orthonormality error is below `1e-12`.
3. Each directional scale is the median absolute matching projected component from valid ±1 coordinate neighbors, with one-sided theta/r boundary handling and periodic phi.
4. The four exact scale triples are tabulated in Section 5.
5. No zero, negative, or nonfinite directional scale exists.
6. Support is the ellipsoid `sqrt(sum_a (delta_a/(3h_a))²)<=1`.
7. Is `1/sin(theta)` used explicitly? **No.**
8. K remains 5.
9. The candidate stencil remains 7³.
10. All five bases are active at every production target.
11. `ZERO_Z_COUNT` is 0.
12. Constant-field normalization passes the full production grid.
13. Theta and r remain nonperiodic, truncated, and renormalized.
14. Phi equivariance passes at all required shifts.
15. The optimized implementation matches the tiny brute-force reference.
16. Input, weight, and bias gradcheck passes.
17. Stage AF is nontrivially distinct from Stage AD.
18. Differences are largest at inner near-pole/corner targets, followed by outer near-pole targets.
19. Parameter count and every initialized trainable tensor match Stage AD exactly.
20. The production RTX 5070 forward/backward feasibility pass succeeds.
21. The estimated 300-epoch cost is approximately 16.004 hours using the stated Stage AD scaling method.
22. Yes: the next stage may run `anisotropic_spherical_disco_controlled_training`, but only after separate authorization.

## 12. Final status

```text
PRIMARY_DECISION = A

REPRODUCTION_SCOPE =
ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION

DISCO_GEOMETRY =
ANISOTROPIC_LOCAL_SPHERICAL_TANGENT_PROXY

KERR_SCHILD_COVARIANT_GEOMETRY = false
VECTOR_COMPONENT_TRANSFORMATION = false

NO_MODEL_TRAINING = true
TRAINING_STARTED = false
TRAINING_COMPLETED = false

DISCO3D_BASIS_COUNT = 5
CANDIDATE_STENCIL_SHAPE = [7,7,7]
DIRECTIONAL_RADIUS_MULTIPLIER = 3

ZERO_DIRECTIONAL_SCALE_COUNT = 0
ZERO_Z_COUNT = 0

ALL_BASES_ACTIVE = true
CONSTANT_FIELD_PASS = true
ANISOTROPIC_DENSE_REFERENCE_MATCH = true
PHI_EQUIVARIANCE_PASS = true
GRADCHECK_PASS = true
ANISOTROPIC_DISCO_UNIT_TESTS_PASS = true

ANISOTROPIC_GEOMETRY_DISTINCT_FROM_STAGE_AD = true

TRAINABLE_PARAMETER_COUNT_MATCH = true
TRAINABLE_INITIALIZATION_HASH_MATCH = true

PRODUCTION_FORWARD_PASS = true
PRODUCTION_BACKWARD_PASS = true
DISCO_BRANCH_FORWARD_ACTIVE = true
DISCO_BRANCH_GRADIENT_ACTIVE = true

ESTIMATED_300_EPOCH_RUNTIME = 16.003981921864007 hours

EXACT_3D_DISCO_IMPLEMENTATION_FOUND = false
EXACT_REPRODUCTION_BLOCKED = true

AUTHORIZE_NEXT_STAGE =
anisotropic_spherical_disco_controlled_training
```
