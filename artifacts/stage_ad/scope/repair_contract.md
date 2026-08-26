# Stage AD Repair Contract

- `REPRODUCTION_SCOPE = ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION`
- `DISCO3D_IMPLEMENTATION = ADAPTED_ISOTROPIC_RADIAL_PIECEWISE_LINEAR`
- `EXACT_3D_DISCO_IMPLEMENTATION_FOUND = false`
- `EXACT_REPRODUCTION_BLOCKED = true`
- sole scientific repair: cutoff radius `1 cell -> 3 cells`
- basis count: 5; basis family unchanged
- computational domain length: [2.0, 2.0, 2.0]
- regular model grid: [64, 64, 64]
- boundary: computational periodic adaptation
- dataset: `data_proc/grmhd_regrid_inner_r200_64_expanded.h5`
- frozen P3 normalizer: `artifacts/stage_s/p3_normalizer_expanded`
- upstream: `86a8bc7812a31b42c4f7895693cf4ac11521c066`; clean

No validation result was used to choose the radius.  R=3Δ is the smallest integer-cell
support that makes every one of the five pre-frozen radial hats discretely active.
