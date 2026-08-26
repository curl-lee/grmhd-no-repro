# Stage I Run A no-H1 comparison

This compares the frozen Stage G paper-adapted Full and Plain pilots with the matched Stage I no-H1 diagnostic extension. no-H1 retains ROI, bounds, envelope, dissipation, channel weights, preprocessing, optimizer, scheduler, initial state, and pair order; it is not Plain L2.

## Validation

| model | average | global |
|---|---:|---:|
| persistence | 0.189088 | 0.235074 |
| stage_g_full | 0.462782 | 0.387339 |
| stage_g_plain | 0.47458 | 0.381627 |
| stage_i_no_h1 | 0.482824 | 0.386497 |

## Morphology and boundary

| category | Full | Plain | no-H1 | best |
|---|---:|---:|---:|---|
| center_morphology | 0.911962 | 0.904979 | 0.938637 | plain |
| polar_morphology | 0.880257 | 0.835741 | 0.856055 | plain |
| magnetic_texture | 2.41274 | 0.899831 | 0.85202 | no_h1 |
| radial_statistics | 6.56373 | 4.84509 | 5.06848 | plain |
| outer_shell_statistics | 7.18638 | 1.206 | 1.8582 | plain |
| model_only_saturation | 0.018526 | 0.0107105 | 0.0177212 | plain |
| step50_100_artifacts | 5 | 3 | 4 | plain |

- Full clipping: `1`
- no-H1 clipping: `1`
- Unit-index/stored-coordinate training executed: `false/false`
