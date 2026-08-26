# Stage I Run B unit-index comparison

Run B replaces only the Stage G current upstream H1 with the Stage H unit-index H1. All other Full terms and paired controls remain frozen. This is a diagnostic extension, not a paper-faithful result.

## Validation

| model | average | global |
|---|---:|---:|
| canonical_oracle | 0 | 0 |
| persistence | 0.189088 | 0.235074 |
| full | 0.462782 | 0.387339 |
| plain | 0.47458 | 0.381627 |
| no_h1 | 0.482824 | 0.386497 |
| unit_index | 0.482485 | 0.385469 |

## Morphology and boundary

| category | Full | Plain | no-H1 | unit-index | best |
|---|---:|---:|---:|---:|---|
| center_morphology | 0.911962 | 0.904979 | 0.938637 | 0.93875 | plain |
| polar_morphology | 0.880257 | 0.835741 | 0.856055 | 0.859956 | plain |
| magnetic_texture | 2.41274 | 0.899831 | 0.85202 | 0.860057 | no_h1 |
| radial_statistics | 6.56373 | 4.84509 | 5.06848 | 5.26359 | plain |
| outer_shell_statistics | 7.18638 | 1.206 | 1.8582 | 1.7158 | plain |
| model_only_saturation | 0.018526 | 0.0107105 | 0.0177212 | 0.0184464 | plain |
| step50_100_artifacts | 5 | 3 | 4 | 4 | plain |

## Gradient control

- Mean H1/base gradient ratio: `0.00474513`
- Mean base/H1 cosine: `0.461873`
- Mean clip scale: `0.142993`
- Mean effective base/H1 gradient norms: `0.986423` / `0.00467276`
- Stored-coordinate/volume H1 training: `not executed`
- Final Stage I decision: `not made in Run B`
