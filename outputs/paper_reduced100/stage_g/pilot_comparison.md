# Stage G Full versus Plain controlled comparison

This is a reduced100, spherical Kerr-Schild, press-adapted, FNO-proxy, 30-epoch resource-scaled pilot. It is not directly comparable to paper Table 2.

## Table-2-style normalized validation

| model | Bcc1 | Bcc2 | Bcc3 | rho | press | vel1 | vel2 | vel3 | average | global |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical_oracle | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| persistence | 0.0691883 | 0.3066 | 0.243158 | 0.116309 | 0.129493 | 0.140437 | 0.369472 | 0.13805 | 0.189088 | 0.235074 |
| full_fno_proxy | 0.505511 | 0.45855 | 0.324556 | 0.589977 | 0.512736 | 0.636946 | 0.493951 | 0.180028 | 0.462782 | 0.387339 |
| plain_l2_fno | 0.464507 | 0.437662 | 0.278945 | 0.626617 | 0.630306 | 0.639081 | 0.510431 | 0.209094 | 0.47458 | 0.381627 |

## Morphology and boundary categories

| category | Full | Plain | better |
|---|---:|---:|---|
| center_morphology | 0.911962 | 0.904979 | plain |
| polar_morphology | 0.880257 | 0.835741 | plain |
| magnetic_texture | 2.41274 | 0.899831 | plain |
| radial_statistics | 6.56373 | 4.84509 | plain |
| outer_shell_statistics | 7.18638 | 1.206 | plain |
| model_only_saturation | 0.018526 | 0.0107105 | plain |
| step50_100_artifacts | 5 | 3 | plain |

- Full-better categories: `none`
- Decision: `B. 先诊断 H1 adaptation`

The comparison does not claim exact paper reproduction or direct numerical equivalence to the paper's 1200-epoch 3D DISCO LocalNO experiments.
