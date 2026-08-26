# paper_reduced100 preprocessing audit

- Status: `passed_with_warnings`
- Protocol: `paper_reduced100_press_spherical_ks`
- Thermal channel: `press` (paper adaptation)
- HDF5 SHA-256: `cd3966bed1f617f49d775151cff87ae92eb2432a560bb82ae9355605e7aea70a`
- Fit snapshots: `11..90`
- Gamma / inverse limit: `6.0` / `5.9399999999999995`
- Validation excluded from fit: `True`

| channel | epsilon | median | MAD scale | >=0.95 gamma | inverse clamp hits | max abs round-trip |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bcc1 | 0.001 | 0 | 1.59447548 | 0 | 0 | 2.38418579e-07 |
| Bcc2 | 0.001 | -0.058162116 | 0.115317572 | 0.0289509773 | 0.00109558105 | 0.0364821441 |
| Bcc3 | 0.01 | 2.19452002e-12 | 0.0118065025 | 0.267527246 | 0.23871932 | 2.06633756 |
| rho | 1e-08 | -3.03405448 | 1.15649502 | 0 | 0 | 1.90734863e-06 |
| press | 1e-10 | -4.10214488 | 1.20992178 | 0 | 0 | 5.96046448e-08 |
| vel1 | 0 | 0.210481167 | 0.183637755 | 0 | 0 | 4.76837158e-07 |
| vel2 | 0 | 2.10038973e-08 | 0.00225010841 | 0.0637510777 | 0.0374410629 | 0.17264038 |
| vel3 | 0 | 0.00437162351 | 0.00624953798 | 0.259980774 | 0.22314415 | 1.36871835 |

## Warnings

- `training_values_exceed_inverse_clamp` on `Bcc2`: fraction `0.00109558105`; paper encode/decode is lossy for these training voxels.
- `training_values_exceed_inverse_clamp` on `Bcc3`: fraction `0.23871932`; paper encode/decode is lossy for these training voxels.
- `training_values_exceed_inverse_clamp` on `vel2`: fraction `0.0374410629`; paper encode/decode is lossy for these training voxels.
- `training_values_exceed_inverse_clamp` on `vel3`: fraction `0.22314415`; paper encode/decode is lossy for these training voxels.

This audit uses train snapshots only. It does not load Round 1--3 normalizer stats,
fit a radial baseline, or authorize any training run.
