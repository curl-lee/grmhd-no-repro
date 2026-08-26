# Paper preprocessing clamp-loss audit

- Status: `completed_with_canonical_information_loss`
- Protocol: `paper_reduced100_press_spherical_ks`
- HDF5 SHA-256: `cd3966bed1f617f49d775151cff87ae92eb2432a560bb82ae9355605e7aea70a`
- Canonical path: `gamma=6`, inverse clamp `0.99*gamma`.
- `gamma=8`, `gamma=12`, and no-soft-clip are diagnostic extensions only.
- No training, checkpoint writing, or canonical-normalizer modification occurred.

## Canonical inverse-clamp and round-trip

| split | channel | + clamp | - clamp | total clamp | round-trip rel L2 | clamped error contribution to all-channel error² |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | Bcc1 | 0 | 0 | 0 | 1.35172e-07 | 0 |
| train | Bcc2 | 0.000469971 | 0.00062561 | 0.00109558 | 0.0785729 | 6.9117e-06 |
| train | Bcc3 | 0.120101 | 0.118619 | 0.238719 | 0.982286 | 0.154877 |
| train | rho | 0 | 0 | 0 | 2.13195e-07 | 0 |
| train | press | 0 | 0 | 0 | 1.41643e-07 | 0 |
| train | vel1 | 0 | 0 | 0 | 1.67304e-08 | 0 |
| train | vel2 | 0.019338 | 0.018103 | 0.0374411 | 0.409533 | 0.000926893 |
| train | vel3 | 0.223144 | 0 | 0.223144 | 0.819729 | 0.844189 |
| validation | Bcc1 | 0 | 0 | 0 | 2.85839e-07 | 0 |
| validation | Bcc2 | 0.0677874 | 0.0366993 | 0.104487 | 0.917466 | 0.00358839 |
| validation | Bcc3 | 0.283865 | 0.281023 | 0.564888 | 0.99935 | 0.983241 |
| validation | rho | 0 | 0 | 0 | 2.84177e-07 | 0 |
| validation | press | 0 | 0 | 0 | 3.10459e-07 | 0 |
| validation | vel1 | 0.000106812 | 0 | 0.000106812 | 0.00228018 | 2.91947e-07 |
| validation | vel2 | 0.0200624 | 0.019256 | 0.0393185 | 0.581782 | 1.55983e-05 |
| validation | vel3 | 0.394723 | 1.50681e-05 | 0.394738 | 0.851449 | 0.0131545 |

## Bcc3 and vel3 concentration

### Bcc3

Train/validation clamp fractions are `0.238719` / `0.564888`.
- train: equatorial/polar/inner enrichment = `0.8292` / `1.112` / `3.641`; top-5 snapshots contain `0.09068` of clamp hits.
- validation: equatorial/polar/inner enrichment = `0.8679` / `1.097` / `1.766`; top-5 snapshots contain `0.2797` of clamp hits.

### vel3

Train/validation clamp fractions are `0.223144` / `0.394738`.
- train: equatorial/polar/inner enrichment = `0.815` / `1.12` / `3.727`; top-5 snapshots contain `0.08096` of clamp hits.
- validation: equatorial/polar/inner enrichment = `0.7062` / `1.243` / `2.427`; top-5 snapshots contain `0.2725` of clamp hits.

## Diagnostic float64 unclamped decode

| split | channel | float32 exact ±gamma | float64 exact ±gamma | diagnostic nonfinite | canonical-vs-diagnostic rel L2 (finite subset) |
| --- | --- | ---: | ---: | ---: | ---: |
| train | Bcc1 | 0 | 0 | 0 | 1.3517199343944142e-07 |
| train | Bcc2 | 0 | 0 | 0 | 0.07857289543519669 |
| train | Bcc3 | 0.126755 | 0.0259338 | 0.0259338 | 0.9480176574974227 |
| train | rho | 0 | 0 | 0 | 2.1319492483450987e-07 |
| train | press | 0 | 0 | 0 | 1.4164301109995315e-07 |
| train | vel1 | 0 | 0 | 0 | 1.6730423706521594e-08 |
| train | vel2 | 0.000387573 | 0 | 0 | 0.40953270434599726 |
| train | vel3 | 0.102062 | 0.0300712 | 0.0300712 | 0.7458721100567521 |
| validation | Bcc1 | 0 | 0 | 0 | 2.8583945653397916e-07 |
| validation | Bcc2 | 0 | 0 | 0 | 0.9174660417069954 |
| validation | Bcc3 | 0.423983 | 0.257661 | 0.257661 | 0.9493097425533648 |
| validation | rho | 0 | 0 | 0 | 2.841770339523064e-07 |
| validation | press | 0 | 0 | 0 | 3.1045931083226575e-07 |
| validation | vel1 | 0 | 0 | 0 | 0.0022801842294307957 |
| validation | vel2 | 0.0027216 | 0.000339699 | 0.000339699 | 0.538188495781145 |
| validation | vel3 | 0.193017 | 0.075318 | 0.075318 | 0.7457281540136241 |

## Gamma sensitivity (diagnostic extensions)

| split | mode | all-channel physical round-trip relative L2 | paper canonical |
| --- | --- | ---: | --- |
| train | gamma_6_paper_canonical | 0.466285 | True |
| train | gamma_8_diagnostic_extension | 0.441281 | False |
| train | gamma_12_diagnostic_extension | 0.395104 | False |
| train | no_soft_clip_diagnostic_extension | 2.02257e-16 | False |
| validation | gamma_6_paper_canonical | 0.526375 | True |
| validation | gamma_8_diagnostic_extension | 0.525525 | False |
| validation | gamma_12_diagnostic_extension | 0.523982 | False |
| validation | no_soft_clip_diagnostic_extension | 7.87842e-16 | False |

The no-soft-clip path is a numerical diagnostic, not an alternative selected by
validation. Canonical training and evaluation remain fixed at gamma=6 and 0.99 gamma.
