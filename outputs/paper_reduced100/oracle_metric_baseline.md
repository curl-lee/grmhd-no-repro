# paper_reduced100 oracle-aware metric baseline

- Status: `passed`
- Split / pairs: `validation` / `19`
- Snapshot pairs: `[91, 92]` through `[109, 110]`
- Device: `cpu`
- Reference semantics: `paper-reference-semantics-v1`
- Gamma / inverse clamp: `6.0` / `0.99`
- Thermal channel: `press` (`press` paper adaptation; unverified EOS conversion disabled)
- Components: native spherical Kerr-Schild coordinate components, not Cartesian vectors
- Reduction: sum squared values over every evaluated pair/voxel within each channel, then take the relative L2 square root
- Channel summary: unweighted arithmetic mean of the eight channel-relative values

## Reference semantics

- `raw_physical_target`: HDF5 target with no canonical round trip.
- `oracle_physical_target`: `decode(encode(raw_physical_target))` under the frozen 0.99-gamma inverse clamp.
- `model_physical_prediction`: `decode(normalized_prediction)` under that same inverse.

## Eight-channel arithmetic averages

| baseline | E_norm | E_model_oracle | E_model_raw | E_oracle_raw floor | excess_absolute |
| --- | ---: | ---: | ---: | ---: | ---: |
| oracle | 0 | 0 | 0.417286886 | 0.417286886 | 0 |
| persistence | 0.189088292 | 0.234101889 | 0.514509437 | 0.417286886 | 0.0972225507 |
| zero_normalized | 1 | 0.968275696 | 0.973288706 | 0.417286886 | 0.55600182 |
| random_upstream_fno | 1.00162165 | 0.968996487 | 0.974522762 | 0.417286886 | 0.557235876 |

The floor is target-side canonical encode/decode loss. `excess_absolute` is a difference of relative norms, not an additive error allocation.

## oracle

| channel | E_norm | E_model_oracle | E_model_raw | E_oracle_raw | excess_absolute |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bcc1 | 0 | 0 | 6.34426387e-07 | 6.34426387e-07 | 0 |
| Bcc2 | 0 | 0 | 0.918700143 | 0.918700143 | 0 |
| Bcc3 | 0 | 0 | 0.999353784 | 0.999353784 | 0 |
| rho | 0 | 0 | 8.78429971e-07 | 8.78429971e-07 | 0 |
| press | 0 | 0 | 9.65840231e-07 | 9.65840231e-07 | 0 |
| vel1 | 0 | 0 | 0.00230182705 | 0.00230182705 | 0 |
| vel2 | 0 | 0 | 0.566810264 | 0.566810264 | 0 |
| vel3 | 0 | 0 | 0.851126595 | 0.851126595 | 0 |

| channel | target clamp | model clamp | intersection | model-only | target-only | Jaccard | precision | recall | sign agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bcc1 | 0 | 0 | 0 | 0 | 0 | NA | NA | NA | NA |
| Bcc2 | 0.104267923 | 0.104267923 | 0.104267923 | 0 | 0 | 1 | 1 | 1 | 1 |
| Bcc3 | 0.573237168 | 0.573237168 | 0.573237168 | 0 | 0 | 1 | 1 | 1 | 1 |
| rho | 0 | 0 | 0 | 0 | 0 | NA | NA | NA | NA |
| press | 0 | 0 | 0 | 0 | 0 | NA | NA | NA | NA |
| vel1 | 0.000112433183 | 0.000112433183 | 0.000112433183 | 0 | 0 | 1 | 1 | 1 | 1 |
| vel2 | 0.0368586088 | 0.0368586088 | 0.0368586088 | 0 | 0 | 1 | 1 | 1 | 1 |
| vel3 | 0.402022111 | 0.402022111 | 0.402022111 | 0 | 0 | 1 | 1 | 1 | 1 |

## persistence

| channel | E_norm | E_model_oracle | E_model_raw | E_oracle_raw | excess_absolute |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bcc1 | 0.0691882924 | 0.105213587 | 0.105213588 | 6.34426387e-07 | 0.105212954 |
| Bcc2 | 0.306599575 | 0.381731666 | 0.922082397 | 0.918700143 | 0.00338225383 |
| Bcc3 | 0.243157803 | 0.247311113 | 0.999357532 | 0.999353784 | 3.74858512e-06 |
| rho | 0.116308718 | 0.315904953 | 0.315904944 | 8.78429971e-07 | 0.315904066 |
| press | 0.129492916 | 0.153771141 | 0.153771144 | 9.65840231e-07 | 0.153770178 |
| vel1 | 0.140437349 | 0.130154636 | 0.130213324 | 0.00230182705 | 0.127911497 |
| vel2 | 0.369471575 | 0.409947629 | 0.637313537 | 0.566810264 | 0.0705032731 |
| vel3 | 0.138050107 | 0.128780384 | 0.852219031 | 0.851126595 | 0.00109243587 |

| channel | target clamp | model clamp | intersection | model-only | target-only | Jaccard | precision | recall | sign agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bcc1 | 0 | 0 | 0 | 0 | 0 | NA | NA | NA | NA |
| Bcc2 | 0.104267923 | 0.103972385 | 0.0961759467 | 0.00779643812 | 0.00809197677 | 0.858220627 | 0.925014338 | 0.922392463 | 0.981763032 |
| Bcc3 | 0.573237168 | 0.560965889 | 0.559359701 | 0.00160618832 | 0.0138774671 | 0.973064566 | 0.997136745 | 0.975791055 | 0.986996518 |
| rho | 0 | 0 | 0 | 0 | 0 | NA | NA | NA | NA |
| press | 0 | 0 | 0 | 0 | 0 | NA | NA | NA | NA |
| vel1 | 0.000112433183 | 2.63013338e-05 | 1.28495066e-05 | 1.34518272e-05 | 9.9583676e-05 | 0.102073365 | 0.488549618 | 0.114285714 | 1 |
| vel2 | 0.0368586088 | 0.039746937 | 0.0329298722 | 0.00681706479 | 0.00392873664 | 0.753963694 | 0.828488299 | 0.89341061 | 0.979849404 |
| vel3 | 0.402022111 | 0.392954575 | 0.388449619 | 0.0045049567 | 0.0135724921 | 0.955531992 | 0.988535681 | 0.966239439 | 0.999922471 |

## zero_normalized

| channel | E_norm | E_model_oracle | E_model_raw | E_oracle_raw | excess_absolute |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bcc1 | 1 | 1 | 1 | 6.34426387e-07 | 0.999999366 |
| Bcc2 | 1 | 0.99967694 | 1.00010065 | 0.918700143 | 0.081400505 |
| Bcc3 | 1 | 1 | 1 | 0.999353784 | 0.000646216424 |
| rho | 1 | 0.999965548 | 0.999965548 | 8.78429971e-07 | 0.999964669 |
| press | 1 | 0.999988908 | 0.999988908 | 9.65840231e-07 | 0.999987942 |
| vel1 | 1 | 0.793082579 | 0.793122629 | 0.00230182705 | 0.790820802 |
| vel2 | 1 | 0.999999985 | 0.999999993 | 0.566810264 | 0.433189729 |
| vel3 | 1 | 0.953491605 | 0.993131926 | 0.851126595 | 0.142005331 |

| channel | target clamp | model clamp | intersection | model-only | target-only | Jaccard | precision | recall | sign agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bcc1 | 0 | 0 | 0 | 0 | 0 | NA | NA | NA | NA |
| Bcc2 | 0.104267923 | 0 | 0 | 0 | 0.104267923 | 0 | NA | 0 | NA |
| Bcc3 | 0.573237168 | 0 | 0 | 0 | 0.573237168 | 0 | NA | 0 | NA |
| rho | 0 | 0 | 0 | 0 | 0 | NA | NA | NA | NA |
| press | 0 | 0 | 0 | 0 | 0 | NA | NA | NA | NA |
| vel1 | 0.000112433183 | 0 | 0 | 0 | 0.000112433183 | 0 | NA | 0 | NA |
| vel2 | 0.0368586088 | 0 | 0 | 0 | 0.0368586088 | 0 | NA | 0 | NA |
| vel3 | 0.402022111 | 0 | 0 | 0 | 0.402022111 | 0 | NA | 0 | NA |

## random_upstream_fno

| channel | E_norm | E_model_oracle | E_model_raw | E_oracle_raw | excess_absolute |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bcc1 | 1.00158848 | 0.999992441 | 0.999992441 | 6.34426387e-07 | 0.999991806 |
| Bcc2 | 1.00585959 | 0.999708186 | 1.00008131 | 0.918700143 | 0.0813811623 |
| Bcc3 | 1.00132525 | 1.00036523 | 0.999999491 | 0.999353784 | 0.000645707281 |
| rho | 0.986228013 | 0.999963829 | 0.999963829 | 8.78429971e-07 | 0.99996295 |
| press | 1.01618435 | 0.999989958 | 0.999989958 | 9.65840231e-07 | 0.999988992 |
| vel1 | 1.01524881 | 0.802973751 | 0.803012164 | 0.00230182705 | 0.800710337 |
| vel2 | 1.00380587 | 1.00194045 | 1.00096816 | 0.566810264 | 0.434157892 |
| vel3 | 0.982732809 | 0.947038052 | 0.992174756 | 0.851126595 | 0.141048161 |

| channel | target clamp | model clamp | intersection | model-only | target-only | Jaccard | precision | recall | sign agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bcc1 | 0 | 0 | 0 | 0 | 0 | NA | NA | NA | NA |
| Bcc2 | 0.104267923 | 0 | 0 | 0 | 0.104267923 | 0 | NA | 0 | NA |
| Bcc3 | 0.573237168 | 0 | 0 | 0 | 0.573237168 | 0 | NA | 0 | NA |
| rho | 0 | 0 | 0 | 0 | 0 | NA | NA | NA | NA |
| press | 0 | 0 | 0 | 0 | 0 | NA | NA | NA | NA |
| vel1 | 0.000112433183 | 0 | 0 | 0 | 0.000112433183 | 0 | NA | 0 | NA |
| vel2 | 0.0368586088 | 0 | 0 | 0 | 0.0368586088 | 0 | NA | 0 | NA |
| vel3 | 0.402022111 | 0 | 0 | 0 | 0.402022111 | 0 | NA | 0 | NA |

## Oracle invariant checks

- Normalized oracle error is zero: `True`.
- Model-to-oracle physical error is zero: `True`.
- Oracle model-to-raw error equals the canonical floor: `True`.

## Squared-error diagnostic

The JSON/CSV record the exact numerator identity `||model-raw||^2 = ||model-oracle||^2 + ||oracle-raw||^2 + 2<model-oracle, oracle-raw>` per channel. Relative norms do not obey this simple additivity, so the report makes no such claim.

The `random_upstream_fno` row is an untrained, seeded random initialization of the fixed upstream FNO. It is a metric-pipeline smoke baseline, not a scientific model.
