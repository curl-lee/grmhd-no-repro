# Raw AMR and diagnostic regrid audit

## Source/grid contract

- Raw directory: `/mnt/d/GRMHD_data`; 20 fixed source files are enumerated in `raw_amr_audit.json`.
- Fixed field snapshots: `[0, 25, 50, 75, 100, 125, 150, 175, 200, 211]`.
- Fixed adjacent-pair sources: `[0, 25, 50, 75, 100, 125, 150, 175, 200, 210]`.
- All selected grid signatures equal `438534dadda3395bfac3365cc2b0a02bef3c05f7804c9f8a497afa68feab4769`: `GRID_LAYOUT_STATIC = true`.
- ATHDF `Coordinates=kerr-schild`; numeric axes establish `(x1,x2,x3)=(r,theta,phi)`.
- Root grid `(r,theta,phi)=[88, 32, 16]`, MeshBlock `[22, 4, 16]`, max level `3`, `2020` leaf blocks and `2844160` actual leaf cells.
- All-domain-equivalent finest resolution is `[704, 256, 128]`; this is not the actual number of cells.
- Bounds: r `[1.100000023841858, 1200.0]`, theta `[0.0, 3.1415927410125732]`, phi `[0.0, 6.2831854820251465]`. Stage X samples only r=`[1.100000023841858, 200.0]` to match the production tensor.

| AMR level | leaf blocks | leaf cells | dr range | dtheta range | dphi range |
|---:|---:|---:|---|---|---|
| 0 | 4 | 5632 | [17.274246215820312, 91.6907958984375] | [0.09817469120025635, 0.0981748104095459] | [0.3926987648010254, 0.3926992416381836] |
| 1 | 96 | 135168 | [0.044597625732421875, 46.75634765625] | [0.0490872859954834, 0.0490875244140625] | [0.1963491439819336, 0.1963496208190918] |
| 2 | 896 | 1261568 | [0.022077202796936035, 23.6104736328125] | [0.02454352378845215, 0.02454376220703125] | [0.0981745719909668, 0.098175048828125] |
| 3 | 1024 | 1441792 | [0.36278533935546875, 11.8638916015625] | [0.012271642684936523, 0.012271881103515625] | [0.0490870475769043, 0.0490875244140625] |

Radial spacing ranges are not monotone in level because levels occupy different radial bands on a geometric source grid; level number alone must not be compared without position.

## Diagnostic sampling contract

Every 32/64/96/128 cube uses the exact production `finest-leaf nearest-cell-center` mapping (`src/build_regrid_from_athdf.py:265-314,446-457`). The independently generated audited 64^3 arrays were bitwise equal to `data_proc/grmhd_regrid_inner_r200_64_expanded.h5` for all 80 checked snapshot/channel arrays. The 128^3 grid is `HIGHER_RES_SAMPLING_REFERENCE`, not raw-AMR truth.

| resolution | median field relative L2 | median increment relative difference | median increment cosine |
|---:|---:|---:|---:|
| 32 | 0.158747 | 0.654077 | 0.781718 |
| 64 | 0.160646 | 0.503429 | 0.871549 |
| 96 | 0.0984533 | 0.378347 | 0.926872 |
| 128 | 0 | 0 | 1 |

At 64^3, detailed channel medians are:

| channel | field rel-L2 | increment rel-difference | increment cosine | shell-increment rel-L2 | radial-increment rel-L2 |
|---|---:|---:|---:|---:|---:|
| Bcc1 | 0.0856162 | 0.299982 | 0.953646 | 1.35543 | 1.41412 |
| Bcc2 | 0.307803 | 0.537001 | 0.844421 | 0.244473 | 0.370095 |
| Bcc3 | 0.153901 | 0.435156 | 0.897095 | 0.532954 | 0.807268 |
| rho | 0.341013 | 0.538795 | 0.850374 | 0.0702244 | 0.297245 |
| press | 0.39045 | 0.545322 | 0.841774 | 0.137217 | 0.192607 |
| vel1 | 0.0726263 | 0.467712 | 0.888906 | 0.0577901 | 0.177824 |
| vel2 | 0.22 | 0.331678 | 0.942846 | 0.933794 | 0.874739 |
| vel3 | 0.0833786 | 0.594394 | 0.812598 | 0.0788352 | 0.169374 |

Inner/middle/outer field loss medians are `{'inner': 0.14971942552350298, 'middle': 0.08266151386917865, 'outer': 0.08918616544353217}`; temporal-increment loss medians are `{'inner': 0.44360202273396504, 'middle': 0.3644246358208748, 'outer': 0.42273226636403505}`. The inner increment loss is largest, although outer loss is also substantial.

`TEMPORAL_INCREMENT_REGRID_FIDELITY = POOR` under the thresholds frozen before the audit in `configs/stage_x/method_gap_audit.yaml`. The overall 64^3 medians are relative difference `0.503429` and cosine `0.871549`. Full snapshot/channel/pair rows, shell vectors, radial profiles, spectra, extrema and q001/q999 are in the four CSV tables.

The frozen rule is GOOD when median relative difference <=0.25 and cosine >=0.95; MODERATE when <=0.50 and >=0.80; otherwise POOR. The 64^3 cosine clears MODERATE, but relative difference `0.503429` does not.

## Conservation caveat

The mapping is non-conservative and not divergence preserving (`src/build_regrid_from_athdf.py:693-712`). A plain Cartesian divergence is not the GRMHD magnetic constraint in this representation; `STRICT_DIVB_AUDIT_NOT_AUTHORIZED = true`.
