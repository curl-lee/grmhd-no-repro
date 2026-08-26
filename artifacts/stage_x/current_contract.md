# CURRENT_REPRO_CONTRACT

Resolved project commit `39dde155093436685c20c4f1d97269a02ae1189d`; pinned upstream `86a8bc7812a31b42c4f7895693cf4ac11521c066`. This is the actually executed Stage-T target, reconstructed from the checkpoint/config/forward path rather than README claims.

| field | value | status | primary evidence |
|---|---|---|---|
| simulation coordinate system | raw Athena++ `Coordinates=kerr-schild`; project represents it as spherical KS (r,theta,phi) | EXPLICIT_IN_ARTIFACT | `/mnt/d/GRMHD_data/mad98.prim.00000.athdf` attrs; `src/build_regrid_from_athdf.py:650-656` |
| spatial domain | raw r=[1.1,1200], theta=[0,π], phi=[0,2π]; processed r=[1.1,200] | EXPLICIT_IN_ARTIFACT | `artifacts/stage_x/regrid/raw_amr_audit.json` |
| final training resolution | 64_phi × 64_theta × 64_r | EXPLICIT_IN_CONFIG | `configs/stage_s/expanded_localno_p3_residual.yaml:5-14,41-51` |
| input variables | 8 normalized spherical-coordinate fields + 8 physical-r shell channels | EXPLICIT_IN_CODE | config lines 13-39; `scripts/train_stage_s.py:91-128` |
| output variables | 8 normalized residuals Δz, reconstructed as z_t+Δz | EXPLICIT_IN_CONFIG | config lines 64-72 |
| thermal variable definition | Athena++ `press`; not converted to internal energy | EXPLICIT_IN_ARTIFACT | ATHDF `VariableNames`; config line 14 |
| vector component basis | retained as Bcc1:3/vel1:3 spherical-KS coordinate-basis representation; no Cartesian conversion | PROJECT_CONTRACT_WITH_INCOMPLETE_SOURCE_PROVENANCE | `src/build_regrid_from_athdf.py:710-712`; ATHDF attrs |
| positional encoding | upstream positional_embedding=null; only 8 shell one-hots | EXPLICIT_IN_CONFIG_AND_FORWARD | config lines 33-55; `scripts/train_stage_s.py:100-128` |
| shell/radial embedding | 8 one-hot bins in physical spherical r, log-spaced | EXPLICIT_IN_CODE | `src/grmhd/shells.py:12-59` |
| number of snapshots | 212 | EXPLICIT_IN_ARTIFACT | processed HDF5 shape and `artifacts/stage_s/data_manifest.csv` |
| temporal cadence | actual ATHDF Time array (approximately 10 code-time units, not assumed constant) | EXPLICIT_IN_ARTIFACT | processed `times` and Stage-S manifest |
| train/validation split | snapshots [0,169) train and [169,212) validation; pair 168→169 dropped | EXPLICIT_IN_CONFIG | config lines 8-12; `artifacts/stage_s/train_val_split.json` |
| preprocessing transforms | P3: canonical except Bcc2/Bcc3/vel3 omit softclip; no press→eint conversion | EXPLICIT_IN_CONFIG | config lines 16-31 |
| normalization | train-only expanded P3 normalizer, gamma6 and inverse 0.99γ where applicable | EXPLICIT_IN_CONFIG | config lines 16-31 |
| architecture | upstream 3D LocalNO differential proxy | EXPLICIT_IN_CONFIG | config lines 41-64 |
| LocalNO layer composition | SpectralConv + FiniteDifferenceConv + linear skip + GELU | EXPLICIT_IN_CODE | pinned `local_no_block.py:273-349,466-481` |
| Fourier branch | enabled, 8×8×8 modes | EXPLICIT_IN_CONFIG | config lines 47,56 |
| differential branch | enabled, four 3×3×3 mixed-derivative blocks | EXPLICIT_IN_CONFIG | config lines 49,53,57-58 |
| DISCO/local integral branch | disabled/unavailable for 3D in pinned upstream | EXPLICIT_IN_CONFIG_AND_CODE | config lines 54-55; `src/grmhd/models.py:127-147`; upstream `local_no_block.py:211-215` |
| number of modes | [8,8,8] | EXPLICIT_IN_CONFIG | config line 47 |
| width | 16 | EXPLICIT_IN_CONFIG | config line 48 |
| depth | 4 | EXPLICIT_IN_CONFIG | config line 49 |
| training epochs | Stage-T full-long ran 1200 epochs | EXPLICIT_IN_CONFIG_AND_CHECKPOINT | `configs/stage_t/optimization_convergence.yaml:14-19`; epoch_1200 checkpoint |
| loss composition | PlainL2Loss on normalized residual only | EXPLICIT_IN_CONFIG | config lines 66-80 |
| rollout procedure | adapted spherical 64^3 autoregressive evaluation; no live coarse solver | EXPLICIT_IN_CODE_AND_ARTIFACT | `artifacts/stage_t/STAGE_T_REPORT.md` and evaluation outputs |
| coarse/fine coupling | not implemented | EXPLICIT_ABSENCE | no Stage-S/T coupling path; README limitation |

`CURRENT_COORDINATE_INFORMATION = [eight log-spaced one-hot bins of physical spherical r]`. There are no raw `r`, `log r`, `theta`, `phi`, Cartesian `x/y/z`, or upstream normalized index-grid channels in the Stage-T forward input.
