# Stage AG Frozen Training Contract

- training only; no Stage AD/AG scientific comparison or rollout
- dataset `data_proc/grmhd_regrid_inner_r200_64_expanded.h5` SHA256 `3582a5c4f52f4fef305578ffd5a63c51423f8583f35d137c288d5c20203b50da`
- chronological train pairs 0..167; drop 168->169; validation pairs 169..210
- frozen expanded-data train-only P3 normalizer; normalized residual Plain L2
- anisotropic spherical DISCO LocalNO, 363480 parameters, K=5, 7x7x7, multiplier=3
- 300 epochs, 168 microbatches/epoch, accumulation=4, 42 updates/epoch
- Adam lr=1e-3, weight decay=1e-4, global clip=1, AMP=false
- frozen 1200-epoch Stage T scheduler horizon: 50400 updates, warmup=3150
- no early stopping and no validation-driven modification
