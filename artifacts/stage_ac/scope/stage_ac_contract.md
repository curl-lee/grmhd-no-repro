# Stage AC Contract

- `REPRODUCTION_SCOPE = ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION`
- `MODEL_NAME = ADAPTED_VOLUMETRIC_3D_DISCO_LOCALNO`
- `DISCO3D_IMPLEMENTATION_CLASS = ADAPTED_ISOTROPIC_RADIAL_PIECEWISE_LINEAR`
- `DISCO3D_DISTANCE = NORMALIZED_COMPUTATIONAL_INDEX_DISTANCE`
- `DISCO3D_BOUNDARY = COMPUTATIONAL_PERIODIC_ADAPTATION`
- resolution: 64 x 64 x 64
- dataset: `data_proc/grmhd_regrid_inner_r200_64_expanded.h5`
- dataset SHA256: `3582a5c4f52f4fef305578ffd5a63c51423f8583f35d137c288d5c20203b50da`
- P3 normalizer SHA256: `c2a36edbb44732efb857165e9a710d393cec68c6ba285521cd9bcc7b783cffa6`
- split: 168 train pairs, dropped 168->169, 42 validation pairs
- target: plain normalized residual
- frozen Stage-T scheduler: 75 warmup epochs in the 1200-epoch schedule
- upstream commit: `86a8bc7812a31b42c4f7895693cf4ac11521c066`; worktree clean

No data, preprocessing, split, loss, boundary, width, modes, or pinned upstream source was changed.
