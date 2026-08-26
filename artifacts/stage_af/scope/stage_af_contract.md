# Stage AF Contract

- Sole scientific intervention: scalar isotropic spherical local scale -> anisotropic local spherical tangent geometry.
- Frozen dataset: `data_proc/grmhd_regrid_inner_r200_64_expanded.h5` (`3582a5c4f52f4fef305578ffd5a63c51423f8583f35d137c288d5c20203b50da`).
- K=5, radius multiplier=3, candidate box=7x7x7.
- Phi is periodic; theta/r are truncated and per-target renormalized.
- No loss, preprocessing, resolution, width, modes, depth, split, initialization, or upstream code is changed.
- `NO_MODEL_TRAINING = true`; `NO_OPTIMIZER_STEP = true`.
