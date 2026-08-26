# Stage AE Contract

- `REPRODUCTION_SCOPE = ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION`
- sole intervention: index-space DISCO fixed geometry -> spherical-coordinate-aware Euclidean embedding proxy
- frozen Z64 dataset SHA256: `3582a5c4f52f4fef305578ffd5a63c51423f8583f35d137c288d5c20203b50da`
- K=5, radius multiplier=3, candidate stencil=[7,7,7]
- phi periodic; theta/r truncated and per-target renormalized
- width/modes/layers, spectral/differential branches, trainable DISCO shapes,
  initialization, P3, split, residual target, Plain L2, optimizer, scheduler,
  pair order, and 300-epoch maximum budget are frozen
- upstream `86a8bc7812a31b42c4f7895693cf4ac11521c066` is pinned and clean
