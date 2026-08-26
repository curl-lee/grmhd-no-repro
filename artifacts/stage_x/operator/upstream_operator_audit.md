# Pinned-upstream LocalNO audit

Pinned commit: `86a8bc7812a31b42c4f7895693cf4ac11521c066` (clean submodule worktree during audit).

- `external/neuraloperator/neuralop/models/local_no.py:24-29` defines LocalNO blocks as Fourier layers with differential and local-integral kernels in parallel.
- `local_no.py:51-69,190-205` makes DISCO and differential branches independently selectable and defaults both to true.
- `local_no_block.py:273-349` constructs spectral, differential, and local-convolution modules.
- `local_no_block.py:466-481` adds enabled spectral, differential, and local-convolution outputs in the forward pass.
- `local_no_block.py:211-215` rejects DISCO when spatial dimension is not 2.
- `discrete_continuous_convolution.py:271-299` implements only `DiscreteContinuousConv2d`; it evaluates a learnable continuous local kernel semi-discretely on a 2D grid.
- `src/grmhd/models.py:127-147` therefore constructs the 3D project proxy with `disco_layers=False, diff_layers=True` and rejects volumetric DISCO.

The pinned package's generic/canonical LocalNO mechanism is `Spectral + differential + local integral`, but the paper's exact per-layer choices cannot be reconstructed from the paper. The **current resolved model** is unambiguously `SpectralConv + FiniteDifferenceConv + linear skip`; it has no local integral branch.
