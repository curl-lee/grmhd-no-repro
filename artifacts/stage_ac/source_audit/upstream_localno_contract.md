# Upstream LocalNO Contract Audit

Classification: `UPSTREAM_EXPLICIT`.

- Pinned `LocalNOBlocks` expands scalar `disco_layers` and `diff_layers` across all layers.
- Enabled differential kernels support up to three dimensions.
- Enabled local-integral DISCO is rejected unless `len(n_modes)==2`.
- The pinned implementation constructs `EquidistantDiscreteContinuousConv2d` branches.
- Post-activation ordering is spectral, differential, local integral, branch sum, optional norm,
  local skip addition, activation, then optional channel MLP/normalization.
- The branch equation is `x_spectral + x_differential + x_local_integral` before skip.

Evidence: `external/neuraloperator/neuralop/layers/local_no_block.py`, constructor checks,
local-convolution construction, and `forward_with_postactivation`.
