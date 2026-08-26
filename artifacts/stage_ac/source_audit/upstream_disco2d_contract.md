# Upstream 2D DISCO Contract Audit

Classification: `UPSTREAM_EXPLICIT` unless otherwise marked.

- Base weight shape is `(out_channels, in_channels/groups, kernel_size)` with scale
  `sqrt(1/groupsize)` and Gaussian initialization; optional bias is zero-initialized.
- For non-Morlet `[2,4]`, `kernel_size=(2-1)*4+1=5`.
- Equidistant 2D cutoff defaults to `max(domain_length[i]/out_shape[i])`.
- Local stencil sizes use `floor(2*R*in_shape[i]/domain_length[i])+1`.
- Quadrature is `L0*L1/(N0*N1)` and the basis is discretely normalized.
- The dense local buffer has shape `(K, psi_local_h, psi_local_w)` and is nonpersistent.
- Kernel synthesis is `einsum('kxy,ogk->ogxy', basis, weight)` after the upstream
  orientation permutation/flip.
- Efficient evaluation maps to grouped `conv2d`; integer stride implements resolution scaling.
- `periodic` sets a padding-mode attribute, but this pinned forward passes integer padding
  directly to `conv2d` and does not itself call circular `pad`. This implementation detail is
  explicit in the pinned source.
- Groups and bias follow PyTorch grouped-convolution semantics.

Evidence: `external/neuraloperator/neuralop/layers/discrete_continuous_convolution.py`,
class `DiscreteContinuousConv` and class `EquidistantDiscreteContinuousConv2d`.
