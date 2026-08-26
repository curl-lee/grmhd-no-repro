# Stage W Spectral Source Audit

## Pinned implementation

- Upstream: `external/neuraloperator@86a8bc7812a31b42c4f7895693cf4ac11521c066`; worktree clean at audit time.
- Exact class: `neuralop.layers.spectral_convolution.SpectralConv` in `external/neuraloperator/neuralop/layers/spectral_convolution.py`; `forward` begins at line 417.
- Local model: `neuralop.models.local_no.LocalNO` in `external/neuraloperator/neuralop/models/local_no.py`.
- Resolved model has 358,296 trainable parameters and four spectral/differential blocks.

## Transform and modal contract

`SpectralConv.forward` obtains all last `order=3` spatial dimensions and calls
`torch.fft.rfftn(x, norm=fft_norm, dim=[-3,-2,-1])`.  The project tensor is
`[B,C,Nphi,Ntheta,Nr]`, hence:

```text
SPECTRAL_AXES = phi index, theta index, r index
REQUESTED_N_MODES = 8 x 8 x 8
STORED_N_MODES = [8, 8, 5]
```

The last real-FFT direction stores `8//2+1=5` radial coefficients.  `fftshift`
centers the full phi and theta spectra, then the layer retains the centered
8-by-8 window and the first five nonredundant r modes.  The inverse performs
IFFT on phi/theta, enforces real zero/Nyquist coefficients on the real-FFT
axis, and then applies radial IRFFT.  Consequently Stage T is exactly a
periodic Fourier representation of all three tensor-index axes.  This is
physically appropriate for phi but imposes periodic spectral continuation on
theta and r.

## Learned contraction

- Per-layer learned complex weight shape: `[16, 16, 8, 8, 5]` with dense input/output-channel mixing.
- Factorization request/resolution: `factorization=None`; this resolves to an unfactorized Dense `FactorizedTensor`.  `implementation=factorized`, `separable=False`.
- FFT normalization: `fft_norm=forward` (the inverse uses the matching normalization).
- Bias: enabled, real shape `[16, 1, 1, 1]` per layer.
- Normalization layer: none; channel MLP: disabled; domain padding: none.

## Block, skip, and positional interactions

Each block adds spectral output, the original upstream 3x3x3 finite-difference
output, and no DISCO output.  A learned linear 1x1 local-NO skip is then added,
followed by GELU except after the final layer.  Lifting and projection are
two-layer channel MLPs.  `positional_embedding=null`, so no raw grid, r,
log-r, theta, phi, or metric coordinate is appended.  The existing eight
shell channels remain the only coarse radial-region input.

## Stage W isolated change

W1 replaces only the spectral convolution with `rFFT_phi x DCT-II_theta x
DCT-II_log-r`; the upstream differential branch remains circular on every
axis.  W2 additionally applies the Stage-U-predeclared boundary policy
`circular_phi/replicate_theta/replicate_r` to the identical 3x3x3 stencil and
identical scalar grid width.  No coordinate scaling or singular metric proxy
is introduced.  The stored log-r uniformity result is
`LOG_R_GRID_UNIFORM=true`.

```text
MIXED_BASIS_IS_NOT_SPHERICAL_HARMONICS = true
MIXED_BASIS_IS_NOT_KERR_SCHILD_COVARIANT = true
```

All eight output fields remain component-wise scalar feature maps; Bcc and
velocity components are not converted to vector harmonics.
