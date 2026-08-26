# Stage U Operator Source Audit

## Frozen contract

```text
DATASET_FROZEN = true
SPLIT_FROZEN = true
PREPROCESSING_FROZEN = true
TARGET_CONTRACT_FROZEN = true
LOSS_FROZEN = true
```

- Dataset: `data_proc/grmhd_regrid_inner_r200_64_expanded.h5` (`3582a5c4f52f4fef305578ffd5a63c51423f8583f35d137c288d5c20203b50da`)
- P3 normalizer: `artifacts/stage_s/p3_normalizer_expanded/normalizer.npz` (`c2a36edbb44732efb857165e9a710d393cec68c6ba285521cd9bcc7b783cffa6`)
- Split: train pair sources `0..167`, dropped `168->169`, validation pair sources `169..210`.
- Target/reconstruction: `z[t+1]-z[t]`, then `z[t]+model_residual`; loss is unmodified `PlainL2Loss`.
- Seed/width/modes/layers: `42` / `16` / `[8, 8, 8]` / `4`.
- Stage T optimization horizon: 1200 epochs, 50,400 updates, 3,150 warmup updates. Stage U pilots stop at epoch 150 but evaluate the same scheduler's first 6,300 updates.

## Spectral branch

- Evidence: `external/neuraloperator/neuralop/layers/spectral_convolution.py:429-449` obtains the last three spatial axes and calls `torch.fft.rfftn(..., dim=[-3,-2,-1])`. Model tensors are `(batch, channel, phi, theta, r)`, so FFT axes are `phi, theta, r`.
- Config requests `n_modes=[8,8,8]`. Source lines 400-415 convert the real-FFT last-axis storage to `[8,8,5]`; this corresponds to 8 total requested modes per axis.
- A standard Fourier basis makes a periodic-extension/translation-invariant index-space assumption on every transformed axis. This is appropriate at the phi seam, but is not a physical boundary model for theta or r.
- `domain_padding=None` because Stage T passes none and `LocalNO` source lines 291-300 disables it. No spectral domain padding/unpadding is active.

## Differential branch

- Evidence: `external/neuraloperator/neuralop/layers/differential_conv.py:6-101` explicitly documents a regular grid and implements a bias-free `Conv3d` followed by coefficient-sum centering and division by one `grid_width` scalar.
- Resolved kernel is `3x3x3`, groups=1 (`mix_derivatives=true`), four layers, each weight shape `(16,16,3,3,3)`.
- `conv_padding_mode=periodic` maps to PyTorch `circular`; it is applied identically on phi, theta, and r (`differential_conv.py:59-79`).
- `local_no_block.py:468-471` computes `grid_width_scaling_factor = 1/(x.shape[-1]/default_in_shape[0])`. At frozen 64^3, this is exactly `1.0`, derived only from the last tensor size, and reused for all three axes.
- It does not read stored `r/theta/phi`, does not know `dr` varies by a factor `154.534`, and does not use `1/r` or `1/(r sin(theta))`.

## Positional and boundary information

- `positional_embedding=null`, so `GridEmbeddingND` is disabled (`local_no.py:267-300,429-444`). The network receives no continuous r, theta, or phi coordinate channels.
- The extra eight inputs are one-hot radial shell memberships constructed from physical r in `scripts/train_stage_s.py:90-96`. They provide coarse radial region identity, not continuous coordinates, theta, or phi.
- Physical data policy: phi is periodic; theta spans pole-to-pole without a periodic seam; r is bounded and nonperiodic.
- Model policy: all three differential axes are circular. Therefore:

```text
PHI_PERIODIC = true
THETA_PERIODIC_IN_MODEL = true
R_PERIODIC_IN_MODEL = true
```

The last two are operator assumptions inconsistent with the stored spherical-domain boundaries. The FFT branch likewise uses periodic Fourier basis functions along all three tensor axes; this is reported as an assumption mismatch, not as a claim that FFT is intrinsically wrong.

## Provenance

- Project branch at audit: `codex/stage-r-residual-localno-pilot`.
- Pinned upstream: `86a8bc7812a31b42c4f7895693cf4ac11521c066`, upstream worktree clean.
- `CARTESIAN_REMAP_NOT_AUTHORIZED`: Bcc1/2/3 and vel1/2/3 remain spherical Kerr-Schild coordinate-basis channel names.
