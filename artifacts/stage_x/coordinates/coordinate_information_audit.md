# Coordinate information audit

## Paper

The simulation and network tensor are Cartesian Kerr–Schild. The default model receives eight one-hot shells based on logarithmic Euclidean distance to the Cartesian index-grid centre (Appendix C.4). The paper's normalized Cartesian Fourier features `(xi,eta,zeta)` are an **ablation** (Appendix D.1), not evidence that the default network consumes continuous Cartesian coordinates.

`PAPER_COORDINATE_INFORMATION = [8 Cartesian-index radial shell one-hots]` for the default model.

## Current forward graph

`configs/stage_s/expanded_localno_p3_residual.yaml:33-63` fixes 16 inputs, `positional_embedding: null`, and eight spherical-r shells. `scripts/train_stage_s.py:91-128` constructs shells from the HDF5 `coords/r` array and concatenates only `(z_input, shells)`. `src/grmhd/shells.py:32-59` uses eight log-spaced physical-r bins, broadcast identically over phi/theta.

`CURRENT_COORDINATE_INFORMATION = [8 physical spherical-r shell one-hots]`.

It does not receive actual `phi`, `theta`, continuous `r`, `log r`, normalized tensor indices, or Cartesian position.

## Frozen no-training test

Checkpoint `artifacts/stage_t/full_long/checkpoints/epoch_0150.pt` was evaluated without gradients or updates. The same 5^3 physical stencil was translated to radial indices 12/32/51; response increments were aligned before comparison. Shell-off distances were `[0.0012878595471922385, 0.0011121390182123785, 0.001182601419805851]`. Shell-on distances were `[0.5712350163251952, 1.0098982948529907, 1.1303662747018084]`, at least `444` times the paired shell-off values. The small nonzero shell-off distances are floating FFT/convolution numerical residuals, not an input coordinate channel.

`RADIAL_POSITION_IDENTIFIABILITY = PARTIAL`: the model distinguishes coarse shell membership but not continuous position, theta, or phi.

`COORDINATE_INFORMATION_GAP = HIGH` because coordinate systems and shell definitions differ and neither continuous spherical coordinates nor the paper's Cartesian shell geometry is present.
