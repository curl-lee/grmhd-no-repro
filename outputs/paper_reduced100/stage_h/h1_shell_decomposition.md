# Stage H H1 shell/radial decomposition

- Inner-two-shell H1 contribution: 0.510206
  for voxel fraction 0.25;
  enrichment 2.04082.
- Full inner contribution mean: 0.4978.
- Plain inner contribution mean: 0.496844.
- Full/Plain inner per-voxel H1:
  1182.03 /
  1317.61.
- Full/Plain inner prediction-gradient fraction:
  0.555957 /
  0.55571.
- Dominant direction: `r` with fractions
  `{'phi': 0.0032608840648682023, 'theta': 0.444938462972641, 'r': 0.5518006497621536}`.
- Dominant channels, descending: `['Bcc3', 'Bcc2', 'vel2', 'vel3', 'vel1', 'Bcc1', 'press', 'rho']`.
- Global H1 overlap means: `{'target_clamp_h1_overlap': 0.597715482711792, 'roi_h1_overlap': 0.3010866007208824, 'envelope_violation_h1_overlap': 0.8358468031883239}`.

Derivative ownership uses central-cell midpoint attribution. Shells 0--7,
inner/middle/outer groups, and the three angular regions independently add to
the full H0 within tolerance. Gradient-norm fractions are prediction-space
diagnostics. No optimizer, Trainer, checkpoint write, or model update was
used.
