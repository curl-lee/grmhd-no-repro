# Stage H H1 spacing audit

- Samples: 5 frozen train pairs and 5 frozen validation pairs.
- States: shared initial, Full best/last, Plain best/last.
- H0/H1 mean amplification: 4096.
- H2/H0 mean: 1; maximum absolute parity deviation:
  0.
- Dominant H0 model-error direction: `r`.
- Direction means: `{'phi': 17.164019245773396, 'theta': 3540.4579953002926, 'r': 4022.217426757813}`.

H0 is exactly the frozen Stage E computational-grid seminorm. H1 holds the
uniform-voxel reduction and periodic centered stencil fixed while setting cell
spacing to one. H2 explicitly restates the upstream `1/N` spacing. H3 uses the
stored `r` or `log(r)` centers with open theta/r boundaries. H4 is only a
spherical scalar-metric proxy; the all-channel form is not covariant for stored
magnetic or velocity components. Volume weighting is
`r^2 sin(theta) dphi dtheta dr`, not a verified Kerr--Schild proper volume.

No Trainer, optimizer, scheduler, checkpoint write, or parameter update was
used. Detailed per-state, per-sample, per-channel, and per-direction values are
in `h1_spacing_audit.json`.
