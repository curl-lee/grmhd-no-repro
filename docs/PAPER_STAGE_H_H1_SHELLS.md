# Stage H shell and radial H1 decomposition

## Attribution

The frozen eight spherical-r shells are used unchanged. Each periodic centered
derivative density is assigned to the shell containing its evaluation cell;
this is the documented central-cell midpoint attribution. Shells 0--7,
inner/middle/outer groups, and angular partitions each reconstruct full H0
within floating-point tolerance.

Angular regions are:

- equatorial: `|theta-pi/2| <= pi/6`;
- polar caps: `theta <= pi/6` or `theta >= 5pi/6`;
- remaining: their complement.

## Radial concentration

Mean model-error H0 contribution fractions for trained states:

| Region | Full | Plain |
| --- | ---: | ---: |
| Shell 0 | 0.2542 | 0.2659 |
| Shell 1 | 0.2436 | 0.2309 |
| Shell 2 | 0.1987 | 0.1808 |
| Shell 3 | 0.0764 | 0.0679 |
| Shell 4 | 0.0714 | 0.0715 |
| Shell 5 | 0.0556 | 0.0604 |
| Shell 6 | 0.0477 | 0.0512 |
| Shell 7 | 0.0524 | 0.0713 |
| Inner two | 0.4978 | 0.4968 |
| Middle four | 0.4021 | 0.3806 |
| Outer two | 0.1001 | 0.1225 |

Across shared and trained states, the inner two shells contain `51.02%` of H0
with `25%` of voxels, for enrichment `2.04x`.

Full/Plain inner per-voxel H0 is `1182.0/1317.6`; their inner
prediction-gradient fractions are `0.55596/0.55571`. Full therefore does not
have the larger absolute inner-shell H1 on these fixed samples. The adaptation
mismatch is supported by concentration and coordinate scaling, not by that
specific Full-greater-than-Plain hypothesis.

## Direction, channel, and angular structure

- Direction fractions: `r=0.5518`, `theta=0.4449`, `phi=0.00326`.
- Dominant channels: `Bcc3`, `Bcc2`, `vel2`, `vel3`.
- Full angular fractions: equatorial `0.4550`, polar `0.2985`, remaining
  `0.2464`.
- Plain angular fractions: equatorial `0.4386`, polar `0.3149`, remaining
  `0.2464`.

## Mask and prior overlap

Mean fractions of global model-error H0 lying inside diagnostic masks:

| Mask | H1 overlap |
| --- | ---: |
| Target preprocessing clamp | 0.5977 |
| Canonical velocity ROI | 0.3011 |
| Radial-envelope violation | 0.8358 |

These are spatial overlaps, not causal attributions. In particular, envelope
violation and H1 can both concentrate around steep radial structure without
one causing the other.
