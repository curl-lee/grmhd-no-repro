# Stage H H1 spacing audit

## Frozen scope

The audit used the shared initial state, Full best/last, and Plain best/last in
evaluation mode on five fixed train and five fixed validation pairs. Batch
size was one and mixed precision was disabled. Every model tensor-state hash
was identical before and after the audit.

H0 is the exact Stage E/Stage G gradient seminorm. H1 keeps its periodic
centered stencil and uniform-voxel reduction but changes derivative spacing to
one. H2 states the upstream `1/N` spacing explicitly. H3 uses the stored `r` or
`log(r)` coordinates. H4 is a non-covariant spherical scalar-metric proxy.

## Global ratios

The following values are mean diagnostic/H0 ratios for model error across all
five states.

| Variant | Reduction | Train | Validation |
| --- | --- | ---: | ---: |
| H0 current upstream | uniform | 1 | 1 |
| H1 unit index | uniform | 0.000244141 | 0.000244141 |
| H2 normalized axis | uniform | 1 | 1 |
| H3 stored r | uniform | 0.04420 | 0.05414 |
| H3 stored r | volume proxy | 0.006916 | 0.005428 |
| H3 stored log-r | uniform | 0.06130 | 0.06712 |
| H3 stored log-r | volume proxy | 0.01066 | 0.007396 |
| H4 naive all-channel | uniform | 0.01030 | 0.02892 |
| H4 naive all-channel | volume proxy | 1.33e-6 | 1.49e-6 |
| H4 rho/press scalar-only | uniform | 0.000170 | 0.000661 |
| H4 rho/press scalar-only | volume proxy | 2.75e-7 | 1.77e-7 |

H0/H1 is exactly `4096 = 64^2` for every state, sample, split, and error-field
case. H2/H0 is bitwise one. The result follows directly from changing a
first derivative from `Delta=1` to `Delta=1/64` while holding the
uniform-voxel mean fixed: derivative energy scales by `64^2`.

H3 and H4 change coordinate units, boundary semantics, and sometimes the
spatial measure. Their ratios quantify sensitivity; they are not evidence
that an alternative is a better training objective.

## Train/validation and Full/Plain structure

The stored-coordinate result is stable across trained states:

| State | H3 stored-r/H0 | H3 log-r/H0 | H4 naive/H0 |
| --- | ---: | ---: | ---: |
| Full best | 0.05370 | 0.07079 | 0.02259 |
| Full last | 0.05370 | 0.07079 | 0.02258 |
| Plain best | 0.05137 | 0.06831 | 0.02094 |
| Plain last | 0.05135 | 0.06828 | 0.02093 |

For validation model error, H0 direction fractions are approximately:

| State | phi | theta | r | H0 total |
| --- | ---: | ---: | ---: | ---: |
| Full best | 0.0059 | 0.4824 | 0.5117 | 6137.17 |
| Full last | 0.0059 | 0.4824 | 0.5117 | 6137.29 |
| Plain best | 0.0052 | 0.4606 | 0.5342 | 6730.71 |
| Plain last | 0.0052 | 0.4606 | 0.5342 | 6730.35 |

`Bcc3`, `Bcc2`, and `vel2` are the largest channel contributors. Plain has a
larger absolute H0 on these fixed validation samples even though its frozen
Stage G morphology/stability scores were better. Therefore H1 magnitude alone
does not explain the Full-versus-Plain outcome.

## Interpretation

Observed: the upstream unit-cube spacing creates an exact 4096 multiplicative
energy scale relative to unit-index derivatives, and stored-coordinate
variants are roughly 15--23 times smaller under uniform reduction. Theta and
r dominate, while phi is negligible.

Inference: treating `(phi,theta,r)` as equally normalized periodic axes is a
material spherical-adaptation mismatch.

Unsupported: these post-hoc values do not identify a scientifically correct
replacement loss, and H4 is neither a vector covariant derivative nor a
Kerr--Schild proper-volume calculation.
