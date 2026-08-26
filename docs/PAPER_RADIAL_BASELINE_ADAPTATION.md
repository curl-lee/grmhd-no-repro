# paper_reduced100 radial baseline adaptation

## Decision

The selected Stage D mode is `appendix_literal_press_proxy`.

Selection used train snapshots `11..90` only. The literal candidate was finite and
implemented successfully, so the protocol rule requires retaining it even though the
stronger spherical adaptation has smaller train residuals. Validation diagnostics did
not participate in candidate selection.

## Appendix-literal press proxy

The fitted train-only formula is

```text
U = rho + press
log10(U + epsilon_U) ~= k*r
intercept = 0
```

with physical spherical Kerr--Schild `r`. The fitted values are:

| quantity | value |
| --- | ---: |
| fitted `k` | `-0.0262420271631` |
| paper reference `k` | `-0.0178` |
| `epsilon_U` | `1e-8` |
| train rho share of `U` | `0.986404716921` |
| train press share of `U` | `0.0135952830794` |

The paper value is provenance-only and is not forced onto this dataset. `epsilon_U` is
computed from the minimum positive train `U` using the canonical order-of-magnitude rule.

The paper defines a scalar `U` curve but the reduced model has separate rho and press
outputs. To expose a same-space two-channel baseline without inventing an EOS conversion,
the fitted `U(r)` curve is partitioned using the global train-only rho/press shares above,
then each positive channel is mapped through its frozen canonical paper preprocessor.
This channel partition is an explicit paper adaptation. The no-intercept scalar fit itself
is unchanged.

## Stronger spherical diagnostic candidate

The non-selected candidate fits each positive pre-transform separately:

```text
B_c(r) = k_c*log10(r) + b_c
```

| channel | `k_c` | `b_c` |
| --- | ---: | ---: |
| rho | `-0.0238147504220` | `-2.87040616997` |
| press | `-1.22685502366` | `-2.71939764914` |

It is marked `stronger_spherical_adaptation` and `exact_paper_formula=false`.

## Candidate diagnostics

All residual diagnostics use the canonical normalized rho/press space and an envelope
threshold of `1.5`.

| split | mode | channel | residual std | q0.001--q0.999 span | physical reconstruction L2 | envelope violation |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| train | literal | rho | `1.60194` | `6.33642` | `3.38482` | `0.836655` |
| train | literal | press | `1.03857` | `4.69584` | `3.32761` | `0.437615` |
| train | spherical adapted | rho | `1.05958` | `4.44494` | `0.997696` | `0.199460` |
| train | spherical adapted | press | `0.670415` | `3.42812` | `0.823551` | `0.0527405` |
| validation | literal | rho | `1.45978` | `6.34106` | `0.982327` | `0.419207` |
| validation | literal | press | `1.07026` | `4.77288` | `0.998353` | `0.309267` |

The large literal residuals are reported rather than used to override the preset
selection rule. Full shell means, inner/outer biases, ranges, and deterministic sampled
quantiles are stored in `outputs/paper_reduced100/priors/radial_*.json` and
`radial_comparison.*`.
