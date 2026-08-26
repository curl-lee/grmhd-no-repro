# Paper-adapted Stage G: Full versus Plain

## Table-2-style validation

These are normalized-domain relative L2 values on the 19 reduced100 validation
pairs. They are a 30-epoch resource-scaled FNO-proxy result and are not
directly comparable to the paper's Table 2.

| Model | Bcc1 | Bcc2 | Bcc3 | rho | press | vel1 | vel2 | vel3 | Average | Global |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Canonical oracle | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| Persistence | 0.0692 | 0.3066 | 0.2432 | 0.1163 | 0.1295 | 0.1404 | 0.3695 | 0.1381 | 0.1891 | 0.2351 |
| Full FNO proxy | 0.5055 | 0.4585 | 0.3246 | 0.5900 | 0.5127 | 0.6369 | 0.4940 | 0.1800 | 0.4628 | 0.3873 |
| Plain L2 FNO | 0.4645 | 0.4377 | 0.2789 | 0.6266 | 0.6303 | 0.6391 | 0.5104 | 0.2091 | 0.4746 | 0.3816 |

Full has the lower arithmetic average, chiefly through `rho`, `press`, and
velocity channels. Plain has the lower global value and lower error on all
three magnetic channels. Both learned models are worse than persistence on
this one-step reduced validation metric, so the pilot is not evidence of
paper-level predictive performance.

## Oracle-aware one-step metrics

| Metric | Full average/global | Plain average/global |
| --- | ---: | ---: |
| Model-to-oracle | 0.6981 / 0.9698 | 0.6998 / 0.9744 |
| Model-to-raw | 0.8710 / 0.9776 | 0.8827 / 0.9810 |
| Oracle floor | 0.4173 / 0.5264 | 0.4173 / 0.5264 |

The physical oracle floor is especially severe for `Bcc3` and `vel3`
(`0.9994` and `0.8511` relative L2 respectively), and also affects `Bcc2` and
`vel2`. Model-to-raw error must therefore not be interpreted without
model-to-oracle and oracle-floor columns.

## Morphology and boundary comparison

Lower is better for the frozen comparison scores.

| Category | Full | Plain | Better |
| --- | ---: | ---: | --- |
| Center morphology | 0.9120 | 0.9050 | Plain |
| Polar morphology | 0.8803 | 0.8357 | Plain |
| Magnetic texture | 2.4127 | 0.8998 | Plain |
| Radial statistics | 6.5637 | 4.8451 | Plain |
| Outer-shell statistics | 7.1864 | 1.2060 | Plain |
| Model-only saturation | 0.01853 | 0.01071 | Plain |
| Step-50/100 artifact flags | 5 | 3 | Plain |

Full did not win two required morphology/stability categories. Its H1/base
value and gradient ratios remained elevated, every optimizer update in both
runs clipped, and Full's long-horizon norm crossed `Rin` while Plain's did not.
The result does not justify changing the frozen H1 weight, clip norm, model
size, resolution, or preprocessing.

## Decision

**B. Pause paper ablations and diagnose the H1 adaptation first.**

The prescribed diagnostic extension is limited to:

- index-grid H1 spacing audit;
- unweighted-gradient comparison;
- shell/radial H1 decomposition.

This decision is not a claim that the paper loss is invalid. It says that the
current spherical-grid, `press`, reduced100 FNO adaptation has not demonstrated
the required morphology/stability benefit over its matched Plain control.
