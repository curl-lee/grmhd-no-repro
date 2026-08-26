# Stage N isolated transform prototypes

## Scope and isolation

Stage N fitted five diagnostic normalizers from exactly HDF5 snapshots 11–90.
Snapshots 91–110 were excluded from fitting and were opened only after the
prototype parameters, readiness rules, Bcc2 selection, and P3 composition were
frozen. Every prototype records:

- `classification: diagnostic_transform_prototype`;
- `canonical_replacement: false`;
- `authorized_for_training: false`;
- `authorized_for_checkpoint_evaluation: false`;
- `paper_faithful: false`.

The canonical normalizer SHA256 remains
`1645714424d95b02de2283fc16a8ef05f3864d15f11025580af13f3e67964001`.
Prototype NPZ files are isolated, Git-ignored runtime artifacts; their small JSON
metadata, checksums, fit indices, and results are reproducible from the Stage N
script.

## Definitions

| prototype | changed channels | mapping |
| --- | --- | --- |
| P0 | none | canonical control |
| P1 | Bcc3, vel3 | bypass forward softclip and matching inverse softclip |
| P2A | Bcc2 | retain forward softclip; replace ±5.94 inverse clamp by `float32 nextafter(6,0)` |
| P2B | Bcc2 | bypass forward and inverse softclip |
| P3 | Bcc2, Bcc3, vel3 | train-selected P2B for Bcc2 plus P1 for Bcc3/vel3 |

P2A's `5.999999523162842` limit is the largest float32 value strictly inside
the `atanh(x/6)` singularity. It is derived from numerical safety, not from a
validation quantile. P0's independently refitted epsilon/median/scale arrays
are bitwise equal to the frozen canonical statistics.

## Train-only selection

Median raw-to-oracle relative L2 and the four core retentions were:

| channel / candidate | canonical L2 | candidate L2 | variance | shell/radial | high-k | span | readiness |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Bcc2 / P2A | 6.67e-7 | 6.67e-7 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | fail: <50% L2 improvement |
| Bcc2 / P2B | 6.67e-7 | 7.76e-8 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| Bcc3 / P1 | 0.970821 | 9.23e-8 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |
| vel3 / P1 | 0.811144 | 2.46e-8 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | pass |

Only P2B passed Bcc2 train readiness, so it was selected before validation.
The preference for P2A when both candidates pass was therefore not invoked.
All candidate/control fields were finite, no control-channel L2 worsened, and no
new exact saturation was introduced.

## Read-only validation

Validation showed a large canonical distribution shift in Bcc2, but it did not
alter candidate selection. P3 median L2 was `1.48e-7` for Bcc2, `2.03e-7` for
Bcc3, and `2.11e-8` for vel3. All twelve P3 core retentions were approximately
one. Hence Bcc2, Bcc3, and vel3 are each `A. PROTOTYPE_READY`, and P3 is
`1. COMBINED_PROTOTYPE_READY_FOR_SHORT_PILOT`.

This is an oracle round-trip result only. It is not model improvement, a
canonical replacement, or authorization to train.
