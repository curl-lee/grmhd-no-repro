# Stage Y Cartesian transform and raw-salvage feasibility

## Decision

| condition | result | reason |
|---|---|---|
| S1 coordinate mapping complete | false | spherical axes are identified, but exact current-run metric/spin/source mapping is not bound |
| S2 vector basis transform complete | false | software-family semantics recovered; exact run and target Cartesian convention/CT path incomplete |
| S3 thermal conversion complete | false | current EOS/Gamma and paper P/eint relation unresolved |
| S4 sufficient raw spatial information | true | 212 original AMR leaf snapshots retain substantially more spatial information than processed 64^3; cell-centred fields are available |

```text
CURRENT_RAW_DATA_SALVAGEABLE = false
VECTOR_TRANSFORM_PROVENANCE = INCOMPLETE
CARTESIAN_VECTOR_CONVERSION_AUTHORIZED = false
PRESS_TO_EINT_CONVERSION_AUTHORIZED = false
```

S4 does not override S1-S3.  It says only that the raw AMR snapshots, unlike
the lossy production 64^3 tensor, are the correct place to begin *if* the
missing provenance is later recovered.  The absence of face-centred magnetic
exports also prevents a claim that a generic cell-centred remap preserves CT.

## Position versus vector transformation

`x1=r`, `x2=theta`, and `x3=phi` are numerically established.  A run-specific
spherical-KS to Cartesian-KS position map still depends on the exact convention
and spin.  The raw header supplies neither, so Stage Y does not select one of
the sign/convention variants and does not design a target cube.

Even a verified position map would not transform `tilde-u^i` or `B^i` by
renaming.  A vector transformation needs the exact primitive definition,
Jacobian/tetrad and metric convention, and magnetic flux handling documented in
the component audit.

## Target-grid/remap blocker

The target paper states a Cartesian `64^3` cube but not its numeric bounds or
the raw/AMR remap algorithm.  It does not state nearest-neighbor, trilinear,
conservative, volume-averaged restriction, or a divergence-preserving magnetic
transfer.  Therefore:

```text
PAPER_REMAP_METHOD = UNKNOWN
```

Part R (Cartesian remap design) is not authorized because
`CURRENT_RAW_DATA_SALVAGEABLE=false`; guessing cube bounds would add a new
scientific contract.

## Numerical-test disposition

`transform_unit_tests.json` and a `vector_transform/` implementation were not
created.  Part F requires complete provenance, which failed.  This is an
intentional scientific skip, not an engineering test failure.
