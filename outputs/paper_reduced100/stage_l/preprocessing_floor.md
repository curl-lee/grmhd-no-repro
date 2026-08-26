# Stage L preprocessing floor

This report compares every raw validation target (snapshots 92--110) with its
canonical encode-once/decode-once oracle. It is a transform audit, not a model
evaluation. Retention is `oracle metric / raw metric`; the frozen severe rule is
strictly `< 0.5`.

| channel | median std | median variance | median q99-q01 span | median TV | median demeaned high-k | median radial-profile variance | oracle detector |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Bcc1 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | never |
| Bcc2 | 0.135732 | 0.018423 | 0.115107 | 0.128949 | 0.002928 | 0.009218 | never |
| Bcc3 | 0.001540 | 0.00000237 | 0.000520 | 0.000840 | 0.000000518 | 0.00000120 | steps 1--19 |
| rho | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | never |
| press | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | never |
| vel1 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | never |
| vel2 | 0.772609 | 0.596924 | 0.747190 | 0.776624 | 0.311851 | 0.490209 | never |
| vel3 | 0.157497 | 0.024806 | 0.084011 | 0.103711 | 0.005669 | 0.019876 | never |

Bcc3 and vel3 each lose more than half of global variance, shell/radial
variance, and combined demeaned high-k energy at all five selected GT steps.
Bcc3 alone crosses the unchanged detector's stricter std-ratio threshold
`< 0.05` at every validation step. The absence of a vel3 detector flag does not
mean absence of a preprocessing floor: its median std retention is about 0.157,
which is severe under Stage L but above the detector threshold.

Bcc2 is a useful sensitivity control: the transform strongly suppresses it even
though the frozen detector does not trigger. Bcc1, rho, press, and vel1 are
essentially preserved; vel2 is moderately affected. This channel contrast rules
out a universal scale bug while showing that the canonical transform is a major
part of the Bcc3/vel3 observation.

The full 19-step values, clamp occupancy, sign changes, shell retentions, and
explicit absolute numerators/denominators are in `preprocessing_floor.json` and
`.csv`. Undefined denominators are represented by `null`, never zero-filled.
