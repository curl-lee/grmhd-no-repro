# Stage J blocker-resolution priority matrix

Scores use `1` (low) through `5` (high). Calendar times are dependency-aware estimates, not
commitments; author-dependent waits cannot be bounded locally.

| rank | component | impact | cost | external | uncertainty | prerequisites | estimated time | next gate |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | author dataset and run provenance | 5 | 2 | 5 | 4 | 0 | 1--6 weeks or author-dependent | obtain 300-state manifest/files and content-linked run/build config |
| 2 | EOS/component/coordinate verification | 5 | 2 | 5 | 4 | 1 | 2--5 days after receipt | verify Gamma, EOS, units, spin, velocity/B basis, Cartesian convention |
| 3 | reference 3D DISCO implementation/specification | 5 | 3 | 4 | 4 | 0 | 1--4 weeks or external-dependent | obtain code or implementation-level basis/support/boundary/config |
| 4 | Cartesian/face-field export or conversion | 5 | 4 | 4 | 4 | 2 | 2--6 weeks | write a separate dataset only after geometry/component gates |
| 5 | conservative/divergence-preserving transfer | 4 | 5 | 4 | 4 | 3 | 4--12 weeks | validate scalar integrals and CT magnetic flux on reference data |
| 6 | coarse/fine coupling interface contract | 4 | 2 | 2 | 3 | 0 | completed in Stage J | keep toy contract separate from ambiguous paper formulas |
| 7 | paired coarse/fine coupling validation data | 5 | 5 | 5 | 5 | 4 | author-dependent | acquire matched grids, clocks, ghost/face fields, fluxes and EMFs |
| 8 | paper-grid shells and exact data protocol | 3 | 2 | 3 | 2 | 2 | 2--5 days after dataset gate | freeze Cartesian shells and ordered last-250 split |
| 9 | final operator integration and training | 5 | 5 | 4 | 4 | 6 | multiple months after prerequisites | begin only after data/geometry/EOS/operator/coupling gates |

The ordering follows the dependency graph: operator training cannot repair missing data identity;
Cartesian vectors cannot precede component-basis verification; and final coupling cannot be
validated without paired solver interface data. Loss tuning is not a blocker-resolution item.
