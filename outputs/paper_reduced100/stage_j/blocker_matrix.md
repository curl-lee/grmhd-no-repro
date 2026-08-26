# Stage J blocker matrix summary

This is the compact output copy of the audited matrix.  The detailed rationale is
in `docs/PAPER_STAGE_J_BLOCKER_MATRIX.md`; the complete structured fields are in
`blocker_matrix.json`.

| component | paper requirement | current state | evidence | missing information | local feasibility | external dependency | next gate |
|---|---|---|---|---|---|---|---|
| 3D DISCO layer | volumetric equidistant DISCO | **BLOCKED** — pinned source is 2D only | dimensional guard and no 3D class in local refs | 3D basis/support/boundaries/reference | generic conv3d is possible but not uniquely paper faithful | author/upstream code or spec | obtain reference before implementation |
| LocalNO architecture | 3D spectral + integral + differential operator | **AVAILABLE_UNVERIFIED** — only differential-only 3D proxy runs | upstream LocalNO and local wrapper | paper hyperparameters and 3D integral path | plumbing reusable | reference model metadata | resolve DISCO/config |
| Cartesian KS grid/components | Cartesian 64^3 state | **REQUIRES_AUTHOR_DATA** — local data are spherical and component basis is unknown | ATHDF/regrid metadata | convention, spin, metric/tetrad, basis | point mapping alone is insufficient | author export/run config | verify grid and basis |
| density | positive rho/dens | **AVAILABLE_VERIFIED** in the local reduced trajectory | raw variables/full audit | exact paper units/identity | usable as adaptation | paper manifest | retain provenance |
| eint and EOS/Gamma | verified internal energy | **BLOCKED** / **REQUIRES_AUTHOR_DATA** — only press is present | 623-file inventory | EOS, Gamma, units, run identity | conversion forbidden now | matching config/data | verify all thermal provenance |
| 300 snapshots / last 250 | paper temporal dataset and window | **REQUIRES_AUTHOR_DATA** / **BLOCKED** — only 111 exist | complete raw inventory | remaining states and manifest | cannot synthesize | author data or exact rerun provenance | acquire data |
| cadence and split | DeltaT and ordered 80/20 last-250 | **AVAILABLE_UNVERIFIED** / **IMPLEMENTABLE_LOCALLY** | raw times and reduced protocol | paper times and boundary policy | protocol logic reusable | paper manifest/data | freeze exact ordered manifest |
| regridding/conservation | Cartesian cell representation | **AVAILABLE_VERIFIED** only as a lossy current method | build/regrid audits | author grid/export, metric volumes | scalar conservative transfer is designable | magnetic/geometry data | do not claim current grid is exact |
| div B | CT-compatible magnetic field | **REQUIRES_NEW_SIMULATION** | no face B/vector potential/EMF | staggered magnetic state | diagnostic proxy only | simulation export | acquire staggered fields |
| radial shells | eight Cartesian index shells | **IMPLEMENTABLE_LOCALLY** after geometry | appendix formula/current adapted shells | exact Cartesian center/domain | straightforward after grid gate | Cartesian metadata | couple to verified geometry |
| coarse/fine coupling and boundary exchange | dynamic hydro + CT/EMF interface | **PAPER_AMBIGUOUS** | local state/time/ghost contract passes 9 toy tests; Appendix E remains incomplete as an API | paper clocks, ownership, magnetic staggering, merge rules | hydro tensor interface is contract-ready; CT integration is not | solver hooks and paired data | validate the engineering contract against solver reference |
| rollout and metrics | autoregression, relative L2, long statistics | **IMPLEMENTABLE_LOCALLY** / **AVAILABLE_VERIFIED** | Stage G/I evaluators | exact data/operator and artifact thresholds | control flow/formulas reusable | author results for replication | preserve non-comparability labels |

Overall status: **BLOCKED** for exact reproduction.  Operator coding does not
remove the upstream data, EOS, component-basis, magnetic-staggering, or
coarse/fine-data dependencies.
