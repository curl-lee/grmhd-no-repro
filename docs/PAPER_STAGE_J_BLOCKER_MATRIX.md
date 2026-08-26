# Stage J: Operator/geometry/data fidelity blocker matrix

## Outcome

Exact paper reproduction is blocked by independent operator, data, thermal,
geometry, and coupled-solver dependencies.  Stage G/I remain a frozen
paper-adapted reduced FNO study; loss tuning cannot substitute for these missing
fidelity layers.

The status vocabulary is restricted to `AVAILABLE_VERIFIED`,
`AVAILABLE_UNVERIFIED`, `IMPLEMENTABLE_LOCALLY`, `REQUIRES_AUTHOR_DATA`,
`REQUIRES_NEW_SIMULATION`, `PAPER_AMBIGUOUS`, and `BLOCKED`.  The complete
machine-readable records are in
`outputs/paper_reduced100/stage_j/blocker_matrix.json`.

| component | paper requirement | current state | evidence | missing information | local feasibility | external dependency | next gate |
|---|---|---|---|---|---|---|---|
| 3D DISCO layer | equidistant DISCO specialized to 3D volumes | **BLOCKED** — fixed upstream only has arbitrary/equidistant 2D classes and rejects DISCO outside 2D | `local_no_block.py` guard, hard-coded 2D class, all local refs | reference 3D basis, support, boundaries, hyperparameters and outputs | pure-PyTorch `conv3d` can test a generic operator, but its paper identity would be guessed | author/upstream implementation or implementation-level spec | obtain reference before coding the paper path |
| LocalNO architecture | 3D spectral + localized integral + differential blocks | **AVAILABLE_UNVERIFIED** — 3D differential-only proxy runs | upstream `LocalNO`; local `localno_diff` forces `disco_layers=False` | exact width/layers/modes/DISCO config and wrapper | surrounding plumbing is reusable | reference model/config | resolve 3D DISCO and architecture metadata |
| Cartesian Kerr–Schild grid | paper 64-cubed Cartesian KS cells | **REQUIRES_AUTHOR_DATA** — local AMR is spherical KS and processed axes are phi/theta/r | raw HDF5 extents and processed axis metadata | Cartesian convention, domain and spacing | point mapping can be designed after conventions; it does not recover author cell averages | author Cartesian export or exact simulation provenance | obtain the paper grid/source data |
| Cartesian vector components | Bx/By/Bz and vx/vy/vz | **REQUIRES_AUTHOR_DATA** — Bcc1:3/vel1:3 remain stored spherical components | raw names and explicit regrid warning | covariant/contravariant/orthonormal basis, primitive velocity definition, spin, metric/tetrad | Jacobian transform only after semantics are verified | matching run code/config or direct export | verify component basis |
| density | positive rho/dens | **AVAILABLE_VERIFIED** for the local 111 states | raw variable list and full trajectory audit | exact paper units and dataset identity | usable for reduced adaptation | paper manifest for numerical comparison | preserve rho provenance |
| eint | internal energy input/output channel | **BLOCKED** — no eint exists, only press | all raw and processed channel lists | EOS, Gamma, pressure definition, units | conversion is forbidden now | author eint or exact run config | pass thermal provenance gate |
| EOS/Gamma | verified press/eint relationship | **REQUIRES_AUTHOR_DATA** — no matching metadata/config | Stage J candidate inventory; unrelated public sample rejected | EOS form/value/units and unique run match | locally auditable after receipt | author/run configuration | acquire and hash exact config |
| 300 snapshots | Ndata=300 fine states | **REQUIRES_AUTHOR_DATA** — 111 local states | full content-hash inventory | 189 states and paper manifest | cannot synthesize states | author dataset or exact rerun provenance | acquire manifest/data |
| last 250 | final 250 of the 300 | **BLOCKED** — local total is below 250 and relation is unknown | count and absence of author manifest | paper indices/order | split code exists but has no inputs | complete dataset | resolve 300-state provenance |
| temporal cadence | every DeltaT=T/Ndata | **AVAILABLE_UNVERIFIED** — local dt is about 10 and continuous | raw `Time` attributes | paper T, units and exact time list | reduced cadence is reproducible only locally | paper manifest/config | compare author times |
| train/validation split | ordered 80/20 of last 250 | **IMPLEMENTABLE_LOCALLY** — reduced ordered analogue exists | protocol and pairing tests | exact boundary-transition policy and data | direct once manifest is present | last-250 states | freeze exact indices/pairs |
| regridding | paper Cartesian cell tensor | **AVAILABLE_VERIFIED** only for the current lossy spherical regrid | build metadata and ablation | paper export/regrid and Cartesian cell geometry | diagnostic comparisons are local | paper representation | never relabel current grid Cartesian |
| divergence preservation | non-monopole/CT-compatible magnetic transfer | **REQUIRES_NEW_SIMULATION** — current regrid is not div-preserving | metadata; no face B/vector potential/EMF | staggering, face areas/fields or vector potential | divergence proxy only from current centers | simulation/export with staggered magnetic data | acquire face/vector-potential/EMF state |
| conservative interpolation | preserve integrals/fluxes across grids | **IMPLEMENTABLE_LOCALLY** for defined scalar finite-volume states; unavailable for current full GRMHD claim | current methods declare non-conservative | metric volumes, conserved variables, target intersections, face fluxes | overlap-volume restriction/prolongation can be built after state definition | magnetic data for full fidelity | specify conserved state/geometry |
| radial shell embedding | eight Cartesian index-distance one-hots | **IMPLEMENTABLE_LOCALLY** after grid gate; current shells use spherical r | paper Appendix C and Stage D artifact | exact Cartesian center/domain handling | simple deterministic construction | Cartesian metadata | build only with verified Cartesian grid |
| coarse/fine coupling | dynamic fine NO inside coarse DNS | **PAPER_AMBIGUOUS** — local engineering contract and toy tests are ready, but no coupled solver/data or complete paper API exists | Appendix E high-level description; `coupling_contract.py`; 9 toy tests | paper resolution/time ratio, ownership, synchronization and failure rules | interface contract is implemented without claiming paper equivalence | solver hooks and paired data | validate contract against author/solver reference |
| boundary exchange | time interpolation, hydro overwrite and CT/EMF magnetic update | **PAPER_AMBIGUOUS** — ghost-zone-only hydro tensor exchange is tested; CT/EMF integration remains undefined | Appendix E equations 14--25; boundary mask/interpolation tests | exact extent/layout/clocks/staggering/merge and corner rules | hydro software semantics ready; magnetic overwrite rejected by default | AthenaK CT interface and exchange data | validate hydro contract and define CT/EMF ownership |
| coarse/fine data | matched states/fluxes/EMFs at exchange times | **REQUIRES_AUTHOR_DATA** — none found | dataset/project inventory | paired manifests/grids/times/interface variables | synthetic tensors only | author data or new coupled run | acquire interface dataset |
| rollout protocol | physical autoregression and long-horizon checks | **IMPLEMENTABLE_LOCALLY** — reduced flow is frozen and tested | Stage G/I transform counts/evaluation | paper starts/horizons and exact operator | reuse control flow | paper model/data for comparison | retain single encode/decode semantics |
| paper evaluation metrics | normalized relative L2, morphology and observables | **AVAILABLE_VERIFIED** at formula level | Appendix C/D and Stage G/I code | author preprocessing/data and quantitative artifact thresholds | formulas reusable; qualitative claims remain qualitative | author outputs/checkpoints for numerical replication | mark reduced numbers non-comparable |

## Dependency order

The blocking graph is not a flat to-do list:

1. author data/run provenance gates the 300/last-250 protocol, EOS/eint, units,
   spin and component semantics;
2. component semantics and staggered magnetic state gate Cartesian conversion,
   conservative/divergence-preserving transfer and CT exchange;
3. a reference 3D DISCO definition gates a paper backbone and only then a formal
   paper configuration;
4. coarse/fine manifests and solver hooks gate scientific validation of the
   local coupling contract;
5. only after these gates does training the final operator become meaningful.

Thus a locally convenient operator prototype is not the highest-priority main
line while the data/physics identity remains unresolved.  Stage J may define a
toy coupling contract because it is explicitly separated from a paper-equivalent
implementation, but it must not generate formal Cartesian data or start a
simulation/training run.
