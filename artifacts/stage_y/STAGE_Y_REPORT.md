| requirement | status | evidence | blocks exact reproduction |
|---|---|---|---|
| paper data | NOT_FOUND | arXiv source, NeurIPS/OpenReview, author repositories, public data-host search | yes |
| BH spin | PAPER `a=0.9`; CURRENT RAW UNKNOWN | paper `main.tex:357`; raw ATHDF attributes contain no spin/input hash | yes, for transforming current raw |
| coordinate mapping | FOUND_PARTIAL | raw `Coordinates=kerr-schild`, numeric `(x1,x2,x3)=(r,theta,phi)`; exact producer/spin absent | yes |
| magnetic basis | FOUND_PARTIAL | `Bcc` cell-centred software convention recovered; exact-run source binding/CT transfer absent | yes |
| velocity convention | FOUND_PARTIAL | public Athena++ uses primitive `tilde-u^i`; exact-run source binding absent | yes |
| EOS/Gamma | NOT_FOUND for current raw/GRMHD contract | 212 headers and local search have no EOS/Gamma; paper GRMHD subsection omits Gamma | yes |
| press/eint mapping | AMBIGUOUS / NOT AUTHORIZED | paper Appendix A/B says `P`, model Appendix C/D says `eint`; no GRMHD conversion | yes |
| 3D DISCO | NOT_FOUND | paper names it; pinned source and author fork implement local-integral DISCO only in 2D | yes |
| paper remap | NOT_FOUND | paper gives Cartesian 64³ but no cube bounds or AMR/remap algorithm | yes |
| coarse/fine coupling | FOUND_PARTIAL | Appendix E gives time interpolation/hydro overwrite/CT prose, not executable B-to-EMF contract | yes |

# Stage Y — Provenance Recovery and Paper Asset Acquisition

## Scope and frozen inputs

Stage Y used project commit `39dde155093436685c20c4f1d97269a02ae1189d`
on branch `codex/stage-r-residual-localno-pilot` and pinned upstream
`86a8bc7812a31b42c4f7895693cf4ac11521c066`.  The worktree already contained
uncommitted Stage-S-through-X material; Stage Y preserved it.

The paper PDF SHA256 is
`fe84a42289da7e86dd8460d223e57627a86aa0b2114a64a13b83f84388441808`.
The processed expanded HDF5 is
`data_proc/grmhd_regrid_inner_r200_64_expanded.h5`, SHA256
`3582a5c4f52f4fef305578ffd5a63c51423f8583f35d137c288d5c20203b50da`,
shape `(212,8,64,64,64)`.  The raw series has 212 contiguous snapshots
00000--00211, times `0.0`--`2110.000838137111`, under
`/mnt/d/GRMHD_data`.  Full details and Stage-X artifact hashes are frozen in
`stage_x_input_manifest.json`; per-snapshot metadata is in
`local_provenance/simulation_file_manifest.csv`.

```text
NO_RAW_MUTATION = true
NO_PROCESSED_MUTATION = true
NO_MODEL_TRAINING = true
```

## 1. 黑洞 spin 是否恢复？

Only on the paper side.  Target-paper Appendix A.2 explicitly states
`a=0.9`.  The current raw files contain no spin, mass, run ID, input-deck hash,
or source hash.  A different-paper public MAD98 archive suggests a simulation
family with `a=0.98`, but no checksum/manifest binds it to the local files; the
filename is not evidence.  Thus `BH_SPIN_VALUE_CURRENT_RAW=UNKNOWN` and the
combined/current-raw provenance is incomplete.

## 2. spherical KS mapping 是否完整恢复？

No.  The raw files explicitly say `Coordinates=kerr-schild`; coordinate values
establish `(x1,x2,x3)=(r,theta,phi)`, and the local versioned public Athena++
source documents standard spherical Kerr--Schild.  However, the exact producer
commit/input and spin are absent, and that source explicitly notes other
Kerr--Schild coordinates exist.  The axis map is recovered; the run-specific
metric/position-map provenance is not.

## 3. Bcc1/2/3 的准确 component convention 是什么？

The raw header alone says only `Bcc1:3` in dataset `B`.  Public Athena++ commit
`9c266692...` shows `Bcc` is copied from reconstructed cell-centred `pfld->bcc`
and its KS routines treat the global values as contravariant coordinate
components `B^i`.  An unrelated MAD98 conversion script corroborates that
software-family convention.  Because neither source is content-linked to the
current run, the exact-current-run convention remains provenance-incomplete;
it must not be treated as an orthonormal or Cartesian vector.

## 4. vel1/2/3 的准确 convention 是什么？

The public Athena++ GR source writes the `IVX:IVZ` primitive slots and defines
them as `tilde-u^i = gamma v^i`, reconstructing `u^mu` with the metric.  They
are not ordinary coordinate `dx^i/dt`.  This is a recovered software convention
but not a complete current-run provenance chain because the producing source
revision is unknown.

## 5. 是否可以科学做 Cartesian vector transformation？

No.  The current spin, exact metric/source mapping, exact producer component
contract, target Cartesian-KS convention, and CT-compatible magnetic transfer
are not jointly established.  Only cell-centred `Bcc` is available.  Stage Y
did not implement a transform and intentionally omitted transform unit tests.

## 6. EOS/Gamma 是否恢复？

No for the current raw data, and the paper GRMHD subsection does not state
Gamma.  Public Athena++ shows that Gamma would be read from runtime
`hydro/gamma` and `press` is primitive gas pressure under that implementation,
but the matching input/build is missing.  Newtonian-paper `gamma=5/3` and an
unrelated MAD98 archive's `4/3` are both excluded as current-GRMHD evidence.

## 7. 是否可以科学做 press→eint？

No.  Current Gamma/EOS is unknown, while the target paper itself alternates
between GRMHD output `P`/temperature `P/rho` and model channel `eint` without a
GRMHD conversion definition.  No array was converted.

## 8. 是否找到 paper official dataset？

No.  The paper specifies 300 generated snapshots, last 250 used, 80/20 split,
Cartesian-KS `64^3`, and symbolic cadence `DeltaT=T/Ndata`, but no official
dataset/manifest/sample/download was found.  Numeric domain bounds, dtype,
times, fitted statistics, and remap are unavailable.  The current raw files and
different-paper Zenodo archive are not paper samples.

## 9. 是否找到 paper official training code？

No.  The arXiv source has TeX/bibliographies/figures only; venue and author
pages provide no code link.  The author's public neuraloperator fork has one
branch, no tags, and no paper/GRMHD/volumetric-DISCO files.  The pinned
neuraloperator tree is only an upstream dependency.

## 10. 是否找到 volumetric 3D DISCO implementation？

No.  The paper names one without publishing its implementation parameters.
Pinned `LocalNOBlocks` explicitly rejects enabled DISCO outside 2D and
hard-codes `EquidistantDiscreteContinuousConv2d`.  The primary LocalNO reference
recovers the general compact-support quadrature operator, but the target 3D
basis, basis count, radius, normalization, boundaries, and per-layer use remain
unknown.  A guessed `conv3d` kernel would not be exact paper DISCO.

## 11. 是否恢复 coarse/fine CT coupling？

Partially.  Appendix E explicitly gives HDF5 rollout, two-time linear
interpolation, direct hydro inner-boundary overwrite, and a CT review.  It does
not define how predicted cell-centred `B` becomes the required face/edge EMF,
nor the spatial exchange, staggering, boundary mask, metric factors, divergence
checks, or executable two-way conditioning.  No coupling code/reference output
was found.

## 12. 当前 raw dataset 是否 salvageable？

Not under the requested scientific gate:

| salvage condition | status |
|---|---|
| S1 coordinate mapping complete | false |
| S2 vector transform complete | false |
| S3 thermal conversion complete | false |
| S4 sufficient raw spatial information | true |

The AMR raw files contain much more spatial information than the lossy 64³
regrid, so S4 passes.  S1-S3 fail independently; therefore
`CURRENT_RAW_DATA_SALVAGEABLE=false`.

## 13. exact paper reproduction 还缺哪些不可替代的信息？

- official paper data or an author-bound manifest/sample;
- current-run spin, source/build/input, EOS/Gamma, units, and exact vector
  conventions if the local raw path is to be used;
- paper Cartesian cube bounds and remap/magnetic-transfer algorithm;
- exact volumetric 3D DISCO implementation or a complete 3D mathematical and
  architectural specification;
- executable model architecture/config and fitted preprocessing artifacts;
- executable coarse/fine boundary and B-to-EMF/CT implementation with reference
  values.

## 14. 下一步选择

Stop the exact-reproduction path with currently public/local assets.  Do not
reprocess the raw data, implement a guessed paper operator, or rerun a
simulation: each would require missing primary contracts.  The only authorized
automatic path is to continue labeling the project as an adapted-workflow
reproduction.  Exact work may resume only after a human obtains author data,
code, run provenance, or an authoritative implementation specification.

## Readiness gates

| gate | result | reason |
|---|---|---|
| R1 Data semantics | FAIL | no paper data; current raw transform provenance incomplete |
| R2 Operator | FAIL | exact 3D DISCO absent and critical 3D choices unknown |
| R3 Preprocessing | FAIL | paper domain/remap/fitted artifacts absent |
| R4 Training | FAIL | exact volumetric architecture/config absent despite detailed optimizer/loss prose |
| R5 Coupling | FAIL | coupling is a main claim and only a partial prose contract exists |

## Final labels

```text
PRIMARY_DECISION = F

BH_SPIN_PROVENANCE = INCOMPLETE
COORDINATE_MAPPING_PROVENANCE = INCOMPLETE
VECTOR_TRANSFORM_PROVENANCE = INCOMPLETE
CARTESIAN_VECTOR_CONVERSION_AUTHORIZED = false

EOS_PROVENANCE = INCOMPLETE
PRESS_TO_EINT_CONVERSION_AUTHORIZED = false

OFFICIAL_PAPER_DATA_FOUND = false
OFFICIAL_PAPER_CODE_FOUND = false
EXACT_3D_DISCO_IMPLEMENTATION_FOUND = false
PAPER_COARSE_FINE_CONTRACT = PARTIAL

CURRENT_RAW_DATA_SALVAGEABLE = false

R1_DATA = FAIL
R2_OPERATOR = FAIL
R3_PREPROCESSING = FAIL
R4_TRAINING = FAIL
R5_COUPLING = FAIL

EXACT_REPRODUCTION_BLOCKED = true

AUTHORIZE_NEXT_STAGE = adapted_workflow_reproduction_only
```
