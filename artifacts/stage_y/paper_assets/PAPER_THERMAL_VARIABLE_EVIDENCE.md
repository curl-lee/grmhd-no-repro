# Target-paper thermal-variable evidence

Primary evidence is the verified arXiv v1 source at
`/tmp/stage_y_arxiv_src.yXyJhp/main.tex` (source archive SHA256
`6e758b1fdd74f27f332a0bfa21dc9a44afd1a458e7459c2048539f0c98c861bf`).

| source position | literal semantic claim | classification |
|---|---|---|
| main text figure caption, lines 159-164 | MHD panels show internal energy `E_int`; observable temperature is shown | INTERNAL_ENERGY |
| GRMHD figure caption, lines 212-218 | GRMHD panels show internal energy `E_int` | INTERNAL_ENERGY |
| Appendix A.1, lines 303-314 | Newtonian MHD defines pressure `P`, total energy, ideal gas `gamma=5/3`, and outputs `(rho,P,v,B)` | PRESSURE; Newtonian only |
| Appendix A.2, lines 340-359 | GRMHD equations use gas pressure `p_g`; output list is `(rho,P,v,B)`; no GRMHD Gamma stated | PRESSURE |
| Appendix B, lines 363-368 | observable temperature is `T=P/rho` | PRESSURE |
| Appendix C.2, lines 378-390 | eight model fields name `eint` and call it internal energy; it receives the positive log transform | INTERNAL_ENERGY |
| Appendix C.5-C.7, lines 437-532 | radial baseline, envelope, bounds, constraints, and loss act on `dens/eint` | INTERNAL_ENERGY |
| Appendix C.8-C.10, lines 535-605 | wrapper and evaluation decode/clamp `dens/eint` | INTERNAL_ENERGY |
| Appendix D, lines 613-723 | ablations and Table 2 use `e`/internal energy | INTERNAL_ENERGY |
| Appendix E, lines 755-759 | coarse hydro overwrite is stated as `(rho,P,vx,vy,vz)` | PRESSURE |

No inspected caption, table, appendix, TeX macro, or source file defines a
GRMHD `P <-> eint` conversion or says that the two names are aliases.  The
Newtonian `gamma=5/3` statement is not labeled as the GRMHD EOS.

```text
PAPER_THERMAL_CONTRACT = AMBIGUOUS
```

Stage Y preserves this inconsistency instead of resolving it by assumption.
