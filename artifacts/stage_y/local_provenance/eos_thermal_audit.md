# Stage Y EOS and thermal provenance audit

## Current raw data

```text
EOS_TYPE = UNKNOWN
ADIABATIC_INDEX = UNKNOWN
PRESS_DEFINITION = raw variable named press in the primitive dataset; public Athena++ convention is rest-frame gas pressure, but exact-run source binding is missing
EINT_DEFINITION = UNKNOWN for the current raw files
UNIT_CONVENTION = UNKNOWN
EOS_PROVENANCE = INCOMPLETE
PRESS_TO_EINT_CONVERSION_AUTHORIZED = false
```

All 212 raw headers have the same root-attribute allow-list and none contains
`gamma`, `gamma_adi`, `EOS`, an energy definition, units, a source hash, or an
input-deck hash.  No matching `.in`, `.par`, build log, problem generator, or
restart metadata was found in the scoped local search or Git history.

The public Athena++ clone at commit
`9c266692b9423743d8e23509b3ab266a232a92d2` is informative but not run-bound:

- `src/eos/adiabatic_mhd_gr.cpp:49-61` reads the adiabatic index from runtime
  key `hydro/gamma`;
- `src/eos/adiabatic_mhd_gr.cpp:626` uses the ideal-gas enthalpy factor
  `gamma/(gamma-1) * pgas`;
- `src/outputs/outputs.cpp:384-392` writes primitive `IPR` under the name
  `press`.

Those lines establish a possible software formula, not the actual value of
Gamma or the binary/module used for this run.  The unrelated public
`athinput.fm_torus` uses different spin, grid, and radial bounds and is
explicitly excluded.  Likewise, the different-paper Zenodo MAD98 script uses
`Gamma=4/3`, but it lacks a checksum link to the local trajectory.

## Paper-side inconsistency

The detailed evidence list is in
`../paper_assets/PAPER_THERMAL_VARIABLE_EVIDENCE.md`.  In brief, target-paper
Appendix A.2 says the GRMHD outputs include pressure `P`, and Appendix B uses
`T=P/rho`; Appendix C, figures, losses, bounds, and Table 2 consistently call
the learned thermal channel `eint`/internal energy.  The GRMHD subsection gives
no Gamma or equation mapping those two descriptions.

```text
PAPER_THERMAL_CONTRACT = AMBIGUOUS
```

The Newtonian MHD subsection's `gamma=5/3` applies to that Newtonian setup and
cannot be transferred to either the paper GRMHD run or current raw data.

## Conversion gate

The algebra `e_int=P/(Gamma-1)` would be valid only after proving an ideal-gas
internal-energy-density definition and the run-specific Gamma on both sides of
the comparison.  Neither condition is established.  Stage Y therefore creates
no converted array, no normalizer, and no transform test.
