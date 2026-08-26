# Stage Y coarse/fine and constrained-transport contract

## Result

```text
PAPER_COARSE_FINE_CONTRACT = PARTIAL
```

The paper gives a high-level and partially mathematical interface, but not an
executable coupling contract.

## Explicit paper statements

From `main.tex:175-181`, the fine domain has size `L^3`, the coarse domain
`(n_L L)^3` with `n_L=6`, the fine trajectory is learned, and the operator is
subsequently coupled to a coarse DNS.

Appendix E (`main.tex:755-815`) explicitly states:

1. NO rollouts are written as HDF5 files.
2. Two files are read and linearly interpolated between output times.
3. `(rho,P,vx,vy,vz)` directly overwrite corresponding inner-boundary hydro
   values.
4. CT evolves face-area-averaged magnetic fields with edge/line-averaged EMFs.
5. The NO predicts cell-centred volume-averaged magnetic field.
6. The text says the face-centred EMF from the Riemann solver is replaced with
   one "from NO prediction," after reviewing the CT reconstruction formulas.

## Missing or ambiguous implementation details

| item | status |
|---|---|
| exact fine/coarse grid bounds and resolution relation | UNKNOWN |
| HDF5 schema, channel units, timestamps, ghost zones | UNKNOWN |
| spatial interpolation/restriction/prolongation | UNKNOWN |
| inner-boundary thickness, indexing, masks, and corner ownership | UNKNOWN |
| conversion from predicted cell-centred `B` to face EMF `E` | UNKNOWN; the prose does not give a dimensionally executable mapping |
| velocity/magnetic temporal interpolation and consistency | UNKNOWN |
| face/edge staggering reconstruction | PARTIAL CT review only |
| divergence diagnostic and correction | UNKNOWN |
| GR metric factors in the coupling interface | UNKNOWN |
| feedback from coarse state into the fine NO input | claimed at framework level; executable conditioning path UNKNOWN |
| source commit, patch, tests, or reference output | NOT_FOUND |

The printed CT equations contain apparent component/index transcription issues
(for example mixed `E_y/E_z` and `B_x/B_y` terms), increasing the need for
source or reference values rather than silent correction.

## Gate implication

Coupling is a main scientific claim rather than an optional visualization.
Because the B-to-EMF step, spatial exchange, and executable boundary contract
are missing, Gate R5 is `FAIL`, not `NOT_REQUIRED`.
