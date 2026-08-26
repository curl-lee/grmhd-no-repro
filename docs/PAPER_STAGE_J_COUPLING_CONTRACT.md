# Stage J coarse/fine coupling contract

## Scope and status

```text
interface/toy-test status: CONTRACT_READY
paper numerical formula status: PAPER_AMBIGUOUS
end-to-end coupling status: REQUIRES_COARSE_FINE_DATA
```

The paper describes writing neural-operator states to HDF5, reading two times for interpolation,
and overwriting hydrodynamic variables at an inner boundary; it also refers to magnetic handling
through constrained transport/electromotive forces. It does not fully specify tensor schemas,
restriction/prolongation formulas, synchronization edge cases, ghost layout, conservation, or
autoregressive ownership. `src/grmhd/coupling_contract.py` is therefore an explicit local
engineering contract for testing interfaces. It is not presented as the paper's missing formula.

No model, trainer, optimizer, or simulation is invoked by this module.

## State schema

All tensors use `[batch, channel, z, y, x]`. `GridMetadata` records spatial shape, ordered channel
names, coordinate system, spacing/origin, component basis, and magnetic staggering.

- `CoarseState`: tensor, metadata, nonnegative time index, physical time, and explicit default
  autoregressive owner `coarse_solver`.
- `FineState`: same, with default owner `fine_solver`.
- `BoundaryState`: a full-grid buffer plus positive ghost width and selected faces. Consumers are
  permitted to read only those ghost faces.

The full-grid boundary buffer is intentional: it makes accidental interior overwrite testable.
Descriptors serialize metadata/time/shape/dtype, not tensor values.

## Reference operators

`RestrictionOperator` uses parameter-free arithmetic `avg_pool3d` with an integer factor. It
preserves constants and arithmetic means on divisible uniform toy grids. It is not proper-volume
conservative on nonuniform Kerr--Schild grids.

`ProlongationOperator` uses parameter-free trilinear interpolation with `align_corners=True`. It
preserves affine fields on aligned uniform toy coordinates. It is not a conservative AMR
prolongation and has no limiter.

`BoundaryExchange` uses a boolean ghost mask and `torch.where`. It overwrites only declared faces
and optional channels, returns a new tensor, and leaves the interior exactly untouched. Declared
magnetic channels are rejected unless the caller explicitly opts into an unverified operation;
the production contract must instead define a CT/EMF update.

All three operations are deterministic, differentiable tensor functions. Their empty
`state_dict`s serialize normally because they contain no learned state.

## Time contract

`TimeStepRatio(fine_steps_per_coarse=n)` fixes the integer cadence. `CouplingSchedule` declares
coarse/fine start indices, physical start time, and coarse `dt`. It computes fine `dt`, exact sync
steps, physical time, and the bracketing coarse indices plus interpolation fraction.

`interpolate_boundary` linearly interpolates compatible full boundary buffers in physical state
space. The caller owns encode/decode placement; normalized arrays must not be silently mixed with
physical coarse states.

## Merge, conservation, and ownership rules

1. Restrict/prolong do not mutate their input.
2. Boundary exchange is ghost-zone only; interior evolution belongs to the declared fine owner.
3. At an exact sync index the bracket collapses to one coarse time and `alpha=0`.
4. Between sync times, boundary interpolation requires compatible grids, faces, and ghost widths.
5. Arithmetic restriction is only a toy expectation; physical mass/energy conservation remains
   outside this contract until verified conserved variables and metric volumes exist.
6. `Bcc` is diagnostic cell-centred data. Magnetic boundary ownership remains with the simulation's
   constrained-transport/EMF path, not a generic tensor overwrite.
7. The neural operator may own an explicitly declared fine autoregressive state, but it may not
   overwrite coarse interior state implicitly.

## Toy verification

`tests/test_coupling_contract.py` covers:

- state shape/channel/time/ownership validation;
- restriction/prolongation shapes and constant preservation;
- affine-field trilinear prolongation;
- temporal synchronization and boundary interpolation;
- selected ghost-face/channel update and exact interior preservation;
- rejection of unverified cell-centred magnetic overwrite;
- finite gradients through restriction, prolongation, and boundary exchange;
- deterministic repeated evaluation and empty state dictionaries;
- JSON round-trip of grid/schedule descriptors.

## Gate to real coupling

End-to-end work requires paired coarse/fine trajectories, matching coordinates/components/EOS,
ghost and face-field layouts, actual time-step ratios, exact boundary conditions, and paper/author
clarification of which fields are overwritten at which stage. Until then the interface is ready for
software tests, while scientific coupling remains `REQUIRES_COARSE_FINE_DATA` and
`PAPER_AMBIGUOUS`.
