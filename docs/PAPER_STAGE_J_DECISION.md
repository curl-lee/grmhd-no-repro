# Stage J operator/geometry/data fidelity decision

## Bounded outcome

Stage J freezes the following five statuses:

| gate | final status | reason |
| --- | --- | --- |
| 3D DISCO | **REQUIRES_NEW_IMPLEMENTATION** | pinned upstream implements and tests 2D DISCO only; a paper-identifiable 3D definition/reference is absent |
| Cartesian Kerr--Schild | **BLOCKED_COMPONENT_BASIS** | raw headers omit run spin, Cartesian convention, velocity/B basis and matching source/config |
| eint/EOS | **BLOCKED_UNVERIFIED_EOS** | only `press` exists; no content-linked EOS/Gamma/units record authorizes conversion |
| 300 snapshots | **NOT_REPRODUCIBLE_FROM_CURRENT_PROVENANCE** | 111 local states are contiguous but lack the author manifest and exact rerun inputs |
| coupling | **CONTRACT_READY** | local shapes/time/ghost-zone semantics and toy tests are ready; paper formulas remain ambiguous and scientific validation requires paired coarse/fine data |

These statuses are deliberately asymmetric. The coupling *software contract* is ready, while an
end-to-end paper-equivalent coupling remains `PAPER_AMBIGUOUS` and
`REQUIRES_COARSE_FINE_DATA`. Likewise, 3D engineering with `conv3d` is technically possible, but
the paper-fidelity prototype gate stays closed until a reference definition is obtained.

## Unique next mainline

```text
B. data acquisition first
```

This choice follows prerequisites rather than implementation convenience. One author-provenance
package can jointly resolve or refine the 300/last-250 identity, EOS/Gamma, units, spin, Cartesian
grid/component semantics, exact architecture metadata, and possibly coarse/fine exchange fields.
Starting an arbitrary operator or geometry conversion now would lock guesses into downstream data
and make later comparisons uninterpretable.

The minimum acquisition request is:

1. the exact 300-snapshot manifest with checksums, times, ordered split, and dataset/version ID;
2. data files or an access method for the Cartesian fine states, including `eint` semantics;
3. exact simulation input/build provenance: code commit, problem generator, spin/metric, units,
   EOS/Gamma, grid/AMR and boundary conditions;
4. definitions/staggering for velocity and magnetic components, ideally face fields or vector
   potential for constrained transfer;
5. 3D DISCO/LocalNO reference code or complete basis/support/boundary/config specification;
6. coarse/fine manifests, time-step ratio, ghost layout, solver hook and CT/EMF interface fields.

If these are unavailable, the honest operational state becomes pending author information; it does
not authorize reconstructing missing parameters from filenames, figures, or unrelated samples.

## Snapshot reconstruction decision

The selected snapshot decision is `D. NOT_REPRODUCIBLE_FROM_CURRENT_PROVENANCE`, not
`C. NEW_SIMULATION_REQUIRED`. No exact Athena/AthenaK commit, matching input, initial condition,
EOS, spin, AMR/boundary prescription or continuation state is content-linked to the 111 snapshots.
Therefore `PAPER_STAGE_J_SIMULATION_PLAN.md` is intentionally not created. A simulation plan would
become meaningful only after the missing provenance changes this gate.

## What Stage J did not do

- no training, optimizer step, smoke, pilot, or simulation;
- no new DISCO prototype whose paper identity would be guessed;
- no `press -> eint` calculation and no Gamma assumption;
- no spherical-to-Cartesian relabeling or generated Cartesian HDF5;
- no rewrite of raw/processed data, checkpoints, pair order, or Stage G/I results;
- no modification to pinned `external/neuraloperator`.

## Remaining-work scenarios

### Optimistic

Author data/config/code arrive complete and checksummed. Allow roughly 6--10 engineering weeks to
validate provenance, implement/reference-test the 3D operator, build a separate verified data path,
and integrate the coupling contract before bounded training. This assumes no need to rerun GRMHD.

### Realistic

Metadata arrive in stages and require clarification; magnetic transfer and 3D DISCO need new local
implementation against reference cases. Allow roughly 3--6 months before a first scientifically
defensible exact-protocol pilot, with full coupled validation longer.

### Blocked by external data

If the 300-state manifest, run EOS/components, reference 3D operator, or coarse/fine interface data
cannot be obtained, exact reproduction remains blocked indefinitely. The frozen deliverable remains
the paper-adapted reduced spherical `press` FNO proxy plus its Stage H/I diagnostics; additional
loss tuning does not change that classification.
