# Stage J regridding fidelity audit

## Decision

The canonical `64^3` HDF5 is verified as a reproducible **nearest leaf cell-centre sampling** of
the available spherical Kerr--Schild AMR snapshots. It is suitable for the frozen paper-adapted
reduced proxy, but it is not suitable for exact reproduction of a Cartesian, conservative,
magnetic-constraint-preserving paper dataset.

Its embedded metadata says both:

```json
{"not_conservative": true, "not_divergence_preserving": true}
```

Stage J did not rebuild or overwrite any data.

## What the current mapping does and does not preserve

Nearest selection takes the highest-level containing leaf cell and copies its cell-centred values
to each target centre. It preserves finite source values and positivity pointwise for `rho/press`,
but it does not integrate source control-volume content. It therefore has no exact guarantee for:

- mass or internal-energy integrals;
- pressure-volume or magnetic-energy proxies;
- face magnetic flux;
- a discrete divergence constraint;
- refinement-interface continuity;
- polar-axis regularity beyond the source samples.

The target is uniform in index, logarithmic in `r`, periodic in `phi`, and bounded at inner/outer
radial faces and theta poles. A correct physical integral needs Kerr--Schild proper-volume weights,
not voxel count or `dr*dtheta*dphi` alone. The absent verified EOS also prevents an internal-energy
integral audit.

## Existing fixed-snapshot diagnostics

Stage J reused the read-only Round 2 comparison instead of generating large arrays. It covers
snapshots `0,10,50,79,90,95,100,110` plus adjacent companions at `64^3`.

| candidate / nearest | TV | block jump | refinement jump | high-k | persistence | adjacent residual RMS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| leaf-local trilinear | 0.953446 | 0.808350 | 0.957923 | 0.992008 | 0.981771 | 0.997352 |
| coordinate-weighted target-bin average | 1.006615 | 0.821512 | 0.974275 | 1.015368 | 0.973053 | 0.995984 |

Both candidates remain finite and keep `rho/press` positive. Both fail the frozen `rho/press`
radial-profile and quantile-span gates. The weighted prototype uses only coordinate
`dr*dtheta*dphi`; `54,016 / 262,144` target bins are empty and fall back to nearest. Neither method
is conservative or divergence preserving. These diagnostics support smoother meshblock boundaries,
not physical conservation.

## Boundary-specific risks

- **AMR block overlap:** selecting one containing leaf is deterministic, but copy/interpolation is
  not a reflux/restriction operation across refinement levels.
- **Refinement boundaries:** local trilinear interpolation clamps at block-local centre extents and
  cannot by itself enforce cross-level flux consistency.
- **Horizon/inner radial face:** a target-centre rule does not define a physical excision or inflow
  boundary condition.
- **Outer radial face:** nearest edge clamping can flatten gradients and cannot reproduce the
  simulation boundary flux.
- **Theta poles:** coordinate singularity requires parity/component rules not encoded in the
  processed grid.
- **Periodic phi:** query wrapping is implementable and was tested for the local trilinear
  diagnostic, but periodic sampling alone does not repair vector-basis or flux errors.

## Minimum scalar conservative transfer

A local scalar prototype would require the exact intersection volumes between source AMR leaf cells
and target cells, using the verified metric volume element; a finest-leaf coverage partition;
deterministic handling of target cells crossed by refinement boundaries; and integral before/after
tests. Restriction must use volume-weighted sums, while prolongation needs a conservative
reconstruction with limiter and boundary policy. `rho` conservation semantics must also be chosen
carefully because the raw channel is a primitive rest density, not automatically the conserved
density integrated by the code.

## Minimum magnetic transfer

Cell-centred `Bcc` interpolation cannot certify `div B`. A defensible route needs either:

1. original face-centred magnetic fluxes and a constrained-transport-compatible prolongation/
   restriction/reflux operator; or
2. a verified vector potential, curl reconstruction, gauge, and boundary treatment.

The current raw variable list exposes neither face fields nor vector potential. Therefore a
divergence-preserving exact conversion is classified `REQUIRES_NEW_SIMULATION` unless a richer
author export exists.

## Exact-reproduction suitability

```text
current nearest regrid: AVAILABLE_VERIFIED for reduced spherical proxy
conservative scalar transfer: IMPLEMENTABLE_LOCALLY after geometry provenance
divergence-preserving magnetic transfer: REQUIRES_AUTHOR_DATA or REQUIRES_NEW_SIMULATION
exact paper dataset: BLOCKED
```
