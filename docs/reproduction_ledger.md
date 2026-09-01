# Adapted reproduction ledger

This ledger separates verified project-local behavior from scientific adaptations
and from paper assets that remain unavailable. The scope is
`ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION`; it is not an exact reproduction of
the paper's numerical results.

## VERIFIED / REPRODUCED

- The frozen 64³ workflow runs on the expanded 212-snapshot dataset with axes
  `(N,C,phi,theta,r)` and eight state channels.
- The temporal contract uses 168 train pairs, drops transition 168→169, and
  evaluates the 42 validation pairs 169→170 through 210→211.
- The train-only P3 normalizer and normalized-residual target are checksum
  locked and are used consistently across the final benchmark.
- Persistence, spectral-only FNO, parameter-matched 3D CNN/U-Net,
  Differential LocalNO, index-space DISCO3D LocalNO, and anisotropic spherical
  DISCO3D LocalNO share the final input, target, loss, split, and evaluation
  contract.
- Differential LocalNO, the repaired volumetric DISCO extension, and the
  anisotropic spherical geometry implementation have unit, numerical, and GPU
  provenance artifacts.
- The Stage AD versus Stage AG geometry intervention is controlled, and formal
  checkpoints are strict-loadable with deterministic probes.
- The final baseline benchmark records one-step, per-channel, transport,
  regional, paired-bootstrap, closed-loop rollout, and computational-cost
  outputs on the same validation population.

## ADAPTED

- The dataset is a spherical Kerr--Schild 64³ workflow derived from unavailable
  original-run provenance, not the paper's official dataset.
- Athena++ AMR fields are mapped by nearest-cell selection to the tensor grid;
  this operation is not conservative and does not preserve magnetic divergence.
- P3 preprocessing, physical-radius shell channels, and the pressure thermal
  channel are project-local frozen adaptations.
- The FNO and CNN are final project baselines; neither is the paper's exact
  operator implementation.
- The project-local volumetric DISCO3D uses compact index-space radial hats.
- The spherical DISCO3D variant uses an anisotropic Euclidean tangent-distance
  proxy and a spherical-volume proxy, not proper Kerr--Schild geometry.
- The final benchmark is a controlled adapted architecture comparison, not a
  direct comparison to paper Table 2.

## BLOCKED / UNAVAILABLE

- Official paper data and exact dataset-generation provenance.
- Official paper training and evaluation code.
- The paper's exact volumetric 3D DISCO implementation.
- Exact Cartesian-Kerr--Schild preprocessing and remap semantics.
- Exact current-run spin, EOS, and source provenance.
- Exact vector-basis transform and definitive pressure/internal-energy
  semantics.
- Exact paper coarse/fine coupling implementation.

`EXACT_REPRODUCTION_BLOCKED = true`
