# Stage J Cartesian Kerr--Schild feasibility audit

## Decision

```text
Cartesian KS status: BLOCKED_COMPONENT_BASIS
Cartesian HDF5 generation: not authorized
Preferred resolution: obtain author data or a verified direct simulation export
```

The current files establish a spherical Kerr--Schild coordinate label and named stored components,
but they do not establish the run-specific spacetime parameters, component basis, or a uniquely
matched output implementation. A coordinate-point map alone cannot transform the velocity and
magnetic channels.

## Current raw evidence

All 111 raw headers consistently report:

- `Coordinates = kerr-schild`;
- primitive variables `rho, press, vel1, vel2, vel3`;
- cell-centred magnetic variables `Bcc1, Bcc2, Bcc3`;
- root grid `[88,32,16]`, AMR `MaxLevel=3`, and coordinate extents
  `r=[1.1,1200]`, `theta=[0,pi]`, `phi=[0,2pi]`.

The headers do **not** contain black-hole mass/spin, run ID, coordinate-map version, vector variance,
tetrad/orthonormal convention, units, or the matching input/source commit. The processed HDF5
retains the names and spherical axis order; it adds no missing physical convention.

The paper states a Cartesian Kerr--Schild `64^3` GRMHD dataset with spin `a=0.9`. That is a paper
requirement, not proof that these independently obtained `mad98` raw files use the same parameters.
The filename is not accepted as spin provenance.

## Point coordinates and component transforms are separate gates

### A. Grid-point map

A map from `(r,theta,phi)` to some Cartesian Kerr--Schild `(x,y,z)` requires a declared Cartesian
KS convention and the Kerr spin. Multiple coordinate conventions are called Kerr--Schild. The raw
header provides neither the run spin nor an implementation/version identifier, so even the
run-specific point map is not uniquely fixed by the current provenance.

### B. Vector components

Transforming points would not transform `vel1:3` or `Bcc1:3`. A component conversion also requires:

1. whether stored components are contravariant, covariant, orthonormal-frame, or tetrad values;
2. the precise primitive-velocity definition;
3. whether `Bcc` is a coordinate component, densitized field, or reconstructed cell-centred field;
4. the matching metric/Jacobian or tetrad convention;
5. the staggering and constrained-transport ownership needed to preserve magnetic constraints.

No such declarations occur in the ATHDF header. Therefore `vel1 -> vx` and `Bcc1 -> Bx` renaming
would be scientifically false.

## Public Athena source: useful hypothesis, not run provenance

The locally visible public Athena tree is clean at commit
`9c266692b9423743d8e23509b3ab266a232a92d2`, but it is not linked to the raw run. Its
`kerr-schild.cpp` (SHA-256
`5e95a63c458628596e724a64a08b2940abf72526eee6d80533c366c103cb0bd3`) explicitly implements
`(t,r,theta,phi)` Kerr--Schild coordinates and reads `coord/m` and `coord/a` from runtime input.
That supports the need for missing run parameters; it does not supply them.

In the same public tree, `outputs.cpp` (SHA-256
`2bb51bac9e3d7aed1d575005e18837d85427a489099f377`) writes `vel1:3` directly from the primitive
array and `Bcc1:3` from the reconstructed cell-centred field. Optional Cartesian vector output
creates separately named `vel_xyz`/`Bcc_xyz`. Its generic `CalculateCartesianVector` implementation
contains branches for `spherical_polar` and `cylindrical`, not `kerr-schild`. Consequently this
unmatched public source neither proves the raw component basis nor provides a verified drop-in KS
Cartesian conversion.

## Jacobian, metric, and magnetic constraint

For coordinate-basis contravariant vectors a verified Jacobian could in principle map components;
covariant components need the inverse Jacobian, and orthonormal/tetrad components need the matching
frame. The primitive GR velocity convention can additionally differ from coordinate three-velocity.
The correct path cannot be selected from variable names alone.

Even with a verified component transform, sampling transformed cell-centred `B` onto a Cartesian
cube by nearest or generic trilinear interpolation does not preserve the face-integrated magnetic
flux or discrete `div B=0`. Preserving the simulation constraint normally requires face fields and
a constrained-transport-compatible transfer, or a vector-potential reconstruction with documented
gauge/boundary handling. Only `Bcc` is present in the audited variable list.

## Resolution gate

Local conversion becomes eligible for a prototype only after obtaining all of:

- a content-linked run input/build record with mass, spin, coordinate convention, and units;
- exact definitions for primitive velocity and magnetic components;
- a verified Jacobian/tetrad implementation for that run;
- a declared target Cartesian grid and boundary convention;
- a magnetic transfer/validation plan, including the discrete divergence operator.

If the component or face-field metadata cannot be recovered, direct simulation export of Cartesian
components/face fluxes is required. Stage J generated no Cartesian arrays and modified no HDF5.
