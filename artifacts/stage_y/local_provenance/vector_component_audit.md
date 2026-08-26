# Stage Y vector-component audit

## Result

The software-family meanings can be recovered, but the exact-current-run
source/input binding cannot.  The strongest non-speculative statements are:

```text
MAGNETIC_COMPONENT_CONVENTION = Athena++ cell-centred Bcc values copied from the reconstructed field array; public KS source treats the underlying global-coordinate magnetic values as contravariant B^i; exact-run source binding UNKNOWN
VELOCITY_COMPONENT_CONVENTION = Athena++ GR primitive slots IVX/IVY/IVZ; public KS source defines them as tilde-u^i = gamma v^i rather than coordinate dx^i/dt; exact-run source binding UNKNOWN
VECTOR_TRANSFORM_PROVENANCE = INCOMPLETE
CARTESIAN_VECTOR_CONVERSION_AUTHORIZED = false
```

These are not orthonormal/tetrad components in the inspected public source, and
they are not the paper's already-Cartesian `Bx/By/Bz` and `vx/vy/vz` fields.

## Raw-file evidence

All 212 files have `DatasetNames=[prim,B]` and
`VariableNames=[rho,press,vel1,vel2,vel3,Bcc1,Bcc2,Bcc3]`.  A representative
`prim` dataset is shaped `(5,2020,16,4,22)` and `B` is
`(3,2020,16,4,22)`, i.e. variable, MeshBlock, x3, x2, x1 storage.  The header
does not declare variance, tetrad, primitive-velocity formula, densitization,
or a Cartesian basis.

## Versioned public Athena++ evidence

The clean public clone used only as a convention reference is
`/home/curl/athena-public-version` at commit
`9c266692b9423743d8e23509b3ab266a232a92d2`.

- `src/outputs/outputs.cpp:384-392` names primitive `IPR` as `press`.
- `src/outputs/outputs.cpp:485-510` copies the three velocity slots from the
  primitive array `phyd->w`; optional Cartesian output would be separately
  named `vel_xyz`.
- `src/outputs/outputs.cpp:586-606`
  copies `pfld->bcc` to a distinct cell-centred `Bcc` dataset; optional
  Cartesian output is separately named `Bcc_xyz`.
- `src/coordinates/kerr-schild.cpp:1062-1081` documents the global inputs as
  `tilde{u}^1,tilde{u}^2,tilde{u}^3` and `B^1,B^2,B^3` before a local-frame
  interface transform.
- `src/eos/adiabatic_mhd_gr.cpp:550-556` stores `gamma*v1:3` in the primitive
  slots.  Lines 583-613 reconstruct four-velocity and comoving magnetic field
  from `uu1:3` and `bb1:3`.

Thus, for that exact public revision, `vel1:3` are projected/primitive
`tilde-u` spatial components and `Bcc1:3` are cell-centred coordinate magnetic
components used as `B^i`, not ordinary Euclidean vector components.

The generic output helper at `src/outputs/outputs.cpp:1605-1663` has Cartesian
conversion branches for `spherical_polar` and `cylindrical`, not
`kerr-schild`.  It is not a verified KS conversion path.

## Independent family-level corroboration

The ranged small-file inspection of the third-party Zenodo MAD98 analysis
script `/tmp/stage_y_zenodo_ath2h5.py` independently reads `x1v,x2v,x3v` as
`r,theta,phi` (lines 47-53), reads the same eight names (57-64), reconstructs
`u^mu` from `uu1:3` (70-92), and reconstructs `b^mu` from `Bcc1:3` (94-102).
That is consistent with the public Athena++ source.  It is not an official
target-paper asset and is not content-linked to the local files, so it cannot
close provenance.

## Why Cartesian conversion remains blocked

A scientifically valid conversion needs the exact run's spin and metric map,
the exact producer convention, a declared target Cartesian-KS convention, and
a validated Jacobian/tetrad treatment.  Magnetic remapping additionally needs
a flux/CT contract: only cell-centred `Bcc` is present, so ordinary
nearest/trilinear resampling cannot be claimed to preserve discrete
`div B=0`.  Renaming 1/2/3 to x/y/z would be false.
