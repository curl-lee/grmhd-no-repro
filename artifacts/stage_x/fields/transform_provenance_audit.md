# Vector-transform provenance audit

Searched sources:

- all files at `/mnt/d/GRMHD_data` (the directory contains only `mad98.prim.?????.athdf` files);
- ATHDF root attributes and datasets on representative and selected Stage-X files;
- project configs, scripts, source, artifacts, README, and notes for spin, metric, EOS, tetrad, orthonormal/coordinate basis, and conversion routines.

Available ATHDF provenance: `Coordinates=kerr-schild`, RootGrid/mesh metadata, face/centre arrays, and variable names. Not available: black-hole spin, exact spherical Kerr–Schild mapping convention, vector tensor/basis definition, Eulerian velocity convention, magnetic-field convention needed for a Cartesian transform, tetrad, or an Athena input/problem-generator file. The filename `mad98` is not interpreted as spin.

Required transformation inputs are therefore incomplete.

`VECTOR_TRANSFORM_PROVENANCE = INCOMPLETE`

`CARTESIAN_VECTOR_CONVERSION_AUTHORIZED = false`
