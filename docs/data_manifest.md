# Data manifest

Dataset files are not committed. The tables below record the local contracts
used by the released WSL workflows. SHA256 values were recomputed from the local
files before publication and match the existing stage manifests.

## Processed datasets

| Dataset | Expected local path | Shape | Bytes | SHA256 | Usage |
|---|---|---:|---:|---|---|
| reduced111 Z64 | `data_proc/grmhd_regrid_inner_r200_64.h5` | `(111,8,64,64,64)` | local symlink target | `cd3966bed1f617f49d775151cff87ae92eb2432a560bb82ae9355605e7aea70a` | Stage F--R reduced100 protocol |
| expanded Z64 | `data_proc/grmhd_regrid_inner_r200_64_expanded.h5` | `(212,8,64,64,64)` | 712187351 | `3582a5c4f52f4fef305578ffd5a63c51423f8583f35d137c288d5c20203b50da` | Stage S--AF frozen data contract |
| expanded Z96 | `data_proc/grmhd_regrid_inner_r200_96_expanded.h5` | `(212,8,96,96,96)` | 1672033912 | `292d3fe62297e91fda8e0f81ab156ca7ceb127ea02a004ca79180705e7411564` | Stage Z/AA/AB resolution comparison |
| expanded Z128 | `data_proc/grmhd_regrid_inner_r200_128_expanded.h5` | `(212,8,128,128,128)` | 3016015911 | `fe62e9c2d8e0311808a1be597002b8813ee6eab6ad42a1f882c8fcce37ae9b1d` | Stage Z information audit only |

All files store `snapshots` as float32 with axes `(N,C,Nphi,Ntheta,Nr)` and
channels `Bcc1,Bcc2,Bcc3,rho,press,vel1,vel2,vel3`. Coordinate arrays are
float64 datasets under `coords/{phi,theta,r}`. The radial edge range is
`[1.100000023841858, 200.0]`; phi spans a full periodic `2*pi`; theta spans
`[0,pi]` by cell centers. Z64/Z96/Z128 use 64/96/128 centers per direction.

## Raw source contract

- Local source directory used during generation: `/mnt/d/GRMHD_data`.
- Files: `mad98.prim.00000.athdf` through `mad98.prim.00211.athdf`.
- Count: 212 snapshots.
- Athena ordering: `(variable,meshblock,x3,x2,x1)`.
- Coordinate mapping: `x1=r`, `x2=theta`, `x3=phi`.
- Coordinate system: Kerr--Schild spherical coordinates.
- Root grid: `(x1,x2,x3)=(88,32,16)`.
- MeshBlock: `(x1,x2,x3)=(22,4,16)`.
- Leaf MeshBlocks: 2020, with level counts `{0:4,1:96,2:896,3:1024}`.
- Source grid bounds: `r=[1.100000023841858,1200]`,
  `theta=[0,pi]`, `phi=[0,2*pi]`.

The full per-snapshot checksum and grid audit is
[`artifacts/stage_s/data_manifest.csv`](../artifacts/stage_s/data_manifest.csv).

## Regridding contract

`src/build_regrid_from_athdf.py` constructs logarithmically spaced radial
targets, linearly spaced theta centers, and linearly spaced periodic phi
centers. For each target cell it chooses the finest leaf block containing the
query and samples the nearest source cell center. Mapping reuse is guarded by a
source-grid signature.

The method is not conservative, performs no interpolation, and does not
preserve `div(B)` to machine precision. Magnetic and velocity components remain
stored coordinate-basis components; they are not converted to Cartesian or an
orthonormal physical frame.

Regrid provenance and coordinate arrays are retained in:

- `outputs/paper_reduced100/manifest.json`
- `artifacts/stage_s/expanded_regrid_report.json`
- `artifacts/stage_z/regrid/z96_manifest.json`
- `artifacts/stage_z/regrid/z128_manifest.json`

## Acquisition

The original scientific dataset is not redistributed here. A user must obtain
authorized source snapshots independently, verify them against the published
per-file manifest, and then run the regrid/preprocessing scripts. No claim is
made that the paper's official dataset or exact Cartesian-KS preprocessing is
publicly recoverable; Stage Y documents the missing provenance/assets.
