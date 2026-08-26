# Stage Y metric and coordinate provenance audit

## Result

The raw files establish a spherical coordinate-axis layout and an Athena++
`Coordinates=kerr-schild` label.  They do **not** contain the black-hole spin,
mass, exact producer commit, or input deck needed to bind those arrays to one
specific Kerr--Schild metric implementation.

```text
BH_SPIN_VALUE_CURRENT_RAW = UNKNOWN
BH_SPIN_VALUE_PAPER = 0.9
BH_SPIN_PROVENANCE = INCOMPLETE
COORDINATE_MAPPING_PROVENANCE = INCOMPLETE
SIMULATION_COORDINATE_MAPPING = kerr-schild label with (x1,x2,x3)=(r,theta,phi); exact run metric implementation/version UNKNOWN
```

The paper value `a=0.9` is an explicit fact about the paper simulation, not a
value that can be copied into the current raw files.

## Direct ATHDF evidence

The audited source series is
`/mnt/d/GRMHD_data/mad98.prim.00000.athdf` through
`mad98.prim.00211.athdf`; `data_raw` is a symlink to that directory.  The
machine-readable inventory is `simulation_file_manifest.csv`.  Representative
files 00000, 00055, 00110, and 00211, plus the frozen 212-row Stage-S manifest,
agree on:

| field | value |
|---|---|
| `Coordinates` | `kerr-schild` |
| `RootGridSize` (x1,x2,x3) | `[88,32,16]` |
| `MeshBlockSize` (x1,x2,x3) | `[22,4,16]` |
| `RootGridX1` | `[1.1,1200,1.0827303846611207]` |
| `RootGridX2` | `[0,pi,1]` |
| `RootGridX3` | `[0,2pi,1]` |
| `MaxLevel` / `NumMeshBlocks` | `3` / `2020` |
| levels | `{0:4,1:96,2:896,3:1024}` |

The numerical ranges, monotonic coordinate arrays, and grid construction make
the axis identity direct rather than name-based: `x1=r`, `x2=theta`, and
`x3=phi`.  This axis identity does not supply the missing spin or prove a
specific source revision.

For snapshots 00000/00055/00110/00211, byte hashes are identical for `Levels`,
`LogicalLocations`, and every `x1f/x1v/x2f/x2v/x3f/x3v` array.  In particular,
`LogicalLocations` SHA256 is
`0a2fa72a56c3e3fb70dea8e24fbb0e7f458317750a0359a4a9ea54afe82bcf7b`
and `Levels` SHA256 is
`91b94288132bb5f5b029bd71246239b426374a80cb4144f202602340ce5d83d0`.
This confirms a static sampled grid layout; it still does not encode metric
parameters.

The complete root-attribute allow-list is `Coordinates`, `DatasetNames`,
`MaxLevel`, `MeshBlockSize`, `NumCycles`, `NumMeshBlocks`, `NumVariables`,
`RootGridSize`, `RootGridX1`, `RootGridX2`, `RootGridX3`, `Time`, and
`VariableNames`.  No attribute stores `a`, `spin`, `M`, `metric_version`, MKS
mapping parameters, units, build hash, or run identifier.

## Public Athena++ source evidence and its limit

The local clean public clone `/home/curl/athena-public-version` is at
`9c266692b9423743d8e23509b3ab266a232a92d2`.  Its
`src/coordinates/kerr-schild.cpp:6-32` explicitly defines Kerr--Schild
`(t,r,theta,phi)` and its metric.  Lines 78-84 read `coord/m` and `coord/a`
from the runtime input.  This proves where a matching Athena++ run would carry
the missing spin and confirms the standard software convention; it does not
prove that this commit or any particular input produced the local files.

A separate public archive, Zenodo `10570521`, contains material for the
different paper *Modeling the inner part of the jet in M87*.  Its small public
conversion script uses spherical `r,theta,phi`, a command-associated MAD98
spin `0.98`, and Gamma `4/3`.  The archive is `THIRD_PARTY` relative to the
target paper and has no checksum/manifest binding its snapshots to the local
00000--00211 files.  It is therefore a simulation-family lead, not run
provenance.  The filename fragment `mad98` is not interpreted as a parameter.

## Paper/current separation

The target paper source `main.tex:354-359` says its GRMHD calculation uses
Cartesian Kerr--Schild coordinates and a Fishbone--Moncrief torus with
`a=0.9`.  Current raw metadata instead describes spherical axes and never
records `a`.  Consequently the paper statement completes the paper-side
contract only.

No modified-KS/FMKS mapping parameters (`hslope`, `R0`, or a theta mapping)
were found in the raw files or a matching input.  It would also be wrong to
label the raw mapping FMKS merely because an unrelated conversion script can
export to MKS/HARM.

## Local-search scope

The audit searched project tracked/ignored files, Git history (including
deleted paths), `/home/curl/projects`, `/home/curl/datasets`, the known raw-data
target, `/home/curl/athena-public-version`, and `/mnt/d/浏览器下载/athena-main`.
`/home/curl/data`, `~/Downloads`, and `~/Documents` do not exist in this WSL
home.  No matching Athena input deck, build log, problem-generator copy, run
manifest, restart record, or source commit was recovered.  Git history contains
no deleted matching provenance file.
