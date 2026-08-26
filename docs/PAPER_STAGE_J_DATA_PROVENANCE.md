# Stage J: GRMHD data and snapshot provenance audit

## Decision

The local raw trajectory contains exactly 111 readable, contiguous snapshots,
not 300.  It is a complete local sequence numbered 0--110, but there is no
author manifest proving that it is the first, last, or sampled subset of the
paper's 300-snapshot trajectory.  The paper `last 250` protocol therefore cannot
be reconstructed from current data.

```text
raw_snapshot_count: 111
status: BLOCKED_300_SNAPSHOT_PROTOCOL
matching_run_config: not_found
processed_hdf5_traceability: verified_file_names_and_times
```

The machine-readable inventory is
`outputs/paper_reduced100/stage_j/data_provenance.json`.  It contains path, size,
complete-file SHA-256, modification time, available HDF5 variables/coordinates,
snapshot/time/cycle, mesh metadata, EOS/Gamma fields, and provenance relation for
every candidate.  The audit stores only allow-listed physics metadata from text
configs and does not copy credentials or unrelated contents.

## Scan scope

The explicit `/home/curl/datasets` scan contains three processed HDF5 files and
no ATHDF.  The project's `data_raw` entry is a symlink to
`/mnt/d/GRMHD_data`; Stage J follows that known data link and finds the actual
111 ATHDF files.  Project YAML/TOML/JSON candidates and the locally available
public Athena sample inputs were inventoried to test for an exact run-config
match.

| record class | count |
|---|---:|
| raw ATHDF through `data_raw` | 111 |
| processed HDF5 under `/home/curl/datasets` | 3 |
| project metadata/config candidates | 430 |
| public Athena sample `athinput` candidates | 79 |
| total records | 623 |

Hashing covered all file contents, including approximately 9.5 GiB of raw
ATHDF.  No data file was opened for writing and field arrays were not loaded by
the Stage J inventory.

## Raw sequence

| item | value |
|---|---|
| path | `/mnt/d/GRMHD_data/mad98.prim.00000.athdf` through `mad98.prim.00110.athdf` |
| indices | 0--110, no missing or duplicate number |
| physical time | 0.0--1100.0006921459083, strictly increasing |
| cadence | mean 10.0000062922; median 10.0006590660; min/max 9.9988639379/10.0006590667 |
| cadence check | approximately constant at relative tolerance `1e-3` |
| cycles | 0--612770 |
| coordinates | `kerr-schild` |
| variables | `rho, press, vel1, vel2, vel3, Bcc1, Bcc2, Bcc3` |
| AMR | max level 3; 2020 blocks; block size `[22,4,16]` |
| root grid | `[88,32,16]`; X1 `[1.1,1200,1.0827303846611207]` |
| first raw SHA-256 | `7f2b88e87a88b2bc653b33b4aa56f5007cfe0095f4e4aca44c79e9f6f9767007` |
| last raw SHA-256 | `275aacfbacc1d0f8f3a2e0137c254ac1dda2f7f284e62384d58fe5c91ba94edf` |

All raw files share the same HDF5 schema and grid signature in the earlier full
field audit.  Their root attributes contain no author dataset ID, parent
trajectory ID, black-hole spin, units, EOS, Gamma, basis convention, or
coarse/fine relation.  The filename fragment `mad98` is not accepted as proof of
spin or any other physical parameter.

## Processed files and traceability

| processed file | snapshots | raw relation | time relation |
|---|---:|---|---|
| `grmhd_regrid_full_64.h5` | 21 | files 00000--00020 all present | exact |
| `grmhd_regrid_inner_r200_64.h5` | 21 | files 00000--00020 all present | exact |
| `grmhd_regrid_inner_r200_64_n111.h5` | 111 | files 00000--00110 all present | exact |

The canonical project symlink resolves to the 111-snapshot file.  Its SHA-256
remains
`cd3966bed1f617f49d775151cff87ae92eb2432a560bb82ae9355605e7aea70a`,
shape `(111,8,64,64,64)`, and axis order `(N,C,Nphi,Ntheta,Nr)`.  Every stored
`source_file` exists locally and every stored source time exactly equals its raw
ATHDF `Time` attribute.  The existing build report also records the source file
list and audit manifest.

This establishes file/time lineage, not a lossless physical transform.  The
processed file is a nearest-cell-centre AMR regrid, explicitly non-conservative
and non-divergence-preserving.  The original build report did not embed each raw
file's SHA-256; the Stage J inventory now records those hashes externally without
rewriting the HDF5.

## Answers to the provenance questions

1. **How many raw snapshots exist?** 111.
2. **Are there exactly 300?** No.
3. **Are the 111 a known subset of a complete 300?** They are a complete,
   contiguous *local* sequence; no author manifest relates them to the paper's
   300, so their paper-trajectory position is unknown.
4. **What is the cadence?** Approximately 10 code-time units with the ranges
   above.  The paper defines `Delta T=T/Ndata` but does not identify this local
   trajectory as that dataset.
5. **Are numbering and physical time continuous?** Numbering has no gap and
   times are strictly increasing with approximately constant cadence.
6. **Is the author's paper-data manifest present?** No.
7. **Is the last-250 range present?** No; 111 is less than 250.
8. **Are matching coarse/fine resolutions present?** No.  The three processed
   files are full/inner views of the same raw sequence, not a verified paired
   coarse/fine simulation dataset.
9. **Is an exact `athinput`/run config present?** No.  Seventy-nine public sample
   inputs were scanned; none match the raw root grid and radial extent.  In
   particular, the public `athinput.fm_torus` uses a different spin, grid,
   radius, and cadence and is explicitly unrelated.
10. **Can the canonical HDF5 be traced to raw files?** Yes at file-name and exact
    time level for all 111 snapshots, with the lossy regrid limitations above.

## 300-snapshot reconstruction status

Current provenance does not contain a reliable continuation state, exact
AthenaK/Athena++ commit, problem generator revision, run input, EOS, spin, units,
boundary conditions, AMR prescription, or author paper manifest.  Consequently
Stage J cannot claim that continuing or rerunning the local simulation would
reconstruct the paper dataset.

```text
snapshot_decision: NOT_REPRODUCIBLE_FROM_CURRENT_PROVENANCE
author_data_path: viable_if_author_supplies_manifest_and_files
new_simulation_path: blocked_until_exact_run_provenance_is_obtained
```

Snapshots must not be copied, interpolated, or cycled to manufacture a count of
300.  No simulation plan is created because the prerequisites for a reliable
rerun are absent.
