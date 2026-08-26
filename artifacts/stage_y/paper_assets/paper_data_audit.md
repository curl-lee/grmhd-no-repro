# Stage Y paper-data acquisition audit

## Decision

```text
OFFICIAL_PAPER_DATA_FOUND = false
PAPER_DATA_SAMPLE_VERIFIED = false
PAPER_REMAP_METHOD = UNKNOWN
```

No official downloadable dataset, manifest, sample tensor, normalization
statistics, coordinate file, snapshot list, or checksum was found.

## Contract recoverable from the paper

| property | paper statement | completeness |
|---|---|---|
| generated snapshots | 300 over `[0,T]` | explicit |
| cadence | `DeltaT=T/Ndata` | symbolic; numeric T/cadence absent |
| used snapshots | last 250 | explicit |
| split | 80/20 train/validation | explicit; exact indices/order absent |
| tensor | eight fields on `64^3`, `(C,D,H,W)` | explicit |
| geometry | GRMHD simulation in Cartesian Kerr--Schild | explicit |
| spin | `a=0.9` | explicit |
| fields | `Bx,By,Bz,rho`, thermal field, `vx,vy,vz` | thermal name contradictory |
| numeric cube bounds | not stated | unknown |
| raw-to-64 remap | not stated | unknown |
| dtype/storage/layout | not stated beyond tensor order | unknown |
| normalization statistics | algorithm given; fitted values absent | unknown |

Evidence locations are `main.tex:176-188`, `226`, `354-359`, and `378-427`.
The GRMHD output list says pressure `P`; the model representation says
`eint`.  See `PAPER_THERMAL_VARIABLE_EVIDENCE.md`.

## Search results

The arXiv source, NeurIPS/OpenReview record, author homepage, author public
repositories, and targeted public data-host searches did not expose paper data.
The source package has no supplementary manifest or URL.

Zenodo record `10570521` is an open 2,011,328,260-byte archive for the different
paper *Modeling the inner part of the jet in M87*.  Only its public metadata,
ZIP central directory, and three small ranged analysis-source members were
inspected; the 2 GB archive was not downloaded.  It contains a different-paper
MAD98 snapshot and analysis utilities.  Similar filenames/grid-family evidence
does not make it the target paper dataset or establish a cryptographic chain to
the current local files.  It is classified `THIRD_PARTY` and excluded from
exact-reproduction evidence.

The current 212 local spherical ATHDF snapshots are likewise not a verified
paper sample: they differ in representation, count, thermal name, and lack an
author manifest.

## Acquisition behavior

Stage Y intentionally did not download any large dataset.  Since no official
target-paper data endpoint was found, there was no target sample to verify for
shape, dtype, channels, coordinates, times, or fitted normalization metadata.
