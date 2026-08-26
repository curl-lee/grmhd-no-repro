# Stage S expanded-data audit

## Counts

- `OLD_SNAPSHOT_COUNT = 111`
- `NEW_SNAPSHOT_COUNT = 101`
- `TOTAL_UNIQUE_SNAPSHOT_COUNT = 212`
- `TIME_SERIES_CONTIGUOUS = true`
- `CADENCE_CONSTANT = true`
- `GRID_LAYOUT_STATIC = true`
- Raw merge gate: `passed`.

## Time and split

- Time range: `0.0..2110.000838137111`.
- Delta-t min/median/max: `9.998863937861415` / `10.000659066036974` / `10.000659066670323`.
- Train snapshots/pairs: `169` / `168` over `[0,169)`.
- Validation snapshots/pairs: `43` / `42` over `[169,212)`.
- Dropped boundary pair: `[168, 169]`.

## Compatibility

- Old/new coordinate system, channel ordering, dataset shapes, root/block grids, AMR levels/layout, domain, and coordinate arrays are identical.
- Component-basis and physical-unit labels are absent from both intervals. Compatibility is established operationally by the continuous file/time/cycle series and identical schema/grid contract; stored components remain spherical Kerr-Schild coordinate components.
- Recycle-bin ATHDF candidates are excluded copies, not additional samples.

## Processed expanded dataset

- Path: `/home/curl/projects/grmhd-no-repro/data_proc/grmhd_regrid_inner_r200_64_expanded.h5`.
- Shape/dtype: `[212, 8, 64, 64, 64]` / `float32`.
- SHA-256: `3582a5c4f52f4fef305578ffd5a63c51423f8583f35d137c288d5c20203b50da`.
- `DISTRIBUTION_SHIFT = STRONG`.
- The shift is reported, not filtered or used to remove difficult snapshots.
