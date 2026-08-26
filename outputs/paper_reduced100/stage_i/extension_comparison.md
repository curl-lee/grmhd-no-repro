# Final Stage I extension comparison

All rows use the frozen reduced100 validation, preprocessing oracle, selected rollout steps, aggregation, and artifact thresholds.

| model | validation average | global |
|---|---:|---:|
| canonical_oracle | 0 | 0 |
| persistence | 0.189088 | 0.235074 |
| full | 0.462782 | 0.387339 |
| plain | 0.47458 | 0.381627 |
| no_h1 | 0.482824 | 0.386497 |
| unit_index | 0.482485 | 0.385469 |
| stored_coordinate | 0.444272 | 0.362543 |

## Decision

`D — No extension rescues Full`
