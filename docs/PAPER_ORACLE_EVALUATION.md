# paper_reduced100 oracle-aware evaluation

## Status

The implementation is fixed by commit `1371a14` and consists of:

- `src/grmhd/paper_references.py`;
- `src/grmhd/paper_metrics.py`;
- `scripts/eval_paper_oracle_metrics.py`;
- `tests/test_paper_reference_metrics.py`.

Stage D does not replace or reinterpret this implementation.

## Reference states

The lossy `0.99 * gamma` inverse clamp makes the following states distinct:

1. `raw_physical_target`: the unchanged HDF5 target;
2. `oracle_physical_target`: canonical `decode(encode(raw_physical_target))`;
3. `normalized_target`: canonical `encode(raw_physical_target)`;
4. `normalized_prediction`: a model or baseline prediction in canonical normalized space;
5. `model_physical_prediction`: canonical `decode(normalized_prediction)`.

`oracle_physical_target` is the repository spelling of the requested
canonical-oracle physical state. It must not be silently replaced by the raw HDF5 target.

## Metrics and masks

The accumulator reports per-channel and aggregate relative L2 for normalized error,
model-to-oracle physical error, model-to-raw physical error, and the oracle-to-raw
preprocessing floor. It also records the exact squared-error numerator identity rather
than claiming that relative norms are additive.

Target and model inverse-clamp masks remain separate. Their intersection, model-only,
target-only, sign agreement, Jaccard, precision, and recall are reported explicitly.

The frozen validation baseline in `outputs/paper_reduced100/oracle_metric_baseline.*`
covers all 19 validation pairs. Its seeded random upstream FNO is untrained and is only a
metric-path check; it is not a training smoke or scientific result.
