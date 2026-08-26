# Stage I Run A no-H1 config difference

- Status: `passed`
- Full config: `configs/paper_reduced100/full_fno_proxy.yaml`
- no-H1 config: `configs/paper_reduced100/extensions/no_h1_control.yaml`
- Selected H1 training contribution: `0`
- All non-H1 training-objective fields: exact
- no-H1 is Plain L2: `false`
- `checkpoint_prefix` is classified as extension output metadata.

## Differences

- `checkpoint_prefix`
- `experiment_name`
- `loss.h1.diagnostic_current_upstream_h1`
- `loss.h1.enabled`
- `loss.h1.implementation`
- `loss.h1.metric_adaptation`
- `loss.h1.mode`
- `loss.h1.paper_reference_weight`
- `loss.h1.weighted_contribution`
- `output_dir`
- `reproduction_metadata.extension_reason`
- `reproduction_metadata.paper_faithful_full`
- `reproduction_metadata.reproduction_level`
