# Stage H training-log correlations

All coefficients are descriptive post-hoc statistics. They do not identify
causal effects and were not used to fit or tune any training parameter.

| Comparison | n | Pearson r | Spearman rho | Status |
| --- | ---: | ---: | ---: | --- |
| h1_base_value_vs_validation_average_l2 | 30 | -0.826203 | -0.983537 | descriptive_only |
| h1_base_gradient_vs_validation_average_l2 | 30 | 0.49053 | -0.23515 | descriptive_only |
| h1_base_value_vs_validation_global_l2 | 30 | -0.751172 | -0.979088 | descriptive_only |
| h1_base_value_vs_clipping_fraction | 30 | NA | NA | undefined_constant_series |
| preclip_total_gradient_vs_parameter_update_norm | 600 | 0.480018 | 0.487825 | descriptive_only |
| h1_base_gradient_ratio_vs_parameter_update_norm | 600 | 0.11535 | 0.0568751 | descriptive_only |
| h1_base_value_vs_outer_shell_error | 0 | NA | NA | not_computed |
| h1_base_value_vs_artifact_flags | 0 | NA | NA | not_computed |
| inner_shell_h1_vs_morphology_score | 0 | NA | NA | not_computed |

Unavailable epoch-aligned outer-shell, artifact, and morphology series
are reported as not computed rather than inferred from final-checkpoint
summaries. Full clipping fraction is constant at 1.0, so its correlation
with H1/base is mathematically undefined.
