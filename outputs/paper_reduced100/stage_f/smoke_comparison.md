# Stage F controlled smoke comparison

This is a two-epoch engineering smoke comparison. It does not establish scientific
superiority or inferiority of either loss.

| field | Full | Plain |
| --- | ---: | ---: |
| status | `passed_with_h1_warning` | `passed` |
| model_parameter_count | `331832` | `331832` |
| initial_state_hash | `00dde6be92d4037f4abfc671cdd2153cfe98901148442903006e745ff3b86b43` | `00dde6be92d4037f4abfc671cdd2153cfe98901148442903006e745ff3b86b43` |
| epochs | `2` | `2` |
| train_batches | `4` | `4` |
| runtime_seconds | `4.795029431999865` | `2.992628899999545` |
| peak_allocated_mib | `0.0` | `0.0` |
| validation_normalized_global_relative_l2 | `1.0183352276751305` | `1.0156894920263964` |
| model_to_oracle_global_relative_l2 | `0.9372144042255971` | `0.9371735405808291` |
| model_to_raw_global_relative_l2 | `0.9551811966889182` | `0.9551236805699226` |
| oracle_floor_global_relative_l2 | `0.5043941588408832` | `0.5043941588408832` |
| rollout3_finite | `True` | `True` |
| rollout3_rho_press_positive | `True` | `True` |
| best_checkpoint_reload | `True` | `True` |
| last_checkpoint_reload | `True` | `True` |
| clipping_fraction | `1.0` | `1.0` |
| h1_value_ratio_max | `21.6084041595459` | `None` |
| h1_gradient_ratio_max | `8.646244609551902` | `None` |
| enabled_loss_components | `['base', 'h1', 'roi', 'bounds', 'envelope', 'dissipation']` | `['plain_l2']` |
| disabled_loss_components | `[]` | `['h1', 'roi', 'bounds_training_penalty', 'radial_envelope', 'dissipation']` |
