# Stage M channel transform-floor sources

| channel | decision | recovered metric counts |
| --- | --- | --- |
| Bcc2 | `E. MULTIPLE_COMPONENTS` | `{"final_inverse_clamp": 4, "forward_softclip": 4, "nonlinear_transform": 0, "normalizer": 0}` |
| Bcc3 | `A. FORWARD_SOFTCLIP_DOMINATED` | `{"final_inverse_clamp": 0, "forward_softclip": 4, "nonlinear_transform": 0, "normalizer": 0}` |
| vel3 | `A. FORWARD_SOFTCLIP_DOMINATED` | `{"final_inverse_clamp": 0, "forward_softclip": 4, "nonlinear_transform": 0, "normalizer": 0}` |

The no-final-clamp diagnostic reports nonfinite values rather than replacing them.
Float64, isolated robust normalization, and isolated channel nonlinearities are reported separately.
