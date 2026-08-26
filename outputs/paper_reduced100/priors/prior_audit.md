# paper_reduced100 Stage D prior audit

- Status: `passed`
- Selected radial mode: `appendix_literal_press_proxy`
- Validation used for fit: `False`
- Validation diagnostics are report-only and do not update any artifact.

## train

| component | diagnostic | value |
| --- | --- | ---: |
| radial rho | envelope violation | 0.836654663 |
| radial press | envelope violation | 0.437615442 |
| bounds rho | raw violation | 0.000610351562 |
| bounds press | raw violation | 0.00165710449 |
| ROI | canonical/raw Jaccard | 0.702727099 |
| ROI | canonical clamp overlap | 0.54408605 |
| dissipation | max state norm | 3818.35846 |
| dissipation | fraction above Rin | 0 |

## validation

| component | diagnostic | value |
| --- | --- | ---: |
| radial rho | envelope violation | 0.41920681 |
| radial press | envelope violation | 0.309267235 |
| bounds rho | raw violation | 0.0417095184 |
| bounds press | raw violation | 0.10508728 |
| ROI | canonical/raw Jaccard | 0.650297416 |
| ROI | canonical clamp overlap | 0.463072918 |
| dissipation | max state norm | 4596.42333 |
| dissipation | fraction above Rin | 0.85 |
