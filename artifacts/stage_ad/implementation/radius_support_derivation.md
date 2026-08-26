# Radius Support Derivation

`delta_cell = max(L_i/N_i) = 0.03125` and
`radius_cutoff = 3 * delta_cell = 0.09375`.  The frozen local-shape formula
resolves to `[7, 7, 7]`.

Deterministic integer-cell audit:

```json
{
  "1": {
    "all_bases_active": false,
    "positive_points_by_basis": [
      1,
      0,
      0,
      0,
      6
    ],
    "radius_cells": 1,
    "radius_cutoff": 0.03125,
    "stencil_shape": [
      3,
      3,
      3
    ]
  },
  "2": {
    "all_bases_active": false,
    "positive_points_by_basis": [
      1,
      0,
      18,
      20,
      14
    ],
    "radius_cells": 2,
    "radius_cutoff": 0.0625,
    "stencil_shape": [
      5,
      5,
      5
    ]
  },
  "3": {
    "all_bases_active": true,
    "positive_points_by_basis": [
      1,
      18,
      56,
      74,
      66
    ],
    "radius_cells": 3,
    "radius_cutoff": 0.09375,
    "stencil_shape": [
      7,
      7,
      7
    ]
  }
}
```

Thus m=1 and m=2 leave at least one hat inactive, while m=3 activates all five.  No
larger radius was tested or authorized.
