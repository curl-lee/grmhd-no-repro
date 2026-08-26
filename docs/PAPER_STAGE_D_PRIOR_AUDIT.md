# paper_reduced100 Stage D prior audit

## Scope and invariants

Stage D fits only snapshots `11..90` from the canonical HDF5 with SHA-256
`cd3966bed1f617f49d775151cff87ae92eb2432a560bb82ae9355605e7aea70a`.
Snapshots `91..110` are used only after fitting for distribution-shift diagnostics.
The transition `90 -> 91` remains dropped.

The preprocessor is frozen at `gamma=6`, inverse clamp fraction `0.99`, and checksum
`1645714424d95b02de2283fc16a8ef05f3864d15f11025580af13f3e67964001`.
The thermal channel remains `press`; no EOS conversion or Cartesian reinterpretation is
performed.

## Physical bounds

| channel | raw physical bounds | normalized bounds | train violation | validation violation |
| --- | --- | --- | ---: | ---: |
| rho | `[3.66563495e-6, 1.20927669]` | `[-1.99673226, 2.52715889]` | `0.000610352` | `0.0417095` |
| press | `[2.03421621e-8, 0.0189266652]` | `[-2.74504451, 1.89891706]` | `0.00165710` | `0.105087` |

Only rho/press can be evaluation-clamped. Magnetic and velocity channels are returned
unchanged, and the API retains the unclamped tensor and full hit mask. Lower-penalty
weights are `0.05`; both upper-penalty weights remain zero. No total paper loss is built
in Stage D.

## Residual envelope

The selected literal baseline and prediction are compared in canonical normalized
rho/press space. Both channels use `Delta=1.5` and weight `0.05`. Literal envelope
violation fractions are:

| split | rho | press |
| --- | ---: | ---: |
| train | `0.836655` | `0.437615` |
| validation | `0.419207` | `0.309267` |

These high rates are retained as an audit result and are not tuned using validation.

## Velocity ROI

The main mask is the per-snapshot top 20% of the canonical paper-decoded
`sqrt(vel1^2+vel2^2+vel3^2)` stored-component proxy. The raw physical mask is diagnostic
only. Neither quantity is a metric-correct relativistic speed.

| split | canonical/raw Jaccard | disagreement | canonical/clamp overlap | raw/clamp overlap |
| --- | ---: | ---: | ---: | ---: |
| train | `0.702727` | `0.0698348` | `0.544086` | `0.699463` |
| validation | `0.650297` | `0.0847614` | `0.463073` | `0.581603` |

The fixed training multiplier is `kappa=8` with `min(1, epoch/375)`. Stage D implements
the ramp and independent normalized velocity ROI relative error but does not invoke them
from a training loss.

## Dissipative global-norm reference

The norm axes are channel plus all three spatial dimensions; batch is excluded.

| quantity | value |
| --- | ---: |
| `Rmax` | `3818.35845738` |
| `Rin=1.05*Rmax` | `4009.27638024` |
| `Rout=1.5*Rin` | `6013.91457037` |
| `beta` | `10` |
| `alpha` | `5e-4` |

No train snapshot exceeds `Rin`. Validation has maximum norm `4596.42332737`, with
`85%` of validation snapshots above `Rin` and none above `Rout`. This is reported as a
distribution shift; `Rmax`, `Rin`, and `Rout` remain train-only.

The gate is a normalized global-array regularizer, not a physical dissipation rate,
covariant GRMHD norm, or the older gradient-dissipation extension.

## Outputs

All JSON artifacts validate HDF5 checksum, protocol, train indices, preprocessing
checksum, and thermal channel when loaded. Full JSON/CSV/Markdown diagnostics are under
`outputs/paper_reduced100/priors/`. These generated outputs are ignored by Git and contain
no model, optimizer, checkpoint, or TensorBoard data.
