# Reproduction status through Stage AH

## Completed controlled run

Stage AG completed the frozen anisotropic spherical DISCO LocalNO run with
300 epochs, 50,400 microbatches, and 12,600 optimizer updates. The model has
363,480 trainable parameters and starts from trainable-state SHA256
`77252855f054a199500fa779f7340b33b6a2ce546b2082f17f93666275f4c588`,
matching Stage AD/AF. The expanded Z64 dataset, train-only P3 preprocessing,
split, pair order, Plain residual L2 objective, optimizer, scheduler, and budget
all match the frozen controlled-comparison contract. Nonfinite count is zero;
formal best and last checkpoints both reload strictly. The formal best epoch is
300 under the 42-pair normalized per-channel state relative-L2 arithmetic mean.

## Stage AH result

| model | state L2 | residual L2 | cosine | shell skill | radial skill | first10x |
|---|---:|---:|---:|---:|---:|---:|
| Stage T LocalNO | 0.26671132 | 0.89185143 | 0.74060210 | -1.67947218 | -0.34808888 | 1 |
| Stage AD index-space DISCO3D | 0.26406202 | 0.89163995 | 0.75746772 | -4.35448118 | -0.39035242 | 1 |
| Stage AG anisotropic spherical DISCO3D | 0.27226335 | 0.91680556 | 0.74758286 | -1.59370979 | -0.30184608 | 1 |

State retention passes the predeclared 1.05x Stage-AD threshold. Median shell
skill improves by 2.76077 and the corresponding median absolute error falls
from 0.939687 to 0.550969, so the shell geometry gate passes. Radial skill
improves by 0.08851 and median absolute error falls, but the skill change does
not reach the frozen +0.10 threshold. Residual L2 and residual direction are
worse, and the first-10x physical-range landmark remains step 1. The spherical
DISCO branch is active; removing it post hoc severely degrades all primary
one-step and transport metrics.

Paired bootstrap confidence intervals show consistent unfavorable state,
residual-L2, and direction deltas. Absolute shell error has a favorable median
and 69% win fraction but a heavy-tailed mean whose 95% interval crosses zero;
radial absolute error has a favorable median and 79% win fraction, also with an
interval crossing zero. Regional attribution is mixed: several equatorial,
middle-r, and north-polar summaries improve, while south-polar and outer-r
summaries worsen. The six-point geometry-difference correlation remains an
exploratory diagnostic, not causal evidence.

`PRIMARY_DECISION = B — SPHERICAL_GEOMETRY_PARTIALLY_IMPROVES_DYNAMICS`

## Scope and limitations

The result applies only to the tested anisotropic spherical tangent-proxy
operator in this adapted spherical Kerr--Schild workflow. The geometry is not
Kerr--Schild covariant, vector components are not transformed, `press` remains
the thermal proxy, and the exact paper volumetric 3-D DISCO implementation and
official data/coupling remain unavailable.

`REPRODUCTION_SCOPE = ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION`

`EXACT_REPRODUCTION_BLOCKED = true`

The frozen details are in `artifacts/stage_ag/` and `artifacts/stage_ah/`.
