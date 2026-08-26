# Stage V exact Plain-objective audit

## Frozen implementation

Stage T calls `PlainL2Loss` on `predicted_residual` and
`residual_target = z[t+1] - z[t]` in `scripts/train_stage_t.py` (the
`train_group` call path).  `PlainL2Loss` delegates to
`PaperComponentFidelityLoss(magnetic_weight=1, velocity_weight=1)` in
`src/grmhd/paper_losses.py`.

For batch size `B`, eight channels, and `V=Nphi*Ntheta*Nr` voxels, the exact
scalar is

```text
L_plain = sum(c=1..8) [ 1/(B V) * sum(b=1..B,v=1..V)
                        (Delta-z-pred[b,c,v] - Delta-z-true[b,c,v])^2 ]
```

Equivalently it is the batch mean of the channel sum of spatial MSEs.  On the
frozen Stage-T `batch_size=1` path, every microbatch produces one such scalar;
an accumulation group divides each of its four microbatch losses by four
before backward.  The final partial-group rule exists generically, although
168 train pairs divide exactly into 42 four-microbatch optimizer updates.

## Weighting consequences

| item | exact Stage-T behavior |
| --- | --- |
| prediction/target space | P3 canonical normalized residual |
| channel aggregation | spatial MSE per channel, then sum over 8 channels |
| channel coefficients | all exactly 1; no magnetic or velocity reweighting |
| voxel weighting | uniform tensor-cell weight `1/V` |
| shell weighting | none |
| radial weighting | none |
| spherical volume/Jacobian weighting | none |
| physical-unit weighting | none |
| H1/ROI/bounds/envelope/dissipation | all disabled |

“Equal channel coefficients” does not mean every channel, shell, or event has
equal realized influence.  Squaring the pointwise residual error means larger
normalized-amplitude errors contribute quadratically more.  A low-amplitude
but dynamically important coherent transport can therefore contribute far
less than a high-amplitude local residual error.  Uniform `(phi,theta,r)`
voxel weighting also does not represent spherical physical volume: no
`r^2 sin(theta)` or Kerr--Schild proper-volume factor is present.

This audit is based on executable code, not report prose:

- `src/grmhd/paper_losses.py`: `PaperComponentFidelityLoss.components` and
  `PlainL2Loss`.
- `scripts/train_stage_s.py`: `StageSBatchPath.pair` defines the P3 residual
  target.
- `scripts/train_stage_t.py`: `train_group` applies Plain loss to predicted
  and target residuals and performs accumulation normalization.
- `configs/stage_s/expanded_localno_p3_residual.yaml`: frozen P3, residual,
  model, optimizer, and disabled-loss contract.
