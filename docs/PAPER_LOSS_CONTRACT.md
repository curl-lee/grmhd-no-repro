# paper_reduced100 loss tensor-space contract

## Scope and evidence

This contract fixes Stage E before any composite-loss code is written. Its paper source
is arXiv:2512.01576v1, especially Appendix C.5--C.7 and Appendix D.1. The software
reference is the pinned `external/neuraloperator` commit
`86a8bc7812a31b42c4f7895693cf4ac11521c066`, specifically
`losses/data_losses.py`, `losses/meta_losses.py`, `training/trainer.py`, and
`scripts/train_mhd64.py`.

The implementation remains a paper-adapted reduced reproduction: `press` is the thermal
proxy for the paper's `eint`, and the three array derivatives are computational-grid
derivatives on the spherical Kerr--Schild `(phi, theta, r)` storage axes. They are not a
covariant or spherical-metric H1 norm.

## Tensor contract

All state tensors are batch-first `(B, 8, Nphi, Ntheta, Nr)` and use the fixed channel
order `[Bcc1, Bcc2, Bcc3, rho, press, vel1, vel2, vel3]`.

| name | exact meaning | allowed training use |
| --- | --- | --- |
| `normalized_input` | current raw snapshot after canonical paper encode | dissipative gate/input norm |
| `normalized_target` | next raw snapshot after canonical paper encode | base, H1, and ROI fidelity target |
| `normalized_prediction` | model output in canonical normalized state space | every prediction-side training term |
| `raw_physical_target` | unchanged next HDF5 snapshot | diagnostics only |
| `oracle_physical_target` | `decode(normalized_target)`, including the `0.99*gamma` inverse-clamp loss | canonical ROI provenance and diagnostics only |
| `canonical_roi_mask` | per-sample top 20% stored-component velocity proxy from `oracle_physical_target` | ROI fidelity mask |
| `raw_roi_diagnostic_mask` | optional top 20% proxy from `raw_physical_target` | Jaccard diagnostics only; never the training mask |
| `radial_baseline_normalized` | selected Stage D literal rho/press baseline mapped to canonical normalized space | radial envelope reference |
| `normalized_bounds` | Stage D train-only rho/press bounds mapped to normalized space | bound penalties and diagnostics |

Model-to-raw and model-to-oracle physical errors remain evaluation metrics. The loss does
not decode its prediction and does not regenerate either ROI mask.

## Resolved component contract

Let `V=Nphi*Ntheta*Nr`. A spatial MSE is the sum of squared voxel errors divided by `V`.
The batch reduction is always a mean. Channel groups are sums of their channel spatial
MSEs, not group means, so a channel weight is applied exactly once.

| loss component | paper formula/location | prediction space | target/reference space | coefficient | reduction | upstream reuse |
| --- | --- | --- | --- | ---: | --- | --- |
| unweighted base fidelity | Appendix C.7 squared L2; Appendix D.1 calls the unit-weight form per-voxel MSE | normalized | `normalized_target` | all channels `1` | spatial mean per channel, channel sum, batch mean | parity only; upstream `LpLoss.__call__` is relative and is not used |
| Full component fidelity | Appendix C.7 | normalized | `normalized_target` | Bcc1:3 `1.2`; rho/press/vel1:3 `1` | same as base | local thin weight wrapper |
| H1 gradient match | Appendix C.7 | normalized | `normalized_target` | `0.05`, once | squared gradient error, spatially normalized; channel sum; batch mean | `H1Loss(d=3,reduction="mean").abs(take_root=False)` minus the matching upstream absolute L2 term, then multiplied by 8 to undo channel averaging |
| velocity ROI | Appendix C.7 | normalized velocity channels | `normalized_target` velocity under supplied canonical mask | `8*min(1,epoch/375)` | per-sample relative L2, batch mean, epsilon `1e-12` | Stage D ROI function |
| rho lower bound | Appendix C.6--C.7 | normalized rho | Stage D normalized lower bound | `0.05` | violating-voxel square mean over batch/spatial | Stage D bounds function |
| press lower bound | Appendix C.6--C.7, with explicit press adaptation | normalized press | Stage D normalized lower bound | `0.05` | violating-voxel square mean over batch/spatial | Stage D bounds function |
| upper bounds | Appendix C.7 | normalized rho/press | Stage D normalized upper bounds | `0` | same as lower bound | Stage D bounds function; logged but exact zero contribution |
| rho envelope | Appendix C.5 | normalized rho | `radial_baseline_normalized` rho | `0.05`, `Delta=1.5` | hinge-square mean over batch/spatial | Stage D envelope function |
| press envelope | Appendix C.5, with explicit press adaptation | normalized press | `radial_baseline_normalized` press | `0.05`, `Delta=1.5` | hinge-square mean over batch/spatial | Stage D envelope function |
| dissipative gate | Appendix C.7 | normalized prediction | `normalized_input` and Stage D train-only `Rmax/Rin/Rout` | `alpha=5e-4` | global state norm per sample, positive-growth batch mean | Stage D dissipative function |

The H1 wrapper uses upstream finite differences on all three computational dimensions and
the pinned upstream default periodicity. This is
`computational_grid_H1_adaptation=true` and `spherical_metric_H1=false`. Upstream
`H1Loss.__call__` is not suitable directly: it is relative, includes the zeroth-order L2
term, and takes a square root. The subtraction above reuses the upstream implementation
without copying its finite-difference code and produces the paper's squared gradient-only
term.

## Full loss

Define raw component values without their outer coefficient. The resolved Full loss is

```text
L_full = L_base_component_weighted
       + 0.05 * L_H1_gradient_raw
       + 8 * min(1, epoch/375) * L_ROI_raw
       + 0.05 * L_bound_rho_low_raw
       + 0.05 * L_bound_press_low_raw
       + 0.05 * L_envelope_rho_raw
       + 0.05 * L_envelope_press_raw
       + 5e-4 * L_dissipation_raw
```

Upper-bound coefficients are zero. No Round 2/3 bounded residual, hybrid target,
rollout-aware term, range loss, Fold B, recency weight, or physical evaluation error is
part of `L_full`.

## Ambiguities and adaptations

1. **Base L2 reduction.** Appendix C writes squared L2 norms without an explicit divisor,
   while Appendix D calls the unit-weight version an unweighted per-voxel MSE. Literal
   summation over all `64^3` voxels would make auxiliary coefficients resolution-dependent.
   The resolved adaptation divides each channel's squared norm by `V`, sums channels, and
   averages samples. Thus unit-weight base loss equals `8 * torch MSE` for eight channels.
2. **H1 coefficient.** The local H1 formula already displays `lambda_H1`, and the total
   formula displays it again. The defaults provide one value, `0.05`. It is applied once.
3. **H1 meaning.** The paper writes a squared gradient difference, whereas upstream
   `H1Loss.__call__` is a relative full H1 norm. The resolved wrapper uses upstream
   absolute squared terms and removes the zeroth-order L2 term.
4. **ROI threshold wording.** The paper says "for each batch", but the reproducible and
   batch-invariant Stage D contract fixes an exact top-20% count independently per sample.
5. **Thermal channel.** Every paper `eint` loss term is explicitly adapted to `press`; no
   EOS conversion is inferred.
6. **Truth equals prediction.** Base, H1, and ROI become zero when
   `normalized_prediction==normalized_target`. Bounds and the literal radial envelope can
   remain nonzero because they constrain the prediction independently of fidelity. The
   dissipative term depends on input/prediction norms. Therefore Full total is not required
   to be zero in this case.

## Plain L2 decision boundary

Appendix D says to remove all auxiliary loss terms and weights and minimize unweighted
per-voxel MSE in normalized space. The loss is therefore

```text
L_plain = batch_mean(sum_c spatial_mean((prediction_c-target_c)^2))
```

It has unit channel weights and no H1, ROI, bounds, envelope, or dissipative term. The
held-constant Appendix D protocol leaves canonical preprocessing and the model's radial
shell/baseline state representation outside the definition of the loss; the separate
"No radial/constraint" ablation is what removes that representation. `PlainL2Loss` accepts
normalized prediction/target tensors only and rejects any enabled prior flags.

## Runtime and logging contract

`PaperLossContext` uses the unambiguous names above and validates protocol metadata,
shapes, device, dtype, epoch, supplied bounds, and ROI provenance. `PaperLossResult.total`
and every component scalar remain in the computation graph and share prediction
dtype/device. Detached logging is produced only after the total is assembled.

The structured result records raw and weighted components plus target clamp fractions,
ROI/clamp overlap, optional canonical/raw ROI Jaccard, bound/envelope violation fractions,
dissipative gate statistics, and input/prediction norms. Every constructor coefficient is
also emitted in loss metadata and the resolved audit output.
