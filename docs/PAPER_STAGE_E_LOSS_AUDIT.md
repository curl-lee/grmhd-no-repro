# paper_reduced100 Stage E loss audit

## Status and scope

Stage E passed its network-free component and gradient audit. The audit reads four fixed
train pairs (`11->12` through `14->15`) and two fixed validation pairs (`91->92` and
`92->93`) at batch size one. It evaluates ten prediction cases per batch: normalized
target, normalized persistence, zero, deterministic random, and isolated perturbations of
`Bcc1`, `Bcc3`, `rho`, `press`, `vel1`, and `vel3`.

The resulting 60 component records and 420 component-specific gradient records are all
finite. No model, `Trainer`, optimizer, checkpoint, or validation fit is involved.

## Truth-equals-prediction audit

When `normalized_prediction == normalized_target`, the fidelity terms behave as required:

| split | base | H1 | ROI | bounds | envelope | dissipation | Full total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train mean, 4 batches | `0` | `0` | `0` | `1.52614e-8` | `0.0447372` | `0.00584848` | `0.0505857` |
| validation mean, 2 batches | `0` | `0` | `0` | `0` | `0.0290732` | `0.0151957` | `0.0442689` |

The nonzero total is intentional. The literal radial envelope constrains the prediction
against the Stage D baseline rather than the target, and the dissipative term compares
prediction growth to the normalized input norm. Stage D already established that the
literal envelope excludes a substantial fraction of true rho/press voxels.

The truth-case total gradient remains finite. Mean total gradient norm is `2.77134e-4` on
the four train batches and `5.33201e-4` on the two validation batches.

## Component and gradient scale

Mean weighted values and gradient norms for representative cases are:

| split/case | base value / grad | H1 value / grad | ROI value / grad | total value / grad |
| --- | ---: | ---: | ---: | ---: |
| train persistence | `2.63071 / 0.00634778` | `134.333 / 0.554872` | `0.949068 / 0.00355708` | `137.959 / 0.559930` |
| validation persistence | `10.6502 / 0.0133969` | `1022.30 / 1.77995` | `2.81030 / 0.00533692` | `1035.79 / 1.79129` |
| train zero | `30.8693 / 0.0226001` | `560.278 / 0.949315` | `4 / 0.00355713` | `595.170 / 0.960488` |
| validation zero | `64.7941 / 0.0333368` | `1429.03 / 1.67230` | `8 / 0.00533692` | `1501.84 / 1.68853` |
| train random | `39.4718 / 0.0255553` | `3017.66 / 3.78658` | `4.24045 / 0.00355708` | `3061.46 / 3.80039` |
| validation random | `73.3339 / 0.0353913` | `3883.58 / 4.02802` | `8.26885 / 0.00533692` | `3965.26 / 4.04516` |

The H1 component dominates persistence, zero, and random cases even after its single
`0.05` weight. Inspection confirmed that this is not a duplicated coefficient or voxel
sum: the wrapper exactly equals eight times the pinned upstream absolute squared H1 minus
its absolute squared L2 term, and upstream quadrature spatially normalizes the result.
The large value comes from unit-cube finite differences (spacing `1/N`) applied to sharp
normalized fields on the spherical storage axes. This is a reported computational-grid
H1 adaptation tension; Stage E does not retune the paper weight using validation.

## Channel isolation

All six requested single-channel perturbations passed:

- Bcc1/Bcc3 change base and H1, but not bounds, envelope, or ROI;
- rho/press can change base, H1, their bound term, and their envelope term, but not ROI;
- vel1/vel3 can change base, H1, and ROI, but not bounds or envelope;
- the canonical ROI mask remains supplied and fixed rather than being recomputed from a
  prediction;
- the dissipative global norm can respond to any output channel;
- tensors with shell or other extra output channels are rejected.

The audit also records target clamp fractions, canonical/raw ROI Jaccard, ROI/clamp
overlap, bounds/envelope violations, gate statistics, and input/prediction norms for every
case. Truth-case canonical/raw ROI Jaccard averages `0.771600` on the four train batches
and `0.585193` on the two validation batches.

## Batch, zero-weight, and parity checks

Tests establish that duplicate samples leave mean-style loss unchanged, batch order does
not matter, a two-sample result equals the mean of individual calls, an incomplete final
batch is valid, and dissipative norms never cross the batch dimension.

Setting H1, ROI, bounds, envelope, or dissipation weight to zero leaves its raw diagnostic
available while producing an exact zero weighted value and exact zero prediction-gradient
contribution. Other components are unchanged.

The strict Plain L2 value equals both `8 * torch MSE` and eight times pinned upstream
`LpLoss.abs(take_root=False, reduction="mean")`. The H1 wrapper has exact parity with the
pinned upstream squared H1-minus-L2 construction. The composite call returns a scalar
compatible with the upstream Trainer loss-call convention, although Stage E does not run
the Trainer.

## Outputs

Generated, Git-ignored outputs are under `outputs/paper_reduced100/losses/`:

- `loss_contract.json`;
- `loss_component_audit.csv`, `.json`, and `.md`;
- `gradient_audit.csv` and `.md`.

The resolved contract records all coefficients, Stage D artifact hashes, protocol and
preprocessing identity, selected literal radial mode, audited indices/epochs, and
`training_or_model_execution=false`.
