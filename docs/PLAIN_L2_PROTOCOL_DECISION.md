# Plain L2 protocol decision

## Decision

`PlainL2Loss` is the literal Appendix-D training-loss ablation: unit-weight squared error
in canonical normalized state space, with no Full loss priors.

```text
L_plain = batch_mean(sum_c spatial_mean((prediction_c-target_c)^2))
```

This equals eight times `torch.nn.functional.mse_loss(..., reduction="mean")` for the
fixed eight-channel output. The factor follows the paper's displayed sum over channel
groups; spatial and batch means make the definition independent of voxel count and batch
duplication.

## Retained protocol and representation

Appendix D holds the data split, normalization, schedule, and reporting protocol constant.
It defines a separate `No radial/constraint` ablation to remove the radial baseline.
Therefore Plain L2 retains, outside the loss itself:

- canonical paper preprocessing (`gamma=6`, inverse clamp fraction `0.99`);
- the paper-reduced100 split and press thermal adaptation;
- whichever shell inputs and selected literal radial state representation the future
  Full/Plain paired model configuration holds constant;
- oracle-aware evaluation and evaluation-only diagnostics.

The paper is less explicit about evaluation-time rho/eint clamping under Plain L2. Its
held-constant reporting paragraph says clamping is retained unless an ablation disables
it, while the Plain paragraph removes all auxiliary loss terms. The resolved decision is:
clamping may remain an evaluation protocol operation, but no bound penalty or clamp enters
`PlainL2Loss`.

## Disabled training terms

Plain L2 has:

- no magnetic `1.2` component weight;
- no H1 term;
- no canonical velocity ROI term or ramp;
- no lower/upper bounds penalty;
- no radial residual envelope;
- no dissipative global-norm term;
- no Round 2/3 extension.

The runtime API accepts only `normalized_prediction` and `normalized_target`. If a future
resolved configuration passes any enabled Full-prior flag, it raises rather than silently
ignoring the flag. No formal Full or Plain training configuration is created in Stage E.
