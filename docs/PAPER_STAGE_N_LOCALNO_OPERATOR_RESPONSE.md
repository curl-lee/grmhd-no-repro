# Stage N frozen LocalNO operator-response audit

## Frozen execution

The audit strictly reloads Stage K best epoch 22 from checkpoint SHA256
`f69008f91da80a4a7a6374adc253bbcdf37f58f0d142ea80f2b3f399e1f02a9f`.
It runs on the RTX 5070 in `eval()` and `torch.no_grad()` with the frozen
canonical normalizer and eight shell inputs. Prototype transforms are never
used. There is no optimizer, scheduler, backward pass, checkpoint write, or
training. Tensor-state SHA256 is identical before and after:
`4320f2dc1ea7f49fed57a68391df2847d1da7003660a73a963fac96f24c18cde`.

Fixed manifest-resolved states are train 11/50/90 and validation
91/92/94/96/100/110. Perturbations use only epsilon `1e-3` and `1e-2`; shell
channels never change. R8 applies the model exactly twice.

## Responses

| probe | result |
| --- | --- |
| R0 fixed point | median `||F(x)-x||/||x||` 0.430; median shell/radial persistence-relative skill −4.17/−6.82 |
| R1 constant | 8×8 response retained; candidate off-diagonal/diagonal signature is large but epsilon-inconsistent |
| R2 compact impulse | inner/middle/outer, ± signs, eight inputs and two epsilons are finite; support/symmetry/mixing are reported |
| R3 shells | full `T[s_out,s_in,c_out,c_in]` is retained as a 4096-cell matrix at each epsilon and aggregated across nine states |
| R4 radial modes | constant, linear/log-r trend, inner, outer, and mid-frequency responses reported |
| R5 directional | frozen Stage L low/mid/high bands along phi/theta/r reported |
| R6 channels | 8×8 constant-basis response matrix reported |
| R7 finite difference | 112 diagonal structural bases compare both epsilons; 112/112 exceed 10% relative gain tolerance |
| R8 two applications | median global/shell/radial variance ratios are 0.882/0.904/0.853; median delta-shrink ratio is 0.390 |

The finite-difference warning is material. Candidate frequency and channel
signatures cannot be promoted to supported mechanisms because the two epsilon
estimates are not approximately consistent. This audit does not claim a full
Jacobian spectrum or a physical Green function.

## Mechanism classification

- `LOW_VARIANCE_ATTRACTOR: supported`: seven of nine states meet the joint R8
  contraction criteria, independent of perturbation linearity.
- `SHELL_RADIAL_TRANSPORT_BIAS: supported`: all nine states have an R3/R4 or
  direct transport signature; on GT-available states the oracle-conditioned
  transport is generally much worse than persistence.
- `FREQUENCY_SELECTIVE_FAILURE: inconclusive`: no stable frequency signature
  survives the two-epsilon qualification.
- `CHANNEL_COUPLING_FAILURE: inconclusive`: the candidate non-diagonal response
  is strong, but zero states meet the 80% epsilon-consistency requirement.

Two mechanisms are supported without a unique single cause. The required
choice is therefore `4. MIXED_OPERATOR_RESPONSE_FAILURE`. This refines the
operator signature without replacing Stage M's frozen
`4. MIXED_TRANSPORT_AND_PREPROCESSING_FAILURE`.
