# Stage P decision

## Frozen-rule mechanism evidence

| mechanism | status | decisive evidence | limiting evidence |
| --- | --- | --- | --- |
| M1 normalized model overshoot | `supported` | teacher-forced OOD/Rout failure, step-1 envelope exit, normalized-direct 19.14x growth, projection effect | OOD and Rout are coincident at step 1 |
| M2 P3 decode tail amplification | `supported` | Bcc2/Bcc3 derivative exceeds train tail, nonlinear physical expansion, projection effect | normalized-direct is not more stable; P3 is not sole source |
| M3 roundtrip/clamp feedback | `weakly_supported` | extreme actual output-feedback discrepancy and coincident sensitivity/any-channel clamp | target channels are no-softclip; repeated `H` does not accumulate; stable gain ratio unavailable |
| M4 recurrent operator gain | `weakly_supported` | normalized-direct grows without decode | teacher-forced already fails; 0/120 gain rows pass epsilon consistency |
| M5 cross-channel feedback | `supported` | frozen projection/reset rule finds Bcc2/Bcc3/vel3 primary drivers across multiple metrics | no stable finite-difference off-diagonal evidence; no unique dominant channel |

At least M1, M2, and M5 are supported, while no single one explains all
observations. M1 explains first-step normalized failure, M2 explains the
nonlinear Bcc2/Bcc3 physical magnitude, and M5 explains the intervention
responses. M3/M4 remain contributory but weaker under the frozen evidence rules.

## Unique decision

**6. `MIXED_CLOSED_LOOP_FAILURE`**

This is a post-hoc attribution result. It does not change Stage O's
`C. P3_PILOT_COMPLETE_BUT_ROLLOUT_UNSTABLE`, does not make P3 canonical, and
does not alter the earlier Stage K--N decisions. P3's target-channel oracle-floor
improvement remains valid.

## Proposed next action only

If a later stage is authorized, change one factor at a time and start with the
earliest observed mechanism: normalized model overshoot at the first prediction.
The first bounded proposal is a no-training persistence/residual anchoring
operator-response audit on the same frozen states. Only after its contract and
toy tests pass should a separately authorized two-epoch paired smoke be
considered. Do not start a new model, clipping fix, transform refit, or training
from this decision alone.

The frozen decision and predicates are in
`outputs/paper_reduced100/stage_p/mechanism_evidence.json` and
`stage_p_decision.json`.
