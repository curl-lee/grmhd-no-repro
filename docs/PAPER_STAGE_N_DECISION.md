# Stage N transform and operator-response decision

## Required choices

| item | decision |
| --- | --- |
| Bcc2 | `A. PROTOTYPE_READY` |
| Bcc3 | `A. PROTOTYPE_READY` |
| vel3 | `A. PROTOTYPE_READY` |
| combined P3 | `1. COMBINED_PROTOTYPE_READY_FOR_SHORT_PILOT` |
| LocalNO operator | `4. MIXED_OPERATOR_RESPONSE_FAILURE` |
| candidate gate | `I. REPORTING_INTERFACE_READY` |

Bcc2 uses P2B because only no-softclip passed the train-only L2-improvement
rule. Validation did not select or tune any prototype parameter. P3 is an
isolated diagnostic mapping with separately fitted train-only statistics; it
has never been supplied to a checkpoint.

The LocalNO result combines supported low-variance attraction and supported
shell/radial transport bias. Frequency/channel response is inconclusive because
both epsilon estimates disagree, despite a candidate channel-mixing signature.

## Frozen history and boundary

- Stage K remains `C. TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE`.
- Stage L remains `3. MIXED_OVERALL`.
- Stage M remains `4. MIXED_TRANSPORT_AND_PREPROCESSING_FAILURE`.
- Canonical preprocessing, checkpoints, HDF5, shell metadata, and pinned upstream
  are unchanged.
- Stage N performs no training and starts no follow-on smoke/pilot.

The next action may only be proposed: use P3 with its isolated train-only stats,
the same Stage K architecture and pair order, first a two-epoch smoke and then at
most a paired 30-epoch pilot, reporting both legacy and `stage_m_v1` gates. It
requires separate authorization.
