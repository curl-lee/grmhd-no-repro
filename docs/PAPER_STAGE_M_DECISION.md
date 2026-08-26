# Stage M transform and evaluation decision

## Transform-floor decisions

| channel | decision | principal evidence |
| --- | --- | --- |
| Bcc2 | `E. MULTIPLE_COMPONENTS` | no-final-clamp and broader no-softclip interventions each recover 4/4; isolated nonlinear/normalizer and float64 do not explain loss |
| Bcc3 | `A. FORWARD_SOFTCLIP_DOMINATED` | exact ±6 numerical saturation makes clamp-only bypass nonfinite; no-softclip recovers 4/4; float64 canonical remains lossy |
| vel3 | `A. FORWARD_SOFTCLIP_DOMINATED` | same 4/4 recovery and exact-saturation pattern as Bcc3 |

These are diagnostic attributions. No transform prototype is installed and the
canonical artifact remains unchanged.

## LocalNO transport decision

**`4. MIXED_TRANSPORT_AND_PREPROCESSING_FAILURE`**

All three primary channels are floor-limited, all have repeated LocalNO-added
state degradation, and none passes joint shell/radial transport. Small positive
skill in one view for Bcc2 and vel3 is insufficient because the paired view and
sign agreement fail.

## Candidate gate decision

**`I. CANDIDATE_GATE_READY_FOR_FUTURE_RUNS`**

The rules are frozen, thresholds are not validation/model-outcome tuned, toys pass,
and replay distinguishes canonical floor, persistence, and model-added failure.
“Ready” means ready to report alongside future pilots; it is not a validated
physical stability metric and does not replace the legacy detector.

## Frozen history

- Stage K remains `C. TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE`.
- Stage L remains `3. MIXED_OVERALL`.

Stage M neither reruns nor reclassifies either stage.
