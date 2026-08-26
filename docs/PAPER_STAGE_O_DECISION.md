# Stage O decision

## C. `P3_PILOT_COMPLETE_BUT_ROLLOUT_UNSTABLE`

The engineering pipeline completed: provenance and full P3 parity passed, the
real GPU preflight and complete-data smoke passed, the formal run completed 30
epochs/2,370 microbatches/600 updates, best and last checkpoints strictly
reloaded, training contained no nonfinite values, and 19/100-step physical
rollouts remained finite with positive rho/press and exact transform counters.

P3 also achieved its narrow transform objective. Bcc2, Bcc3, and vel3
raw-to-oracle validation floor is approximately `3.07e-7`, `2.99e-7`, and
`6.22e-8`, respectively. This removes the canonical target-channel floor as an
explanation for the new model behavior.

The model does not turn that recovery into skill. It is `5.80871x` worse than
P3 persistence on one-step normalized average. Bcc2 and Bcc3 fail Gate 2;
all target channels fail Gate 3. Every validation prediction is above frozen
Rout. GT rollout normalized average reaches `28.0025` at step 19, decoded
Bcc2/Bcc3 reach extreme finite ranges, and no-GT clamp occupancy remains near
one through step 100. This persistent decoded-range and structure instability
directly satisfies the frozen Stage O C rule.

The disappearance of old collapse flags is not a successful stability result:
legacy ripple/stripe flags and a much more severe range explosion replace the
collapse signature. `stage_m_v1` makes this distinction explicit by separating
the recovered P3 floor, model-added degradation, transport skill, and Gate 0
engineering range.

This decision does not authorize another model, transform, loss, ablation,
hyperparameter change, or longer run. The next action is proposal-only: diagnose
P3/LocalNO extreme-range physical feedback and the already observed mixed
operator response using the frozen artifacts.

Historical decisions remain exactly:

- Stage K: `C. TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE`.
- Stage L: `3. MIXED_OVERALL`.
- Stage M: `4. MIXED_TRANSPORT_AND_PREPROCESSING_FAILURE`.
- Stage N: P3 ready; mixed operator-response failure.
