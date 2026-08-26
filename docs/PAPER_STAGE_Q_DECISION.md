# Stage Q decision

## Unique decision

**D. `NO_VALID_ANCHOR_CANDIDATE`**

All contract, provenance, strict-reload, finite-output, positivity, transform
counter, and parameter-immutability checks pass. The failure is scientific,
not an engineering blocker: no middle alpha simultaneously passes frozen
train-only readiness groups A--F.

Anchoring clearly suppresses normalized overshoot. Alpha 0.125 and 0.25 reduce
the median q001/q999 OOD fraction by 92.42% and 79.79% relative to direct.
That does not establish stability: all alphas have 79/79 train Rout failures,
and Gate-2 reductions are below the required 50%. It also does not establish
dynamical skill: every middle alpha is more than 5% worse than persistence in
normalized arithmetic average and is worse in both median shell and radial
transport skill. The candidates are nontrivial updates with six positive
median channel cosines, so this is not a numerical collapse-to-persistence
classification; it is a failed joint stability/skill contract.

The Stage Q route stops here. Do not search additional alpha values, introduce
channel/state/timestep-specific alpha, add clipping, or start a paired smoke
under this decision. A different next step would require separate user
authorization.

Historical results remain frozen: Stage K is
`C. TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE`, Stage L is `3. MIXED_OVERALL`,
Stage M is `4. MIXED_TRANSPORT_AND_PREPROCESSING_FAILURE`, Stage N remains P3
ready with mixed operator-response failure, Stage O is
`C. P3_PILOT_COMPLETE_BUT_ROLLOUT_UNSTABLE`, and Stage P is
`6. MIXED_CLOSED_LOOP_FAILURE`.
