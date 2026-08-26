# Stage R decision

## C. RESIDUAL_PILOT_COMPLETE_BUT_ROLLOUT_UNSTABLE

Stage R passes its engineering contract: provenance and P3 train-only
statistics, residual/state Plain-L2 equivalence, RTX 5070 64-cube preflight,
complete-data smoke, 30 epochs, 2,370 microbatches, 600 updates, nonzero learning,
best/last strict reload, deterministic metric parity, finite 19/100-step
execution, positive rho/press, and exact transform counters.

The residual model substantially improves on Stage O direct-state normalized
overshoot, including step 1. That improvement is insufficient for A: it does not
beat P3 persistence, validation residual relative L2 is worse than the zero
baseline, selected closed-loop residual cosine becomes negative after step 1,
all validation predictions remain above Rout, physical ranges explode by step
3, Gate 2 still fails Bcc3, and Gate 3 has negative shell/radial skill for all
three target channels. It is not B because range and closed-loop structure are
not stable. It is not D because the residual engineering contract itself passed.

Historical decisions remain frozen:

- Stage K: `C. TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE`;
- Stage L: `3. MIXED_OVERALL`;
- Stage M: `4. MIXED_TRANSPORT_AND_PREPROCESSING_FAILURE`;
- Stage N: P3 ready; mixed operator-response failure;
- Stage O: `C. P3_PILOT_COMPLETE_BUT_ROLLOUT_UNSTABLE`;
- Stage P: `6. MIXED_CLOSED_LOOP_FAILURE`;
- Stage Q: `D. NO_VALID_ANCHOR_CANDIDATE`.

Stage R authorizes no further training. A possible next stage may only be
proposed: diagnose why unscaled normalized residuals enter P3 inverse tails and
why closed-loop residual transport reverses direction, without presuming a
larger model, new transform, residual scale, clipping, or ablation.
