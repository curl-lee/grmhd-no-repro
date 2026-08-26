# Stage V transport-objective engineering audit

The first V2 attempt used decoded physical shell variances in `Lshell`.  This
was chosen to match the frozen evaluation statistic as literally as possible,
and its lambda was derived solely from the fixed train subset.  No validation
was evaluated.

That formulation was not numerically admissible.  Despite float64 variance
arithmetic, the nonlinear rho/press inverse tail generated heavy-tailed epoch
means (for example transport `4.246e9` at epoch 7 and `6.806e9` at epoch 10)
and the combined parameter gradient became nonfinite during epoch 15.  The
attempt stopped immediately; it produced no scientific model result.

This failure revealed a definition-level confound: a physical `Lshell`
re-injected `DECODER_PHYSICAL_TAIL_AMPLIFICATION` into an experiment intended
to test normalized temporal-dynamics alignment, contrary to Stage V Part M.
The frozen replacement therefore keeps:

- the exact eight radial shells;
- population variance as the shell statistic;
- predicted/true evolution relative squared error;
- all eight channels and equal channel aggregation;

but evaluates those shell variances on the P3 normalized state, matching the
already-required normalized radial-profile objective.  It adds no clamp,
changes neither P3 nor the data/model, and was fixed before any V2 validation
result existed.  Train-only gradient scaling is recomputed for this corrected
definition.  V2 restarts from the shared initial state and epoch-0 optimizer.

V1 is unaffected: its enabled computation graph contains only Plain and
direction objectives.  Its checkpoint records the earlier unused shell
definition in provenance, so that historical difference remains explicit
rather than being rewritten retrospectively.
