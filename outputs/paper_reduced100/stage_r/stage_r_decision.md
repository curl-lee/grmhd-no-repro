# Stage R decision

## C. RESIDUAL_PILOT_COMPLETE_BUT_ROLLOUT_UNSTABLE

The residual contract, preflight, complete-data smoke, 30 epochs, 2,370 microbatches, 600 updates, strict reload, and exact 19/100-step transform counters passed. All rollout states remained finite and rho/press positive.

Stage R reduced the Stage O normalized one-step average by a factor of `0.214565`, but remained `1.24634x` P3 persistence. Mean validation residual cosine was `0.131921`, while selected closed-loop cosine turned negative after step 1.

All validation predictions remained above frozen Rout. Physical ranges exploded by GT step 3, and step 100 retained widespread high-frequency artifact flags. Therefore the model is not stable enough for choice A or B; the engineering contract itself did not fail, so choice D is not applicable.
