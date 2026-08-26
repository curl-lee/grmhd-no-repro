# Stage O decision

## C. P3_PILOT_COMPLETE_BUT_ROLLOUT_UNSTABLE

Preflight, full smoke, 30 epochs, 2,370 microbatches, 600 updates, strict reload, and the 19/100-step execution completed. All states remained finite and rho/press positive, and transform counters were exact.

P3 strongly reduced the Bcc2/Bcc3/vel3 oracle floor, but that recovery did not become stable model skill. The best checkpoint was worse than P3 persistence by `5.80871x` on the P3 normalized one-step average. Validation predictions were above frozen Rout for every pair, physical ranges expanded persistently, and the step-100 evaluation clamp fraction approached one. These satisfy the frozen Stage O C decoded-range/structure-instability rule.

This does not rewrite Stage K–N and does not authorize another training run.
