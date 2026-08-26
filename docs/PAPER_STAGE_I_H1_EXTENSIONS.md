# Stage I H1 diagnostic extensions

## Scope

Stage I investigates the Stage H finding that current-upstream H1 on the
reduced spherical grid is amplified by `64^2 = 4096` relative to the same
stencil with unit-index spacing. It does not change the frozen Stage G
paper-adapted Full result or its H1 weight of 0.05.

All Stage I runs are separately classified `diagnostic_extension` experiments
on reduced100 spherical Kerr--Schild data with `press`, an FNO proxy, and a
30-epoch resource-scaled budget. They are not the paper's 1200-epoch Cartesian
3D DISCO LocalNO experiment.

## Frozen variants

| variant | selected H1 | status | best validation average/global |
|---|---|---|---:|
| Stage G Full | current upstream, `1/N` spacing | frozen main result | 0.462782 / 0.387339 |
| Run A no-H1 | no selected H1 contribution | completed | 0.482824 / 0.386497 |
| Run B unit-index | centered periodic, spacing `[1,1,1]`, weight 0.05 | completed | 0.482485 / 0.385469 |
| Run C stored-coordinate/volume proxy | physical-r/open theta-r, periodic phi, normalized spherical-coordinate volume proxy, weight 0.05 | completed | 0.444272 / 0.362543 |

The variants share the same 331,832-parameter initial FNO state, 30 epoch pair
orders, 79/19 temporal split, preprocessing, shells, non-H1 Full terms,
optimizer, scheduler, gradient clip, and evaluation protocol.

## What Run A and Run B establish

Run A showed that removing H1 relieves the severe Stage G compression of the
base gradient, although every update still clips. Run B retained a finite H1
signal while behaving similarly to no-H1:

| diagnostic mean | Full | no-H1 | unit-index |
|---|---:|---:|---:|
| H1/base gradient ratio | 7.85 | not applicable | 0.00475 |
| clip scale | 0.01079 | 0.14355 | 0.14299 |
| effective base-gradient norm | 0.16747 | 0.98877 | 0.98642 |
| effective H1-gradient norm | dominated combined gradient | 0 | 0.00467 |
| clipping fraction | 1.0 | 1.0 | 1.0 |
| parameter-update norm | 0.20707 | 0.20806 | 0.20806 |

Both extensions completed 30 epochs/600 updates, strictly reloaded best and
last checkpoints, and stayed finite with positive rho/press through 19 GT and
100 total rollout steps. Both improved all selected GT rollout normalized
errors relative to Full. Unit-index was marginally better than no-H1 in
one-step validation and selected GT errors, while no-H1 retained better scores
in most aggregate morphology categories. Plain remained the strongest model
in most frozen morphology and boundary categories.

## Final decision

Run C completed its 30-epoch budget, strict reload, 19-step GT evaluation and
100-step no-GT rollout. Its mean H1/base gradient ratio was 0.09180: larger
than unit-index 0.00475 but far below Full 7.85. All three extensions improve
five frozen morphology/boundary categories relative to Full, but no-H1,
unit-index, and stored-coordinate improve only 1, 1, and 0 categories relative
to Plain. Final Stage I decision is **D: no extension rescues Full**. See
`PAPER_STAGE_I_STORED_COORDINATE.md` and `PAPER_STAGE_I_DECISION.md`.
