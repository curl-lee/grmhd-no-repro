# Stage I Run B: unit-index H1 diagnostic

## Classification and frozen pairing

Run B is a `diagnostic_extension`, not paper-faithful Full. It replaces only
the Stage G current-upstream H1 training term with the Stage H unit-index H1:
the H1 weight remains 0.05, while base weights, ROI, bounds, radial envelope,
dissipation, preprocessing, shells, radial representation, optimizer,
scheduler, seed, shared initial state, pair order, clipping, and evaluation
remain identical to Stage G Full.

The launch manifest records project commit
`95b53c0cea9e926f73f34bdbe48c288eeb53e2cc`, pinned upstream commit
`86a8bc7812a31b42c4f7895693cf4ac11521c066`, unit config SHA256
`d9b82811e07c9f89e8d0f9ff9ba8ded795cbc3e478a4ef90926b2ca23fe5371b`,
shared tensor-state SHA256
`02d83278d5e74a27c83130e99b488ca18eae5477dad9732a4552b694056b6cb1`,
and pair-order SHA256
`5e1e361ea75e01ffd667822d8bfe870d70c5bc69570d52cdf806e637b2df2b52`.
The FNO proxy has 331,832 parameters. Stage G/H and Run A hashes were
revalidated before Run B.

## Unit-index definition and preflight

The selected H1 uses second-order centered differences, periodic wrapping on
all three stored axes, spacing `[1, 1, 1]`, and a uniform voxel mean. It does
not use spherical metric factors, physical-r spacing, volume weights, or shell
weights. The unchanged current-upstream H1, which uses unit-cube `1/N`
spacing, is logged only as a detached diagnostic.

On the frozen Stage H reference, current-upstream raw H1 divided by unit-index
raw H1 was 4096 within floating-point tolerance. The real `64^3` CUDA
preflight passed on the NVIDIA GeForce RTX 5070 with PyTorch 2.12.1+cu130.
Input, model, output, and loss were on CUDA; output, loss, and all parameter
gradients were finite; backward did not change the model state. Selected raw
and weighted H1 were 2.70121 and 0.135060, while the detached current-upstream
raw value was 11064.2. Peak allocated/reserved memory was approximately
776.28/972.00 MiB. Preflight created no optimizer, scheduler step, checkpoint,
or training state.

## Training and checkpoints

Run B completed exactly 30 epochs, 2,370 microbatches, and 600 Adam updates.
Every epoch used all 79 frozen train pairs, made 20 optimizer steps, and
normalized the final update over its actual three microbatches. There were no
NaN/Inf values. The epoch-27 best checkpoint and epoch-30 last checkpoint both
strictly restored model, optimizer, scheduler, epoch, ROI ramp, shared-state
hash, pair-order hash, extension identity, non-H1 loss contract, provenance,
and deterministic validation prediction.

Across 600 updates:

- H1/base value ratio mean/median/q95/max was
  0.01415/0.01482/0.02377/0.02739.
- H1/base parameter-gradient ratio was
  0.00475/0.00421/0.00890/0.01342.
- Base/H1 gradient cosine was
  0.46187/0.46959/0.81766/0.90431.
- Clipping fraction remained 1.0, but mean clip scale was 0.14299.
- Mean effective base/H1 gradient norms were 0.98642/0.00467.
- Mean effective base/H1 projections were 0.97601/0.00215.
- Mean parameter-update norm was 0.20806.

The unit-index signal was therefore finite and nonzero but did not dominate
the base gradient. Its clip scale and effective base gradient were close to
Run A no-H1 (0.14355 and 0.98877), not Stage G Full (0.01079 and 0.16747).
Training wall time was 333.37 s, end-to-end entry time was 337.47 s, and peak
allocated/reserved GPU memory was 831.50/1012.00 MiB. Final parameter
displacement from the common initial state was 111.33.

## Best validation

The selector remained normalized per-channel relative-L2 arithmetic average
over all 19 validation pairs.

| channel | relative L2 |
|---|---:|
| Bcc1 | 0.494388 |
| Bcc2 | 0.442613 |
| Bcc3 | 0.277311 |
| rho | 0.642651 |
| press | 0.642631 |
| vel1 | 0.620327 |
| vel2 | 0.514031 |
| vel3 | 0.225930 |
| arithmetic average | **0.482485** |
| global | **0.385469** |

Oracle-aware arithmetic-average/global values were 0.699212/0.973082 for
model-to-oracle, 0.879501/0.980011 for model-to-raw, and 0.417287/0.526358 for
the preprocessing oracle floor. The saved evaluation also retains the exact
squared-error numerator decomposition and target/model/model-only/target-only
saturation masks, Jaccard, and sign agreement.

Validation was finite with positive rho/press. Mean evaluation clamp fraction
was 0.03082, mean prediction norm was 4031.66, 63.16% of predictions exceeded
`Rin`, and none exceeded `Rout`. Mean rho/press envelope violations were
0.76978/0.22010; ROI/target-clamp overlap was 0.46126.

## Controlled interpretation

Relative to Stage G Full, unit-index improved all selected GT rollout errors,
both long-horizon norms, and six of seven frozen morphology/boundary
categories, but its validation arithmetic average was worse by 0.01970. It
slightly improved Full's global validation L2 by 0.00187.

Relative to no-H1, unit-index slightly improved validation average/global and
all selected GT rollout errors. It improved aggregate outer-shell statistics,
but no-H1 retained better center, polar, magnetic, radial, and saturation
scores and slightly lower step-50/100 norms.

Relative to Plain, unit-index was better only for GT step 1 and aggregate
magnetic texture; Plain retained better one-step validation, most morphology
and boundary scores, and step-100 norm. Run B therefore proves that removing
the 4096 spacing amplification yields a finite, weak H1 signal and relieves
Stage G gradient compression. It does not establish that unit-index H1 is the
correct replacement.

## Partial decision

**A. Unit-index passed and the next separately authorized comparison is
stored-coordinate/volume H1.**

This is not the final Stage I decision. Stored-coordinate training was not
started, and no conclusion is made about stored-coordinate rescue,
unit-index sufficiency, or removing H1 as the final objective.
