# Stage I Run A, Run B, and Run C training record

## Frozen budget and pairing

The run used the Stage G FNO proxy with 331,832 parameters, seed 42, batch size
1, gradient accumulation 4, Adam at `1e-3` with weight decay `1e-4`, two-epoch
warmup, cosine decay to `1e-6`, norm-1 gradient clipping, no mixed precision,
and no early stopping. It used the shared Stage G initial state and the exact
30-epoch pair order.

The selected H1 contribution was exactly zero. The detached current-upstream
H1 was diagnostic only and did not enter total loss or backward. Logs record
selected H1 gradient norm as zero and H1/base gradient ratio as
`null/not_applicable`, never as a fabricated zero ratio.

## Curves

| epoch | train total mean | base mean | other-prior mean | validation average | validation global | displacement |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 43.1567 | 43.1371 | 0.01955 | 0.984310 | 0.984012 | 7.8745 |
| 5 | 9.08870 | 9.03700 | 0.05170 | 0.664394 | 0.555157 | 54.5063 |
| 10 | 2.51826 | 2.44482 | 0.07344 | 0.550081 | 0.451354 | 85.6547 |
| 15 | 1.90398 | 1.81885 | 0.08512 | 0.506687 | 0.404893 | 101.2502 |
| 20 | 1.71192 | 1.61584 | 0.09608 | 0.494377 | 0.395065 | 108.4601 |
| 25 | 1.62629 | 1.51935 | 0.10694 | **0.482824** | 0.386497 | 111.0040 |
| 30 | 1.62169 | 1.50182 | 0.11987 | 0.483336 | **0.385874** | 111.3354 |

Epoch 25 is the required best checkpoint because selection uses the arithmetic
average, not global L2. The last checkpoint is epoch 30.

## Gradient and update behavior

- Optimizer updates: 600/600.
- Microbatches: 2370/2370.
- Per-epoch steps: exactly 20.
- Per-epoch final accumulation count: exactly 3.
- Selected H1 gradient: exactly 0 for all 600 steps.
- Selected H1/base gradient ratio: not applicable for all 600 steps.
- Clipping fraction: 1.0.
- Mean clip scale: 0.14355, versus 0.01079 for Stage G Full.
- Mean effective base-gradient norm: 0.98877, versus 0.16747 for Stage G Full.
- Mean effective other-prior gradient norm: 0.11482.
- Mean parameter-update norm: 0.20806.
- Nonfinite count: 0.

Removing H1 did not eliminate clipping, but it removed the Stage G H1-driven
compression of the base-gradient projection. Occasional large other-prior
gradients are retained in the optimizer log; their aggregate effective scale
and weighted loss do not dominate base.

## Runtime and checkpoints

Training wall time was 335.50 s and end-to-end training entry time was 340.11
s. Peak allocated/reserved memory was 795.98/966.00 MiB. Both
`best_validation_l2` and `last` strictly restored the model, Adam state,
scheduler, epoch, ROI ramp, shared-state hash, pair-order hash, extension
classification, selected H1 coefficient zero, non-H1 loss contract, and
provenance. Saved and recomputed validation metrics agreed within the frozen
floating-point tolerance.

This run is a 30-epoch resource-scaled FNO-proxy diagnostic on reduced100
spherical Kerr--Schild data with `press`; it is not the paper's 1200-epoch 3D
DISCO LocalNO experiment.

## Run B unit-index H1

Run B changed only the selected H1 definition to centered periodic
unit-index differences with spacing `[1,1,1]`; weight 0.05 and every non-H1
control remained frozen. The current-upstream H1 was detached diagnostic data
only. The current/unit-index raw H1 ratio retained the expected value 4096.

| epoch | train total mean | base mean | selected H1 weighted | other-prior mean | validation average | validation global | displacement |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 43.3453 | 43.1371 | 0.18870 | 0.01955 | 0.984307 | 0.984012 | 7.8745 |
| 5 | 9.14678 | 9.03172 | 0.06346 | 0.05160 | 0.664189 | 0.554956 | 54.5054 |
| 10 | 2.54439 | 2.44134 | 0.02969 | 0.07336 | 0.550054 | 0.451080 | 85.6526 |
| 15 | 1.92602 | 1.81546 | 0.02556 | 0.08499 | 0.506080 | 0.404503 | 101.2471 |
| 20 | 1.73359 | 1.61372 | 0.02391 | 0.09596 | 0.494086 | 0.394696 | 108.4576 |
| 25 | 1.64726 | 1.51726 | 0.02323 | 0.10676 | 0.482545 | 0.386097 | 111.0016 |
| 27 | 1.63885 | 1.50426 | 0.02315 | 0.11144 | **0.482485** | **0.385469** | 111.2687 |
| 30 | 1.64261 | 1.49980 | 0.02312 | 0.11968 | 0.483025 | 0.385482 | 111.3330 |

Run B completed 2,370/2,370 microbatches and 600/600 updates, including the
three-microbatch partial group in every epoch. Clipping fraction was 1.0.
Across optimizer steps, mean H1/base gradient ratio was 0.00475, base/H1
cosine 0.46187, clip scale 0.14299, effective base/H1 norms
0.98642/0.00467, and parameter-update norm 0.20806. The H1 signal was finite
and nonzero without dominating or materially compressing the base gradient.

Training wall time was 333.37 s; peak allocated/reserved memory was
831.50/1012.00 MiB; nonfinite count was zero. Epoch-27 best and epoch-30 last
checkpoints both strictly restored model, Adam, scheduler, epoch, ROI ramp,
pair order, shared-state identity, extension metadata, and provenance.

## Run C stored-coordinate/volume H1

Run C preserved weight 0.05 and changed only the selected H1 to the frozen
physical-coordinate derivative with normalized spherical-coordinate volume
proxy. It completed 2,370 microbatches and 600 updates; best epoch was 25 with
validation average/global 0.444272/0.362543, and epoch 30 was
0.445051/0.361986. Mean H1/base value and gradient ratios were 0.25381 and
0.09180. Mean base/H1 cosine, clip scale, and effective base/H1 norms were
0.04065, 0.13899, and 0.98115/0.08987. Clipping fraction was 1.0, nonfinite
count zero, final displacement 111.023, wall time 599.54 s, and peak
allocated/reserved memory 840.57/1012 MiB. Best and last strict reload and
validation parity passed.

The selected proxy was theta- and outer-shell-dominated. This is recorded as
an adaptation diagnostic, not evidence for a covariant GRMHD H1.
