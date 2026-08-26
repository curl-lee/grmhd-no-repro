# Paper-adapted Stage G: matched 30-epoch pilots

## Scope

Stage G is a resource-scaled pilot on the frozen `paper_reduced100` protocol. It
is not the paper's 1200-epoch experiment and it uses an upstream FNO proxy, not
the paper's 3D DISCO LocalNO. The data remain spherical Kerr--Schild with
`press` standing in for the unverified `eint` channel.

The two pilots used the same 331,832-parameter model, preprocessing, split,
seed, initial tensor state, and epoch-by-epoch pair order. Only the loss mode
and its associated logging differ.

## GPU preflight

The WSL CUDA path was revalidated on an NVIDIA GeForce RTX 5070:

- PyTorch `2.12.1+cu130`, CUDA build `13.0`, capability `(12, 0)`;
- a real `64^3`, 16-input, 8-output upstream FNO forward plus Full loss
  backward was finite;
- forward/loss/backward times were 0.2585/0.1488/0.2304 seconds;
- peak allocated/reserved memory was 672,342,528/876,609,536 bytes;
- no optimizer step was taken during preflight.

The preflight exposed and fixed a local integration bug: calling the upstream
abstract `DataProcessor.to` did not move the registered shell tensor. The
paper processor now uses `torch.nn.Module.to` explicitly, with a regression
test.

## Frozen pairing

| Item | Value |
| --- | --- |
| Shared tensor-state SHA256 | `02d83278d5e74a27c83130e99b488ca18eae5477dad9732a4552b694056b6cb1` |
| Pair-order SHA256 | `5e1e361ea75e01ffd667822d8bfe870d70c5bc69570d52cdf806e637b2df2b52` |
| Seed | 42 |
| Train/validation pairs | 79/19 |
| Epochs | 30 |
| Batch/accumulation | 1/4 |
| Optimizer | Adam, `lr=1e-3`, `weight_decay=1e-4` |
| Schedule | 2-epoch warmup, cosine to `1e-6`, epoch update |
| Gradient clip | norm 1.0 |
| Mixed precision | disabled |
| Early stopping | disabled |

Every epoch consumed all 79 train pairs exactly once. The accumulation groups
were 19 groups of four and one final group of three, producing 20 optimizer
steps per epoch and 600 per run. The final group was normalized by its actual
count. Validation used all 19 pairs without shuffle.

The Full and Plain epoch-zero state hashes both matched the shared state, and
their raw outputs on the same batch were bitwise identical. The paired config
audit found no unexpected differences.

## Training results

| Metric | Full FNO | Plain L2 |
| --- | ---: | ---: |
| Status | `passed_with_h1_warning` | `passed` |
| Best validation epoch | 28 | 27 |
| Best validation average relative L2 | 0.462782 | 0.474580 |
| Last validation average relative L2 | 0.462829 | 0.475203 |
| Best-to-last gap | 0.0000477 | 0.0006229 |
| Microbatches / optimizer steps | 2370 / 600 | 2370 / 600 |
| Clipping fraction | 1.000 | 1.000 |
| Nonfinite count | 0 | 0 |
| Training wall time | 305.04 s | 228.08 s |
| Total wall time | 309.16 s | 232.00 s |
| Peak allocated / reserved | 828.92 / 962 MiB | 709.91 / 858 MiB |

For Full, the weighted H1/base value ratio had
mean/median/q95/max `29.61/26.37/55.85/74.98`; the H1/base gradient ratio was
`7.85/5.94/18.65/34.81`. The frozen H1 weight remained 0.05. Plain explicitly
disabled H1, ROI, bounds training penalty, radial envelope, and dissipation;
it did not emit fabricated zero-valued Full components.

Both `best_validation_l2` and `last` checkpoints strictly reloaded model,
optimizer, scheduler, epoch, ROI ramp, pair order, and provenance. A
deterministic batch prediction and the saved validation metrics reproduced
within tolerance. No NaN, Inf, OOM, or positivity safety stop occurred.

## Artifacts

Run manifests, logs, checkpoints, shared initial state, and pair order live
under `outputs/paper_reduced100/stage_g/`. They are intentionally Git ignored.
The corresponding small, tracked implementation is in
`scripts/prepare_paper_stage_g.py`, `scripts/train_paper_reduced.py`, and
`src/grmhd/paper_stage_g.py`.
