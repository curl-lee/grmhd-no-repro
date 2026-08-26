# Stage K Differential LocalNO Plain Pilot

## Scope

Stage K is an `adapted_method_reproduction` of the standalone-surrogate
workflow. It uses the pinned public 3D differential LocalNO with its spectral
branch enabled and DISCO integral branch disabled. It is not the paper's 3D
DISCO model and is not an exact numerical reproduction.

The frozen data remain 111 spherical Kerr--Schild snapshots with `press` as
the thermal channel. The reduced100 split uses 79 train transitions and 19
validation transitions; `90 -> 91` is dropped. Canonical preprocessing,
train-only statistics, the eight input-only radial shells, direct normalized
`t -> t+1` prediction, and Plain L2 are unchanged from Stage G Plain.

## Provenance and architecture

- project training commit: `4f9d9074436eba89117d07a5aa7e37f52541195f`;
- upstream commit: `86a8bc7812a31b42c4f7895693cf4ac11521c066`;
- HDF5 SHA256: `cd3966bed1f617f49d775151cff87ae92eb2432a560bb82ae9355605e7aea70a`;
- preprocessing SHA256: `1645714424d95b02de2283fc16a8ef05f3864d15f11025580af13f3e67964001`;
- pair-order SHA256: `5e1e361ea75e01ffd667822d8bfe870d70c5bc69570d52cdf806e637b2df2b52`;
- LocalNO initial tensor-state SHA256:
  `7e10895cb283be8532e4c036e8a526c536a632cce880c500080ea1d14fd89311`.

The model has 358,296 parameters versus 331,832 for Stage G FNO Plain, a
ratio of 1.079751. It has four spectral layers, four learned 3x3x3 periodic
finite-difference layers, four linear skips, no channel MLP or normalization,
and zero DISCO/Conv2d modules. Input/output shapes are 16/8 channels and the
prediction mode is direct.

## CUDA preflight and smoke

The real-batch preflight ran on an NVIDIA GeForce RTX 5070 with PyTorch
2.12.1+cu130 and CUDA runtime 13.0. The real input/output shapes were
`(1,16,64,64,64)` and `(1,8,64,64,64)`. Forward, Plain L2, backward, and all
24 trainable-tensor gradients were finite. Backward did not change the model
tensor hash. Forward/backward took 0.3732/0.0688 s and peak allocated/reserved
memory was 515.41/686.00 MiB. No optimizer, scheduler, checkpoint, or training
state was created by preflight.

The full-data smoke then passed two epochs, 158 microbatches, and 40 optimizer
updates. Every epoch consumed 79 pairs, made 20 updates, and normalized the
last accumulation group by its actual three microbatches. Best and last
checkpoints strictly restored model, optimizer, scheduler, epoch, pairing,
architecture, loss, and provenance. The three-step physical rollout was
finite and rho/press remained positive, with exactly three prediction decodes
and three next-input encodes. The smoke is an engineering result, not a
scientific ranking.

## Frozen 30-epoch result

The formal pilot completed exactly 30 epochs, 2,370 microbatches, and 600
updates. All epochs satisfy `79/20/final-3`; mixed precision and early stopping
were disabled. No nonfinite value occurred. All 600 updates triggered the
frozen norm-1 gradient clip. The training phase took 235.75 s (239.42 s total
entry-point time) and peak allocated/reserved memory was 621.99/786.00 MiB.

Best validation was epoch 22; last was epoch 30. Their normalized
average/global values were 0.624225/0.540968 and 0.625578/0.541519. The final
parameter displacement from the frozen LocalNO initialization was 109.319.
Both best and last checkpoints strictly reload and reproduce the stored
one-batch prediction; best validation recomputation also matches the
training-time average and global metrics within the frozen tolerance.

## One-step validation

All values below use the same 19 validation transitions and normalized
per-channel relative-L2 aggregation.

| model | Bcc1 | Bcc2 | Bcc3 | rho | press | vel1 | vel2 | vel3 | average | global |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical oracle | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| persistence | 0.06919 | 0.30660 | 0.24316 | 0.11631 | 0.12949 | 0.14044 | 0.36947 | 0.13805 | 0.18909 | 0.23507 |
| Stage G FNO Plain | 0.46451 | 0.43766 | 0.27894 | 0.62662 | 0.63031 | 0.63908 | 0.51043 | 0.20909 | 0.47458 | 0.38163 |
| Stage G FNO Full | 0.50551 | 0.45855 | 0.32456 | 0.58998 | 0.51274 | 0.63695 | 0.49395 | 0.18003 | 0.46278 | 0.38734 |
| Stage K LocalNO Plain | 0.49927 | 0.60738 | 0.41250 | 0.78151 | 0.67907 | 0.89020 | 0.75936 | 0.36449 | 0.62422 | 0.54097 |

LocalNO/persistence is 3.3012 for the arithmetic average and 2.3013 for the
global value; every individual channel ratio is above one. Stage K therefore
does not beat persistence or the prior FNO proxies on one-step validation.

The oracle-aware LocalNO average/global values are:

- model-to-oracle: 0.823681/0.994860;
- model-to-raw: 0.931080/0.995782;
- preprocessing oracle floor: 0.417287/0.526358.

Evaluation remained finite and rho/press positive. Mean evaluation-bound clamp
occupancy was 0.04156 and no validation prediction exceeded the frozen
dissipation `Rout`. Bcc3 and vel3 retain the known preprocessing-oracle floor;
raw-space values for those channels must not be interpreted without that
caveat.

Machine-readable training, validation, persistence, and provenance summaries
are under `outputs/paper_reduced100/stage_k/`. Checkpoints and the shared
initial state remain Git ignored.
