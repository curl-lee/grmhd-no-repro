# Stage O P3 differential LocalNO training

## Architecture and controlled pairing

The model is identical in names, shapes, and tensor initialization to the
Stage K differential LocalNO: 16 inputs (eight P3 state channels plus eight
input-only radial shells), eight direct state outputs, width 16, four spectral
blocks, four periodic 3x3x3 differential `Conv3d` modules, and 358,296
parameters. The model has zero DISCO and zero `Conv2d` modules.

Stage O reuses the Stage K initial tensor state and Stage G/K 30-epoch pair
order. It also retains seed 42, Plain L2, AdamW at `1e-3` with weight decay
`1e-4`, two warmup epochs, cosine minimum LR `1e-6`, batch one, accumulation
four, norm-one clipping, and validation every epoch. Only the isolated P3
processor and Stage O provenance differ.

## GPU preflight and smoke

The real 64-cube preflight ran on an NVIDIA GeForce RTX 5070 with PyTorch
`2.12.1+cu130` and capability 12.0. Forward, Plain L2, backward, every trainable
gradient, and a decoded physical state were finite; rho and press were positive.
Forward/backward took `0.415738/0.074806 s`, peak allocated/reserved memory was
`515.41/686.00 MiB`, and the tensor-state hash was unchanged. No optimizer,
scheduler, update, or checkpoint was created.

The complete-data two-epoch smoke then passed exactly 158 microbatches and 40
optimizer updates. Best and last checkpoint strict reload passed, as did a
three-step finite and positive physical rollout with exact transform counts.

## Formal 30-epoch pilot

| Field | Result |
| --- | ---: |
| epochs | 30 |
| microbatches | 2,370 |
| optimizer updates | 600 |
| best epoch | 23 |
| best P3 normalized average/global | `1.0471111 / 0.3443392` |
| last P3 normalized average/global | `1.0524799 / 0.3431509` |
| clipping fraction | `1.0` |
| final parameter displacement from initial | `109.5847` |
| nonfinite count | 0 |
| training / total wall time | `272.627 / 276.976 s` |
| peak allocated/reserved | `622.44 / 782.00 MiB` |

Every epoch consumed all 79 pairs, performed 20 optimizer updates, and handled
the final partial accumulation with count three. Best and last model,
optimizer, scheduler, epoch, pair-order, P3, and provenance metadata strictly
reloaded; the saved validation metric was reproduced within the frozen floating
tolerance.

## Best-checkpoint one-step validation

| Channel | model P3-normalized | P3 persistence | model-to-P3-oracle | model-to-raw | P3 floor |
| --- | ---: | ---: | ---: | ---: | ---: |
| Bcc1 | 1.23512 | 0.06919 | `1.5703e21` | `1.5703e21` | `7.04e-7` |
| Bcc2 | 0.65517 | 0.31646 | 0.80838 | 0.80838 | `3.07e-7` |
| Bcc3 | 0.34702 | 0.17736 | 0.62525 | 0.62525 | `2.99e-7` |
| rho | 1.52359 | 0.11631 | 0.98471 | 0.98471 | `1.42e-6` |
| press | 1.38477 | 0.12949 | 0.99795 | 0.99795 | `1.11e-6` |
| vel1 | 0.93300 | 0.14044 | 0.82309 | 0.82310 | 0.00230 |
| vel2 | 1.99614 | 0.36947 | 1.68605 | 1.25323 | 0.56681 |
| vel3 | 0.30208 | 0.12341 | 0.30001 | 0.30001 | `6.22e-8` |

The arithmetic-average model/persistence ratio is `5.80871`; Stage O does not
beat P3 persistence. The very large Bcc1 physical error and the fact that every
validation prediction exceeds frozen Rout are early decoded-range warnings,
not evidence of useful physical skill.
