# Stage R GPU preflight and training

The RTX 5070 real-64-cube preflight used input shape `(1,16,64,64,64)`, raw
residual/state shapes `(1,8,64,64,64)`, and Stage R Plain L2. Forward, loss,
backward, gradients, reconstructed state, P3 decode, and rho/press positivity
all passed. Forward/backward took 0.375654/0.074516 s and peak
allocated/reserved memory was 531.41/684.00 MiB. No optimizer, scheduler,
optimizer step, or checkpoint was created.

The first smoke launch exposed an uninitialized P3 diagnostic-envelope local
before any optimizer update. The engineering wiring was fixed and covered by
targeted tests; the failed launch is not a scientific run. The fresh complete
two-epoch smoke then passed with 158 microbatches, 40 updates, two final partial
accumulations of three, strict best/last reload, deterministic probe parity,
and a finite positive three-step physical rollout with exact transform counts.

The authorized formal run started again from the frozen common initial state:

| field | result |
| --- | --- |
| epochs / train microbatches / updates | 30 / 2,370 / 600 |
| train / validation pairs per epoch | 79 / 19 |
| batch / accumulation | 1 / 4 (final count 3) |
| optimizer | Adam, lr `1e-3`, weight decay `1e-4` |
| scheduler | warmup 2 epochs, cosine to `1e-6`, epoch stepped |
| AMP / early stopping | disabled / disabled |
| gradient clip | fixed norm `1.0` |
| best epoch / normalized average | 9 / `0.224673118` |
| clipping fraction | `0.986667` |
| nonfinite / model updated | 0 / yes |
| wall / total wall | 314.417 / 318.636 s |
| peak allocated / reserved | 654.44 / 824.00 MiB |

Best epoch 9 and last epoch 30 strictly reload model, optimizer, scheduler,
epoch, pairing, P3 provenance, residual contract, and deterministic one-batch
prediction. Recomputed normalized average/global metrics match their saved
training-time values.

This is a 30-epoch, resource-scaled, P3/spherical/press-adapted LocalNO proxy
pilot. It is not the paper's 1,200-epoch 3D DISCO result.
