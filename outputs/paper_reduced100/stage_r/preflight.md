# Stage R residual LocalNO GPU preflight

- Status: `passed`.
- GPU: `NVIDIA GeForce RTX 5070`; capability `[12, 0]`.
- Input/residual/state: `[1, 16, 64, 64, 64]` -> `[1, 8, 64, 64, 64]` -> `[1, 8, 64, 64, 64]`.
- Parameters: `358296`; differential/DISCO: `4/0`.
- Forward/backward: `0.375654/0.074516` s.
- Peak allocated/reserved: `531.41/684.00` MiB.
- Residual/state Plain L2 equivalence, finite gradients/decode, and positivity: passed.
- Optimizer, scheduler, optimizer step, and checkpoint write: absent.
