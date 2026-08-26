# Stage K 3D differential LocalNO GPU preflight

- Status: `passed`
- Classification: `adapted_method_reproduction`
- Backbone: `3D differential LocalNO`
- DISCO integral: `disabled`
- Exact paper backbone: `false`
- GPU: `NVIDIA GeForce RTX 5070`
- Input shape: `[1, 16, 64, 64, 64]`
- Output shape: `[1, 8, 64, 64, 64]`
- Parameters: `358296`
- Differential/spectral/DISCO modules: `4/4/0`
- Forward/loss/backward finite: `True/True/True`
- Forward/backward seconds: `0.373188/0.068783`
- Peak allocated/reserved MiB: `515.41/686.00`
- Tensor state unchanged by backward: `True`
- Optimizer/scheduler/checkpoint created: `false/false/false`

This preflight uses one real reduced100 training pair and the frozen PaperDataProcessor/Plain-L2 contract. It is not a training run.
