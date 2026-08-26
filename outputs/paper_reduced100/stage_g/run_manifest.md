# Stage G controlled run manifest

- Status: `ready`
- Project commit: `4686c2e73a3c150c5ac1f96fbe2ca38204f605e3`
- Upstream commit: `86a8bc7812a31b42c4f7895693cf4ac11521c066`
- GPU: `NVIDIA GeForce RTX 5070`
- Parameter count: `331832`
- Shared initial tensor-state SHA256: `02d83278d5e74a27c83130e99b488ca18eae5477dad9732a4552b694056b6cb1`
- Pair-order SHA256: `5e1e361ea75e01ffd667822d8bfe870d70c5bc69570d52cdf806e637b2df2b52`
- Full/Plain epoch-0 output bitwise equal: `true`
- Epochs: `30`
- Train/validation pairs: `79/19`
- Gradient accumulation: `4` (`20` steps/epoch; final count `3`)
- Checkpoint metric: validation normalized per-channel relative L2 arithmetic average
- Early stopping: disabled

This is a reduced100, spherical Kerr-Schild, press-adapted, FNO-proxy, 30-epoch resource-scaled pilot. It is not directly comparable to the paper's 1200-epoch 3D DISCO LocalNO result.
