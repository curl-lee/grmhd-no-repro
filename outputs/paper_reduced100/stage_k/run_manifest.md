# Stage K controlled run manifest

- Classification: `adapted_method_reproduction`
- Backbone: `3D differential LocalNO`
- DISCO integral: `disabled`
- Exact paper reproduction: `false`
- Project commit: `4f9d9074436eba89117d07a5aa7e37f52541195f`
- Upstream commit: `86a8bc7812a31b42c4f7895693cf4ac11521c066`
- GPU: `NVIDIA GeForce RTX 5070`
- Dataset SHA256: `cd3966bed1f617f49d775151cff87ae92eb2432a560bb82ae9355605e7aea70a`
- Preprocessing SHA256: `1645714424d95b02de2283fc16a8ef05f3864d15f11025580af13f3e67964001`
- Pair-order SHA256: `5e1e361ea75e01ffd667822d8bfe870d70c5bc69570d52cdf806e637b2df2b52`
- LocalNO initial tensor-state SHA256: `7e10895cb283be8532e4c036e8a526c536a632cce880c500080ea1d14fd89311`
- LocalNO/FNO Plain parameters: `358296/331832` (ratio `1.079751`)
- Epochs/train pairs/validation pairs: `30/79/19`
- Accumulation: `4` (`20` updates/epoch, final group `3`)
- Total microbatches/updates: `2370/600`
- Early stopping: `disabled`
- Selector: validation normalized per-channel relative-L2 arithmetic average

The architecture differs from the Stage G FNO proxy, so fairness freezes the seed and complete data/optimization/evaluation protocol rather than claiming a shared tensor initialization.
