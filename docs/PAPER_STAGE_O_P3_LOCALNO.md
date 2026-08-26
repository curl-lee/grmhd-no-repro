# Stage O P3 differential LocalNO protocol

Stage O is an `adapted_transform_model_pilot`. It evaluates the frozen Stage N
P3 transform with the already-audited three-dimensional differential LocalNO.
It is neither a canonical preprocessing replacement nor an exact paper
reproduction, and the model contains no 3D DISCO integral layer.

## Frozen inputs and isolation

| Object | Frozen value |
| --- | --- |
| Stage N base | `2c762a47f7235b6b608a00f780d8f2cc18df6627` |
| upstream `neuraloperator` | `86a8bc7812a31b42c4f7895693cf4ac11521c066` |
| HDF5 | `cd3966bed1f617f49d775151cff87ae92eb2432a560bb82ae9355605e7aea70a` |
| P3 config | `41409e803cc0f54bede7cc73a84d9986e081bb07265c84dcac2fd64a44778809` |
| P3 statistics NPZ | `aea6d3f91e54865deeb9b2b1142c49a633ae8b67223075273e6c0b88bf541948` |
| P3 decision JSON | `71919bffd5a13d96a500b76fa06ea8308dc578c524893984fd4f168cd9eba0e2` |
| LocalNO tensor initial state | `7e10895cb283be8532e4c036e8a526c536a632cce880c500080ea1d14fd89311` |
| 30-epoch pair order | `5e1e361ea75e01ffd667822d8bfe870d70c5bc69570d52cdf806e637b2df2b52` |

P3 uses independently fitted train-only statistics from snapshots 11--90.
Validation snapshots 91--110 do not participate in fitting or channel-policy
selection. Bcc2, Bcc3, and vel3 use the frozen no-softclip policies; Bcc1,
rho, press, vel1, and vel2 retain their canonical policies inside the isolated
P3 object. The canonical normalizer and every Stage K--N artifact remain
unchanged.

## Full round-trip reproduction

The preparation command recomputed every train and validation snapshot and
compared 344 frozen summary values. Train and validation parity both passed at
maximum scaled difference `0.0`; the statistics identity check was bitwise.

| Target channel | train median raw-to-P3-oracle L2 | validation median |
| --- | ---: | ---: |
| Bcc2 | `7.7597451e-8` | `1.4783347e-7` |
| Bcc3 | `9.2294075e-8` | `2.0250436e-7` |
| vel3 | `2.4604000e-8` | `2.1097570e-8` |

Tensor encode/decode follows the frozen NumPy transform. The tensor path keeps
the original float32 result in its trained domain and uses wider intermediates
only when extreme finite autoregressive states would otherwise overflow a
float32 intermediate. These engineering guards do not refit statistics or
change the frozen P3 formulas.

The complete provenance and parity rows are in
`outputs/paper_reduced100/stage_o/run_manifest.json`.
