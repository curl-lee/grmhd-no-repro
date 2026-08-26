# Stage R normalized-residual contract

Stage R is an `adapted_residual_contract_model_pilot`. It is a newly trained
model, not a Stage Q alpha anchor, checkpoint wrapper, bounded residual model,
or exact paper reproduction. The Stage N P3 transform/statistics, Stage K
differential LocalNO architecture, shared tensor state, pair order, data split,
optimizer, scheduler, and 30-epoch budget are frozen. The sole training factor
changed from Stage O is the prediction/target contract.

For P3-normalized states `z_t` and `z_{t+1}`:

```
delta_z_true = z_{t+1} - z_t
r_theta = LocalNO(concat(z_t, shells))
z_hat_{t+1} = z_t + r_theta
```

The raw eight-channel LocalNO output is the unscaled residual. There is no
learnable or fixed residual scale, alpha, residual normalization, bounded
activation, residual/output clamp, detach, or output repair. The model still
has 16 inputs (eight P3 state channels plus eight fixed radial shells), eight
outputs, 358,296 parameters, four spectral modules, four differential modules,
and no DISCO integral or Conv2d module.

Training minimizes strict unit-channel `PlainL2Loss(r_theta, delta_z_true)`.
The implementation checks every batch against
`PlainL2Loss(z_t + r_theta, z_{t+1})`; preflight observed an absolute difference
of exactly zero. A zero predicted residual is exactly P3 persistence.

Frozen pairing:

- initial tensor-state SHA256: `7e10895cb283be8532e4c036e8a526c536a632cce880c500080ea1d14fd89311`;
- pair-order SHA256: `5e1e361ea75e01ffd667822d8bfe870d70c5bc69570d52cdf806e637b2df2b52`;
- P3 statistics SHA256: `aea6d3f91e54865deeb9b2b1142c49a633ae8b67223075273e6c0b88bf541948`;
- upstream commit: `86a8bc7812a31b42c4f7895693cf4ac11521c066`.

Stage R explicitly disables rho/press evaluation output clamping so that the
closed loop contains no hidden physical clipping. The fixed optimizer gradient
clip remains `1.0`; this is parameter-update control, not output repair.
