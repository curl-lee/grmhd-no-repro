# Stage P train-only envelope

- Source: snapshots `11..90` only; validation leakage: `false`.
- Normalized and physical quantiles: `0.001/0.01/0.5/0.99/0.999`.
- Decoder sensitivity: analytic frozen-P3 derivative, checked at fixed `1e-5/1e-4`.
- All OOD and projection bounds in Stage P use this artifact.
