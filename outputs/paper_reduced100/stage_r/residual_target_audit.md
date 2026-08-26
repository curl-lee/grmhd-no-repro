# Stage R residual-target audit

- Contract: `delta_z_true = z_t1 - z_t`; reconstruction uses an identity skip.
- Persistence is exactly the zero predicted residual.
- Train calibration source: 79 transitions 11->12 through 89->90.
- Validation: 19 confirmatory transitions, not used to alter the contract.
- Residual scale: `1.0`; clipping/renormalization: disabled.
