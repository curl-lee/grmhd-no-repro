# Stage H H1 gradient audit

Across the five frozen model states and ten fixed pairs:

- weighted H1/base parameter-gradient norm ratio mean:
  12.6434;
- weighted H1/total norm fraction mean:
  0.94281;
- base-versus-weighted-H1 cosine mean:
  0.43912;
- base-versus-weighted-H1 conflict fraction mean:
  0.133362;
- Full total norm before clip mean:
  392.237;
- simulated norm-1 clip scale mean:
  0.00607965.

The simulation scales the combined gradient exactly as global norm clipping
would, but does not write `.grad`, call `backward`, construct an optimizer, or
update parameters. Per-sample prediction-space/channel/shell/concentration
statistics, per-module parameter norms, dot products, cosines, and sign
conflicts are stored in `h1_gradient_audit.json`.
