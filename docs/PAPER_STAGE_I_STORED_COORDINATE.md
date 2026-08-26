# Stage I Run C stored-coordinate/volume H1

## Frozen definition

Run C is a `diagnostic_extension`, not paper-faithful Full. It uses the actual
HDF5 center arrays in `(phi, theta, r)`. Phi uses the frozen periodic centered
stencil and its measured uniform spacing. Theta uses open three-point
Lagrange stencils, including one-sided boundaries. Radius uses the physical,
nonuniform `r` centers with the same open local three-point rule; it is neither
an average radial spacing nor log-r.

The reduction is the normalized spherical-coordinate volume proxy
`r^2 sin(theta) dphi dtheta dr`. Weights are finite, nonnegative, explicitly
sum to one, exclude the batch axis, and use theta-center `sin(theta)` clamped
at zero with pole floor 0. This is not a Kerr--Schild proper-volume element,
not a covariant GRMHD H1, and not a covariant derivative of stored vectors.

- Config SHA256: `4b3f46adcc7cbf1e0d72f83a068a67ebdb9c5461005f1a36c59d00f67a7e1eb6`
- Coordinate SHA256: `68b3cd5f9f7c852007fb3b619013620ba77b8c97e5c0a4f2aeb1463986b54a11`
- Canonical CPU volume-weight SHA256: `64ca16b6bb0fa9c1379b97eac21cbe39779b154df4be3b42cc91fe8b7ffbe17d`
- CUDA runtime volume-weight SHA256: `6fa9a978558865d1f29eb83eb7473c7bfcb90d4e428b54761e0bf043c366dcf5`
- Shared state: `02d83278d5e74a27c83130e99b488ca18eae5477dad9732a4552b694056b6cb1`
- Pair order: `5e1e361ea75e01ffd667822d8bfe870d70c5bc69570d52cdf806e637b2df2b52`

The CPU/CUDA weight hashes record device-specific floating construction; both
use the same frozen formula and coordinates and pass finite, nonnegative, and
unit-sum checks.

## GPU preflight and training

The stored-only 64-cubed preflight passed on NVIDIA GeForce RTX 5070,
capability 12.0, PyTorch 2.12.1+cu130. Input/model/output/loss and gradients
were CUDA and finite, model state was unchanged, and all non-H1 components
were exactly equal to Stage G Full. Peak allocated/reserved memory was
0.758/0.951 GiB.

Run C completed 30 epochs, 2,370 microbatches, and 600 optimizer updates.
Every epoch contained 79 microbatches, 20 updates, and a final accumulation
count of three. Best epoch was 25. Best validation average/global was
0.444272/0.362543; last was 0.445051/0.361986. Best and last strictly restored
model, Adam, scheduler, epoch, ROI ramp, pairing, H1 definition, and
provenance, with deterministic prediction and validation parity.

Mean selected-H1/base value and gradient ratios were 0.25381 and 0.09180.
Mean current/stored and unit-index/stored raw ratios were 332.55 and 0.08119.
The mean base/H1 cosine was 0.04065; mean clip scale was 0.13899; effective
base/H1 gradient norms were 0.98115/0.08987. All updates clipped, mean update
norm was 0.20857, final displacement was 111.023, and nonfinite count was
zero. Wall time was 599.54 s; peak allocated/reserved was 840.57/1012 MiB.

The raw selected-H1 direction means were phi `1.82e-5`, theta `13.1380`, and
r `0.00112`; the proxy is overwhelmingly theta-dominated. Mean inner,
middle, and outer shell contributions were 0.00280, 0.42258, and 12.71379,
so its coordinate-volume reduction is overwhelmingly outer-shell dominated.

## Validation and rollout

Best normalized per-channel relative L2 was: Bcc1 0.44464, Bcc2 0.40070,
Bcc3 0.27410, rho 0.55102, press 0.57180, vel1 0.57085, vel2 0.54015, and
vel3 0.20091. Model-to-oracle, model-to-raw, and oracle-floor average/global
were 0.69074/0.96733, 0.87212/0.97586, and 0.41729/0.52636. Validation was
finite and rho/press positive; mean eval-bound clamp was 0.03165 and no
prediction exceeded Rout.

GT rollout normalized average errors at steps 1/5/10/19 were
0.36597/0.81080/0.89666/0.98735. All steps were finite and positive. At steps
50/100, prediction norms were 4532.69/4515.47, bound-clamp fractions
0.00208/0.00244, and artifact counts 2/1. No GT error is reported beyond step
19. Exactly 100 physical prediction decodes and 100 next-input encodes were
used.

The large Bcc3/vel3 target clamp fractions and oracle floor remain explicit;
Run C does not remove the canonical preprocessing limitation.
