# Stage H adaptation decision

## Decision

**A. Index-grid H1 adaptation mismatch strongly supported.**

## Observed results

1. H0 is exactly `4096 = 64^2` larger than the same periodic centered
   unit-index derivative energy; H2 has exact H0 parity.
2. Stored-r/log-r uniform reductions are only about `4--7%` of H0.
3. Inner shells contain `51.0%` of H0 in `25%` of voxels.
4. `r` and `theta` contribute `55.2%` and `44.5%`; `phi` contributes `0.33%`.
5. Paper-weighted H1/base parameter-gradient ratio averages `12.64`; weighted
   H1 contributes `94.3%` of total gradient norm.
6. Base/H1 cosine averages `0.439`, with conflict weight `0.133`.
7. Simulated norm-1 clipping scales the total and all component projections by
   `0.00608` on average.
8. H0 overlap with target clamp and envelope-violation masks is `59.8%` and
   `83.6%`.
9. All five model hashes and every frozen Stage G file checksum remained
   unchanged.

## Inference

The pinned upstream H1 is internally consistent, but its `1/N` unit-cube
spacing, periodic theta/r boundaries, and uniform voxel measure are a material
mismatch when adapted to stored spherical `(phi,theta,r)` data with nonuniform
physical radius. This mismatch can strongly control the Full gradient and
compress the base signal under the frozen norm-1 clip.

This is consistent with Stage G finding no Full morphology/stability advantage
over Plain. It is not proof that H1 caused every Stage G difference.

## Unsupported claims and counterevidence

- Plain has larger absolute validation H0 and larger inner per-voxel H0 than
  Full on the fixed samples, despite better Stage G morphology. Magnitude alone
  is not a morphology predictor.
- Base/H1 cosine is positive, not globally anti-parallel.
- H1/base value ratio is negatively correlated with validation L2 over the 30
  epochs because validation improved while the ratio grew. This is descriptive
  training-time co-evolution, not evidence that increasing H1 improves error.
- Epoch-aligned outer-shell, artifact, and morphology series were not saved;
  their requested correlations are reported as unavailable, not reconstructed.
- H3/H4 do not establish the correct covariant GRMHD loss.

## Reproduction impact

The paper main path remains frozen:

```text
H1 weight = 0.05
current pinned upstream H1
gradient clip = 1.0
```

No Stage G checkpoint, output, config, prior, or preprocessing artifact was
modified. No training, smoke, pilot, Trainer, optimizer step, or scheduler step
was run. The controlled Stage G Full-versus-Plain comparison remains valid for
the original paper-adapted objective; Stage H changes its interpretation, not
its recorded result.

## Proposed next stage — not executed

Only after explicit authorization, three separately named diagnostic
extensions could be run:

1. no-H1 diagnostic control;
2. unit-index-spacing H1 diagnostic;
3. stored-coordinate/volume-proxy H1 diagnostic.

All three would be `EXTENSION`, not paper-faithful Full. They would require the
same shared initial state, pair order, data, and 30-epoch budget. This Stage H
work created no training configuration or output for them.
