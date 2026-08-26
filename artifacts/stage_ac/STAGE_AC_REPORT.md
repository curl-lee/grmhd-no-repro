# Stage AC — Adapted Volumetric 3D DISCO LocalNO

| model | params | local integral | state L2 | residual L2 | cosine | shell skill | radial skill | first10x |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Persistence | 0 | no | 0.310282 | 1 | 0 | 0 | 0 | stable |
| Stage-T differential LocalNO | 358296 | no | 0.266711 | 0.891851 | 0.740602 | -1.679472 | -0.348089 | 1 |
| Adapted 3D-DISCO LocalNO | not constructed | intended | not run | not run | not run | not run | not run | not run |

## Mandatory implementation-gate result

The upstream audit completed and the pinned source remains clean. The exact frozen Stage AC
formulas produce a 3x3x3 stencil with only radii 0 and R inside support. K=5 radial hats require
centers at 0, R/4, R/2, 3R/4, and R, so basis indices [1, 2, 3] are identically zero. Their
normalizations fail (`Z_k=0`), and the associated weights cannot receive gradients.

This is not a GPU, I/O, or training failure and is not a scientific negative result about DISCO.
It is an internally inconsistent adapted-discretization contract discovered before model
construction. No radius, K, basis, boundary, loss, data, or upstream code was changed; no CUDA
preflight or training was run.

## Required contract decision

Exactly one of these scientific choices must be newly authorized and frozen before repair:

1. increase cutoff to at least three computational cell spacings (changing the frozen radius),
2. reduce the radial basis count to the two radii represented by the stencil (changing K), or
3. define a precise subcell/voxel-integrated basis projection and quadrature rule.

Until then, dense-reference, gradcheck, branch activity, controlled training, attribution, and
rollout cannot be meaningfully evaluated.
