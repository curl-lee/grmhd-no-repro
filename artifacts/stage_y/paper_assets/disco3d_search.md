# Stage Y volumetric 3D DISCO search and contract

## Search decision

```text
EXACT_3D_DISCO_IMPLEMENTATION_FOUND = false
```

No paper-bound volumetric implementation was found in the arXiv source,
official venue assets, author public repositories/branches/tags, the pinned
dependency, or its available history.  No speculative implementation was
created.

## Decisive pinned-source evidence

The pinned dependency is `external/neuraloperator` at
`86a8bc7812a31b42c4f7895693cf4ac11521c066`.

- `neuralop/layers/local_no_block.py:140-144` states local integral kernels are
  implemented only for 2D.
- Lines 214-215 raise when DISCO is enabled and `len(n_modes) != 2`.
- Lines 333-348 instantiate `EquidistantDiscreteContinuousConv2d`.
- `neuralop/layers/discrete_continuous_convolution.py:144-169` requires 2D
  input/output point clouds and constructs radius/angle support.
- Lines 271 onward define `DiscreteContinuousConv2d`; lines 682 and 858 define
  the equidistant 2D forward and transpose classes; the forward uses `conv2d`
  at line 845.

The author fork at commit `8719ad2...` contains the same dimensional
limitation and no hidden public branch/tag.  Current project `localno_diff`
experiments disable DISCO and are not the paper operator.

## Mathematical contract recovered from primary sources

The target paper explicitly names a volumetric equidistant DISCO LocalNO but
gives no formulas beyond the LocalNO citation.  The primary LocalNO reference
(arXiv `2402.16845v2`, audited PDF SHA256
`485956551f9b683c2ba621bb1d02973238df5d90732c2d090403525401287125`)
defines the general local integral/DISCO construction:

\[
(\mathcal K v)(y) = \int_D \kappa(x-y)v(x)\,dx
\approx \sum_{x_j\in D^h}\kappa(x_j-y)v(x_j)q_j,
\]

with the sum restricted to the compact support of `kappa`.  Its group form is
`kappa(g^{-1}x)` with quadrature weights.  The reference and pinned 2D code use
continuous basis evaluation so the same physical receptive field can be
evaluated at multiple resolutions.

| target-paper 3D item | recovered contract | status |
|---|---|---|
| local support definition | compact/radius-local support is part of DISCO; target numeric support/radius absent | EXPLICIT general / UNKNOWN target |
| kernel basis | primary 2D reference/code offers radial/angle piecewise-linear (also Morlet/Zernike in code) | EXPLICIT 2D / UNKNOWN 3D |
| quadrature | weighted discrete approximation `sum kappa*v*q`; equidistant 2D implementation precomputes filters | EXPLICIT general / UNKNOWN target weights |
| physical/grid coordinates | target says equidistant volumetric inputs; cube bounds and coordinate normalization absent | EXPLICIT partial / UNKNOWN |
| number of basis functions | paper does not report 3D kernel shape | UNKNOWN |
| support radius | paper does not report it | UNKNOWN |
| channel mixing/groups | general code has learned in/out/group weights; target setting absent | INFERRED family / UNKNOWN target |
| normalization | 2D helper can discretely normalize basis filters; target 3D choice absent | EXPLICIT 2D / UNKNOWN target |
| boundary handling | target paper does not state periodic/open/padding behavior | UNKNOWN |
| 3D extension | no formula/code chooses radial-only, spherical-angle, or tensor-product 3D basis | UNKNOWN |
| per-layer use | target does not state which layers enable DISCO/differential/spectral branches | UNKNOWN |
| branch relationship | primary LocalNO adds spectral, differential, local-integral and skip branches in parallel; target exact composition absent | EXPLICIT general / UNKNOWN target |

## Consequence

The integral-operator idea is recoverable, but its volumetric realization is not
unique.  A `conv3d` radial kernel, a spherical angular basis, and a tensor-product
basis are materially different operators.  Critical target choices are
UNKNOWN, so the mathematical specification is not sufficient to reproduce the
paper's 3D operator and Gate R2 fails.
