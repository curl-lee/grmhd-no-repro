# Stage J: 3D DISCO and LocalNO source audit

## Decision

The pinned dependency does **not** contain a 3D DISCO convolution.  It contains
2D arbitrary-grid and 2D equidistant DISCO layers, while `LocalNO` explicitly
rejects any enabled DISCO layer when `len(n_modes) != 2`.  The resulting status
is:

```text
pinned_upstream: BLOCKED_IN_PINNED_UPSTREAM
stage_j_3d_disco: REQUIRES_NEW_IMPLEMENTATION
paper_faithful_prototype_gate: not_authorized
```

A generic equidistant 3D local integral layer could be prototyped with pure
PyTorch `conv3d`, but the paper and pinned source do not uniquely specify its 3D
basis, support, boundary handling, or architecture hyperparameters.  Implementing
one now would select those choices by guess.  Stage J therefore does not create
`src/grmhd/experimental/disco3d_prototype.py` and does not call a generic local
convolution the paper's DISCO.

## Audited source and refs

- project dependency: `external/neuraloperator`
- pinned commit: `86a8bc7812a31b42c4f7895693cf4ac11521c066`
- local `main` and `origin/main`: the same commit
- local tags: `0.1.0`, `0.1.1`, `0.2.0`, `0.3.0`, `1.0.0`, `1.0.1`,
  `1.0.2`, `2.0.0`
- no local branch, tag, or all-history search contains
  `EquidistantDiscreteContinuousConv3d`
- the relevant local history begins with `adb99fe` (DISCO layers),
  `69061e7` (LocalFNO), and `ff9bea7` (LocalNO rename); no later local commit
  adds a 3D DISCO implementation

No checkout, fetch, or upstream modification was performed.

## What is actually available

| concern | source-backed result |
|---|---|
| arbitrary-grid DISCO | `DiscreteContinuousConv2d`; input/output point clouds must each have coordinate dimension 2 |
| equidistant DISCO | `EquidistantDiscreteContinuousConv2d` and transpose variant; implemented through `torch.nn.functional.conv2d` |
| kernel parameterization | learnable weights multiply a 2D radius/angle basis; available bases are piecewise-linear, Morlet, and Zernike; `LocalNO` defaults to piecewise-linear with kernel shape `[2,4]` |
| neighbor/support construction | arbitrary grids form all pairwise 2D offsets, retain basis support, and use a sparse COO matrix in the forward; equidistant grids precompute only a local dense filter and use regular convolution, so there is no runtime neighbor search |
| nonuniform grids | only the arbitrary **2D** point-cloud path accepts explicit grids and quadrature weights; `LocalNO` instantiates the equidistant 2D path |
| batching | both paths preserve a batch dimension; the equidistant path delegates batching to `conv2d` |
| CUDA | operations are ordinary PyTorch tensors/sparse matmul/`conv2d`; the local CPU audit produced finite forward/backward results, but this Stage J process had no CUDA device and did not establish a 3D CUDA path |
| serialization | weights and bias are persistent state; precomputed filter buffers are `persistent=False` and are rebuilt from constructor arguments, so reload also requires identical grid/kernel configuration |
| Trainer | the generic upstream `Trainer` can call a `LocalNO(x=...)`; there is no GRMHD-specific or 3D-DISCO Trainer integration |
| examples | the DISCO tutorial and all DISCO layer tests are 2D; the `LocalNO` docstring example is 2D |
| tests | `LocalNO` tests cover 1D/2D/3D only with DISCO disabled outside 2D; enabled-DISCO tests are fixed to `n_dim=2` |

The decisive guards are in
`external/neuraloperator/neuralop/layers/local_no_block.py`: differential
convolutions are allowed through three dimensions, followed by a separate guard
that raises `NotImplementedError("Local conv layers only implemented for
dimension 2.")` whenever DISCO is enabled outside 2D.  Construction then
hard-codes `EquidistantDiscreteContinuousConv2d`.

## Boundary behavior

The arbitrary-grid helper can wrap pairwise offsets on a unit-periodic 2D
domain.  The equidistant class exposes a `periodic` flag and sets an internal
`padding_mode` string, but its forward calls functional `conv2d` with numeric
padding and never applies circular padding.  A constant-field probe with
`periodic=True` produced different corner and center values.  Thus periodic
boundary behavior in the equidistant path is not verified at this commit and
must not be assumed for a 3D extension.  Zero-style edge truncation is the
observed behavior.  The tests check shapes and backward use, not constant-field
preservation or periodic translation equivariance.

## Is current LocalNO the paper backbone?

No.

- `src/grmhd/models.py` constructs `localno_diff` with
  `disco_layers=False, diff_layers=True`.  It is a 3D spectral plus finite-
  difference model, not the paper's 3D DISCO LocalNO.
- requesting `localno_disco` at `64^3` reaches the upstream dimensionality guard
  and fails during construction.
- Stage G/I use the pinned upstream FNO and explicitly classify it as an FNO
  proxy.  Exact upstream FNO reuse does not make it the paper backbone.
- earlier FNO/LocalNO comparisons are useful engineering baselines only; neither
  may be reported as a trained paper 3D DISCO result.

## Minimum implementation needed

A verifiable 3D DISCO path needs at least:

1. a documented 3D continuous kernel basis and learnable coefficient layout;
2. 3D support/radius construction and quadrature normalization;
3. an equidistant `Conv3d` path and, if nonuniform points are required, a sparse
   radius-neighbor path that never materializes every point pair;
4. explicit per-axis periodic/open boundary semantics;
5. persistent constructor metadata for exact state reconstruction;
6. a `LocalNOBlocks` integration that accepts three-dimensional DISCO without
   changing the pinned dependency;
7. tests for constants, locality, interior translation equivariance, boundary
   behavior, backward, CPU/GPU parity, determinism, and strict state reload;
8. memory/latency benchmarks at `8^3`, `16^3`, then forward/backward-only
   `64^3` before any GRMHD integration.

The paper states “3D Local Neural Operator with equidistant DISCO specialized to
volumetric inputs,” but does not publish the above implementation choices or a
code locator.  The 2D source's polar radius/angle basis cannot be lifted uniquely
to 3D: spherical-angle, radial-only, and tensor-product bases are materially
different models.  This is why the isolated prototype gate remains closed.

## CUDA kernel requirement

A first equidistant feasibility implementation would not inherently require a
custom CUDA kernel: precompute a small basis filter and use PyTorch `conv3d`,
which supplies CPU/CUDA autograd.  A scalable arbitrary/nonuniform 3D operator
would need a radius-neighbor structure and sparse/segmented accumulation; a
custom kernel may become useful, but is not justified before the operator
definition and reference values exist.

## `64^3`, batch 1 memory and complexity

These are estimates, not measurements of a 3D implementation that does not yet
exist.

| quantity | estimate |
|---|---:|
| grid points | `64^3 = 262,144` |
| one float32 channel | 1 MiB |
| 16-channel input / 8-channel output | 16 MiB / 8 MiB |
| one hidden activation, width 16 / 32 / 64 | 16 / 32 / 64 MiB |
| dense all-pairs scalar distance matrix | 256 GiB |
| local equidistant convolution complexity | `O(B * Cout * (Cin/groups) * Ksupport * 64^3)` |
| local-filter storage | `O(Kbasis * Ksupport)` plus effective convolution weights |

The dense arbitrary-grid 2D precompute generalized naively to 3D would be
infeasible: `262,144^2` float32 distances alone require 256 GiB before offsets,
indices, basis values, activations, or gradients.  A regular-grid local `conv3d`
does not have this `N^2` storage and is plausibly compatible with 12 GB at small
width/support, but backward workspaces and the surrounding spectral LocalNO
blocks can add multiple activation copies.  Since the paper does not give the
exact hidden width, layer count, 3D support, or basis, Stage J cannot certify a
12-GB fit.  It can only require measured preflight after a reference
implementation is obtained.

## Evidence versus estimates

Source evidence covers the class names, dimensional guards, basis choices,
quadrature/filter construction, padding call, state persistence, tests, local
refs, and the successful 2D CPU forward/backward probe.  The proposed modules,
the suitability of `conv3d`, and all `64^3` memory figures are engineering
estimates.  No 3D DISCO output, GPU benchmark, optimizer, smoke, pilot, or GRMHD
training was run.
