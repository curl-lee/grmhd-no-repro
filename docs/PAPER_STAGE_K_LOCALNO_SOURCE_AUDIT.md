# Stage K 3D Differential LocalNO Source Audit

## Decision

- Selected backbone: `3D differential LocalNO`
- Experiment classification: `adapted_method_reproduction`
- DISCO integral: `disabled`
- Exact paper backbone: `false`
- Pinned upstream: `external/neuraloperator` at
  `86a8bc7812a31b42c4f7895693cf4ac11521c066`
- Upstream modifications: none

The public implementation supports the selected spectral-plus-differential
three-dimensional operator. It does not support a volumetric DISCO integral
branch, so this Stage K method cannot be described as the paper's 3D DISCO
LocalNO.

## Audited sources

The audit directly inspected:

- `neuralop/models/local_no.py`
- `neuralop/layers/local_no_block.py`
- `neuralop/layers/differential_conv.py`
- `neuralop/layers/discrete_continuous_convolution.py`
- `neuralop/models/tests/test_local_no.py`
- `neuralop/layers/tests/test_local_no_block.py`

The task referred to `finite_difference_convolution.py`; that path does not
exist in the pinned tree. The actual pinned implementation is
`layers/differential_conv.py` and defines `FiniteDifferenceConvolution`.

## Constructor and branch audit

`LocalNO` infers `n_dim` from `len(n_modes)`. The frozen Stage K constructor
uses `n_modes=(8,8,8)` and `default_in_shape=(64,64,64)`, therefore `n_dim=3`.
It explicitly passes `disco_layers=False` and `diff_layers=True` through the
project model factory.

`LocalNOBlocks` constructs one spectral `SpectralConv` and, when enabled, one
`FiniteDifferenceConvolution` per block. With the frozen four-layer Stage K
configuration the resulting counts are:

- differential modules: 4;
- spectral modules: 4;
- DISCO `EquidistantDiscreteContinuousConv2d` modules: 0;
- `torch.nn.Conv2d` modules: 0;
- finite-difference `torch.nn.Conv3d` modules: 4.

The spectral branch is explicitly recorded as enabled in the Stage K config;
Stage K is therefore a spectral-plus-differential LocalNO, not a
differential-only convolutional network.

## Dimensional support and explicit rejection

The pinned block rejects differential convolutions only above three spatial
dimensions, so `n_dim=3` is supported. `FiniteDifferenceConvolution` selects
`torch.nn.Conv3d` and `torch.nn.functional.conv3d` when `n_dim=3`.

The pinned DISCO branch constructs
`EquidistantDiscreteContinuousConv2d` and rejects any dimension other than two
with a `NotImplementedError`. The project config resolver and model factory
therefore reject `n_dim=3` with `use_disco=true` before upstream construction,
using the explicit message:

> volumetric 3D DISCO is unavailable in pinned upstream

This prevents accidentally invoking the upstream two-dimensional-only code.

## Differential stencil and boundary behavior

The frozen differential settings are:

- kernel size: 3 (odd, centered same-size convolution);
- derivative mixing: enabled (`groups=1`);
- padding: `periodic`, mapped by upstream to PyTorch `circular` padding;
- bias: disabled;
- resolution reference: `default_in_shape=(64,64,64)`.

For an input `x` and learned kernel `K`, the upstream layer computes a local
convolution and subtracts the convolution using the spatially summed kernel,
then divides by its runtime grid-width scale. This gives the learned stencil a
finite-difference-like zero-order cancellation. Periodicity here is a
computational-grid adaptation; it is not a claim about exact spherical
Kerr–Schild differential geometry.

## Upstream test evidence and project gates

The pinned model tests exercise LocalNO without DISCO for dimensions 1, 2, and
3, including forward and backward. The pinned block tests also exercise the
three-dimensional differential path while enabling DISCO only for dimension
2.

Project Stage K tests additionally require:

- a `16 -> 8` three-dimensional forward shape;
- finite forward values and finite gradients for every trainable tensor;
- deterministic seed-42 initialization;
- strict `state_dict` reload;
- zero DISCO/Conv2d module count;
- unchanged Stage G FNO construction and Plain-L2 semantics;
- fixed shells as input-only channels;
- exact physical rollout transform counters.

The mandatory CUDA preflight then repeats forward, Plain L2, and backward on a
real `(1,16,64,64,64)` processed training batch on the RTX 5070 without
creating an optimizer, scheduler, checkpoint, or training state.

## Reproduction boundary

Stage K reproduces an end-to-end adapted standalone-surrogate workflow using a
publicly runnable three-dimensional differential LocalNO. It does not
reproduce the paper's unavailable 3D DISCO integral operator, 1200-epoch
budget, 300-snapshot dataset, verified `eint`, Cartesian Kerr–Schild
representation, or coarse/fine coupled simulation.
