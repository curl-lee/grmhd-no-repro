# Stage H: pinned upstream H1 definition

## Audited source

This definition is tied to
`external/neuraloperator@86a8bc7812a31b42c4f7895693cf4ac11521c066`.
The relevant implementation is `H1Loss`/`LpLoss` in
`neuralop/losses/data_losses.py` and `FiniteDiff` in
`neuralop/losses/differentiation.py`. Toy and exact parity tests are in
`tests/test_paper_h1_definition.py` and
`tests/test_paper_h1_spacing_diagnostics.py`.

## Upstream H1 semantics

For an array with shape `(B,C,Nphi,Ntheta,Nr)`,
`H1Loss(d=3, reduction="mean").abs(pred,target,take_root=False)` computes an
**absolute, squared H1 norm**. It includes both the L2 value term and all three
first-derivative terms. The default uniform quadrature is

```text
h_phi   = 1/Nphi
h_theta = 1/Ntheta
h_r     = 1/Nr
```

and the product `h_phi*h_theta*h_r` multiplies each spatial sum. Thus spatial
reduction is a uniform-voxel integral/mean over a unit cube. With
`reduction="mean"`, the resulting `(B,C)` values are averaged jointly over
batch and channel. `take_root=False` means no square root is applied.

This is not the relative `H1Loss.__call__` path. Stage E directly calls
`.abs(..., take_root=False)`.

## Derivative and boundary convention

`FiniteDiff` uses second-order centered first differences in the interior:

```text
D f_i = (f_{i+1} - f_{i-1}) / (2 h)
```

All three periodic flags default to `True`. Stage E does not override them, so
`phi`, `theta`, and `r` all use `torch.roll` wrap boundaries. No cropping,
padding, spectral differentiation, or one-sided boundary stencil is used on
the current path. The same helper does provide third-order one-sided
boundaries when periodic flags are false, but Stage E does not select that
mode.

Consequently the stored tensor axes `(phi,theta,r)` are treated as three
equally measured unit-cube axes. Their physical extents, nonuniform radius,
spherical metric factors, pole geometry, and Kerr--Schild proper volume are
not represented.

## Exact Stage E wrapper

Let `e = prediction - target`, let `C=8`, and let `D_a` be the pinned periodic
centered derivative with `h_a=1/N_a`. The upstream squared absolute H1 is

```text
mean_(b,c) mean_(phi,theta,r)
    [ e^2 + (D_phi e)^2 + (D_theta e)^2 + (D_r e)^2 ].
```

The upstream squared L2 is the matching
`mean_(b,c) mean_(phi,theta,r)[e^2]`. `PaperH1GradientLoss` subtracts this L2
term and multiplies by eight:

```text
L_H1_raw =
    mean_b sum_c mean_(phi,theta,r)
    [ (D_phi e)^2 + (D_theta e)^2 + (D_r e)^2 ].
```

It is therefore an eight-channel, absolute, squared **gradient seminorm**, not
a rooted H1 norm and not a relative H1 norm. `PaperCompositeLoss` includes it
once as `0.05 * L_H1_raw`.

Exact wrapper-vs-upstream and additive-density parity are locked by tests. H0
is this frozen path. H1--H4 in `paper_h1_diagnostics.py` are post-hoc
diagnostics only and are not imported by the paper loss, Trainer, checkpoint,
or Full configuration.

## Stored-coordinate facts

The frozen HDF5 centers are:

| Axis | Range | Center spacing |
| --- | --- | --- |
| phi | 0.0490874 to 6.2340981 | uniform, 0.0981748 |
| theta | 0.0245437 to 3.1170490 | uniform, 0.0490874 |
| r | 1.14563 to 192.03332 | nonuniform, 0.09703 to 14.99399 |
| log(r) | 0.13594 to 5.25767 | uniform, 0.0812970 |

H3 therefore differentiates the radial coordinate point-by-point; it never
substitutes one average `Delta r`. H4 uses the scalar proxy
`dr^2 + dtheta^2/r^2 + dphi^2/(r^2 sin^2(theta))` with an explicit
`sin(theta)` floor of `1e-3`. H4 is not a vector covariant derivative and its
volume weighting is only the coordinate proxy
`r^2 sin(theta) dr dtheta dphi`, not a verified Kerr--Schild proper volume.
