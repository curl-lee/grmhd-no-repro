# Stage AH Scientific Result Analysis

| model | geometry | params | state L2 | residual L2 | cosine | shell skill | radial skill | first10x |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Persistence | none | 0 | 0.31028248 | 1 | 0 | 0 | 0 | stable |
| Stage-T LocalNO | no DISCO | 358296 | 0.26671132 | 0.89185143 | 0.7406021 | -1.6794722 | -0.34808888 | 1 |
| Stage-AD DISCO3D | index-space | 363480 | 0.26406202 | 0.89163995 | 0.75746772 | -4.3544812 | -0.39035242 | 1 |
| Stage-AG spherical DISCO3D | anisotropic spherical tangent | 363480 | 0.27226335 | 0.91680556 | 0.74758286 | -1.5937098 | -0.30184608 | 1 |

## Controlled result

- formal selector: `validation_normalized_per_channel_relative_l2_arithmetic_average`
- formal best epoch: `300`
- Stage AG minus Stage AD state L2: `0.008201333501826058`
- Stage AG minus Stage AD residual L2: `0.025165614125623348`
- Stage AG minus Stage AD residual cosine: `-0.00988485554464813`
- Stage AG minus Stage AD shell skill: `2.7607713891793453`
- Stage AG minus Stage AD radial skill: `0.0885063445781622`
- Stage AG first 10x physical-range step: `1`
- spherical DISCO branch active: `true`

The paired, channel, theta-region, radial-region, geometry-benefit, branch-ablation,
rollout, and efficiency tables are stored in their respective Stage AH directories.
Kernel-geometry correlation is exploratory and is not treated as causal proof.

## Interpretation boundary

This is an adapted spherical Kerr--Schild workflow using a Euclidean spherical
tangent proxy. It is not a Kerr--Schild covariant operator and is not the exact
volumetric 3-D DISCO implementation from the paper.

`REPRODUCTION_SCOPE = ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION`

`EXACT_REPRODUCTION_BLOCKED = true`
