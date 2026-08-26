# Operator mechanism comparison

| mechanism | paper contract | current Stage T | scientific consequence |
|---|---|---|---|
| global spectral coupling | exact branch flags not reported | enabled, 8^3 modes | current has global index-grid Fourier mixing |
| local finite-difference coupling | not separately reported | enabled, 3^3 kernel in each of 4 blocks | current has local differential coupling |
| local integral/DISCO coupling | explicitly volumetric equidistant DISCO | absent | HIGH mechanism gap; localized learned integral transport is unavailable |
| skip/MLP | not reported | linear LocalNO skip; channel MLP disabled | cannot match paper implementation |
| positional conditioning | 8 Cartesian-index radial shells | 8 physical spherical-r shells | both supply coarse radial context, with different geometry |
| parameters | not reported | 358,296 | no parameter-count comparison is possible |

Parameter count cannot resolve the comparison because the paper count is `NOT_SPECIFIED` and the missing operator branch changes receptive mechanism, not just size.
