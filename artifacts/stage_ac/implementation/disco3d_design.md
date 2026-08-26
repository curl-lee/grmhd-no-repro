# DISCO3D Design Feasibility

The frozen formulas resolve to:

- `radius_cutoff = 0.03125`
- `psi_local_phi/theta/r = [3, 3, 3]`
- `q = 3.0517578125e-05`
- sampled radii within support: `[0.0, 0.03125]`
- zero-support basis indices: `[1, 2, 3]`

For K=5 the centers are 0, R/4, R/2, 3R/4, R. The exact 3x3x3 cell-centre stencil has
supported radii only 0 and R. Consequently bases 1, 2, and 3 have `Z_k=0`; division by
`Z_k+eps` leaves their quadrature integral at zero, not one. Their trainable coefficients would
also have identically zero forward contribution and gradient.

Implementing a trainable operator would require an unapproved scientific choice: increase the
cutoff/stencil, reduce K, or define a subcell/voxel-integrated projection rule. Stage AC forbids
changing radius or basis, and supplies no subcell quadrature contract, so implementation stops
before LocalNO integration and training.
