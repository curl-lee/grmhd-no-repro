# Paper operator audit

Primary paper evidence is Appendix C.8 of [From Black Hole to Galaxy: Neural Operator Framework for Accretion and Feedback Dynamics](https://arxiv.org/pdf/2512.01576v1) (verified PDF SHA256 `fe84a42289da7e86dd8460d223e57627a86aa0b2114a64a13b83f84388441808`). It explicitly says the backbone is a **3D Local Neural Operator with equidistant discrete–continuous convolutions (DISCO) specialized to volumetric inputs**. Thus a local-integral/DISCO mechanism is part of the claimed paper backbone.

The paper does **not** resolve per-layer flags, modes, width, depth, local kernel basis/support/radius, or whether every layer simultaneously includes Fourier and differential branches. Consequently:

1. Exact per-layer branch composition: `NOT_SPECIFIED` in this paper.
2. Reliance on DISCO/local integral: `EXPLICIT_IN_PAPER`.
3. DISCO's detailed role: the paper identifies it as the equidistant discrete–continuous volumetric convolution; detailed quadrature/kernel mechanics are not stated here.
4. Current missing mechanism: every local integral branch.
5. Mechanism plausibility: removing a learnable localized integral kernel can plausibly alter local transport, radial redistribution, inner/outer transfer, and boundary-local structure. This is a mechanism hypothesis, **not causal proof**.

`PAPER_OPERATOR_GAP = HIGH`.
