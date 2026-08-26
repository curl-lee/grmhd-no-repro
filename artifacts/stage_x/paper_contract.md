# PAPER_PIPELINE_CONTRACT

Primary source: [From Black Hole to Galaxy: Neural Operator Framework for Accretion and Feedback Dynamics](https://arxiv.org/pdf/2512.01576v1), arXiv `2512.01576v1`, locally verified PDF SHA256 `fe84a42289da7e86dd8460d223e57627a86aa0b2114a64a13b83f84388441808`. The table deliberately keeps paper facts separate from upstream implementation assumptions.

| field | value | status | primary evidence |
|---|---|---|---|
| simulation coordinate system | Cartesian Kerr–Schild | EXPLICIT_IN_PAPER | Appendix A.2, PDF p.9 |
| spatial domain | bounded fine domain L^3 and coarse domain (n_L L)^3, n_L=6; numeric coordinate bounds not given | EXPLICIT_IN_PAPER | §2, PDF p.3 |
| final training resolution | 64_x × 64_y × 64_z | EXPLICIT_IN_PAPER | Appendix C.2, PDF p.10 |
| input variables | eight fields at t: bcc1:3, dens, eint, velx:y:z | EXPLICIT_IN_PAPER | Appendix C.2 |
| output variables | same eight fields at t+ΔT | EXPLICIT_IN_PAPER | §2 and Appendix C.2 |
| thermal variable definition | C.2 calls it internal energy eint; A.2 separately says GRMHD outputs P; exact P↔eint relation is absent | EXPLICIT_IN_PAPER | Appendices A.2 and C.2 |
| vector component basis | Cartesian-KS x/y/z components | EXPLICIT_IN_PAPER | Appendices A.2/C.2 |
| positional encoding | default uses radial-shell one-hot; normalized Cartesian Fourier features are an ablation, not default | EXPLICIT_IN_PAPER | §2, C.4, D.1 |
| shell/radial embedding | 8 one-hot shells from log distance to Cartesian index-grid centre; rmax=10 | EXPLICIT_IN_PAPER | §2, C.4, Table 1 |
| number of snapshots | Ndata=300 generated; last 250 used | EXPLICIT_IN_PAPER | §2, PDF pp.3–4 |
| temporal cadence | ΔT=T/Ndata | EXPLICIT_IN_PAPER | §2 |
| train/validation split | last 250, 80/20 | EXPLICIT_IN_PAPER | §2 |
| preprocessing transforms | positive log10 dens/eint; signed-log bcc1:3; linear velocities; inverse atanh at 0.99γ | EXPLICIT_IN_PAPER | Appendix C.3 |
| normalization | per-channel median/MAD robust z-score and γ=6 tanh soft clip | EXPLICIT_IN_PAPER | Appendix C.3 |
| architecture | 3D Local Neural Operator with equidistant DISCO, volumetric input | EXPLICIT_IN_PAPER | Appendix C.8 |
| LocalNO layer composition | exact per-layer branch placement is not stated | NOT_SPECIFIED | Appendix C.8 gives only backbone family/DISCO |
| Fourier branch | not separately specified for the instantiated paper model | NOT_SPECIFIED | paper never resolves layer flags |
| differential branch | not separately specified for the instantiated paper model | NOT_SPECIFIED | paper never resolves layer flags |
| DISCO/local integral branch | equidistant discrete–continuous convolution specialized to volumetric inputs | EXPLICIT_IN_PAPER | Appendix C.8 |
| number of modes | not reported | NOT_SPECIFIED | paper PDF |
| width | not reported | NOT_SPECIFIED | paper PDF |
| depth | not reported | NOT_SPECIFIED | paper PDF |
| training epochs | 1200 | EXPLICIT_IN_PAPER | Appendix C.9/Table 1 |
| loss composition | component-weighted L2 + 0.05 H1 + velocity ROI + dissipation + radial envelope + constraints | EXPLICIT_IN_PAPER | Appendix C.7 |
| rollout procedure | autoregressive fine-level rollout; reported 50/100-step behavior | EXPLICIT_IN_PAPER | §2, figures and C.10 |
| coarse/fine coupling | HDF5 rollout/time interpolation; hydro inner-boundary overwrite; magnetic CT/EMF treatment | EXPLICIT_IN_PAPER | Appendix E |

## Primary-evidence caveats

The paper explicitly claims a volumetric 3D DISCO backbone, but does not publish modes, width, depth, parameter count, resolved boundary policy, local-kernel support/radius, or per-layer Fourier/differential flags. The paper also contains a thermal-description mismatch: Appendix A.2 says the GRMHD output includes pressure `P`, whereas Appendix C.2 says the network tensor contains internal energy `eint`. No GRMHD EOS conversion resolving this is specified. Those absences are `NOT_SPECIFIED`, not inferred facts.
