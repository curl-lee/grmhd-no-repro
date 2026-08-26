# Stage V paper-loss provenance audit

## Evidence boundary

No paper PDF is present anywhere in the local repository (`find . -iname
'*.pdf'` returned no files).  Therefore this audit does not claim a fresh
PDF-to-code verification.  It uses the already frozen source audit in
`docs/PAPER_LOSS_CONTRACT.md`, which identifies arXiv:2512.01576v1 Appendix
C.5--C.7 and D.1, plus the executable implementation and prior Stage D/G/H/I
artifacts.  The pinned software reference remains
`external/neuraloperator@86a8bc7812a31b42c4f7895693cf4ac11521c066`.

## Classification

Each concrete term below has one Stage-V classification.  “Paper-explicit
family” records provenance without pretending that this reduced spherical
implementation is literal paper code.

| term | classification | provenance and current meaning |
| --- | --- | --- |
| Plain unweighted normalized per-voxel L2 | `PAPER_EXPLICIT` | Frozen Appendix-D baseline; implemented by `PlainL2Loss`.  Stage T applies it to normalized residuals under the residual-output adaptation. |
| Full component fidelity (magnetic coefficient 1.2, other channels 1) | `PAPER_ADAPTED` | Paper-explicit family from Appendix C.7, but local reduction is resolved as per-voxel MSE and the reduced protocol predicts P3 residuals rather than the paper's exact state/operator setup. |
| H1 gradient match, coefficient 0.05 | `PAPER_ADAPTED` | Paper-explicit family, implemented with upstream periodic index-grid differences on spherical storage axes.  This is not a covariant or coordinate-aware GRMHD H1. |
| velocity ROI, coefficient `8*min(1,epoch/375)` | `PAPER_ADAPTED` | Paper-explicit family; frozen implementation uses per-sample top-20% stored-component velocity proxy from the canonical oracle, not validation. |
| rho lower bound, coefficient 0.05 | `PAPER_ADAPTED` | Paper-explicit family, with train-only empirical bound mapped into normalized space. |
| thermal lower bound, coefficient 0.05 | `PAPER_ADAPTED` | Paper-explicit family; `press` replaces the unavailable/unverified paper `eint`; EOS conversion is disabled. |
| rho/thermal upper-bound diagnostics, coefficient 0 | `PAPER_ADAPTED` | Paper-explicit family retained at exactly zero training contribution. |
| rho radial envelope, coefficient 0.05, Delta=1.5 | `PAPER_ADAPTED` | Paper-explicit family, using the frozen `appendix_literal_press_proxy` radial adaptation. |
| thermal radial envelope, coefficient 0.05, Delta=1.5 | `PAPER_ADAPTED` | Paper-explicit family with the explicit `press` adaptation. |
| dissipative gate, coefficient 5e-4 | `PAPER_ADAPTED` | Paper-explicit family represented by a train-only normalized global-array norm gate; it is not physical/covariant dissipation. |
| Stage-V residual-direction loss | `REPRODUCTION_ONLY` | New causal objective diagnostic; not a paper loss. |
| Stage-V shell-evolution loss | `REPRODUCTION_ONLY` | New normalized-state objective aligned structurally (same frozen eight shells and variance evolution) with the physical evaluation metric; not a paper loss. It intentionally excludes decoder-tail amplification. |
| Stage-V normalized radial-profile loss | `REPRODUCTION_ONLY` | New objective aligned with temporal radial-profile evolution; not a paper loss. |
| physical error, decoder-tail fraction, shell/radial skill, OOD score | `DIAGNOSTIC_ONLY` | Evaluation and attribution only; never checkpoint selectors or training terms. |
| prior Round-2/3 bounded residual, hybrid target, rollout-aware/range losses, Fold-B/recency terms | `REPRODUCTION_ONLY` | Earlier extensions explicitly excluded from Stage E and Stage V. |

## Why V4 is not authorized

Stage H measured that the current upstream H1 uses unit-cube spacing `1/N` on
each array dimension and amplifies the comparable unit-index gradient term by
`64^2 = 4096`.  Its periodic assumptions also mismatch theta and radial
boundaries.  Stage I showed that removing or replacing this H1 relieves the
specific gradient compression but does not establish a scientifically correct
covariant paper H1 or recover a broad advantage over Plain.

Consequently a literal reuse of the old Stage-E Full objective would knowingly
restore an index-grid pathology; choosing one of the Stage-I diagnostic H1
variants would instead invent a new “paper Full” without unambiguous paper
provenance.  Stage V therefore freezes:

```text
PAPER_FULL_PILOT_NOT_AUTHORIZED
```

No V4 model will be trained.  V1--V3 are explicitly reproduction-only causal
objective diagnostics.

Evidence files include `docs/PAPER_LOSS_CONTRACT.md`,
`docs/PAPER_STAGE_D_PRIOR_AUDIT.md`, `docs/PAPER_STAGE_G_PILOTS.md`,
`outputs/paper_reduced100/stage_h/h1_spacing_audit.json`, and
`docs/PAPER_STAGE_I_DECISION.md`; executable definitions are in
`src/grmhd/paper_losses.py`.
