# Frozen Stage K field-collapse detector contract

- Function: `grmhd.paper_stage_g_evaluation.artifact_diagnostics`
- Function-source SHA256: `3a9ea391b663df6b58a0c75942283658ee0214669d234e2c5a9594cc5f36e95d`
- Full source-file SHA256: `64125810309c129ca56c326c6a6be7f60f94c706b8362b62810947587d7101e8`
- Contract SHA256: `2e0d4df70e9f074f410ea29ad806d03fa44e9b24aa95650e3305d291d996511b`
- Prediction domain: canonical decoded physical prediction after the rho/press-only evaluation bound clamp and the frozen inverse clamp.
- no-GT reference: raw physical validation snapshot 91.
- Reduction: separately per channel, one standard deviation over all `(phi, theta, r)` voxels; PyTorch correction is 1.
- Formula: `prediction.std() / max(reference.std(), 1e-12)`.
- Collapse flag: strict ratio `< 0.05`.
- Shell aggregation: none. Time aggregation: none; every step is flagged independently.
- NaN/Inf: the rollout rejects nonfinite decoded predictions before calling the detector; the detector has no separate nonfinite branch.
- Saturation: there is no mask exemption. Bcc/velocity predictions can be limited by the frozen inverse clamp; only rho/press additionally receive the evaluation bounds clamp.
