# Stage L collapse-detector audit

## Scope and gate status

Stage L is a post-hoc attribution audit. Its first gate performed no training,
backward pass, optimizer or scheduler construction, checkpoint write, preprocessing
fit, or model update. The frozen Stage K decision remains
`C. TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE`. Phase 2 has now completed the
authorized variance, shell, radial, and spectral attribution under the unchanged
gate and detector contract.

The repository gate passed at project commit
`980ed310d117ca1625b4afd91c2788ef3b645592`: the repository path and branch were
correct, the initial project worktree was clean, and pinned upstream commit
`86a8bc7812a31b42c4f7895693cf4ac11521c066` had a clean worktree. The existing
suite passed with 248 tests and 13 warnings before Stage L changes.

## Detector source and call path

The frozen detector is
`grmhd.paper_stage_g_evaluation.artifact_diagnostics` in
`src/grmhd/paper_stage_g_evaluation.py`. Its function-source SHA256 is
`3a9ea391b663df6b58a0c75942283658ee0214669d234e2c5a9594cc5f36e95d`; the
full source-file SHA256 is
`64125810309c129ca56c326c6a6be7f60f94c706b8362b62810947587d7101e8`.

`scripts/evaluate_paper_stage_g.py:evaluate_rollout` calls the detector after a
model output is decoded to physical space. For no-GT steps, the arguments are:

- prediction: canonical decoded physical prediction, including the frozen
  normalized inverse clamp (`0.99 * gamma`, `gamma=6`) and the rho/press-only
  evaluation bounds clamp;
- reference: raw physical validation snapshot 91;
- reference input: the same raw physical snapshot 91;
- reference kind: `initial_snapshot_91`.

Thus this detector is neither normalized-domain nor a model-to-oracle metric. It
compares a decoded model prediction with a raw physical initial state. Bcc and
velocity channels do not receive the rho/press evaluation bounds clamp, but all
decoded channels remain subject to the frozen inverse clamp. No saturation mask is
removed or exempted.

## Exact contract

The detector processes each of the eight channels independently. After selecting
batch element 0 and one channel, it reduces all `(phi, theta, r)` voxels with
PyTorch `std()` using the default correction of 1. It computes

`std_ratio = prediction.std() / max(reference.std(), 1e-12)`

and appends `<channel>:possible_field_collapse` exactly when `std_ratio < 0.05`.
The comparison is relative and strictly less-than. It does not use range,
variance, shells, radial profiles, masks, or time aggregation. Every rollout step
is classified independently. The surrounding rollout raises on a nonfinite
prediction before the detector is called; the detector itself has no separate
NaN/Inf branch.

The canonical detector-contract SHA256 is
`2e0d4df70e9f074f410ea29ad806d03fa44e9b24aa95650e3305d291d996511b`, and the
detector-config SHA256 is
`864f62987b4a36fcfeef88bb03da33598de94b8b9e8b5b2e210d81fd8c207930`.
The resolved attribution config is frozen at
`configs/paper_reduced100/stage_l_collapse_attribution.yaml`; it fixes the
preprocessing/model severe-loss threshold at `< 0.5` before any attribution
metrics are read.

## Exact Stage K reproduction

The reproduction reused `selected_states.pt` from the Stage K best checkpoint
(epoch 22), loaded snapshot 91 through the frozen `paper_reduced100` protocol, and
called the unchanged detector. It did not run a model. Full flag lists and collapse
subsets match the recorded CUDA rollout at all four selected no-GT steps:

| Step | Bcc3 std ratio | Bcc3 flag | vel3 std ratio | vel3 flag |
|---:|---:|:---:|---:|:---:|
| 25 | 0.00654031 | yes | 0.05824152 | no |
| 50 | 0.00618126 | yes | 0.06666264 | no |
| 75 | 0.00614293 | yes | 0.04491106 | yes |
| 100 | 0.00571497 | yes | 0.04961326 | yes |

The maximum difference between the CPU-recomputed standard-deviation ratios and
the ratios recorded during the CUDA rollout is `6.3891e-08`. This numerical device
difference changes no flags. The exact reproduced collapse sets are Bcc3 at steps
25/50/75/100 and vel3 at steps 75/100.

## Frozen provenance

The HDF5, paper manifest, preprocessing, all six Stage D artifacts, Stage E loss
contract, oracle baseline/semantics, and shared pair order all match their frozen
hashes. FNO Full, FNO Plain, and LocalNO Plain best/last checkpoint directories
were discovered from the config checksums in the Stage G/K manifests. All six
models passed metadata validation and CPU `strict=True` state loading in eval mode;
no optimizer or scheduler was instantiated.

Detailed hashes and machine-readable evidence are in
`outputs/paper_reduced100/stage_l/run_manifest.json`,
`detector_contract.json`, and `detector_reproduction.json`/`.csv`.

## Gate and Phase 2 decision

The Stage L detector gate is `passed`. The subsequent read-only attribution found
both Bcc3 and vel3 to be `C. MIXED_PREPROCESSING_AND_MODEL`, giving overall
`3. MIXED_OVERALL`. Those results are documented separately so the frozen detector
contract and the broader structural attribution remain distinguishable. Stage K C
is unchanged.
