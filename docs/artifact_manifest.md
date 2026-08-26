# Artifact publication manifest

## Included scientific artifacts

The release retains reports, decisions, resolved configurations, provenance,
CSV/JSON metrics, train/validation logs, rollout summaries, implementation and
geometry audits, bootstrap results, and review-sized figures for confirmed WSL
stages through AF.

| Stage range | Included directory |
|---|---|
| F--R | `outputs/paper_reduced100/stage_f/` through `stage_r/` |
| S--AF | `artifacts/stage_s/` through `artifacts/stage_af/` |
| Shared contracts | `outputs/paper_reduced100/{manifest,stats,losses,priors,...}` |

The largest retained individual files are lightweight structured result files:
`stage_h/h1_shell_decomposition.json` (18.6 MB), Stage AB one-step metrics
(16.8 MB), and `preprocessing_loss_audit.json` (9.8 MB). No retained file
exceeds GitHub's ordinary 100 MB per-file limit.

## Excluded checkpoint binaries

All `.pt`, `.pth`, and `.ckpt` files were omitted. They comprise 226 files and
1,705,098,858 bytes in the private recovery snapshot. They are reproducible
model/optimizer/scheduler bundles, shared initial states, or selected-state
caches; the formal metrics, configuration, initialization hashes, checkpoint
selection records, and strict-reload results remain published.

Every omitted binary's original repository-relative path, byte size, and
SHA256 is recorded in
[`excluded_checkpoint_sha256.csv`](excluded_checkpoint_sha256.csv). Recreate a
checkpoint by using the corresponding stage's resolved configuration and the
training entry point named in its report. Reproduction can then verify the new
binary against the recorded hash where the software/hardware determinism
contract applies.

Git LFS was available but was not used: no checkpoint was required to inspect
the source, numerical reports, decisions, or operator tests, and omitting the
1.7 GB binary set keeps the repository reviewable.

## Excluded data and transient files

| Excluded class | Reason | Recovery/verification |
|---|---|---|
| raw `.athdf` snapshots | scientific dataset is not redistributed here | obtain separately and verify with `artifacts/stage_s/data_manifest.csv` |
| processed `.h5`/`.hdf5` | 0.7--3.0 GB per expanded dataset | paths and SHA256 are in `data_manifest.md` |
| later/in-progress stage artifacts | outside the Stage-AF publication ceiling | retained only in the private pre-cleanup snapshot |
| Python caches/test caches | generated transient content | regenerated automatically |
| TensorBoard events/temp logs | machine-local transient content | rerun the documented stage command |
| raw Stage-AB process/PATH dumps | contains machine-local process and editor paths | formal report and GPU gate are retained |

Recovery archives were created outside the repository before branch cleanup.
They are intentionally not tracked or pushed.
