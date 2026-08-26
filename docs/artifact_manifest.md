# Artifact publication manifest

## Included scientific artifacts

The release retains reports, decisions, resolved configurations, provenance,
CSV/JSON metrics, train/validation logs, rollout summaries, implementation and
geometry audits, bootstrap results, and review-sized figures for confirmed WSL
stages through AH.

| Stage range | Included directory |
|---|---|
| F--R | `outputs/paper_reduced100/stage_f/` through `stage_r/` |
| S--AF | `artifacts/stage_s/` through `artifacts/stage_af/` |
| AG--AH | completed controlled-training records and post-hoc scientific analysis |
| Shared contracts | `outputs/paper_reduced100/{manifest,stats,losses,priors,...}` |

The largest retained individual files are lightweight structured result files:
`stage_h/h1_shell_decomposition.json` (18.6 MB), Stage AB one-step metrics
(16.8 MB), and `preprocessing_loss_audit.json` (9.8 MB). No retained file
exceeds GitHub's ordinary 100 MB per-file limit.

## Checkpoint publication policy

The original through-AF publication omitted all `.pt`, `.pth`, and `.ckpt`
files. They comprise 226 files and
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

Stage AG adds two review-sized formal checkpoints: `best.pt` and `last.pt`
(about 8.4 MB each). Fixed-epoch checkpoints, the resume state, and the 42-file
raw validation prediction cache remain excluded. Their individual paths,
SHA256 values, byte sizes, reasons, and regeneration routes are recorded in
[`stage_ag_excluded_artifacts_sha256.csv`](stage_ag_excluded_artifacts_sha256.csv).
Git LFS is unnecessary because every retained file is below GitHub's ordinary
per-file limit.

## Excluded data and transient files

| Excluded class | Reason | Recovery/verification |
|---|---|---|
| raw `.athdf` snapshots | scientific dataset is not redistributed here | obtain separately and verify with `artifacts/stage_s/data_manifest.csv` |
| processed `.h5`/`.hdf5` | 0.7--3.0 GB per expanded dataset | paths and SHA256 are in `data_manifest.md` |
| work after Stage AH | outside the current publication ceiling | retain separately until formally completed |
| Python caches/test caches | generated transient content | regenerated automatically |
| TensorBoard events/temp logs | machine-local transient content | rerun the documented stage command |
| raw Stage-AB process/PATH dumps | contains machine-local process and editor paths | formal report and GPU gate are retained |

Recovery archives were created outside the repository before branch cleanup.
They are intentionally not tracked or pushed.
