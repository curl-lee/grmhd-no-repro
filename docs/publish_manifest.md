# Publication manifest

| Field | Value |
|---|---|
| Repository | `curl-lee/grmhd-no-repro` |
| Branch | `main` |
| Main commit | the commit containing this manifest; resolve exactly with `git rev-parse main` |
| Publication base | `d721e3d82f0a3ccdad441bc6a9c8bba18b7c0052` |
| Stage H frozen starting commit | `65f94a5252b4dbed559df2b39ee02ad529eb634f` |
| First included WSL stage | F |
| Last included stage | AH |
| Included stages | F, G, H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z, AA, AB, AC, AD, AE, AF, AG, AH |
| Confirmed Windows stages included | none |
| Ambiguous stages excluded | A--E (no direct stage-level WSL artifact evidence) |
| Dataset files committed | no |
| Checkpoint binaries committed | Stage AG formal best/last only |
| Pinned neuraloperator | `86a8bc7812a31b42c4f7895693cf4ac11521c066` |
| Test summary | `536 passed, 2 skipped, 13 warnings` in 32.11 s |

## Included implementation families

FNO, differential LocalNO, P3 preprocessing/residual contracts, mixed-basis
spectral diagnostics, adapted volumetric DISCO3D, isotropic spherical DISCO3D,
anisotropic spherical DISCO3D, model attachments, dataset/preprocessing code,
shell construction, loss/prior code, training/checkpoint utilities, and
evaluation/rollout metrics.

## Included artifact directories

- `outputs/paper_reduced100/` shared manifests/stats/losses/priors and Stage F--R.
- `artifacts/stage_s/` through `artifacts/stage_af/`.
- `artifacts/stage_ag/` controlled-run records and formal best/last checkpoints.
- `artifacts/stage_ah/` integrity, one-step, paired, regional, attribution, rollout, efficiency, figures, report, and decision.
- `outputs/environment_wsl.json`.

## Excluded large files

- Four local processed HDF5 datasets totaling about 5.4 GB; see
  `docs/data_manifest.md` for exact sizes/hashes.
- 226 checkpoint/state files totaling 1,705,098,858 bytes; see
  `docs/excluded_checkpoint_sha256.csv` for per-file hashes.
- All raw Athena++ snapshots and transient caches/events.
- Stage AG fixed-epoch/resume checkpoints and raw validation prediction arrays;
  see `docs/stage_ag_excluded_artifacts_sha256.csv`.
- All work beyond the Stage-AH publication ceiling.

The literal tip SHA cannot be embedded inside the commit that determines that
same SHA without creating a self-reference. The immutable publication base is
recorded above; the exact final tip is reported by the repository ref and the
final release verification.
