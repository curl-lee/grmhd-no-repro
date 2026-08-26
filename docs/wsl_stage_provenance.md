# WSL stage provenance through Stage AF

## Inclusion rule

Only stage artifacts with positive Linux/WSL evidence were selected. A stage
was accepted when its formal report, resolved configuration, environment record,
or immediate frozen-input provenance identified the `/home/curl` project,
WSL2/Linux platform, `/mnt` source data, `/dev/dxg`, or the RTX 5070 CUDA chain.
No candidate stage directory through AF was classified as Windows-native.
Stages A--E were excluded conservatively because no independent stage artifact
directory with direct WSL execution evidence was present in the audited tree.

| Stage | Execution environment | Evidence | Included | Reason |
|---|---|---|---|---|
| F | WSL2/Linux, CPU smoke | `outputs/paper_reduced100/stage_f/smoke_full_fno/environment.json` | yes | Linux project path and environment capture |
| G | WSL2/Linux, RTX 5070 | `outputs/paper_reduced100/stage_g/gpu_preflight.json` | yes | CUDA device and WSL project provenance |
| H | WSL2/Linux, read-only | `outputs/paper_reduced100/stage_h/frozen_inputs.json` | yes | frozen Stage-G inputs and local project provenance |
| I | WSL2/Linux, RTX 5070 | `outputs/paper_reduced100/stage_i/gpu_preflight.json` | yes | CUDA and local environment metadata |
| J | WSL2/Linux, read-only | `outputs/paper_reduced100/stage_j/data_provenance.json` | yes | `/mnt/d` raw-data and `/home/curl` project evidence |
| K | WSL2/Linux, RTX 5070 | `outputs/paper_reduced100/stage_k/localno_preflight.json` | yes | GPU preflight and Linux environment |
| L | WSL2/Linux, read-only | `outputs/paper_reduced100/stage_l/run_manifest.json` | yes | frozen Stage-K checkpoint/provenance chain |
| M | WSL2/Linux, read-only | `outputs/paper_reduced100/stage_m/run_manifest.json` | yes | local run manifest and frozen inputs |
| N | WSL2/Linux, read-only | `outputs/paper_reduced100/stage_n/run_manifest.json` | yes | local run manifest and pinned upstream |
| O | WSL2/Linux, RTX 5070 | `outputs/paper_reduced100/stage_o/run_manifest.json` | yes | CUDA, local paths, and pinned upstream |
| P | WSL2/Linux, read-only | `outputs/paper_reduced100/stage_p/run_manifest.json` | yes | Stage-O frozen-state replay provenance |
| Q | WSL2/Linux, read-only | `outputs/paper_reduced100/stage_q/run_manifest.json` | yes | local train-only audit manifest |
| R | WSL2/Linux, RTX 5070 | `outputs/paper_reduced100/stage_r/run_manifest.json` | yes | CUDA run and local project provenance |
| S | WSL2/Linux, RTX 5070 | `artifacts/stage_s/STAGE_S_REPORT.md` | yes | `/home/curl`, `/mnt/d`, and GPU evidence |
| T | WSL2/Linux, RTX 5070 | `artifacts/stage_t/full_long/resolved_config.json` | yes | Linux/WSL platform and CUDA metadata |
| U | WSL2/Linux, RTX 5070 | `artifacts/stage_u/variants/coordinate_fd/resolved_config.json` | yes | Linux/WSL resolved environment |
| V | WSL2/Linux, RTX 5070 | `artifacts/stage_v/variants/direction/resolved_config.json` | yes | Linux/WSL resolved environment |
| W | WSL2/Linux, RTX 5070 | `artifacts/stage_w/gpu_preflight.json` | yes | CUDA device and local project path |
| X | WSL2/Linux, read-only | `artifacts/stage_x/regrid/raw_amr_audit.json` | yes | `/mnt/d/GRMHD_data` source and local audit chain |
| Y | WSL2/Linux, read-only | `artifacts/stage_y/STAGE_Y_REPORT.md` | yes | explicit local project and no-mutation audit |
| Z | WSL2/Linux, mixed CPU/GPU gate | `artifacts/stage_z/STAGE_Z_REPORT.md` | yes | local regrid paths and WSL resource preflight |
| AA | WSL2/Linux, GPU blocked | `artifacts/stage_aa/gpu/gpu_gate.json` | yes | explicit WSL GPU-bridge diagnosis |
| AB | WSL2/Linux, RTX 5070 | `artifacts/stage_ab/STAGE_AB_REPORT.md` | yes | WSL bridge recovery and CUDA sanity pass |
| AC | WSL2/Linux, no training | `artifacts/stage_ac/scope/stage_ac_contract.md` | yes | immediate Stage-AB frozen chain; same pinned checkout; no Windows markers |
| AD | WSL2/Linux, RTX 5070 | `artifacts/stage_ad/training/disco3d_localno/resolved_config.json` | yes | platform explicitly contains `microsoft-standard-WSL2` |
| AE | WSL2/Linux, no training | `artifacts/stage_ae/scope/stage_ae_contract.md` | yes | frozen Stage-AD project/data/upstream chain |
| AF | WSL2/Linux, RTX 5070 preflight | `artifacts/stage_af/preflight/cuda_feasibility.json` | yes | RTX 5070 forward/backward feasibility record |

## Exclusions

- Confirmed Windows-only stage artifact directories: none identified among the
  audited candidate directories.
- Ambiguous pre-WSL stages: A--E, excluded because direct stage-level execution
  evidence was unavailable.
- Any work after Stage AF: excluded by the publication ceiling, regardless of
  local existence or execution environment.
- Raw Stage-AB process/PATH dumps: excluded as machine-local diagnostics; the
  formal report and GPU gate retain the scientific recovery evidence.

The first included stage is F and the last included stage is AF.
