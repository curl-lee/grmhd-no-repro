# Reproduction status through Stage AI

## Final status

```text
PRIMARY_DECISION = B
PRIMARY_DECISION_NAME = MIXED_ARCHITECTURE_RESULT
REPRODUCTION_SCOPE = ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION
EXACT_REPRODUCTION_BLOCKED = true
AUTHORIZE_NEXT_STAGE = REPRODUCTION_CLOSEOUT
```

Stage AI is complete. The final benchmark includes Persistence, a canonical
spectral-only FNO, a parameter-matched 3D CNN/U-Net, Stage T Differential
LocalNO, Stage AD index-space DISCO3D LocalNO, and Stage AG anisotropic
spherical DISCO3D LocalNO. All five trainable formal checkpoints pass strict
reload, optimizer/scheduler recovery, deterministic prediction, and unified
metric recomputation.

## Frozen benchmark outcome

| result | model / value |
|---|---|
| one-step accuracy winner | 3D CNN/U-Net, state L2 `0.24252603` |
| trained shell leader | FNO |
| trained radial-absolute leader | 3D CNN/U-Net |
| trained radial-skill leader | anisotropic spherical DISCO3D |
| overall transport reference | Persistence |
| relative trained rollout leader | 3D CNN/U-Net, but first10x is 1 |
| compute-efficient baseline | FNO |

Every trained model improves aggregate normalized state L2 over Persistence,
but none has positive aggregate shell or radial persistence-relative skill.
Every trained model reaches the 10x physical-range landmark on rollout step 1.
No trained neural baseline is physically stable in closed loop under the frozen
adapted preprocessing/model contract.

## Geometry conclusion

`SPHERICAL_GEOMETRY_CONCLUSION = PARTIAL_TRANSPORT_GAIN`.

Stage AG improves Stage AD's median shell/radial transport summaries and some
inner/middle regions, while worsening aggregate one-step state, residual, and
direction metrics. Paired absolute-transport intervals cross zero, first10x is
unchanged, and spherical DISCO remains below Persistence. This is limited to
the project-local anisotropic spherical tangent/volume proxy.

## Publication boundary

Source, configs, tests, lightweight logs, CSV/JSON summaries, reports, and
review-sized figures are publishable. Raw Athena++ data, processed HDF5, Stage
AI checkpoint binaries, and transient training state are excluded. Local Stage
AI checkpoint paths, sizes, SHA256 values, and regeneration commands are listed
in `docs/stage_ai_excluded_checkpoint_sha256.csv`.

The verified/adapted/blocked classification is in
[`reproduction_ledger.md`](reproduction_ledger.md). Exact paper reproduction
remains blocked by unavailable official data/code and unresolved exact
preprocessing, operator, and coupling provenance.
