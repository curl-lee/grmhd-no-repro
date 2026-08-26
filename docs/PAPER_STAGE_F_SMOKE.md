# Stage F paired training smoke

## Scope and status

Stage F ran two sequential engineering smokes, each with 2 epochs, 2 train
batches per epoch, 1 validation batch per epoch, batch size 1, and smoke
accumulation 1.  Both executed real Adam optimizer updates.  These runs are not
scientific models and their metrics cannot establish that Full is better or
worse than Plain.

The controlled identity gate passed:

- both models have 331,832 parameters;
- both start from tensor-state SHA256
  `00dde6be92d4037f4abfc671cdd2153cfe98901148442903006e745ff3b86b43`;
- dataset, split, preprocessing, shell/radial representation, optimizer,
  scheduler, seed, batch, and evaluation configuration are identical;
- resolved differences are confined to experiment/output names and `loss.*`.

The Stage F process could not access the WSL GPU (`GPU access blocked by the
operating system`), so both smokes ran on CPU.  CUDA peak memory is therefore
explicitly unavailable and recorded as 0 MiB, not inferred.

## Full FNO smoke

Status: `passed_with_h1_warning`.

All four forward/backward/Adam updates were finite and changed model state.
H1 weight remained exactly 0.05.  Weighted H1/base value ratios were
`16.20, 18.66, 21.61, 17.72`; weighted H1/base gradient ratios were
`7.93, 5.86, 8.65, 8.52`.  All four updates exceeded gradient norm 1 and were
clipped successfully; this is reported rather than used to retune the paper
weight.

The observed engineering timing was 4.71 s for the bounded training/validation
loop and 7.09 s end-to-end, including strict reload and short evaluation.  CPU
throughput was 0.85 sampled train batches/s.  Timings are environment-specific.

Best and last upstream training-state bundles both restored model, Adam,
scheduler, epoch 2, ROI ramp `2/375`, metadata, and deterministic validation
prediction exactly.  The best one-step global relative L2 diagnostics were:

| metric | value |
| --- | ---: |
| normalized | 1.01833523 |
| model to canonical oracle | 0.93721440 |
| model to raw target | 0.95518120 |
| canonical oracle floor | 0.50439416 |

The 3-step physical autoregression was finite, kept rho/press positive, decoded
once per step, and re-encoded once for the next step.  It is only a pipeline
smoke.

## Plain L2 smoke

Status: `passed`.

Plain used the same input/target representation and initial FNO.  Its training
log contains only strict unit-channel normalized squared error, clamp
diagnostics, gradients, and clipping.  It explicitly lists H1, ROI, bounds
training penalty, radial envelope, and dissipation as disabled; none is emitted
as a zero-valued Plain loss component.  Evaluation-only rho/press clamp remains
enabled and does not enter `PlainL2Loss`.

All four Adam updates were finite and changed model state.  All four also
triggered norm-1 clipping.  The observed bounded loop was 2.63 s and the
end-to-end run 4.95 s on CPU, or 1.52 sampled train batches/s.  Best/last strict
reload and deterministic prediction parity passed.  Best one-step diagnostics
were:

| metric | value |
| --- | ---: |
| normalized | 1.01568949 |
| model to canonical oracle | 0.93717354 |
| model to raw target | 0.95512368 |
| canonical oracle floor | 0.50439416 |

Its 3-step physical autoregression also remained finite and rho/press positive
with exact transform-count checks.

## Decision

Stage F meets the engineering pass criteria with an H1 adaptation warning.  It
permits Stage G's two controlled 30-epoch pilots, provided the Full run retains
per-batch H1/base value and gradient ratios plus clipping monitoring.  The
warning does not authorize changing H1 weight, gamma, inverse clamp, radial
mode, or any other paper parameter.

Generated, Git-ignored evidence is under
`outputs/paper_reduced100/stage_f/`, including pre-smoke tests, paired-config
diff, both checkpoint bundles/logs/runtime/metrics, and the three-format
controlled smoke comparison.
