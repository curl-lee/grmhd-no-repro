# Stage M canonical transform source audit

## Scope

This is a post-hoc source and numerical audit. It does not alter the canonical
processor, statistics, checkpoints, detector, or any Stage K/L conclusion. Every
counterfactual is labelled `diagnostic_counterfactual` and is evaluated only in
memory.

The frozen Stage M config SHA256 is
`91a360fe9087f413fccaeaa6cf56874e980773aa7f584b3bf5b2917e8a677206`.
The normalizer artifact SHA256 remains
`1645714424d95b02de2283fc16a8ef05f3864d15f11025580af13f3e67964001`,
and the resolved channel-parameter SHA256 is
`917dc105eb8e1a2c2e4401e249b2e31aab595152d35f2471df263a7c426c6ddb`.

## Actual pipeline

The implementation order differs from the generic T0--T8 wording: robust
normalization occurs before tanh soft clipping. Stage M records the real order.

| stage | function/formula | input domain | output domain | channel parameters | mathematically invertible | observed/explicit loss | frozen |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T0 | identity | raw physical | raw physical | none | yes | no | yes |
| T1 | signed `sign(x) log10(1+abs(x)/epsilon)`, positive `log10(x+epsilon)`, or linear | physical | nonlinear representation | transform, epsilon | yes | no on finite valid input | yes |
| T2 | `(T1-median)/scale` | nonlinear representation | unbounded robust z | train-only median/MAD scale | yes | only ordinary floating error | yes |
| T3 | `6*tanh(T2/6)` | robust z | canonical normalized | gamma 6 | analytically yes for `abs(y)<6` | finite-precision saturation at exact ±6 | yes |
| T4 | identity before decode | canonical normalized | decoder input | none | yes | no additional forward clamp exists | yes |
| T5 | `clamp(y,-5.94,5.94)` then `6*atanh(y/6)` | decoder input | recovered robust z | inverse fraction 0.99 | clamp no; atanh yes | explicit clipping | yes |
| T6 | `T5*scale+median` | robust z | nonlinear representation | median/scale | yes | ordinary floating error | yes |
| T7 | inverse channel transform plus dtype-max finite guard | nonlinear representation | canonical physical oracle | epsilon, output dtype | yes before guard | guard is lossy only if hit | yes |
| T8 | rho/press normalized bounds clamp, then canonical decode | normalized prediction | evaluation physical | train-only rho/press bounds | no | evaluation-only rho/press clipping | yes |

T8 does not modify Bcc2, Bcc3, or vel3. It is a model-prediction evaluation path,
not part of the raw target oracle constructed in `PaperDataProcessor.preprocess`.
The robust statistics were fitted only on snapshots 11--90 and were not refitted.

## Channel parameters

| channel | transform | epsilon | median | scale |
| --- | --- | ---: | ---: | ---: |
| Bcc2 | signed_log | 0.001 | -0.0581621 | 0.115318 |
| Bcc3 | signed_log | 0.01 | 2.19452e-12 | 0.0118065 |
| vel3 | linear | 0 | 0.00437162 | 0.00624954 |

The very small Bcc3/vel3 robust scales map broad validation tails deep into tanh
saturation. This statement is supported by the trace, not inferred from parameter
names alone.

## Source hashes

- `paper_preprocessing.py`: `ce72b6...4547d`
- `paper_bounds.py`: `cda59d...6d9eb`
- `paper_data_processor.py`: `debd4a...2bb2`
- tensor encode/decode function hashes: `3ebe6e...2a08` / `46dcc9...60dd`
- evaluation bounds clamp function hash: `b6f468...8b32`

Machine-readable stages, complete hashes, and source locations are in
`outputs/paper_reduced100/stage_m/transform_source_audit.json`.
