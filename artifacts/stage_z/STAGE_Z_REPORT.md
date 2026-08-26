# Stage Z — Higher-Resolution Adapted Spherical-KS Reproduction and Regrid-Loss Test

`REPRODUCTION_SCOPE = ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION`

Stage Z is not an exact paper reproduction, a paper-faithful LocalNO, or an
exact volumetric 3D DISCO reproduction. Exact reproduction remains blocked by
Stage Y. The data/information half completed; the model half was stopped at the
mandatory CUDA gate without CPU fallback.

## Frozen controls and generated datasets

- All 212 raw snapshots, chronological split, eight stored spherical-component
  fields, nearest-leaf sampling, residual contract, Stage-T architecture/loss,
  optimizer, seed, and pair order are frozen.
- Z64 regression: all 40 arrays at snapshots 0/50/100/150/211 are bitwise
  identical; `Z64_REGRID_REGRESSION_PASS = true`.
- Z64/Z96/Z128 SHA256: `3582a5c4...b50da`, `292d3fe6...1564`,
  `fe62e9c2...9b1d`. Z128 is only `HIGHER_RES_SAMPLING_REFERENCE`.
- All generated HDF5 arrays are float32 `(N,C,Nphi,Ntheta,Nr)`, finite, with
  strictly positive rho/press. Direct raw sampling equals decompressed HDF5
  bitwise on 120 checked cross-resolution arrays (80 at Z96/Z128).
- Uncompressed float32 estimates are 6,002,049,024 bytes for Z96 and
  14,227,079,168 bytes for Z128; the pre-generation project filesystem had
  about 935 GiB free and RAM had about 13 GiB available. Chunked generation
  never loaded a complete trajectory into RAM.
- Physical-r internal shell boundaries are identical across resolutions; they
  are not re-fit to equal index counts.
- Common trainable tensor hash: `7e10895cb283be8532e4c036e8a526c536a632cce880c500080ea1d14fd89311`;
  358,296 parameters at all resolutions; common pair-order hash:
  `e8f20d8ff8ea67160d9655f6bc0e9fc41eedf2f3b2737e48075ea5cdbae94bb7`.

## Information-fidelity result

| resolution | class | median increment rel. difference | median cosine | shell error | radial error |
|---:|---|---:|---:|---:|---:|
| 64 | POOR | 0.503429 | 0.871549 | 0.206029 | 0.328343 |
| 96 | MODERATE | 0.378347 | 0.926872 | 0.103076 | 0.205074 |
| 128 | GOOD/reference | 0 | 1 | 0 | 0 |

Z96 improves relative difference by 24.85%,
shell error by 49.97%,
radial error by 37.54%,
and cosine by 0.055324.
All four predeclared conditions pass, so `HIGHER_RES_DATA_INFORMATION_GAIN = true`.
Regional relative improvements are inner 25.08%, middle
26.92%, outer 9.18%; middle improves most,
while outer remains the least rescued.

Per-channel, Bcc2 and rho have the largest relative-difference reductions
(about 25.4% and 23.3%). Bcc1 is essentially unchanged/slightly worse; vel2
improves only about 2.8%. Z96 remains MODERATE rather than close enough to call
the Z128 sampling reference reproduced.

## Resolution-specific P3 and oracle

P3 transform family/policies are identical, while numerical epsilon/median/MAD
statistics were independently fit from train snapshots 0..168 for each
resolution. Validation was never used for fit. Normalizer hashes are Z64
`c2a36e...ffa6`, Z96 `63c5dd...127`, Z128 `aa737d...d4a`.

For the priority channels Bcc2/Bcc3/vel3/rho/press, validation encode/decode
relative floors remain approximately 1e-8--3e-7 at every resolution. Thus
higher resolution did not reintroduce their preprocessing floor. The large
canonical softclip floors on Bcc1 and vel2 remain a frozen P3 limitation.

Same-resolution persistence was evaluated on all 42 validation pairs: normalized
state L2 is 0.310282
/ 0.310069
/ 0.309407
for Z64/Z96/Z128. Physical L2 is
0.571584 /
0.565225 /
0.566144.
The spherical-coordinate volume-proxy persistence error is recorded separately;
it is not strict Kerr-Schild proper-volume error and does not replace the selector.

## GPU gate and model status

`nvidia-smi` failed with `GPU access blocked by the operating system`;
PyTorch `2.12.1+cu130` / CUDA `13.0` reported
`cuda_available=false`, device count 0. Z96 forward/backward therefore could
not begin. No CPU fallback, optimizer step, epoch, checkpoint, one-step model
evaluation, volume-proxy model error, or rollout was produced. Z128 preflight
was not attempted after the prerequisite Z96 CUDA failure.

The reusable Z64 Stage-T epoch-150 reference is contract-compatible (same
split/P3 family/model/loss/scheduler prefix, 6,300 updates): state L2
0.266711, residual L2
0.891851, residual cosine
0.740602, shell/radial skill
-1.679472/-0.348089,
and first-10x physical range step 1. It cannot answer the cross-resolution
learned-dynamics question alone.

## Direct answers

1. **Does Z96 preserve more increment than Z64?** Yes: all four frozen
   information gates improve and classification moves POOR -> MODERATE.
2. **Is Z96 already close to Z128?** No under the frozen fidelity thresholds;
   its median relative difference is still 0.378. Z128 is a sampling
   reference, not truth.
3. **Most resolution-sensitive channels?** Bcc2 and rho by relative-difference
   reduction; vel3/vel1/press also improve materially. Bcc1 and vel2 change least.
4. **Most improved region?** Middle (26.9%), then inner
   (25.1%); outer improves only 9.2%.
5. **Does higher resolution lower learned residual L2?** Not available: GPU block.
6. **Does learned residual cosine improve?** Not available: GPU block.
7. **Does shell skill improve?** Sampling shell fidelity improves; learned shell
   skill is not available.
8. **Does radial skill improve?** Sampling radial fidelity improves; learned
   radial skill is not available.
9. **Is first-10x delayed beyond step 1?** Not available for Z96/Z128.
10. **Does rho/press decoder-tail improve?** Their oracle floor remains tiny and
    stable; model decoder-tail behavior is not available.
11. **Does information gain become learned-dynamics gain?** Unresolved because
    no high-resolution model could run.
12. **Is 64^3 the main adapted bottleneck?** It is a demonstrated information
    bottleneck, but dominance over operator/objective limitations is unresolved.
13. **Use 64, 96, or 128?** `UNRESOLVED_PENDING_GPU_MODEL_COMPARISON`; Z96 is the
    first resource-prudent candidate because it passes the information gate,
    but it cannot be scientifically selected without its controlled model run.
14. **Proceed to unified Persistence/FNO/CNN/LocalNO benchmark?** Authorized only
    as a resource-feasible adapted suite after GPU access is restored and the
    Z96 controlled model comparison resolves the selected resolution.

## Final labels

```text
PRIMARY_DECISION = E
PRIMARY_DECISION_LABEL = HIGH_RES_MODEL_RESOURCE_LIMITED

REPRODUCTION_SCOPE =
ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION

Z64_REGRID_REGRESSION_PASS = true
HIGHER_RES_DATA_INFORMATION_GAIN = true

TEMPORAL_INCREMENT_FIDELITY_64 = POOR
TEMPORAL_INCREMENT_FIDELITY_96 = MODERATE
TEMPORAL_INCREMENT_FIDELITY_128 = GOOD_REFERENCE_SELF

Z96_TRAINING_RESOURCE_FEASIBLE = false
Z128_TRAINING_RESOURCE_FEASIBLE = false
SELECTED_ADAPTED_RESOLUTION = UNRESOLVED_PENDING_GPU_MODEL_COMPARISON

EXACT_REPRODUCTION_BLOCKED = true
AUTHORIZE_NEXT_STAGE = resource_feasible_adapted_baseline_suite
```
