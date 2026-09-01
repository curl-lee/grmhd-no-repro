# Stage AI Final Adapted Baseline Benchmark

## 1. Executive summary

Stage AI completed the frozen six-model benchmark on the adapted spherical
Kerr--Schild Z64/P3 residual task. The scientific decision is
`B — MIXED_ARCHITECTURE_RESULT`.

- The 3D CNN/U-Net is the one-step winner: state L2 `0.24252603`, residual L2
  `0.84369137`, and cosine `0.76828410`.
- Persistence remains the transport reference winner. Every trained model has
  negative aggregate shell and radial persistence-relative skill.
- Trained-model transport is split: FNO is best on shell skill/absolute error,
  CNN has the smallest radial absolute error, and spherical DISCO has the best
  radial skill.
- Every trained model reaches the 10x physical-range landmark at rollout step
  1. CNN is the least-bad trained rollout by the declared lexicographic
  tie-break, not a stable closed-loop model.
- Stage AG retains the Stage AH conclusion: spherical geometry improves median
  transport relative to Stage AD while sacrificing one-step accuracy, but the
  paired absolute-error transport intervals cross zero and both skills remain
  below Persistence.

`REPRODUCTION_SCOPE = ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION`

## 2. Canonical adapted benchmark contract

| item | frozen value |
|---|---|
| data | `data_proc/grmhd_regrid_inner_r200_64_expanded.h5` |
| SHA256 | `3582a5c4f52f4fef305578ffd5a63c51423f8583f35d137c288d5c20203b50da` |
| shape / axes | `(212,8,64,64,64)` / `(N,C,phi,theta,r)` |
| train pairs | sources 0--167 (168 pairs) |
| excluded boundary | 168→169 |
| validation pairs | sources 169--210 (42 pairs) |
| preprocessing | frozen train-only Z64 P3 |
| input | 8 transformed state + 8 frozen physical-r shell channels |
| target | normalized residual `z[t+1]-z[t]` |
| reconstruction | `z[t] + predicted residual` |
| loss | plain normalized residual L2 |
| new-baseline budget | 300 epochs, 168 microbatches/epoch, accumulation 4, 12,600 updates |
| optimizer | Adam, lr `1e-3`, weight decay `1e-4`, global clip 1 |

No validation pair was used to fit P3, tune an architecture, stop training, or
change the objective.

## 3. Comparability audit

The old reduced100 FNO runs and the Stage U spectral diagnostic were not final
contract substitutes: the former used a different data/budget contract and the
latter used a deliberately truncated diagnostic scheduler without the final
300-epoch selector protocol. `FINAL_FNO_REUSE=false`. No comparable canonical
CNN existed, so one parameter-count-frozen U-Net was defined before training.

Stage T epoch 150, Stage AD epoch 300, and Stage AG best/epoch 300 match the
data, P3, split, residual target, loss, and formal-selector contract and were
reused without retraining. Full evidence is in `audit/comparability_audit.csv`.

## 4. Models

| model | parameters | geometry / implementation |
|---|---:|---|
| Persistence | 0 | exact zero residual |
| FNO | 330,648 | spectral-only pinned LocalNO backbone, width 16, modes 8³, 4 layers |
| 3D CNN/U-Net | 346,488 | widths 16/32/64; 2 encoders, bottleneck, 2 decoders |
| Differential LocalNO | 358,296 | spectral + tensor finite differences |
| Index-space DISCO3D LocalNO | 363,480 | repaired index-space local integral |
| Anisotropic spherical DISCO3D LocalNO | 363,480 | spherical tangent/volume proxies |

The CNN is `-4.675%` from the 363,480-parameter reference. It uses 3³
convolutions, SiLU, GroupNorm(4), max-pool downsampling, transposed-convolution
upsampling, circular phi padding, and zero theta/r padding. It does not reuse
the spherical DISCO geometry.

## 5. Training/reuse provenance

Both missing baselines completed the mandatory 300 epochs, 50,400
microbatches, and 12,600 optimizer updates with no NaN/Inf. FNO's fixed-epoch
selector chose epoch 75 (`0.291041652364`); CNN chose epoch 300
(`0.242526025637`). FNO validation values at epochs 2/10/30/75/150/300 were
`0.39268/0.32525/0.39203/0.29104/0.32311/0.30035`; CNN values were
`0.33219/0.36390/0.29803/0.26252/0.24780/0.24253`.

FNO and CNN mean clipping fractions were `0.98857` and `0.99992`. This is
reported, not used to alter the frozen clip. Their total runtimes were
2,897.36 s and 3,477.05 s. All five formal trainable checkpoints passed fresh
construction, strict model load, optimizer/scheduler restoration,
deterministic probes, and metric recomputation. Cross-evaluator selector
differences were at most `1.04e-11` under the declared `1e-10` tolerance.

## 6. One-step results

| model | state L2 | persistence ratio | residual L2 | cosine | physical L2 |
|---|---:|---:|---:|---:|---:|
| Persistence | 0.31028248 | 1.000000 | 1.000000 | 0.000000 | 0.571584 |
| FNO | 0.29104165 | 0.937989 | 1.021080 | 0.651508 | 7.2119e7 |
| 3D CNN/U-Net | **0.24252603** | **0.781630** | **0.843691** | **0.768284** | 3.8756e6 |
| Differential LocalNO | 0.26671132 | 0.859576 | 0.891851 | 0.740602 | 297.777 |
| Index-space DISCO3D LocalNO | 0.26406202 | 0.851037 | 0.891640 | 0.757468 | 2.9973e12 |
| Anisotropic spherical DISCO3D LocalNO | 0.27226335 | 0.877469 | 0.916806 | 0.747583 | 1.6020e12 |

All trained models improve aggregate state L2 over Persistence; only FNO has
residual L2 slightly above one. CNN's state, residual, and cosine improvements
over FNO and spherical DISCO have paired 95% intervals excluding zero.

## 7. Increment dynamics

CNN is best on aggregate normalized residual magnitude and direction. Compared
with FNO, CNN's paired mean differences are `-0.16207` residual L2 and
`+0.10915` cosine; both bootstrap intervals exclude zero. Differential LocalNO
improves over pure FNO in state, residual L2, and direction, also with intervals
excluding zero.

Stage AD versus Differential LocalNO is not a stable general one-step gain:
state and residual-L2 intervals cross zero, while Stage AD's cosine is robustly
higher. Stage AG is robustly worse than Stage AD on state L2, residual L2, and
cosine in the paired comparison.

## 8. Transport results

| model | shell skill | shell absolute | radial skill | radial absolute |
|---|---:|---:|---:|---:|
| Persistence | **0.000000** | **0.0197543** | **0.000000** | **0.0769317** |
| FNO | -0.539260 | **0.0664414** | -0.983433 | 0.188690 |
| 3D CNN/U-Net | -2.145730 | 0.371180 | -0.353718 | **0.0972837** |
| Differential LocalNO | -1.679472 | 0.684244 | -0.348089 | 0.118887 |
| Index-space DISCO3D LocalNO | -4.354481 | 0.939687 | -0.390352 | 0.173822 |
| Anisotropic spherical DISCO3D LocalNO | -1.593710 | 0.550969 | **-0.301846** | 0.147548 |

No trained model beats Persistence on either skill. FNO is the trained shell
leader; CNN has the smallest trained radial absolute error; spherical DISCO has
the least-negative trained radial skill. The benchmark deliberately does not
combine these metrics into an arbitrary score. Skill and absolute error are
both retained because small denominators produce extreme per-channel skills.

## 9. Regional results

Relative to Stage AD, Stage AG lowers mean residual L2 in the equatorial,
north-mid, south-mid, and north-pole theta regions, but worsens it near the
south pole. Radially it improves inner (`0.6891` vs `0.7041`) and middle
(`0.7729` vs `0.8572`) residual L2 but worsens outer (`4.3873` vs `3.7211`).
Thus the Stage AH interpretation remains regional and partial, not uniform.

All trainable models were evaluated over the same five theta and three radial
partitions. Full model/channel tables are in `regional/`.

## 10. Rollout stability

Every model was rolled from snapshot 169 for 100 physical-state steps with no
teacher forcing. All runs remained finite and retained positive rho/press, but
every trained model crossed the 10x decoded physical-range threshold at step
1. Persistence never crossed it.

| model | first10x | first negative cosine | first normalized OOD | step-42 state L2 |
|---|---:|---:|---:|---:|
| FNO | 1 | 19 | 83 | 0.575892 |
| 3D CNN/U-Net | 1 | not reached | 98 | **0.530864** |
| Differential LocalNO | 1 | 12 | 4 | 246.143 |
| Index-space DISCO3D LocalNO | 1 | 11 | 6 | 65.2491 |
| Anisotropic spherical DISCO3D LocalNO | 1 | 8 | 7 | 69.4230 |

CNN is labeled the relative rollout leader only after the common first10x tie,
using later direction failure, later OOD, then lower step-42 state error. This
does not mean it is stable. **No trained neural baseline achieved stable
physical closed-loop rollout under the current adapted preprocessing/model
contract.**

## 11. Computational cost

| model | checkpoint epoch | reported training s | inference s/sample | forward+backward s | 100-step s | peak train MiB |
|---|---:|---:|---:|---:|---:|---:|
| FNO | 75 | 2,897.36 (300 epochs executed) | **0.05258** | 0.08007 | 12.34 | **490.34** |
| 3D CNN/U-Net | 300 | 3,477.05 | 0.05311 | 0.11382 | **12.46** | 515.26 |
| Differential LocalNO | 150 | 1,675.63 (cost to formal checkpoint) | 0.05435 | **0.06857** | 13.12 | 562.82 |
| Index-space DISCO3D LocalNO | 300 | 8,874.89 | 0.06126 | 0.18540 | 13.70 | 648.08 |
| Anisotropic spherical DISCO3D LocalNO | 300 | 92,127.67 | 0.63221 | 1.88355 | 71.10 | 1,608.60 |

FNO is the designated compute-efficient baseline because it has the fewest
parameters, lowest inference time, and lowest peak training allocation; Stage
T has the shortest cost-to-selected-checkpoint and lowest measured
forward+backward time. Spherical DISCO's substantial cost reinforces, rather
than reverses, the mixed scientific conclusion.

## 12. Paired statistical analysis

All intervals use 10,000 fixed-seed resamples of the 42 temporal pairs.

- Every trained model's state improvement over Persistence excludes zero.
- Every trained model's shell and radial absolute-error degradation versus
  Persistence excludes zero.
- CNN beats FNO and spherical DISCO robustly on state, residual L2, and cosine.
- Differential LocalNO beats FNO robustly on one-step metrics; FNO is robustly
  better in shell absolute error, while Differential LocalNO is robustly better
  in radial absolute error.
- Stage AD versus Stage T state/residual differences are inconclusive; Stage AD
  has better cosine, but Stage T has robustly smaller absolute transport errors.
- Stage AG's one-step degradation versus Stage AD is robust. Its median
  transport aggregates improve, but paired shell/radial absolute-error
  intervals cross zero.

Aggregate rankings are descriptive and are not treated as significance tests.

## 13. Architecture comparison

1. **Does FNO beat Persistence?** Yes for state L2 with a paired interval
   excluding zero, but no for residual L2 or transport; physical tails are
   catastrophic.
2. **Does Differential LocalNO beat FNO?** Yes robustly for state, residual L2,
   and direction. Transport is split: FNO shell, LocalNO radial.
3. **Does index-space DISCO stably improve Differential LocalNO?** No. Only
   cosine improves robustly; state/residual are inconclusive and absolute
   transport is worse.
4. **Does spherical DISCO trade one-step accuracy for transport?** Yes at the
   aggregate-median level: one-step metrics worsen while shell/radial summaries
   improve relative to Stage AD. Paired transport uncertainty prevents a
   stronger claim.
5. **Does spherical DISCO exceed Persistence on transport?** No; both skills
   remain negative and absolute errors remain larger.
6. **Does any trained model improve first10x?** No; all equal one.
7. **CNN or neural operators for one-step?** CNN is best and robustly exceeds
   the specified FNO and spherical-DISCO comparisons.
8. **CNN or neural operators for transport?** Split. FNO leads shell, CNN leads
   radial absolute error, spherical DISCO leads radial skill; Persistence beats
   all.
9. **CNN or neural operators for rollout?** CNN is least bad after the common
   first-step physical failure; no trainable architecture is stable.
10. **Does cost change the conclusion?** It strengthens the practical case for
    FNO/CNN and weakens a claim for spherical DISCO, but cannot repair their
    transport/rollout failures.

## 14. Spherical-geometry interpretation

`SPHERICAL_GEOMETRY_CONCLUSION = PARTIAL_TRANSPORT_GAIN`.

Spherical geometry improves Stage AD's aggregate median shell and radial
transport and selected inner/middle regions, but worsens all aggregate
one-step metrics, does not improve first10x, remains below Persistence, and is
roughly 10x slower for single-sample inference. This is evidence about the
tested anisotropic tangent/volume proxy only, not exact DISCO or covariant
Kerr--Schild geometry.

## 15. Paper Table 2 contextual comparison

The paper's public Table 2 values are copied to `paper_context/` solely as
architecture-level context. They use different data, coordinates, thermal
semantics, preprocessing provenance, operator implementation, and coupling.
Current values are not converted into paper percentages and are not claimed to
numerically reproduce Table 2.

## 16. Reproduction ledger

The final verified/adapted/blocked classification is maintained in
`docs/reproduction_ledger.md`. The 64³ workflow, temporal contract, P3 residual
task, operator implementations, and controlled comparisons are verified
project results. Dataset/remap/operator choices are adaptations. Official data,
official code, exact volumetric DISCO, exact Cartesian-KS processing, and exact
coarse/fine coupling remain unavailable.

## 17. Limitations

Physical inverse-P3 errors and range ratios are enormous for several models
despite moderate normalized errors. These catastrophic tails may be amplified
by the preprocessing inverse and cannot be equated directly with normalized
model failure; nonetheless, they invalidate claims of physical closed-loop
stability. The dataset is nearest-cell regridded, nonconservative, and not
divergence preserving. Pressure, spherical coordinates, P3, shell channels,
and all project-local DISCO variants are adapted choices.

## 18. Final conclusions

`PRIMARY_DECISION = B — MIXED_ARCHITECTURE_RESULT`.

- `ONE_STEP_ACCURACY_WINNER = 3D CNN/U-Net`
- `TRANSPORT_BEST_MODEL = SPLIT: FNO shell; CNN radial absolute; spherical DISCO radial skill`
- `ROLLOUT_BEST_TRAINED_MODEL = 3D CNN/U-Net` (relative only; first10x is 1)
- `COMPUTE_EFFICIENT_MODEL = FNO`
- `ANY_TRAINED_MODEL_FIRST10X_GT_1 = false`
- `EXACT_REPRODUCTION_BLOCKED = true`
- `AUTHORIZE_NEXT_STAGE = REPRODUCTION_CLOSEOUT`

No automatic new architecture, geometry, radius, basis, or objective search is
authorized by this closeout.
