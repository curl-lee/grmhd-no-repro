# Stage AD Report

| model | params | local integral | state L2 | residual L2 | cosine | shell skill | radial skill | first10x |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| Persistence | 0 | no | 0.310282 | 1 | 0 | 0 | 0 | stable |
| Stage-T differential LocalNO | 358296 | no | 0.266711 | 0.891851 | 0.740602 | -1.67947 | -0.348089 | 1 |
| Adapted 3D-DISCO LocalNO | 363480 | yes | 0.264062 | 0.89164 | 0.757468 | -4.35448 | -0.390352 | 1 |

## Scope and outcome

Stage AD tested one authorized repair only: the project-local isotropic radial, piecewise-linear adapted DISCO3D cutoff changed from one cell to three cells while K=5, the spherical 7×7×7 support, the Stage-S/T data, preprocessing, common LocalNO initialization, Plain L2 objective, scheduler, resolution, and periodic computational boundary remained frozen. This is an adapted spherical-KS workflow test, not the paper's unavailable exact volumetric DISCO implementation.

The implementation and training are valid, but the scientific outcome is not consistent. Epoch 300 is selected only by the frozen 42-pair arithmetic-average state-L2 selector. Its state L2 is 0.264062018, a 0.993% improvement over Stage T, while the paired bootstrap CI for DISCO−StageT is [-0.00386932, 4.57507e-05] and includes zero. Residual L2 is 0.891639951 versus 0.891851431; cosine improves to 0.757467715. Aggregate shell and radial skills remain negative and change by -2.67501 and -0.0422635, respectively. The 100-step rollout stays finite and positive but FIRST_10X remains 1.

## Implementation evidence

- The Stage-AC R=Δ, K=5 contract is invalid on the point-sampled lattice: positive support counts are `[1, 0, 0, 0, 6]`, so three interior hats are inactive and their quadrature normalizers vanish.
- Integer-cell audits found m=2 still inactive (`[1, 0, 18, 20, 14]`) and m=3 first activates every hat (`[1, 18, 56, 74, 66]`). No radius above three cells was tested.
- R=3Δ=0.09375; the 7³ bounding stencil contains 343 offsets, 123 inside the spherical cutoff and 220 exactly masked out.
- All Zk are finite and positive: `3.0517578e-05, 0.00016395822, 0.00062860208, 0.0015378076, 0.0013927767`. Normalized quadrature-integral maximum error is 3.28e-08.
- Partition-of-unity, compact support, dense explicit reference, cross-correlation orientation, float64 gradcheck, repository tests, and CPU/CUDA agreement all pass. Dense-reference relative L2 is 2.39e-17; CUDA relative L2 is 2.69e-07.
- The four branches add 5,184 parameters: 363,480 total versus 358,296 in Stage T (1.447% increase). Pinned `external/neuraloperator` is not modified.

## Controlled training and computational cost

- Training completed 300 epochs, 50,400 microbatches and 12,600 optimizer updates on `NVIDIA GeForce RTX 5070` with no NaN/Inf.
- The frozen 30→75→150 state-L2 sequence is 0.409307→0.288551→0.265138; 75→150 improves 8.114%, satisfying the predeclared extension rule. Epoch 300 was therefore authorized.
- Full-model CUDA forward/backward preflight times are 0.1318/0.1665 s; the isolated DISCO forward measurement is 0.00847 s. Preflight peak memory is 672.2/972.0 MiB allocated/reserved.
- Actual 300-epoch runtime is 2.465 h with 648.1/908.0 MiB peak memory. The exact epoch-150 cumulative cost is recorded in `comparison/efficiency_comparison.csv`.
- DISCO gradients were nonzero at every optimizer update, the trained DISCO state hash differs from initialization, and all checkpoints strictly reload with exact metric recomputation.

## Per-channel DISCO minus Stage-T deltas

Negative L2 deltas and positive cosine/skill deltas are favorable.

| channel | Δ state L2 | Δ residual L2 | Δ cosine | Δ shell skill | Δ radial skill |
|---|---:|---:|---:|---:|---:|
| Bcc1 | -0.0047682 | -0.0147472 | +0.0277889 | +5.32889 | -27.6864 |
| Bcc2 | +0.0153587 | +0.0273639 | +0.0166436 | -37.5851 | -1.88678 |
| Bcc3 | +0.00543037 | +0.0176339 | +0.0141441 | -4.7586 | +0.040707 |
| rho | +0.00208528 | +0.0134618 | +0.00138362 | +888.174 | +13.421 |
| press | -0.00211883 | -0.0146454 | +0.0532987 | -4.51409e+06 | -724.957 |
| vel1 | -0.000211599 | -0.00219892 | +0.0666301 | +1.20719 | -0.32148 |
| vel2 | -0.0405508 | -0.0538866 | +0.0198055 | +0.791707 | +1.88354 |
| vel3 | +0.0035806 | +0.0253267 | +0.0694997 | +0.227144 | +0.158614 |

The largest state-L2 benefit is vel2 (-0.0405508), followed by Bcc1, press, and vel1. Residual L2 improves most for vel2, Bcc1, press, and vel1. Directional cosine improves for every channel, especially vel3 and vel1. These gains are mixed: Bcc2/Bcc3/rho/vel3 state errors worsen, aggregate transport does not improve, and physical L2 is dominated by the press inverse-preprocessing tail.

## Paired validation, attribution, and rollout

- The 42-pair paired comparison gives state win fraction 0.548; its confidence interval includes no reliable aggregate state advantage. Residual-L2 CI also crosses zero, while cosine improves consistently.
- The formal-best branch is active at every layer: outputs and parameter gradients are nonzero, and trained weights moved from initialization.
- Removing DISCO changes the probe residual by norm 22317.9. Full-minus-DISCO state L2 is 0.502101, showing that the branch materially affects prediction, but ablation does not establish that the effect is beneficial.
- Physical closed-loop rollout starts at snapshot 169, uses no teacher forcing, encodes/decodes once per step, completes 100 finite steps, and preserves positive rho/press. Residual cosine first becomes negative at step 11; shell/radial skill first become negative at steps 1/1; physical range exceeds 10× at step 1.

## Required questions

1. **Why was Stage AC R=Δ invalid?** Three of five point-sampled radial hats had no positive stencil point, making the basis/normalization contract degenerate.
2. **Why is R=3Δ the minimum integer-cell repair?** m=1 and m=2 fail the all-bases-active audit; m=3 is the first passing integer and no larger radius was tried.
3. **Are all five radial bases active?** Yes.
4. **Is every Zk finite and positive?** Yes.
5. **Does partition of unity pass?** Yes.
6. **Does the dense reference match?** Yes, relative L2 2.39e-17.
7. **Is kernel orientation correct?** Yes; implementation and explicit reference use cross-correlation, and the asymmetric test distinguishes a flipped kernel: True.
8. **Does gradcheck pass?** Yes for input, weight, and bias.
9. **Does the DISCO branch have nonzero output and gradient?** Yes, in every layer; gradients were nonzero throughout training.
10. **What is the 7³ GPU cost?** Isolated DISCO forward 0.00847 s in preflight; full forward/backward 0.1318/0.1665 s; 150 epochs took 4435.58 s and 300 took 8874.89 s.
11. **Did controlled training complete 150 epochs?** Yes, and the frozen convergence rule authorized completion through epoch 300.
12. **Is state L2 better/retained versus Stage T?** Retained and nominally 0.993% better, but the paired 95% CI crosses zero.
13. **Do residual L2/cosine improve?** Aggregate residual L2 improves by only 0.000211481; its paired-delta CI crosses zero. Cosine improves by +0.0168656.
14. **Does shell skill improve?** No; -1.67947→-4.35448.
15. **Does radial skill improve?** No; -0.348089→-0.390352.
16. **Is rollout FIRST_10X delayed?** No; it remains step 1.
17. **Which channels benefit most?** vel2 most clearly in state/residual L2; Bcc1, press, and vel1 have smaller L2 gains; vel3/vel1 have the largest cosine gains. Benefits do not align consistently with aggregate transport.
18. **Does trained ablation prove DISCO affects prediction?** Yes, it proves material influence, not scientific benefit.
19. **Does this support the missing local-integral mechanism as the important method gap?** No. The valid tested adaptation is active but does not rescue aggregate transport or rollout.
20. **Proceed to the final adapted baseline benchmark?** Yes, as the frozen next-stage action for a scientifically completed A–E result; include DISCO as a tested adapted baseline, not as a rescued/exact paper method.

## Provenance and limitations

Evidence is in `artifacts/stage_ad/implementation`, `preflight`, `training/disco3d_localno`, `attribution`, and `comparison`. The workflow remains adapted spherical Kerr–Schild data on a regular 64³ computational grid with periodic computational padding and a project-local isotropic radial basis. `EXACT_3D_DISCO_IMPLEMENTATION_FOUND=false` and `EXACT_REPRODUCTION_BLOCKED=true`.
