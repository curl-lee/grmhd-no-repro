# Stage K Physical Rollout and Morphology

## Autoregressive contract

The best epoch-22 LocalNO checkpoint starts from real snapshot 91. Every step
performs normalized prediction, exactly one physical decode, exactly one
canonical encode of that physical state for the next input, and reattaches the
same fixed shells. There is no teacher forcing, normalized-tensor feedback,
residual shortcut, output repair, or ground-truth correction.

The 100-step counter delta is exactly 100 for input encode, target encode,
oracle decode, and prediction decode. Only input-encode and prediction-decode
form the autoregressive state path; target/oracle transforms are evaluation
bookkeeping and do not enter the prediction.

## Nineteen-step ground-truth rollout

| step | normalized average | persistence | ratio | global | model-to-oracle | model-to-raw | oracle floor | flags |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.42039 | 0.22947 | 1.8320 | 0.35855 | 0.67981 | 0.85866 | 0.40034 | 4 |
| 3 | 0.77483 | 0.64509 | 1.2011 | 0.82225 | 0.90649 | 0.93107 | 0.38306 | 8 |
| 5 | 0.84462 | 0.71321 | 1.1842 | 0.88286 | 0.94396 | 0.95177 | 0.39247 | 9 |
| 10 | 0.96869 | 0.74165 | 1.3061 | 0.97082 | 0.98736 | 0.98396 | 0.39025 | 11 |
| 19 | 1.03662 | 0.79046 | 1.3114 | 1.01047 | 0.99332 | 0.98687 | 0.39064 | 9 |

All 19 steps are finite and rho/press positive, but LocalNO is worse than the
same frozen-state persistence rollout at every selected GT step. The frozen
artifact detector reports increasing collapse/stripe flags; step 19 includes
collapse flags for Bcc1, Bcc3, rho, press, and vel3 plus stripe flags. These
are detector outcomes, not a metric-correct claim about physical GRMHD modes.

## One-hundred-step no-GT rollout

No ground-truth or oracle error is reported after step 19.

| step | prediction norm | rho range | press range | clamp | frozen flags |
|---:|---:|---|---|---:|---|
| 25 | 2788.56 | 1.84e-5..1.1490 | 1.27e-6..0.01125 | 0 | Bcc1 ripple; Bcc3 collapse |
| 50 | 2756.39 | 7.35e-6..1.2093 | 5.67e-7..0.01112 | 4.67e-4 | Bcc3 collapse |
| 75 | 2578.36 | 6.06e-6..1.2093 | 6.06e-7..0.01474 | 5.76e-4 | Bcc3, vel3 collapse |
| 100 | 2685.57 | 8.51e-6..1.2093 | 1.06e-6..0.01893 | 8.20e-5 | Bcc3, vel3 collapse |

The entire rollout is finite, rho/press positive, and below the frozen `Rout`;
there is no decoded-range explosion. This does not establish physical
correctness. The frozen detector marks Bcc3 field collapse at all selected
no-GT steps and vel3 collapse at steps 75 and 100. Because persistence of a
collapse flag is an explicit Stage K C condition, it controls the final
decision even though numerical range and positivity gates pass. Bcc3/vel3 are
also affected by the canonical preprocessing floor; the caveat is retained
but the threshold is not changed for LocalNO.

## Frozen morphology and boundary comparison

Scores reuse the Stage G functions and thresholds. Lower is better. GT
categories average steps 1, 5, 10, and 19; the artifact count sums steps 50
and 100.

| category | oracle | persistence | FNO Plain | FNO Full | LocalNO Plain |
|---|---:|---:|---:|---:|---:|
| center morphology | 0 | 0.82165 | 0.90498 | 0.91196 | 0.89646 |
| polar morphology | 0 | 0.59403 | 0.83574 | 0.88026 | 0.95309 |
| magnetic texture | 0 | 0.57954 | 0.89983 | 2.41274 | 4.27951 |
| radial statistics | 0 | 1.00086 | 4.84509 | 6.56373 | 2.96560 |
| outer-shell statistics | 0 | 0.22357 | 1.20600 | 7.18638 | 3.53896 |
| model-only saturation | 0 | 0.01523 | 0.01071 | 0.01853 | 0.00646 |
| step-50/100 flags | n/a | 0 | 3 | 5 | 3 |

LocalNO has lower center, radial, and model-only-saturation scores than both
FNO proxies, but it is worse than persistence in all five truth-referenced
morphology/boundary categories and has the worst magnetic-texture score.

Twelve untracked figures cover steps 1/5/10/19/50/100 in fixed-phi theta-r and
equatorial phi-r views. They are explicitly titled `spherical-coordinate
adapted views`; they are not Cartesian central slices. Raw truth and canonical
oracle appear only where GT exists, and the captions preserve the
preprocessing-floor/clamp caveat.
