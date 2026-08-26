# Stage I Run A: matched no-H1 diagnostic

## Classification

Run A is a `diagnostic_extension`, not paper-faithful Full and not Plain L2.
It removes only the selected H1 training contribution. Magnetic/base channel
weights, velocity ROI, lower bounds, radial residual envelope, dissipative
gate, shells, radial representation, preprocessing, optimizer, scheduler,
initial state, pair order, and evaluation remain matched to Stage G Full.

The run manifest locks project commit
`df2891bb4085d44eee710b4761cd8c40bb5f1385`, upstream commit
`86a8bc7812a31b42c4f7895693cf4ac11521c066`, config SHA256
`2e8e2ebb611842214c60fe207222ffb401f25b229172c76beab68e205e2278c6`,
shared tensor-state SHA256
`02d83278d5e74a27c83130e99b488ca18eae5477dad9732a4552b694056b6cb1`,
and pair-order SHA256
`5e1e361ea75e01ffd667822d8bfe870d70c5bc69570d52cdf806e637b2df2b52`.
The Stage G checkpoint and Stage H frozen-input hashes were revalidated before
training.

## GPU preflight

The repeated no-H1-only real `64^3` preflight passed on the RTX 5070 with
PyTorch 2.12.1+cu130. Input, model, output, and loss were on CUDA; forward,
loss, and parameter gradients were finite. Selected H1 value, prediction
gradient, and parameter gradient were exactly zero. Every non-H1 component had
exact Stage G Full parity, and backward did not change the model state.
Peak allocated/reserved memory was 770,999,296/977,272,832 bytes. No optimizer,
scheduler step, checkpoint, or training state was created by preflight.

## Training result

The authorized run completed exactly 30 epochs, 2,370 microbatches, and 600
Adam updates. Each epoch consumed all 79 train pairs in the frozen order,
performed 20 optimizer steps, and normalized the last step over its actual
three microbatches. There were no NaN/Inf values. Best and last checkpoint
epochs were 25 and 30; both strictly reloaded model, optimizer, scheduler,
epoch, pair order, initial state, extension identity, provenance, and the
deterministic validation probe.

Every update still clipped at norm 1.0, so removing H1 did not reduce clipping
*fraction*. It materially changed clipping scale: the mean scale increased
from Stage G Full's 0.01079 to 0.14355, and the mean effective base-gradient
norm increased from 0.16747 to 0.98877. The mean no-H1 effective other-prior
gradient norm was 0.11482. Mean parameter-update norm remained matched:
0.20806 versus 0.20707 for Stage G Full.

Training wall time was 335.50 s; peak allocated/reserved GPU memory was
795.98/966.00 MiB. Parameter displacement from the shared initial state was
111.00 at best epoch and 111.34 at last epoch.

## Best validation

The checkpoint selector remained the normalized per-channel relative-L2
arithmetic average over all 19 validation pairs.

| channel | normalized relative L2 |
|---|---:|
| Bcc1 | 0.495621 |
| Bcc2 | 0.445423 |
| Bcc3 | 0.276243 |
| rho | 0.637070 |
| press | 0.638744 |
| vel1 | 0.621347 |
| vel2 | 0.521612 |
| vel3 | 0.226533 |
| arithmetic average | 0.482824 |
| global | 0.386497 |

Oracle-aware arithmetic-average/global values were 0.69961/0.97280 for
model-to-oracle, 0.87944/0.97981 for model-to-raw, and 0.41729/0.52636 for the
preprocessing oracle floor. The exact squared-error numerator decomposition
was saved with the evaluation.

Validation stayed finite and rho/press remained positive. Mean evaluation
bound-clamp fraction was 0.02909; mean prediction norm was 4018.96; 63.16% of
validation predictions were above `Rin` and none above `Rout`. Mean envelope
violation was 0.76577/0.21344 for rho/press, dissipation gate mean was 0.15789,
and target-clamp overlap with canonical ROI was 0.46126.

## Controlled interpretation

No-H1 did not beat Stage G Full on teacher-forced one-step average
(0.48282 versus 0.46278), and it was 1.7% above Plain (0.47458). It improved
all selected GT rollout errors relative to Full and improved six frozen
morphology/boundary categories: polar morphology, magnetic texture, radial
statistics, outer-shell statistics, model-only saturation, and step-50/100
artifact count. Magnetic texture was also slightly better than Plain; Plain
remained best in the other aggregate morphology/boundary categories.

Other Full priors did not dominate the no-H1 objective on average: weighted
base was 6.08384 while ROI, bounds, envelope, and dissipation summed to
0.08230. Their respective means were 0.04767, `5.57e-8`, 0.03455, and
`9.21e-5`. The continued 100% clipping is therefore a warning, but the logged
values do not support stopping Run A for a newly dominant prior stack.

## Partial decision

**A. no-H1 run passed and supports continuing.**

This is only the Run A decision. It does not decide whether unit-index H1 is
sufficient, stored-coordinate H1 rescues the adaptation, or removing H1 is the
best final objective. Neither remaining variant was trained.
