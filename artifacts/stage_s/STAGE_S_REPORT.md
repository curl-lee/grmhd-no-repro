# Stage S — Expanded-Data LocalNO Reproduction and Data-Scarcity Audit

## 1. Expanded data summary

Total unique snapshots: 212
Train snapshots: 169 (0..168)
Train pairs: 168
Validation snapshots: 43 (169..211)
Validation pairs: 42
Time range: 0.0 .. 2110.000838137111
Median dt: 10.000659066036974
Grid: static Kerr-Schild AMR source; processed tensor `(212,8,64,64,64)` in `(N,C,phi,theta,r)` order
Channels: Bcc1, Bcc2, Bcc3, rho, press, vel1, vel2, vel3

The 101 new active files are a continuous extension of the 111 old files. All 212 have the same schema/grid signature, strictly increasing time, finite fields, and positive rho/press. Recycle-bin copies were hash-audited and excluded. The processed old prefix is bitwise identical to the frozen 111-snapshot artifact. `DISTRIBUTION_SHIFT=STRONG`; no difficult samples were removed.

## 2. Model

3D differential LocalNO, P3 preprocessing refit on snapshots 0..168 only, normalized-residual target, 358,296 trainable parameters. Inputs are 8 P3 state channels plus 8 frozen spherical-r shell channels; output is 8 residual channels. The model, Plain L2 loss, Adam settings, clip=1, seed=42, and initial tensor hash are frozen from Stage R. This is an adapted method reproduction, not exact volumetric 3D DISCO.

## 3. Core comparison table

| model | train pairs | updates | avg norm rel-L2 | persistence ratio | residual rel-L2 | shell skill | radial skill | first unstable step |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Persistence | - | - | 0.310282 | 1 | 1 | 0 | 0 | stable baseline |
| S-small matched | 79 | 600 | 0.333462 | 1.0747 | 1.1668 | -0.00620851 | -0.038014 | 1 |
| S-full matched | 168 | 600 | 0.343186 | 1.10604 | 1.22404 | -0.0290467 | -0.0584656 | 1 |
| S-full 30epoch | 168 | 1260 | 0.297125 | 0.957594 | 1.15347 | -3.53789 | -0.374952 | 1 |

All learned rows use their best checkpoint under the same 42-pair held-out normalized-average selection metric. Matched runs both use exactly 2,400 microbatches and 600 optimizer updates. S-full 30epoch uses 5,040 microbatches and 1,260 updates.

The matched scheduler has exactly 600 update calls for both populations and a 40-update warmup (two Stage-R 20-update epochs). The natural run has an 84-update warmup (two complete 42-update full-data epochs). Physical-space average relative L2 is `0.571584` for persistence, `6.41753e21` for S-small, `15.0064` for S-full matched, and `1.14409e22` for S-full 30epoch. These extreme learned values are dominated by rho/press inverse-tail excursions; q001/q999 and saturation evidence is preserved in each `one_step_metrics.json`.

## 4. Direct answers

**Q1. Are the 101 new snapshots compatible?** Yes operationally: exact channel/schema/grid/layout agreement, continuous indices/times, and a bitwise-identical processed old prefix. Component basis and physical units are not explicit in ATHDF metadata; compatibility is supported by the continuous `mad98.prim` series and identical stored spherical-KS contract, not by undocumented unit metadata.

**Q2. What is the actual total?** 212 unique active snapshots: 111 old + 101 new. The chronological split is 169 train snapshots/168 pairs, one dropped boundary pair 168→169, and 43 held-out validation snapshots/42 pairs.

**Q3. Does more data reduce one-step error at matched updates?** No. S-small is `0.333462` and S-full is `0.343186`; `Improvement_data=-0.029161` (-2.916%). Full is worse, not better.

**Q4. Does it improve residual direction?** No at matched updates. Global residual cosine changes from `0.319241` to `0.0773009`. The 30-epoch full run improves it to `0.741839`, which is extra-optimization evidence.

**Q5. Does it improve shell/radial transport?** No. Matched-full shell/radial skills are `-0.0290467`/`-0.0584656` versus small `-0.00620851`/`-0.038014`. All are below persistence (skill <= 0); the 30-epoch run is more negative.

**Q6. Is the Stage-R step~3 physical-range explosion delayed?** No. Under the explicit GT-aware Stage-S rule, all three Stage-S rollouts first exceed the 10× physical range criterion at step 1. S-full 30epoch normalized rollout error grows from `0.301995` at step 1 to `10.1603` at step 5 and `160.716` at step 19.

**Q7. Does any model beat persistence?** Only S-full 30epoch in normalized one-step average: `0.297125` versus persistence `0.310282` (ratio `0.957594`). It does not beat persistence in physical-space average or shell/radial transport, and its rollout rapidly diverges.

**Q8. Is improvement from more data or more optimizer updates?** Evidence supports more optimization, not more data. Matched-full is 2.92% worse than matched-small, while 30epoch-full is `13.422%` better than matched-full after 1,260 rather than 600 updates.

**Q9. Is geometry/operator mismatch still the likely bottleneck?** Data scarcity is not supported as the primary explanation. The remaining evidence is consistent with operator/geometry mismatch plus P3 inverse-tail amplification: normalized one-step improves with updates, but physical errors are tail-dominated, transport skill remains negative, and closed-loop residual direction reverses after early steps. This audit does not isolate those mechanisms causally.

## 5. Engineering and provenance

- Both 2-epoch CUDA smokes passed; training/backward gradients were finite and non-zero.
- All formal runs began from tensor SHA256 `7e10895cb283be8532e4c036e8a526c536a632cce880c500080ea1d14fd89311`.
- Best and last checkpoints strictly reload for all runs.
- RTX 5070 peak allocated memory was 562.816 MiB; no AMP was used.
- All 100-step rollouts remained numerically finite with positive decoded rho/press, but this is not scientific stability: decoded ranges and normalized norms grow severely.
- Stage R numbers in `stage_r_vs_stage_s.csv` are contextual only because its validation time block and P3 fit population differ.

## 6. Decision

**D. MORE_OPTIMIZATION_NOT_MORE_DATA**

The matched-update experiment does not show a benefit from the larger pair population. The natural 30-epoch full-data run improves normalized one-step error and residual direction only after receiving more optimizer updates, even slightly beating P3 persistence in normalized one-step average, while physical-space tails, shell/radial transport, and autoregressive stability remain poor. Therefore the evidence attributes the limited improvement to optimization exposure rather than data volume.
