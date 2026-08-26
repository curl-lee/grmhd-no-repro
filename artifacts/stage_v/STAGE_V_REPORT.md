# Stage V — Objective Alignment and Temporal Distribution-Shift Audit

| variant | state L2 | residual L2 | cosine | shell skill | radial skill | first 10x step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage-T Plain | 0.266711 | 0.891851 | 0.740602 | -1.67947 | -0.348089 | 1 |
| Direction | 0.276819 | 0.959922 | 0.736947 | -3.309 | -0.628465 | 1 |
| Transport | 0.263992 | 0.892073 | 0.730396 | -2.7226 | -0.819424 | 1 |
| Direction+Transport | 0.272443 | 0.918276 | 0.733153 | -3.43598 | -0.897406 | 1 |
| Paper Full | not run | not run | not run | not run | not run | not run |

`Paper Full` was not authorized because the recoverable old objective contains the known index-grid H1 pathology.

## Frozen controls and training

All learned variants use the Stage-T differential LocalNO (358,296 parameters), P3 residual contract, shared tensor initialization, identical 168-pair order for 150 epochs, Adam, unchanged clip=1, and the first 6,300 updates of the frozen 1,200-epoch warmup/cosine schedule. Validation was not used for weights, training, or stopping. The formal selector remained validation normalized per-channel state relative-L2 arithmetic average.

- Shared initial tensor-state SHA256: `7e10895cb283be8532e4c036e8a526c536a632cce880c500080ea1d14fd89311`.
- Shared 150-epoch pair-order SHA256: `e8f20d8ff8ea67160d9655f6bc0e9fc41eedf2f3b2737e48075ea5cdbae94bb7`; every variant records `stage_t_first_150_order_match=true`.
- V1 retains an earlier checksum for an unused shell-definition field; its active Plain+direction graph is unaffected. The pre-validation correction and V2 restart are documented in `transport_objective_engineering_audit.md`.

- Stage-T Plain: 6300 updates, runtime 1675.6s, mean clip fraction 0.9963, peak allocated 562.8 MiB.
- Direction: 6300 updates, runtime 1704.5s, mean clip fraction 0.9978, peak allocated 562.8 MiB.
- Transport: 6300 updates, runtime 1946.7s, mean clip fraction 0.9984, peak allocated 562.8 MiB.
- Direction+Transport: 6300 updates, runtime 2048.0s, mean clip fraction 0.9984, peak allocated 562.8 MiB.

## No-training alignment evidence

Validation Spearman correlations of Plain pair loss were state=0.938, residual=0.701, cosine-error=0.533, shell-error=0.647, and radial-error=0.469. Thus the strict pairwise weak-transport rule is `false`.
The train-only initialization audit classified aggregate objective-gradient conflict as `WEAK`. The auto-scaled lambdas are direction-only 0.148958616, transport-only 2.7283075e-05, and combined direction/transport 0.074479308/1.36415375e-05.

## Required scientific answers

1. **Plain L2 versus shell/radial statistics.** The strict misalignment flag is `false`; the actual correlations above quantify partial rather than absent alignment.
2. **Gradient conflict.** `WEAK`; the most conflicting focus channels, lowest cosine first, are Bcc3, Bcc2, press, rho, vel3.
3. **Direction objective.** No dynamics improvement: Direction changes state L2 by +3.79% and residual L2 by +7.63% versus Stage T, changes cosine from 0.740602 to 0.736947, makes both transport skills more negative, and leaves first-10x at step 1.
4. **Shell skill.** No candidate turns shell skill positive. Transport=-2.7226, Direction+Transport=-3.43598; relative to Stage T the changes are Transport=-1.04313 and Direction+Transport=-1.75651 skill points.
5. **Radial skill.** No candidate turns radial skill positive. Transport=-0.819424, Direction+Transport=-0.897406; relative to Stage T the changes are Transport=-0.471335 and Direction+Transport=-0.549317 skill points.
6. **State retention.** O1 results are Direction=True, Transport=True, Direction+Transport=True; the fixed ceiling is 1.05*0.266711=0.280047.
7. **Step-1 physical-range failure.** First 10x steps are Stage-T Plain=1, Direction=1, Transport=1, Direction+Transport=1.
8. **Physical-tail separation.** Yes: normalized state errors remain O(0.26--0.28) while decoded rho/press amplification ratios span orders of magnitude, and normalized shell/radial skills are independently negative. The two failures coexist but are not the same metric; no clamp or P3 change was made.
9. **Error/OOD coupling.** Not strong: `MODERATE` under train-only feature fitting. For Stage T the state-error Spearman coefficient is 0.680; this is association, not causal proof.
10. **Late degradation versus shift.** Stage-T late/early state error is 1.546 and is temporally synchronized with the train-derived OOD score (`true`), but MODERATE aggregate coupling is insufficient to claim that shift is the main or causal explanation.
11. **Next change.** Spectral geometry: `non_euclidean_spectral_operator_design` follows because decision `C` rejects objective mismatch as the supported bottleneck and shift coupling is not STRONG.

## Channel attribution

Lowest per-channel Plain/transport correlation among the focus channels: vel3, Bcc3, press, rho, Bcc2. Strongest transport-gradient conflict: Bcc3, Bcc2, press, rho, vel3. Largest summed shell+radial skill improvements occur first in: press, vel3, Bcc3, Bcc2, rho. The full per-channel state/residual/cosine/shell/radial/physical table is `comparison/per_channel_comparison.csv`.

## Distribution shift

The train-only 168-feature snapshot score gives `SHIFT_FINDING=LOW` and `DISTRIBUTION_SHIFT_ERROR_COUPLING=MODERATE`. It does not alter the chronological split and does not establish causality.

## Limitations

This remains a 64^3 resampled spherical Kerr--Schild, `press`-adapted, P3 residual, reduced LocalNO reproduction. It is not the paper's exact 3D DISCO/coarse-coupled operator. Physical rho/press metrics remain exposed to nonlinear inverse tails; normalized dynamics and decoded physical amplification are reported separately.

PRIMARY_DECISION = C
OBJECTIVE_GRADIENT_CONFLICT = WEAK
PLAIN_L2_TRANSPORT_MISALIGNMENT = false
SHIFT_FINDING = LOW
DISTRIBUTION_SHIFT_ERROR_COUPLING = MODERATE
AUTHORIZE_NEXT_STAGE = non_euclidean_spectral_operator_design
