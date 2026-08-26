# Stage M OracleConditionedStructureGate proposal

## Status and calibration

`OracleConditionedStructureGate` is a candidate for future runs. This Stage M
replay is explicitly `counterfactual_gate_replay_only`; it does not replace the
legacy detector or reclassify Stage G/K/L.

Thresholds were frozen before model replay:

- severe retention: strict `< 0.5`, inherited from Stage L;
- Gate 2 repetition: at least two selected GT steps;
- transport skill boundary: zero versus persistence;
- transport sign agreement: at least 0.5.

No Stage K flag, epoch, validation score, or FNO/LocalNO ranking selected a
threshold. No data-derived validation quantile is used. The declared calibration
split remains train snapshots 11--90; the calibration artifact explicitly records
that the current proposal needs no additional quantile threshold.

## Four levels

### Gate 0 — engineering validity

Requires finite tensors, positive rho/press, valid transform counters, checkpoint
provenance, decoded ranges, Rout, and shape/device checks. It reuses existing
Stage K engineering evidence.

### Gate 1 — preprocessing floor qualification

A channel is `floor_limited_channel` if its canonical oracle triggers the legacy
detector or median raw→oracle variance, shell/radial, or high-k retention is below
0.5. Bcc2, Bcc3, and vel3 are all floor-limited. A raw-reference detector flag is
therefore not sufficient model-failure evidence for them.

### Gate 2 — model-added degradation

The reference is the target canonical oracle. Failure requires at least two of
global variance, shell/radial variance, dynamic span, and high-k to be severe,
including global or shell/radial, on at least two selected GT steps.

### Gate 3 — oracle-conditioned transport skill

Shell and radial transport errors must both be no worse than persistence, at least
one skill must be strictly positive, and sign agreement must be at least 0.5.
Absolute state retention alone cannot pass this gate.

## Replay behavior

- Canonical oracle is correctly identified as a floor-limited reference, not a
  model failure.
- Persistence is retained as the zero-transport comparator and does not gain a
  false positive-skill pass.
- FNO and LocalNO produce distinct Gate 2 failure patterns but all fail the joint
  transport gate on the three primary channels.
- No historical result is overwritten.

The replay contains no logical contradiction under these rules. The decision is
`I. CANDIDATE_GATE_READY_FOR_FUTURE_RUNS`.
