# Stage N oracle-conditioned gate integration

Stage N turns the Stage M proposal into the unified read-only function
`evaluate_oracle_conditioned_structure_gate(...)`. Schema version is
`stage_m_v1`. Each report contains the unchanged caller-supplied legacy detector
payload plus:

1. Gate 0 engineering validity;
2. Gate 1 preprocessing-floor qualification;
3. Gate 2 model-added structural degradation;
4. Gate 3 persistence-relative shell/radial transport skill.

The Stage M thresholds remain unchanged. The interface rejects
`training_blocking=true`, defaults to reporting only, deep-copies the legacy
payload, and does not write historical JSON. The Stage N config demonstrates
explicit future enablement while retaining `training_blocking_default: false`.

The frozen-state replay has 64 channel/state rows. Gate 0 passes all 64; Gate 1
qualifies 32 as floor-limited; Gate 2 does not trigger under the strict two-step
rule; and Gate 3 passes none because the joint persistence-relative transport
conditions fail. These are parallel reports, not retrospective Stage K/L/M
reclassifications and not validated physical-stability claims.

Legacy implementation source hash remains covered by its existing regression
test. Candidate schema, immutability, required fields, disabled training block,
and historical decision preservation have dedicated Stage N tests. The decision
is `I. REPORTING_INTERFACE_READY`.
