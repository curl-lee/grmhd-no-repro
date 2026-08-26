# Stage O legacy detector and stage_m_v1 comparison

Stage O reports the unchanged collapse/ripple/stripe detector beside frozen
`stage_m_v1`. Gate thresholds and definitions were not retuned after seeing
Stage O.

| Channel | Gate 0 | Gate 1 floor-limited | Gate 2 degradation | Gate 3 transport | shell/radial skill |
| --- | --- | --- | --- | --- | --- |
| Bcc2 | fail: range/Rout | no | fail at 1/3/5 | fail | `-0.833 / -1.216` |
| Bcc3 | fail: range/Rout | no | fail at 3/5 | fail | `-1.212 / -108.193` |
| vel3 | fail: range/Rout | no | pass | fail | `-12.182 / -5.410` |

Finite state, rho/press positivity, shape/device, checkpoint provenance, and
exact transform counters pass Gate 0. Its overall result fails because decoded
range and frozen Rout fail. Gate 1 correctly uses the P3 raw-to-oracle floor:
P3 preserves essentially all variance and high-k energy for the three target
channels, so their current failures cannot be dismissed as the old canonical
floor. Gate 2 finds repeated model-added degradation for Bcc2 and Bcc3. Gate 3
finds negative persistence-relative shell and radial skill for all three.

## Frozen core questions

| Question | Answer | Evidence class |
| --- | --- | --- |
| Did P3 lower the Bcc2/Bcc3/vel3 floor? | Yes, by several million-fold relative to the canonical target-channel floors. | observed |
| Did P3 LocalNO reduce model-added variance degradation? | Not stably: Bcc2/Bcc3 fail Gate 2; vel3 alone passes. | observed |
| Did it improve shell/radial transport? | No; all target channels fail Gate 3 with negative skills. | observed |
| Did it reduce legacy collapse flags? | Yes, targeted collapse flags disappear, but severe ripple/stripe and range explosion replace them. | observed |
| Does `stage_m_v1` differ from legacy? | Yes. Legacy collapse alone looks better; Gate 0/2/3 exposes worse engineering range and transport behavior. | observed |
| Does it beat P3 persistence? | No; one-step normalized average is `5.80871x` persistence. | observed |
| Is improvement from transform or model? | Floor recovery is transform-side; stable skill did not follow and model behavior dominates failure. | inference |
| Does mixed operator-response failure remain? | The Gate 3 and feedback evidence supports that interpretation. | inference |

The pilot does not prove a unique causal defect inside LocalNO, nor does it
prove that P3 caused the operator failure. Those causal claims remain
unsupported. Stage G FNO Plain is retained only as a background structural
reference, and cross-transform normalized L2 is never treated as a common
ranking scale.

Machine-readable Gate 0--3 details and frozen morphology comparisons are in
`outputs/paper_reduced100/stage_o/stage_m_v1_gate.json` and
`morphology_comparison.json`.
