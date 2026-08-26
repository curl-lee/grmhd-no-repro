# Stage M oracle-conditioned variance transport

## Reference contract

For each available transition, Stage M computes input and target canonical oracles
from the frozen raw snapshots. Model metrics are read from already saved rollout
statistics and Stage L selected-step metrics; no model rollout is regenerated.
Persistence is the input oracle and therefore has exactly zero predicted
transport.

Shell transport uses the frozen eight physical-r shells. Radial profile transport
uses the existing mean over phi/theta at each of 64 stored r centers. It is a
stored-coordinate structural diagnostic, not physical invariant flux transport.
Persistence transport is available for all 19 transitions; matched model
shell/radial statistics are available at the frozen selected steps 1/3/5/10/19.

## Selected-step transport replay

Persistence-relative skill greater than zero means lower transport error than
persistence. Gate 3 additionally requires both shell and radial errors not to
exceed persistence and minimum sign agreement at least 0.5.

| model/channel | shell skill | radial skill | minimum sign agreement | Gate 3 |
| --- | ---: | ---: | ---: | --- |
| FNO Plain / Bcc2 | -0.923 | -0.386 | 0.475 | fail |
| FNO Plain / Bcc3 | -7.445 | -11.645 | 0.250 | fail |
| FNO Plain / vel3 | -0.269 | -3.506 | 0.191 | fail |
| FNO Full / Bcc2 | -1.701 | -0.228 | 0.453 | fail |
| FNO Full / Bcc3 | -7.897 | -14.496 | 0.350 | fail |
| FNO Full / vel3 | -0.644 | -2.715 | 0.350 | fail |
| LocalNO / Bcc2 | -1.551 | +0.0246 | 0.450 | fail |
| LocalNO / Bcc3 | -12.109 | -5.107 | 0.200 | fail |
| LocalNO / vel3 | +0.1187 | -4.695 | 0.338 | fail |

LocalNO has partial positive skill in Bcc2 radial and vel3 shell transport, but
neither channel satisfies the joint candidate rule. Bcc3 is much worse than
persistence in both transport views.

## State retention versus transport

These are deliberately separate. LocalNO model-to-target-oracle state degradation
meets the two-step Gate 2 rule at:

- Bcc2: steps 10 and 19;
- Bcc3: steps 3, 5, 10, and 19;
- vel3: steps 3, 5, 10, and 19.

Thus the operator both inherits a preprocessing floor and adds incorrect
oracle-conditioned state/transport structure. The supported LocalNO decision is
`4. MIXED_TRANSPORT_AND_PREPROCESSING_FAILURE`.
