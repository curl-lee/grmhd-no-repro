# Stage M candidate gate replay

This is `counterfactual_gate_replay_only`; no Stage G/K/L result is reclassified.

- persistence / Bcc2: `transport_gate_not_passed_without_two_step_state_failure`
- persistence / Bcc3: `transport_gate_not_passed_without_two_step_state_failure`
- persistence / vel3: `transport_gate_not_passed_without_two_step_state_failure`
- fno_plain / Bcc2: `transport_gate_not_passed_without_two_step_state_failure`
- fno_plain / Bcc3: `transport_gate_not_passed_without_two_step_state_failure`
- fno_plain / vel3: `model_added_degradation_and_transport_failure`
- fno_full / Bcc2: `model_added_degradation_and_transport_failure`
- fno_full / Bcc3: `model_added_degradation_and_transport_failure`
- fno_full / vel3: `transport_gate_not_passed_without_two_step_state_failure`
- localno_plain / Bcc2: `model_added_degradation_and_transport_failure`
- localno_plain / Bcc3: `model_added_degradation_and_transport_failure`
- localno_plain / vel3: `model_added_degradation_and_transport_failure`
- canonical_oracle / Bcc2: `floor_limited_reference_not_model_failure`
- canonical_oracle / Bcc3: `floor_limited_reference_not_model_failure`
- canonical_oracle / vel3: `floor_limited_reference_not_model_failure`
