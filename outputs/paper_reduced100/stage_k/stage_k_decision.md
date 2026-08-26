# Stage K decision

## C. TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE

Classification: `adapted_method_reproduction` (3D differential LocalNO; DISCO integral disabled; not exact paper reproduction).

- LocalNO/persistence one-step average: `0.624225/0.189088` (ratio `3.30123`)
- Training microbatches/updates: `2370/600`
- GT/no-GT finite and positive: `True/True`
- Transform counters exact: `True`
- Any frozen Rout exceedance: `False`
- Persistent selected-step collapse flags: `True`

Training, checkpoint, validation, and transform-count engineering gates passed, and decoded ranges stayed finite and positive. The frozen artifact detector nevertheless marked field collapse at every selected no-GT step 25/50/75/100, so the explicit Stage K C rule takes precedence.
Bcc3 and vel3 also carry a known preprocessing-oracle floor; that caveat is reported but does not change the frozen Stage K decision rule.
