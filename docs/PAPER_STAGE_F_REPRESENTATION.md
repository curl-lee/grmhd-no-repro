# Stage F representation decision

## Runtime semantics

The selected `appendix_literal_press_proxy` radial artifact is only the Full
loss envelope reference.  The existing canonical paper preprocessor directly
encodes each eight-channel physical state; neither it nor the Stage D radial
code subtracts a baseline.  Stage F therefore does not introduce baseline
subtraction or add-back.

Both paired experiments use exactly the same tensors:

```text
normalized_input  = canonical_encode(physical_input)
normalized_target = canonical_encode(physical_target)
model_input        = concat(normalized_input, eight spherical-r shell channels)
model_output       = normalized_prediction with eight state channels
```

Shell channels are inputs only.  They never enter the target, model output, or
loss.  `radial_baseline_normalized` is carried separately in
`PaperLossContext`; only `PaperCompositeLoss` reads it for the rho/press
envelope.  `PlainL2Loss` receives the unchanged normalized target and does not
read the baseline or any training prior.

## Object and transform boundaries

- raw physical target, canonical normalized target, and lossy canonical oracle
  physical target have distinct keys;
- input and target are each encoded once per batch;
- canonical ROI is computed from the canonical oracle physical target, never
  from the model prediction;
- the raw-physical ROI is a separate diagnostic and is not placed in the
  training `PaperLossContext`;
- training postprocessing leaves the prediction normalized;
- rho/press evaluation clamping and physical decode occur only in the explicit
  evaluation decode method;
- all normalizer, shell, radial, and bound objects are loaded from frozen
  train-only artifacts after checksum validation; validation data performs no
  fit.

This is a spherical Kerr--Schild adaptation.  Stored vector components remain
marked non-Cartesian, and the velocity ROI remains a stored-component speed
proxy rather than a metric-correct relativistic velocity norm.
