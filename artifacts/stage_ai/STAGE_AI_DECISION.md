# Stage AI Decision

```text
PRIMARY_DECISION = B
PRIMARY_DECISION_NAME = MIXED_ARCHITECTURE_RESULT
REPRODUCTION_SCOPE = ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION
CANONICAL_RESOLUTION = 64^3
CANONICAL_TRAIN_PAIRS = 168
CANONICAL_VALIDATION_PAIRS = 42
CANONICAL_TARGET = NORMALIZED_RESIDUAL
CANONICAL_LOSS = PLAIN_L2
PERSISTENCE_INCLUDED = true
FNO_INCLUDED = true
CNN_INCLUDED = true
DIFFERENTIAL_LOCALNO_INCLUDED = true
INDEX_DISCO3D_INCLUDED = true
SPHERICAL_DISCO3D_INCLUDED = true
ALL_CHECKPOINTS_VALID = true
SPHERICAL_GEOMETRY_CONCLUSION = PARTIAL_TRANSPORT_GAIN
ANY_TRAINED_MODEL_BEATS_PERSISTENCE_STATE = true
ANY_TRAINED_MODEL_BEATS_PERSISTENCE_SHELL = false
ANY_TRAINED_MODEL_BEATS_PERSISTENCE_RADIAL = false
ANY_TRAINED_MODEL_FIRST10X_GT_1 = false
ONE_STEP_ACCURACY_WINNER = 3D CNN/U-Net
TRANSPORT_BEST_MODEL = SPLIT: FNO shell; CNN radial absolute; spherical DISCO radial skill
ROLLOUT_BEST_TRAINED_MODEL = 3D CNN/U-Net
COMPUTE_EFFICIENT_MODEL = FNO
EXACT_REPRODUCTION_BLOCKED = true
AUTHORIZE_NEXT_STAGE = REPRODUCTION_CLOSEOUT
NO_AUTOMATIC_NEW_MODEL_VARIANT = true
```

## Basis for decision B

The CNN is the robust one-step winner, trained transport leadership is split
across FNO/CNN/spherical DISCO, and Persistence remains better than every
trained model on both aggregate transport skills. All five trained models reach
the 10x physical-range landmark at step 1. Consequently no operator family has
a robust advantage in more than one successful scientific dimension, while
the architectures do show materially different one-step, transport, cost, and
relative rollout behavior.

The spherical variant improves Stage AD's aggregate median transport summaries
but worsens one-step accuracy and does not rescue rollout. Its geometry result
therefore remains `PARTIAL_TRANSPORT_GAIN`, not a general spherical-operator
win.

`TRANSPORT_BEST_MODEL` is deliberately split rather than computed from an
arbitrary weighted score. `ROLLOUT_BEST_TRAINED_MODEL` is a relative tie-break
among models whose primary first10x gate all failed at step 1; it is not a
stability claim.
