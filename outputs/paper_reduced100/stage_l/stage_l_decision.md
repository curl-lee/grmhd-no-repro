# Stage L decision

## Channel decisions

- Bcc3: `C. MIXED_PREPROCESSING_AND_MODEL`
- vel3: `C. MIXED_PREPROCESSING_AND_MODEL`

## Overall decision

`3. MIXED_OVERALL`

Both primary channels suffer a large canonical preprocessing floor before model
inference, and LocalNO then adds material global-variance and shell/radial
degradation. Combined demeaned high-k energy is not severely lost by LocalNO;
the model instead redistributes spectral content anisotropically while entering
a lower-variance regime. Hence neither preprocessing-only nor model-only
attribution is consistent with the frozen rules.

This result does not alter Stage K's decision:
`C. TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE`. It does not authorize further
training, threshold changes, preprocessing refits, checkpoint changes, or a new
rollout. The scientifically appropriate next action is to separate and repair
the canonical transform floor before using the current Bcc3/vel3 detector as a
model-quality gate, and independently audit LocalNO's shell/radial variance
compression. Any such work requires a new authorization.
