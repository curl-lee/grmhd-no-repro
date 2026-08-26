# Stage L collapse-attribution decision

## Frozen-rule result

| channel | preprocessing core classes | LocalNO-added core classes | decision |
| --- | ---: | ---: | --- |
| Bcc3 | 3/3 | 2/3 | `C. MIXED_PREPROCESSING_AND_MODEL` |
| vel3 | 3/3 | 2/3 | `C. MIXED_PREPROCESSING_AND_MODEL` |

The overall result is **`3. MIXED_OVERALL`**.

Preprocessing is severe for both channels in global variance, shell/radial
variance, and combined demeaned high-k energy. LocalNO then adds severe global and
shell/radial variance loss but not severe combined high-k loss. The channel-level
rules therefore select C exactly; A, B, and D do not fit, and provenance/metric
checks provide no basis for E.

## Consequence

The frozen Stage K decision remains
**`C. TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE`**. Stage L explains the observed
flags; it does not retroactively pass Stage K, redefine the detector, or authorize
training. In particular:

1. Bcc3's detector flag is already present in the canonical oracle and canonical
   persistence reference, so it cannot be used as model-only evidence.
2. Vel3's late LocalNO flags include genuine model-added global/radial compression,
   but occur on top of a large preprocessing floor and are non-monotonic.
3. The LocalNO mechanism is not simple spectral smoothing: combined high-k energy
   is retained or amplified while correct large-scale/radial variance is lost.
4. FNO and LocalNO failure modes differ; choosing one solely by the frozen collapse
   flag would obscure that distinction.

The next scientifically defensible work would be separately authorized diagnostic
work: first isolate/repair the canonical transform floor for Bcc2/Bcc3/vel3, then
audit differential LocalNO's shell/radial variance transport using an evaluation
gate that distinguishes oracle floor from model-added degradation. No threshold,
checkpoint, preprocessing artifact, model, or rollout was changed in Stage L.
