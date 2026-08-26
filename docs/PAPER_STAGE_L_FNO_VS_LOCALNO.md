# Stage L FNO versus differential LocalNO

## Controlled comparison

This is a post-hoc comparison of existing frozen best-checkpoint trajectories.
It does not retrain any model and does not imply architecture parity: Stage G FNO
and Stage K differential LocalNO differ in backbone and initial tensors, while
sharing the reduced100 split, canonical preprocessing, training order, optimizer
schedule, and evaluation semantics.

Median GT-step model/oracle retentions are:

| model | channel | global variance | total variation | combined demeaned high-k |
| --- | --- | ---: | ---: | ---: |
| Persistence | Bcc3 | 0.7905 | 2.449 | 1.347 |
| FNO Plain | Bcc3 | 0.5428 | 2.622 | 0.608 |
| FNO Full | Bcc3 | 0.5590 | 2.618 | 0.584 |
| LocalNO Plain | Bcc3 | 0.2385 | 3.255 | 0.930 |
| Persistence | vel3 | 0.9239 | 1.504 | 1.222 |
| FNO Plain | vel3 | 0.3812 | 1.829 | 0.650 |
| FNO Full | vel3 | 0.6699 | 1.815 | 0.736 |
| LocalNO Plain | vel3 | 0.2018 | 2.180 | 1.568 |

LocalNO compresses global and shell/radial variance more than either FNO. Both
FNOs lose more combined high-k energy than LocalNO. LocalNO's higher total
variation and retained/amplified high-k energy do not demonstrate physical detail:
correlation with the oracle is low and the spectrum is redistributed unevenly by
axis. These observations separate “amount of variation” from “correct structure.”

## Long-horizon behavior without GT

After step 19, no target or oracle metric is computed. Every row is explicitly
marked `ground_truth_available=false` and `oracle_available=false`. The comparison
uses the raw and canonical snapshot-91 states only as fixed diagnostic references,
plus state changes, historical validation envelopes, detector ratios, shells,
profiles, and spectra.

LocalNO Bcc3 detector ratios at steps 25/50/75/100 are approximately
0.00654/0.00618/0.00614/0.00571; vel3 ratios are
0.0582/0.0667/0.0449/0.0496. Vel3 crosses the detector threshold only at 75 and
100 and is non-monotonic. LocalNO's selected-interval relative state changes remain
large, while combined high-k energy relative to canonical snapshot 91 is preserved
or amplified. The rollout is therefore not a literal constant fixed point. It is
consistent with an unstable low-global-variance regime containing anisotropic
texture, but that phrase is an inference, not a new detector.

Persistence exactly preserves canonical snapshot-91 variance and spectrum after
GT ends, but by construction supplies no dynamics. It is a reference, not a
successful long-horizon surrogate. Detailed comparisons are in
`outputs/paper_reduced100/stage_l/fno_localno_comparison.{json,csv,md}` and the
four `no_gt_*` products.
