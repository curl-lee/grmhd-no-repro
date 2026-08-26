# Stage I final decision

## D — No extension rescues Full

All three diagnostic extensions passed their engineering gates. Run C was
finite, strictly reloadable, positive through GT step 19 and no-GT step 100,
and improved five frozen morphology/boundary categories relative to Stage G
Full. Run A and Run B each also improved five categories relative to Full.

The decisive controlled comparison is against Plain. Lower is better:

| category | Full | Plain | no-H1 | unit-index | stored-coordinate |
|---|---:|---:|---:|---:|---:|
| center morphology | 0.91196 | 0.90498 | 0.93864 | 0.93875 | 0.91375 |
| polar morphology | 0.88026 | 0.83574 | 0.85606 | 0.85996 | 0.83685 |
| magnetic texture | 2.41274 | 0.89983 | 0.85202 | 0.86006 | 0.97595 |
| radial statistics | 6.56373 | 4.84509 | 5.06848 | 5.26359 | 4.96409 |
| outer-shell statistics | 7.18638 | 1.20600 | 1.85820 | 1.71580 | 2.12554 |
| model-only saturation | 0.01853 | 0.01071 | 0.01772 | 0.01845 | 0.01802 |

No-H1 and unit-index each beat Plain in one category; stored-coordinate beats
Plain in none. Thus adapting or removing H1 clearly rescues several failures
of Stage G Full, but does not produce the paper-relevant multi-category
advantage over Plain required to proceed with paper ablations. The remaining
gap is not attributable only to H1.

## Required questions

1. **Full vs no-H1.** Observed: no-H1 improves five categories and all four
   selected GT errors relative to Full. Inference: removing upstream H1
   relieves gradient compression. Unsupported: that H1 removal is the correct
   paper objective.
2. **Full vs unit-index.** Observed: the raw 4096 amplification disappears;
   mean H1/base gradient ratio falls from 7.85 to 0.00475, and five categories
   improve. Inference: spacing amplification is a material Stage G adaptation
   mismatch. Unsupported: unit-index is a physical or covariant H1.
3. **no-H1 vs unit-index.** Observed: unit-index has slightly better one-step
   and selected GT errors but only one better morphology category. Inference:
   a weak H1 has no broad, repeatable morphology advantage over removal.
   Unsupported: either result predicts 1200-epoch behavior.
4. **unit-index vs stored-coordinate.** Observed: stored improves center,
   polar, radial, saturation, and GT steps 1/5/10; unit-index retains better
   magnetic texture and outer-shell score. Inference: real coordinates and
   the volume proxy add a reproducible but mixed effect. Unsupported: this is
   Kerr--Schild proper volume or covariant GRMHD H1.
5. **stored-coordinate vs Plain.** Observed: stored has lower one-step
   average/global and fewer step-100 artifacts, but no frozen aggregate
   morphology/boundary category is better. Inference: retained Full priors
   plus adapted H1 do not establish the required advantage. Unsupported: that
   Plain is scientifically superior outside this reduced proxy.

## Reproduction consequence

Stage G Full with current upstream H1 and weight 0.05 remains the frozen
paper-adapted main path. Runs A/B/C remain diagnostic extensions and do not
replace it. Do not begin paper loss ablations from an extension. The next
reproduction work should improve operator/geometry/data fidelity—especially
the blocked 3D DISCO LocalNO, Cartesian Kerr--Schild components, verified
`eint`, 300-snapshot protocol, and coarse/fine coupling—before interpreting
more loss ablations. No Stage J or additional training was started.
