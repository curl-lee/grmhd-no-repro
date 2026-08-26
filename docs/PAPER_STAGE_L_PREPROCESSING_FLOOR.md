# Stage L canonical preprocessing floor

## Question and method

This audit asks how much spatial structure is lost before a learned model acts.
For each validation target snapshot 92--110, the raw physical tensor is passed
through the frozen canonical encode/decode transform exactly once. All comparisons
are raw target versus canonical oracle at the same snapshot. There is no model in
this experiment.

The analysis reports population variance, standard deviation, robust span,
total variation, first-difference energy, center/polar/outer-region variance,
eight physical-r shells, radial profiles, sign/correlation statistics, and
orthonormal FFT energy. Spectrum results are diagnostics in stored
`(phi, theta, r)` index space; they are not coordinate-invariant Kerr--Schild
physical spectra. The frozen severe threshold is retention strictly below 0.5.

## Results

Median retention over all 19 validation targets is:

| channel | std | variance | q99-q01 span | TV | combined demeaned high-k | radial-profile variance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bcc1 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| Bcc2 | 0.135732 | 0.018423 | 0.115107 | 0.128949 | 0.002928 | 0.009218 |
| Bcc3 | 0.001540 | 0.00000237 | 0.000520 | 0.000840 | 0.000000518 | 0.00000120 |
| rho | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| press | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| vel1 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| vel2 | 0.772609 | 0.596924 | 0.747190 | 0.776624 | 0.311851 | 0.490209 |
| vel3 | 0.157497 | 0.024806 | 0.084011 | 0.103711 | 0.005669 | 0.019876 |

Bcc3 and vel3 meet severe loss in all three predeclared core evidence classes at
all selected steps 1/3/5/10/19. Bcc3's canonical oracle also triggers the frozen
collapse detector at every GT step 1--19. Vel3 does not trigger because its std
retention, while strongly degraded, remains above the detector's much lower 0.05
threshold. This distinction is why detector flags and structural attribution must
not be treated as equivalent.

The control channels are informative. Bcc2 also has strong transform sensitivity,
and vel2 loses moderate high-k/radial variance; Bcc1, rho, press, and vel1 are
nearly unchanged. The effect is therefore channel-selective rather than a universal
tensor scaling error.

## Interpretation boundary

The canonical oracle is not raw truth. Any model evaluated after this transform
inherits a substantial Bcc3/vel3 floor. This does not absolve the model: it only
establishes that raw-to-model degradation cannot be assigned wholly to the
operator. Detailed 19-step and clamp/sign evidence is in
`outputs/paper_reduced100/stage_l/preprocessing_floor.{json,csv,md}`.
