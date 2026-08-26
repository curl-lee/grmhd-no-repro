# Stage L collapse attribution

The frozen severe-loss criterion is retention `< 0.5`; a core evidence class is
severe only when at least three of the five selected GT steps are severe.

| channel | preprocessing global | preprocessing shell/radial | preprocessing high-k | model global | model shell/radial | model high-k | decision |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | --- |
| Bcc3 | yes | yes | yes | yes | yes | no | `C. MIXED_PREPROCESSING_AND_MODEL` |
| vel3 | yes | yes | yes | yes | yes | no | `C. MIXED_PREPROCESSING_AND_MODEL` |

For Bcc3, preprocessing is already severe in all three evidence classes and the
same-snapshot canonical oracle triggers the unchanged detector on steps 1--19.
LocalNO adds severe global-variance loss on steps 3/5/10/19 and severe shell or
radial-profile loss on the same steps. It does not add severe combined high-k
loss. The strongest preprocessing loss is concentrated in inner/middle shells;
LocalNO adds compression mainly across shells 1--6 while the outermost shell can
retain or amplify variance.

For vel3, preprocessing is also severe in all three core classes even though its
std remains above the detector's `< 0.05` threshold. LocalNO adds severe global
and shell/radial loss on steps 3/5/10/19 but not severe combined high-k loss.
Some per-shell ratios have tiny oracle denominators and are therefore null or
large; the classification excludes undefined values and uses the frozen radial
and median-defined-shell rule.

There are 136 undefined ratios across all detailed products. Each is explicitly
stored as JSON `null` with an undefined flag and excluded from medians/evidence;
the required global, shell/radial, and combined spectral evidence remains defined
at enough steps for both channel classifications. No engineering failure is
present.
