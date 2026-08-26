# Stage L FNO versus LocalNO comparison

All values below are medians across GT steps 1, 3, 5, 10, and 19, measured
against the same canonical preprocessing oracle. They are post-hoc metrics from
frozen best checkpoints and saved trajectories.

| model | channel | global variance retention | total variation retention | combined demeaned high-k retention |
| --- | --- | ---: | ---: | ---: |
| persistence | Bcc3 | 0.790490 | 2.449 | 1.347 |
| FNO Plain | Bcc3 | 0.542834 | 2.622 | 0.608 |
| FNO Full | Bcc3 | 0.559038 | 2.618 | 0.584 |
| LocalNO Plain | Bcc3 | 0.238543 | 3.255 | 0.930 |
| persistence | vel3 | 0.923865 | 1.504 | 1.222 |
| FNO Plain | vel3 | 0.381182 | 1.829 | 0.650 |
| FNO Full | vel3 | 0.669925 | 1.815 | 0.736 |
| LocalNO Plain | vel3 | 0.201820 | 2.180 | 1.568 |

LocalNO has the strongest additional global-variance and shell/radial
compression. It does not have severe combined demeaned high-k loss: Bcc3 retains
about 0.93 and vel3 about 1.57 at the median. FNO instead preserves more global
variance while losing more combined high-k energy. LocalNO therefore cannot be
described as a uniform low-pass smoother.

Axis-specific spectra show redistribution. For LocalNO Bcc3, median demeaned
high-k retention is very large on phi, roughly 0.86 on theta, and 1.40 on r;
for vel3 it is about 0.12 on phi, 9.43 on theta, and 1.03 on r. Ratios with
near-zero oracle energy are undefined or numerically sensitive, so absolute
energies in the JSON/CSV remain primary evidence. The combined spectrum and
Parseval checks are stable; maximum relative Parseval error is
`6.08325654478406e-16`.

Frozen no-GT detector behavior also differs. Bcc3 is flagged at every selected
step for persistence and both FNOs as well as LocalNO, reflecting the already
collapsed canonical initial state. vel3 is not flagged for persistence or either
FNO; LocalNO is flagged only at steps 75 and 100. LocalNO's no-GT state-to-state
changes remain large, so “constant fixed point” is not supported. A
low-variance, anisotropic autoregressive regime is a better, explicitly
post-hoc description.
