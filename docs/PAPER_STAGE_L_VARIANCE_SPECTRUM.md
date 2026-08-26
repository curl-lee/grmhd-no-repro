# Stage L variance, shell, radial, and spectrum attribution

## Frozen metric contract

Every field is one channel over `(phi, theta, r)`; batch and channel axes are never
FFT or shell axes. Variance uses `ddof=0`. Eight radial shells reuse the frozen
physical-r edges. Radial profiles average over phi/theta. The spectrum uses no
window, orthonormal FFT normalization, both original and mean-subtracted variants,
and mutually exclusive stored-index frequency bands:

- low: `k <= 0.125`;
- mid: `0.125 < k <= 0.25`;
- high: `k > 0.25`.

For the combined spectrum, `k` is the maximum absolute cycles-per-index value over
the three axes. This is a reproducible array diagnostic, not a covariant physical
wavenumber. Maximum observed Parseval relative error is
`6.08325654478406e-16`.

Retention is always decomposed as raw to oracle and oracle to model. A denominator
at or below `1e-30` makes the ratio undefined; the absolute numerator/denominator
are retained, JSON uses `null`, and the ratio is excluded from aggregation. There
are 136 such ratios, without loss of the required core evidence.

## Where preprocessing removes structure

For Bcc3, preprocessing shell-variance median retentions at the selected GT steps
are 0.231, 0.214, 0.179, 0.0368, and 0.0376. The median shell pattern is extremely
small in shells 1--5 and close to one in shells 7--8. Combined demeaned high-k and
radial-profile variance are also nearly eliminated.

For vel3, preprocessing removes most variance in inner shells 1--4 while preserving
the outer shells. Its median global variance retention is 0.0248, combined high-k
retention 0.00567, and radial-profile variance retention 0.0199. Thus its lack of a
same-snapshot detector flag is threshold-specific, not evidence of transform
fidelity.

## What LocalNO adds

Relative to the canonical oracle, LocalNO median global variance retention is
0.2385 for Bcc3 and 0.2018 for vel3. Both channels are below 0.5 on steps
3/5/10/19. Shell/radial evidence is also severe on those four steps. Bcc3 shows
additional compression mainly in shells 1--6, while the outermost shell may be
amplified. Vel3 has severe radial-profile loss and later middle/outer-shell
compression; ratios for oracle-near-zero inner shells are not interpreted as
healthy preservation.

Combined demeaned high-k retention is different: LocalNO medians are 0.930 for
Bcc3 and 1.568 for vel3, with no selected step below 0.5. Per-axis results are
strongly anisotropic and some ratios are large because an oracle axis-band energy
is tiny. The supported mechanism is therefore global/radial variance compression
with retained or redistributed high-frequency texture, not uniform low-pass
smoothing.

The complete values are in `gt_variance_retention`, `gt_shell_retention`,
`gt_radial_retention`, `gt_spectral_retention`, and
`shell_retention_heatmap_data` under `outputs/paper_reduced100/stage_l/`.
