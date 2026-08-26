# Stage P transform and feedback trace

Stage P is a read-only, post-hoc audit of the frozen Stage O best checkpoint
(epoch 23). It performs no training, parameter backward, optimizer or scheduler
construction, normalizer fitting, or checkpoint write. The replayed physical
states at the frozen selected steps are bitwise identical to the Stage O states.

## Closed-loop contract

For P3-normalized state `z_t`, fixed spherical radial-shell channels `s`, frozen
LocalNO `F`, P3 decoder `D`, and P3 encoder `E`, the deployed diagnostic path is

```text
y_(t+1) = F(concat(z_t, s))
x_(t+1) = D(y_(t+1))
z_(t+1) = E(x_(t+1))
G(z_t)  = E(D(F(concat(z_t, s))))
```

The trace records `y`, `x`, the re-encoded feedback, their discrepancy, and the
next model input separately. It never assumes `E(D(y)) == y`.

## Train-only envelope

The diagnostic envelope uses only snapshots 11--90 and the frozen P3 statistics;
validation snapshots are not used to fit any bound. Key target-channel ranges are:

| channel | normalized train min/max | normalized q0.001/q0.999 | physical train min/max | decoder derivative q0.999 |
| --- | ---: | ---: | ---: | ---: |
| Bcc2 | `-17.322 / 17.686` | `-15.447 / 15.100` | `-0.1127 / 0.0948` | `1.947e-2` |
| Bcc3 | `-196.369 / 196.368` | `-157.112 / 157.247` | `-2.0717 / 2.0717` | `2.462e-2` |
| vel3 | `-5.001 / 234.891` | `-1.443 / 191.916` | `-0.0269 / 1.4723` | `6.250e-3` |

P3 assigns `no_softclip` to Bcc2, Bcc3, and vel3. Their inverse-clamp occupancy
is therefore exactly zero; the large Bcc2/Bcc3 sensitivity is signed-log inverse
tail amplification, not a hidden `0.99 gamma` clamp. The separate rho/press
evaluation bounds remain active and are reported per channel.

## First failure

All three target channels leave the train q0.001--q0.999 envelope and train
min/max at step 1. Their normalized OOD and the global Rout condition are
coincident at step 1, not sequential. Bcc2 and Bcc3 decoder sensitivity exceeds
the train q0.999 derivative at step 1; vel3 never does because its inverse is
linear. Target-channel inverse clamp never occurs. An any-channel clamp occurs at
step 1 through bounded control channels, coincident with target sensitivity.

Bcc2 Gate 2 fails at step 1 and Bcc3 Gate 2 at step 2. Gate 3 fails for Bcc2,
Bcc3, and vel3 at step 1. Bcc2 and Bcc3 tie for earliest target failure; vel3 is
also OOD at step 1, but driver/passenger attribution requires the reset and
projection evidence rather than the timeline alone.

At step 19 the physical-loop normalized Bcc2/Bcc3 ranges are approximately
`[-344,329]` and `[-4330,3468]`. Their decoded values reach about `5.39e36` and
the float32 extrema respectively, while vel3 remains linear in decode. The
failure is therefore already present at the first prediction and then grows
recursively.

Machine-readable sources are
`outputs/paper_reduced100/stage_p/train_envelope.json`,
`trajectory_trace.json`, and `first_failure_timeline.json`.
