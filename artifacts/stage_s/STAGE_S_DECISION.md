# Stage S Decision

## D. MORE_OPTIMIZATION_NOT_MORE_DATA

Matched update budget: S-full (`0.343186`) did not improve over S-small (`0.333462`) and had worse residual cosine and transport skill. The S-full 30-epoch run improved normalized one-step error to `0.297125` only with 1,260 updates and beat P3 persistence (`0.310282`) by `4.241%`, but it retained catastrophic physical-tail errors, negative shell/radial skill, and step-1 range failure. This supports more optimization rather than more data; it does not support data scarcity as the primary Stage-R failure mechanism.
