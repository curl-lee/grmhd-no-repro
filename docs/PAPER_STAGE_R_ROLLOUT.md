# Stage R residual closed-loop evaluation

The best epoch-9 checkpoint was evaluated on all 19 validation pairs and then
rolled from snapshot 91 for 100 physical steps. Each step performs raw residual
prediction, identity reconstruction, exactly one P3 decode, physical feedback,
exactly one P3 encode, and shell append. There is no teacher forcing, alpha,
residual/output clamp, GT correction, direct normalized feedback, or double
transform. Counts are exactly input/target/oracle/prediction
`100/19/19/100`. All 100 states are finite and rho/press positive.

| GT step | normalized avg | residual cosine | normalized state norm | max physical magnitude |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.221565 | 0.758703 | 41,913.9 | 1.980e3 |
| 3 | 0.731428 | -0.310092 | 44,547.8 | 5.353e6 |
| 5 | 0.947915 | -0.337201 | 47,487.1 | 2.090e22 |
| 10 | 1.463119 | -0.432930 | 56,454.3 | 2.090e22 |
| 19 | 2.320825 | -0.602293 | 82,405.5 | 2.090e22 |

Step 1 shows real normalized improvement and a correct average residual
direction. The direction reverses by step 3 and remains negative through step
19. Physical range explosion therefore begins by step 3 even though normalized
growth is much smaller than frozen Stage O.

| no-GT step | residual norm | normalized state norm | max physical magnitude | legacy flags |
| ---: | ---: | ---: | ---: | ---: |
| 25 | 9,523.1 | 114,212.7 | 2.090e22 | 8 |
| 50 | 75,059.6 | 724,577.5 | 3.403e38 | 8 |
| 75 | 146,700.2 | 1,604,894.5 | 3.403e38 | 8 |
| 100 | 168,415.7 | 1,878,179.5 | 3.403e38 | 8 |

No GT error or future target oracle is constructed after step 19. The no-GT
records contain state/residual change, ranges, train-envelope/Rout diagnostics,
radial/shell statistics, total variation, stored-index frequency energy,
temporal autocorrelation/PSD, and legacy artifacts only. By step 100 all eight
channels carry high-frequency ripple flags. Finite float32-extreme values are
not interpreted as stable physical behavior.
