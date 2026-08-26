# Stage Q train-only alpha sweep

Selection used exactly train transitions `11->12` through `89->90`; validation was not read.

| alpha | normalized average | / persistence | q-tail OOD median | Rout failures | Gate 2 count | Gate 3 passes | readiness |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0.0 | 0.1329131 | 1 | 0 | 79 | 4 | 10 | `control` |
| 0.125 | 0.14317961 | 1.0772423 | 0.00073242188 | 79 | 7 | 22 | `False` |
| 0.25 | 0.17233666 | 1.2966116 | 0.001953125 | 79 | 7 | 19 | `False` |
| 0.5 | 0.25164262 | 1.8932868 | 0.0028800964 | 79 | 8 | 17 | `False` |
| 1.0 | 0.44310191 | 3.3337716 | 0.0096626282 | 79 | 11 | 13 | `control` |

Frozen candidate: `None`.
Reason: `no_intermediate_alpha_passed_all_train_only_readiness_groups`.
