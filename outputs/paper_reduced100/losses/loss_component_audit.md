# paper_reduced100 Stage E loss-component audit

No model, Trainer, optimizer, or checkpoint was used. Each row below is the mean
over the fixed batches for that split/case.

| split | case | total | base | H1 | ROI | bounds | envelope | dissipation | finite |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| train | deterministic_random | 3061.4558 | 39.471782 | 3017.6643 | 4.2404496 | 0.00031599824 | 0.079074567 | 0 | True |
| train | isolated_Bcc1 | 6.6541187 | 0.40000308 | 6.2015624 | 0 | 1.5261445e-08 | 0.044737234 | 0.0078161322 | True |
| train | isolated_Bcc3 | 6.6543193 | 0.40000308 | 6.2015624 | 0 | 1.5261445e-08 | 0.044737234 | 0.0080167549 | True |
| train | isolated_press | 6.5952028 | 0.33333588 | 6.2015624 | 0 | 0.00023073389 | 0.052214021 | 0.0078597721 | True |
| train | isolated_rho | 6.6019703 | 0.33333588 | 6.2015624 | 0 | 0.00025195953 | 0.058961106 | 0.0078587653 | True |
| train | isolated_vel1 | 7.0577453 | 0.33333588 | 6.2015624 | 0.47024819 | 1.5261445e-08 | 0.044737234 | 0.0078616031 | True |
| train | isolated_vel3 | 7.0577344 | 0.33333588 | 6.2015624 | 0.47024819 | 1.5261445e-08 | 0.044737234 | 0.0078506777 | True |
| train | normalized_target | 0.050585729 | 0 | 0 | 0 | 1.5261445e-08 | 0.044737234 | 0.0058484804 | True |
| train | persistence_normalized_input | 137.95861 | 2.6307148 | 134.33312 | 0.9490677 | 1.5304139e-08 | 0.045707164 | 0 | True |
| train | zero_tensor | 595.17027 | 30.869346 | 560.27827 | 4 | 0 | 0.022633323 | 0 | True |
| validation | deterministic_random | 3965.2595 | 73.333878 | 3883.5775 | 8.2688527 | 0.00031730162 | 0.079082213 | 0 | True |
| validation | isolated_Bcc1 | 6.6511474 | 0.40000308 | 6.2015619 | 0 | 0 | 0.029073202 | 0.0205094 | True |
| validation | isolated_Bcc3 | 6.6521699 | 0.40000308 | 6.2015626 | 0 | 0 | 0.029073202 | 0.021530946 | True |
| validation | isolated_press | 6.5903568 | 0.33333588 | 6.2015624 | 0 | 5.8113283e-06 | 0.034651447 | 0.020801758 | True |
| validation | isolated_rho | 6.5940971 | 0.33333588 | 6.2015624 | 0 | 0.00015262342 | 0.038286848 | 0.0207594 | True |
| validation | isolated_vel1 | 7.2902668 | 0.33333588 | 6.2015624 | 0.70552373 | 0 | 0.029073202 | 0.020771363 | True |
| validation | isolated_vel3 | 7.2902861 | 0.33333588 | 6.2015624 | 0.70552373 | 0 | 0.029073202 | 0.020790895 | True |
| validation | normalized_target | 0.044268882 | 0 | 0 | 0 | 0 | 0.029073202 | 0.015195679 | True |
| validation | persistence_normalized_input | 1035.7893 | 10.650178 | 1022.2975 | 2.8103044 | 0 | 0.031356963 | 0 | True |
| validation | zero_tensor | 1501.8418 | 64.79409 | 1429.0251 | 8 | 0 | 0.022633323 | 0 | True |

For `normalized_target`, base, H1, and ROI are exactly zero. Bounds, envelope,
and dissipation are independent priors and are not required to vanish.
