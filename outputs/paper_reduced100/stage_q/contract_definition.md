# Stage Q persistence-anchored output contract

`y_direct = F_theta(concat(z_t, fixed_shells))`

`r_theta = y_direct - z_t`

`y_alpha = z_t + alpha * r_theta = (1-alpha)z_t + alpha*y_direct`

The scalar grid is `0/0.125/0.25/0.5/1`. Endpoints are controls; only intermediate values may become future-smoke candidates. The contract contains no decode, encode, clamp, clipping, repair, or shell mutation.
