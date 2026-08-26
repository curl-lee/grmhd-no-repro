# paper_reduced100 radial candidate comparison

- Selected mode: `appendix_literal_press_proxy`
- Selection reason: literal candidate is finite; lower adapted residual is not a selection criterion.
- Selection used train diagnostics only; validation is report-only.
- A smaller adapted residual is not sufficient to replace the literal candidate.

| split | mode | channel | mean | std | q span | reconstruction L2 | envelope violation |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| train | appendix_literal_press_proxy | rho | -1.55895511 | 1.60193835 | 6.33641932 | 3.38481534 | 0.836654663 |
| train | appendix_literal_press_proxy | press | -1.03674307 | 1.03857223 | 4.69583901 | 3.32760888 | 0.437615442 |
| train | spherical_logr_channelwise | rho | -0.00923733697 | 1.05958355 | 4.44494106 | 0.997695594 | 0.199459839 |
| train | spherical_logr_channelwise | press | 0.00273990189 | 0.670415263 | 3.4281162 | 0.823551479 | 0.0527404785 |
| validation | appendix_literal_press_proxy | rho | -0.717616666 | 1.45978343 | 6.3410628 | 0.982327285 | 0.41920681 |
| validation | appendix_literal_press_proxy | press | -0.665566485 | 1.07026225 | 4.77288211 | 0.998352711 | 0.309267235 |
| validation | spherical_logr_channelwise | rho | 0.832101104 | 1.22454926 | 5.44066119 | 0.999950633 | 0.42416687 |
| validation | spherical_logr_channelwise | press | 0.373916489 | 0.772180313 | 3.6385871 | 0.999823198 | 0.0971122742 |
