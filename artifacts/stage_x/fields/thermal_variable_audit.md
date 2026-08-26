# Thermal-variable audit

The current ATHDF `VariableNames` contains `press` and no `eint`; its attributes have no Gamma, EOS, or adiabatic-index field. The raw directory has no Athena input/problem-generator file. The project does not perform a conversion.

The paper's GRMHD Appendix A.2 says outputs include pressure P, while network Appendix C.2 says the fifth tensor channel is internal energy `eint`. The paper does not provide the GRMHD EOS relation used to resolve that mismatch. Gamma=5/3 appears in Appendix A.1 for the separate Newtonian MHD setup and cannot be transferred to the GRMHD data.

`PRESS_TO_EINT_CONVERSION_AUTHORIZED = false`

`THERMAL_VARIABLE_GAP = HIGH`
