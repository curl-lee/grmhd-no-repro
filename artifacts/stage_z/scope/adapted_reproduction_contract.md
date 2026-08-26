# Stage Z adapted reproduction contract

`REPRODUCTION_SCOPE = ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION`

This experiment is an adapted spherical-KS workflow reproduction. It is not an exact
paper reproduction, a paper-faithful LocalNO, or an exact volumetric 3D DISCO
reproduction. The sole scientific variable is target spherical regrid resolution.

- Raw snapshots: 212 (`00000..00211`), frozen.
- Split: train snapshots `0..168` / 168 pairs; drop `168->169`; validation
  snapshots `169..211` / 42 pairs.
- Fields: `['Bcc1', 'Bcc2', 'Bcc3', 'rho', 'press', 'vel1', 'vel2', 'vel3']` with no Cartesian vector conversion, press-to-eint
  conversion, EOS inference, or spin inference.
- Regrid: finest-covering leaf, nearest cell centre, non-conservative, no
  interpolation, not divergence preserving.
- Datasets: Z64 `3582a5c4f52f4fef305578ffd5a63c51423f8583f35d137c288d5c20203b50da`, Z96 `292d3fe62297e91fda8e0f81ab156ca7ceb127ea02a004ca79180705e7411564`,
  Z128 `fe62e9c2d8e0311808a1be597002b8813ee6eab6ad42a1f882c8fcce37ae9b1d`.
- Sampling reference: Z128 is `HIGHER_RES_SAMPLING_REFERENCE`, not raw truth.
- Frozen gates: `Z64_REGRID_REGRESSION_PASS = true` and
  `HIGHER_RES_DATA_INFORMATION_GAIN = true`.
- Exact reproduction remains blocked by Stage Y.
