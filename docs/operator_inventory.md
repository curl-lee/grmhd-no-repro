# Operator and model inventory through Stage AH

| Operator/model | Stage(s) | Geometry / basis | Implementation | Status |
|---|---|---|---|---|
| Persistence | F--AH comparisons | n/a | `src/grmhd/models.py` | baseline |
| Canonical preprocessing oracle | F--AH | spherical grid, channel transforms | `src/grmhd/paper_preprocessing.py` | tested; selected channels have a material oracle floor |
| FNO proxy | F--J | regular tensor Fourier modes | pinned upstream via `src/grmhd/models.py` | tested/trained; not paper 3-D DISCO |
| Differential LocalNO | K--V | tensor-index finite difference plus spectral branch | pinned upstream `LocalNO`, project wrappers | tested/trained; long rollout unstable |
| P3 differential LocalNO | N--T | differential LocalNO with isolated P3 transform | `src/grmhd/paper_stage_n.py`, `paper_stage_o.py`, `paper_stage_r.py` | adapted pilots completed |
| Spherical finite-difference variants | U | coordinate and spherical-distance proxies | `src/grmhd/stage_u_geometry.py`, `stage_u_training.py` | diagnostic variants completed |
| Mixed-basis LocalNO | W | phi Fourier, theta cosine, log-r cosine | `src/grmhd/stage_w_mixed_basis.py`, `stage_w_training.py` | tested/trained; not supported by outcome |
| Adapted volumetric DISCO3D | AC--AD | index-space isotropic 7x7x7, five radial hats | `src/grmhd/operators/disco3d.py`, `src/grmhd/localno3d_disco.py` | AC contract invalid; AD three-cell repair tested/trained |
| Isotropic spherical DISCO3D | AE | Euclidean spherical-coordinate proxy with scalar local scale | `src/grmhd/operators/spherical_disco3d.py` | implementation invalid at four theta/r corners; not trained |
| Anisotropic spherical DISCO3D | AF--AH | target-local spherical tangent frame and directional scales | `src/grmhd/operators/anisotropic_spherical_disco3d.py`, `src/grmhd/anisotropic_spherical_disco_localno.py` | AF implementation gates pass; AG controlled training complete; AH decision B (partial improvement) |

## Supporting code

- Dataset and coordinate contracts: `src/grmhd/dataset.py`,
  `src/grmhd/stage_s_data.py`, `src/build_regrid_from_athdf.py`.
- Shell channels and preprocessing: `src/grmhd/shells.py`,
  `paper_preprocessing.py`, `paper_data_processor.py`, `stage_z.py`.
- Losses and train-only priors: `paper_losses.py`, `paper_bounds.py`,
  `paper_radial.py`, `paper_velocity_roi.py`, `paper_dissipation.py`.
- Training/checkpoints: `paper_trainer.py`, `paper_checkpoint.py`,
  `stage_s_training.py`, `stage_t_training.py`, `stage_w_training.py`.
- Evaluation/rollout: `paper_metrics.py`, `paper_stage_g_evaluation.py`,
  `paper_stage_l_attribution.py`, `stage_v_analysis.py`, `stage_ah.py`.

## Test coverage

The published tests include dense-reference comparisons, float64 gradcheck,
constant-field normalization, boundary semantics, phi equivariance, CPU/CUDA
agreement, geometry normalization, basis activity, and kernel-orientation
checks. The most direct files are `tests/test_disco3d.py`,
`tests/test_spherical_disco3d.py`, and
`tests/test_anisotropic_spherical_disco3d.py`, supplemented by the Stage
S--AC and paper-workflow suites.

The exact volumetric 3-D DISCO operator from the paper was not found in the
pinned upstream source or recovered paper assets. All project-local DISCO3D
implementations are explicitly labeled adaptations.
