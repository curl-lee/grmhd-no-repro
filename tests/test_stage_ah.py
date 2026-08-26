import numpy as np
import pytest

from grmhd.stage_ah import (
    RADIAL_REGION_BOUNDS,
    THETA_REGION_BOUNDS,
    field_metrics,
    paired_bootstrap,
    primary_gates,
    scientific_decision,
    validate_region_bounds,
)


def test_regions_exhaust_axes_without_overlap():
    validate_region_bounds(THETA_REGION_BOUNDS, size=64)
    validate_region_bounds(RADIAL_REGION_BOUNDS, size=64)
    with pytest.raises(ValueError):
        validate_region_bounds({"a": (0, 2), "b": (3, 4)}, size=4)


def test_field_metrics_exact_prediction():
    target = np.arange(8, dtype=np.float64).reshape(2, 2, 2) + 1.0
    residual = target / 10.0
    result = field_metrics(target, target, residual, residual)
    assert result == {
        "state_l2": 0.0,
        "residual_l2": 0.0,
        "residual_cosine": pytest.approx(1.0),
        "absolute_residual_error": 0.0,
    }


def test_paired_bootstrap_is_reproducible_and_direction_aware():
    rows = [
        {"stage_ag_state_l2": 1.0, "stage_ad_state_l2": 2.0,
         "stage_ag_residual_cosine": 0.8, "stage_ad_residual_cosine": 0.5},
        {"stage_ag_state_l2": 2.0, "stage_ad_state_l2": 3.0,
         "stage_ag_residual_cosine": 0.7, "stage_ad_residual_cosine": 0.6},
    ]
    left = paired_bootstrap(rows, ("state_l2", "residual_cosine"), samples=100)
    right = paired_bootstrap(rows, ("state_l2", "residual_cosine"), samples=100)
    assert left == right
    assert left[0]["win_fraction"] == 1.0
    assert left[1]["win_fraction"] == 1.0


def test_primary_gates_and_decision_b():
    ad = {"state_l2": 1.0, "residual_l2": 1.0, "cosine": 0.5,
          "shell_skill": -0.5, "radial_skill": -0.5}
    ag = {"state_l2": 1.01, "residual_l2": 1.01, "cosine": 0.49,
          "shell_skill": -0.2, "radial_skill": -0.5}
    gates = primary_gates(
        ad, ag,
        stage_ad_shell_absolute_error=2.0,
        stage_ag_shell_absolute_error=1.0,
        stage_ad_radial_absolute_error=2.0,
        stage_ag_radial_absolute_error=2.0,
        first_10x_step=1,
    )
    assert gates["SHELL_GEOMETRY_GAIN"]
    assert scientific_decision(gates, ad, ag, integrity_valid=True) == "B"
    assert scientific_decision(gates, ad, ag, integrity_valid=False) == "F"
