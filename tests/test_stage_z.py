import numpy as np
import pytest

from grmhd.stage_z import (
    REPRODUCTION_SCOPE,
    fixed_shell_indices,
    fixed_shell_tensor,
    information_gain,
    spherical_volume_proxy_weights,
    weighted_relative_l2,
)


def test_stage_z_scope_is_explicitly_adapted():
    assert REPRODUCTION_SCOPE == "ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION"


def test_information_gain_is_predeclared_two_of_four():
    baseline = {
        "temporal_increment_relative_difference": 0.50,
        "temporal_increment_cosine": 0.87,
        "shell_increment_relative_l2": 0.20,
        "radial_increment_relative_l2": 0.32,
    }
    candidate = {
        "temporal_increment_relative_difference": 0.39,
        "temporal_increment_cosine": 0.90,
        "shell_increment_relative_l2": 0.18,
        "radial_increment_relative_l2": 0.30,
    }
    result = information_gain(baseline, candidate)
    assert result["condition_count"] == 2
    assert result["higher_res_data_information_gain"] is True


def test_information_gain_rejects_invalid_error():
    with pytest.raises(ValueError):
        information_gain(
            {"temporal_increment_relative_difference": 0.0, "temporal_increment_cosine": 0,
             "shell_increment_relative_l2": 1, "radial_increment_relative_l2": 1},
            {"temporal_increment_relative_difference": 0.0, "temporal_increment_cosine": 1,
             "shell_increment_relative_l2": 0, "radial_increment_relative_l2": 0},
        )


def test_fixed_shells_use_physical_boundaries_not_equal_index_counts():
    r64 = np.geomspace(1.1, 200, 65)
    centres64 = np.sqrt(r64[:-1] * r64[1:])
    edges = np.geomspace(centres64[0], centres64[-1], 9)
    r96 = np.geomspace(1.1, 200, 97)
    centres96 = np.sqrt(r96[:-1] * r96[1:])
    indices = fixed_shell_indices(centres96, edges)
    tensor = fixed_shell_tensor(centres96, 3, 2, edges)
    assert tensor.shape == (8, 3, 2, 96)
    assert np.all(tensor.sum(axis=0) == 1)
    assert np.array_equal(indices, tensor[:, 0, 0].argmax(axis=0))
    assert len(np.unique(np.bincount(indices))) > 1


def test_spherical_volume_proxy_and_weighted_l2():
    r_edges = np.geomspace(1.1, 2.0, 5)
    r = np.sqrt(r_edges[:-1] * r_edges[1:])
    theta = np.linspace(0, np.pi, 5)[:-1] + np.pi / 8
    phi = np.linspace(0, 2 * np.pi, 5)[:-1] + np.pi / 4
    weights = spherical_volume_proxy_weights(
        r, theta, phi, r_bounds=(1.1, 2.0), theta_bounds=(0, np.pi),
        phi_bounds=(0, 2 * np.pi),
    )
    assert weights.shape == (4, 4, 4)
    assert np.all(weights > 0)
    reference = np.ones((2, 4, 4, 4))
    value = 2 * reference
    assert weighted_relative_l2(value, reference, weights) == pytest.approx(1.0)
