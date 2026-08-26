from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from grmhd.dataset import sha256_file
from grmhd.paper_checkpoint import (
    build_paper_checkpoint_metadata,
    stage_i_selected_h1_definition,
    validate_paper_checkpoint_metadata,
)
from grmhd.paper_config import load_paper_experiment_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/paper_reduced100/extensions/no_h1_control.yaml"
)
UNIT_INDEX_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/paper_reduced100/extensions/unit_index_h1.yaml"
)
STORED_CONFIG_PATH = (
    PROJECT_ROOT
    / "configs/paper_reduced100/extensions/stored_coordinate_volume_h1.yaml"
)


def stage_i_metadata(config):
    return {
        "diagnostic_h1_mode": "no_h1",
        "reproduction_level": "diagnostic_extension",
        "paper_faithful_full": False,
        "extension_reason": (
            "isolate_H1_contribution_under_spherical_grid_adaptation"
        ),
        "comparison_parent": "stage_g_paper_adapted_full",
        "stage_g_parent_config": (
            "configs/paper_reduced100/full_fno_proxy.yaml"
        ),
        "selected_h1_training_coefficient": 0.0,
        "diagnostic_current_upstream_h1_trained": False,
        "non_h1_loss_contract": {
            name: config.values["loss"][name]
            for name in ("base", "roi", "bounds", "envelope", "dissipation")
        },
    }


def test_no_h1_checkpoint_metadata_has_strict_extension_identity():
    config = load_paper_experiment_config(
        CONFIG_PATH,
        project_root=PROJECT_ROOT,
    )
    metadata = build_paper_checkpoint_metadata(
        config,
        project_commit="test",
        config_checksum=sha256_file(CONFIG_PATH),
        epoch=1,
        experiment_name="stage_i_no_h1",
        gradient_accumulation=4,
        stage_g={
            "shared_initial_state_sha256": "1" * 64,
            "pair_order_sha256": "2" * 64,
            "optimizer_step": 20,
            "h1_weight": 0.0,
            "gradient_clip_norm": 1.0,
            "resource_scaled_training": True,
        },
        stage_i=stage_i_metadata(config),
    )
    validate_paper_checkpoint_metadata(
        metadata,
        config,
        expected_config_checksum=sha256_file(CONFIG_PATH),
    )
    assert metadata["stage_i"]["selected_h1_training_coefficient"] == 0.0
    assert metadata["stage_i"][
        "diagnostic_current_upstream_h1_trained"
    ] is False
    assert metadata["stage_g"]["h1_weight"] == 0.0


def test_unit_index_checkpoint_metadata_freezes_selected_h1_definition():
    config = load_paper_experiment_config(
        UNIT_INDEX_CONFIG_PATH,
        project_root=PROJECT_ROOT,
    )
    selected_definition = stage_i_selected_h1_definition(config)
    assert selected_definition == {
        "mode": "unit_index",
        "weight": 0.05,
        "spacings": [1.0, 1.0, 1.0],
        "stencil": "second_order_centered",
        "boundary": "periodic_wrap_all_three_axes",
        "reduction": "uniform_voxel_mean",
        "spherical_metric_factors": False,
        "physical_r_spacing": False,
        "volume_weighting": False,
        "shell_weighting": False,
    }
    stage_i = {
        **stage_i_metadata(config),
        "diagnostic_h1_mode": "unit_index",
        "extension_reason": "isolate_unit_cube_spacing_amplification",
        "selected_h1_training_coefficient": 0.05,
        "selected_h1_definition": selected_definition,
    }
    metadata = build_paper_checkpoint_metadata(
        config,
        project_commit="test",
        config_checksum=sha256_file(UNIT_INDEX_CONFIG_PATH),
        epoch=1,
        experiment_name="stage_i_unit_index_h1",
        gradient_accumulation=4,
        stage_g={
            "shared_initial_state_sha256": "1" * 64,
            "pair_order_sha256": "2" * 64,
            "optimizer_step": 20,
            "h1_weight": 0.05,
            "gradient_clip_norm": 1.0,
            "resource_scaled_training": True,
        },
        stage_i=stage_i,
    )
    validate_paper_checkpoint_metadata(
        metadata,
        config,
        expected_config_checksum=sha256_file(UNIT_INDEX_CONFIG_PATH),
    )
    changed = deepcopy(metadata)
    changed["stage_i"]["selected_h1_definition"]["spacings"] = [
        1.0,
        1.0,
        1.01,
    ]
    with pytest.raises(ValueError, match="selected H1 definition"):
        validate_paper_checkpoint_metadata(changed, config)


def test_stored_checkpoint_metadata_freezes_coordinates_and_volume_proxy():
    config = load_paper_experiment_config(
        STORED_CONFIG_PATH,
        project_root=PROJECT_ROOT,
    )
    selected_definition = stage_i_selected_h1_definition(config)
    assert selected_definition is not None
    assert selected_definition["mode"] == "stored_coordinate_volume_proxy"
    assert selected_definition["radial_coordinate"] == "physical_r"
    assert set(selected_definition["coordinate_sha256"]) == {
        "phi",
        "theta",
        "r",
    }
    assert all(
        len(value) == 64
        for value in selected_definition["coordinate_sha256"].values()
    )
    assert len(selected_definition["coordinate_combined_sha256"]) == 64
    volume = selected_definition["volume_proxy"]
    assert len(volume["weight_sha256"]) == 64
    assert volume["normalization_sum"] == pytest.approx(1.0)
    assert volume["finite"] is True
    assert volume["nonnegative"] is True
    assert volume["pole_sin_floor"] == 0.0
    assert volume["pole_handling"] == "theta_center_sin_clamp_min_0"

    stage_i = {
        **stage_i_metadata(config),
        "diagnostic_h1_mode": "stored_coordinate_volume_proxy",
        "extension_reason": (
            "stored_spherical_coordinate_H1_adaptation_diagnosis"
        ),
        "selected_h1_training_coefficient": 0.05,
        "selected_h1_definition": selected_definition,
    }
    metadata = build_paper_checkpoint_metadata(
        config,
        project_commit="test",
        config_checksum=sha256_file(STORED_CONFIG_PATH),
        epoch=1,
        experiment_name="stage_i_stored_coordinate_volume_h1",
        gradient_accumulation=4,
        stage_g={
            "shared_initial_state_sha256": "1" * 64,
            "pair_order_sha256": "2" * 64,
            "optimizer_step": 20,
            "h1_weight": 0.05,
            "gradient_clip_norm": 1.0,
            "resource_scaled_training": True,
        },
        stage_i=stage_i,
    )
    validate_paper_checkpoint_metadata(
        metadata,
        config,
        expected_config_checksum=sha256_file(STORED_CONFIG_PATH),
    )
    changed = deepcopy(metadata)
    changed["stage_i"]["selected_h1_definition"]["volume_proxy"][
        "weight_sha256"
    ] = "0" * 64
    with pytest.raises(ValueError, match="selected H1 definition"):
        validate_paper_checkpoint_metadata(changed, config)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("diagnostic_h1_mode", "unit_index", "diagnostic H1 mode"),
        ("selected_h1_training_coefficient", 0.05, "selected H1 coefficient"),
        (
            "diagnostic_current_upstream_h1_trained",
            True,
            "diagnostic H1 entered training",
        ),
    ],
)
def test_no_h1_checkpoint_rejects_extension_identity_changes(
    field,
    bad_value,
    message,
):
    config = load_paper_experiment_config(
        CONFIG_PATH,
        project_root=PROJECT_ROOT,
    )
    metadata = build_paper_checkpoint_metadata(
        config,
        project_commit="test",
        config_checksum=sha256_file(CONFIG_PATH),
        epoch=1,
        experiment_name="stage_i_no_h1",
        gradient_accumulation=4,
        stage_g={
            "shared_initial_state_sha256": "1" * 64,
            "pair_order_sha256": "2" * 64,
            "optimizer_step": 20,
            "h1_weight": 0.0,
            "gradient_clip_norm": 1.0,
            "resource_scaled_training": True,
        },
        stage_i=stage_i_metadata(config),
    )
    changed = deepcopy(metadata)
    changed["stage_i"][field] = bad_value
    with pytest.raises(ValueError, match=message):
        validate_paper_checkpoint_metadata(changed, config)
