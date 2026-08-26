from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from grmhd.dataset import sha256_file
from grmhd.paper_config import (
    STAGE_I_PAIR_ALLOWED_DIFFERENCES,
    build_paper_training_loss,
    load_paper_experiment_config,
    recursive_config_differences,
    stage_i_difference_is_allowed,
)
from grmhd.paper_h1_extensions import DiagnosticPaperCompositeLoss
from grmhd.paper_losses import PaperCompositeLoss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_PATH = PROJECT_ROOT / "configs/paper_reduced100/full_fno_proxy.yaml"
EXTENSION_PATHS = {
    "no_h1": PROJECT_ROOT
    / "configs/paper_reduced100/extensions/no_h1_control.yaml",
    "unit_index": PROJECT_ROOT
    / "configs/paper_reduced100/extensions/unit_index_h1.yaml",
    "stored_coordinate_volume_proxy": PROJECT_ROOT
    / "configs/paper_reduced100/extensions/stored_coordinate_volume_h1.yaml",
}


@pytest.fixture(scope="module")
def resolved_configs():
    return {
        "full": load_paper_experiment_config(FULL_PATH, project_root=PROJECT_ROOT),
        **{
            name: load_paper_experiment_config(path, project_root=PROJECT_ROOT)
            for name, path in EXTENSION_PATHS.items()
        },
    }


def test_stage_i_configs_are_explicit_full_config_extensions(resolved_configs):
    full = resolved_configs["full"]
    assert STAGE_I_PAIR_ALLOWED_DIFFERENCES == {
        "experiment_name",
        "output_dir",
        "checkpoint_prefix",
        "loss.h1.*",
        "reproduction_metadata.reproduction_level",
        "reproduction_metadata.paper_faithful_full",
        "reproduction_metadata.extension_reason",
    }
    for name, path in EXTENSION_PATHS.items():
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert raw["extends"] == "../full_fno_proxy.yaml"
        assert set(raw) == {
            "extends",
            "experiment_name",
            "output_dir",
            "checkpoint_prefix",
            "loss",
            "reproduction_metadata",
        }
        assert set(raw["loss"]) == {"h1"}

        config = resolved_configs[name]
        differences = recursive_config_differences(
            full.as_dict(),
            config.as_dict(),
        )
        assert differences
        assert all(
            stage_i_difference_is_allowed(item["path"])
            for item in differences
        )
        assert config.diagnostic_h1_mode == name
        assert config.values["reproduction_metadata"][
            "reproduction_level"
        ] == "diagnostic_extension"
        assert config.values["reproduction_metadata"]["paper_faithful_full"] is False
        expected_reason = {
            "no_h1": "spherical_grid_H1_adaptation_diagnosis",
            "unit_index": "isolate_unit_cube_spacing_amplification",
            "stored_coordinate_volume_proxy": (
                "stored_spherical_coordinate_H1_adaptation_diagnosis"
            ),
        }[name]
        assert (
            config.values["reproduction_metadata"]["extension_reason"]
            == expected_reason
        )


def test_stage_i_h1_contracts_keep_reference_weight(resolved_configs):
    no_h1 = resolved_configs["no_h1"].values["loss"]["h1"]
    unit_index = resolved_configs["unit_index"].values["loss"]["h1"]
    stored = resolved_configs["stored_coordinate_volume_proxy"].values["loss"]["h1"]
    for h1 in (no_h1, unit_index, stored):
        assert h1["weight"] == 0.05
        assert h1["paper_reference_weight"] == 0.05
        assert h1["diagnostic_current_upstream_h1"] is True
    assert no_h1["enabled"] is False
    assert no_h1["weighted_contribution"] == 0.0
    assert unit_index["spacings"] == [1.0, 1.0, 1.0]
    assert unit_index["reduction"] == "uniform_voxel_mean"
    assert stored["coordinate_source"] == "frozen_hdf5_centers"
    assert stored["radial_coordinate"] == "physical_r"
    assert stored["pole_sin_floor"] == 0.0
    assert stored["pole_handling"] == "theta_center_sin_clamp_min_0"
    assert stored["volume_weight_normalization"] == "explicit_sum_to_one"
    assert stored["covariant_GRMHD_H1"] is False
    assert stored["proper_Kerr_Schild_volume"] == "unverified"
    assert stored["stored_components_covariant_derivative"] is False
    assert stored["stored_vector_covariant_derivative"] is False
    assert stored["diagnostic_proxy_only"] is True


def test_frozen_stage_g_full_config_and_loss_are_unchanged(resolved_configs):
    manifest = json.loads(
        (
            PROJECT_ROOT / "outputs/paper_reduced100/stage_g/run_manifest.json"
        ).read_text(encoding="utf-8")
    )
    assert sha256_file(FULL_PATH) == manifest["checksums"]["full_config"]
    full_loss = build_paper_training_loss(resolved_configs["full"])
    assert type(full_loss) is PaperCompositeLoss
    assert full_loss.metadata["schema_version"] == "paper-composite-loss-v1"
    assert full_loss.metadata["h1_weight"] == 0.05


@pytest.mark.parametrize("name", sorted(EXTENSION_PATHS))
def test_extension_builder_selects_only_diagnostic_wrapper(
    resolved_configs,
    name,
):
    loss = build_paper_training_loss(resolved_configs[name])
    assert isinstance(loss, DiagnosticPaperCompositeLoss)
    assert isinstance(loss, PaperCompositeLoss)
    assert loss.diagnostic_h1_mode.value == name
    assert loss.h1_weight == 0.05
    assert loss.metadata["paper_faithful_full"] is False


def test_extension_config_rejects_non_h1_override(tmp_path):
    raw = yaml.safe_load(
        EXTENSION_PATHS["no_h1"].read_text(encoding="utf-8")
    )
    raw["runtime"] = {"seed": 43}
    path = tmp_path / "bad_extension.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="non-paired override"):
        load_paper_experiment_config(path, project_root=PROJECT_ROOT)
