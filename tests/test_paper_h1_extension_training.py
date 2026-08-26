from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_training_module():
    path = PROJECT_ROOT / "scripts/train_paper_reduced.py"
    spec = importlib.util.spec_from_file_location(
        "train_paper_reduced_for_stage_i_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_h1_gradient_ratio_is_not_fabricated_as_zero():
    module = load_training_module()
    assert (
        module.selected_h1_gradient_ratio(
            base_gradient_norm=4.0,
            selected_h1_gradient_norm=0.0,
            diagnostic_h1_mode="no_h1",
        )
        is None
    )
    assert module.selected_h1_gradient_ratio(
        base_gradient_norm=4.0,
        selected_h1_gradient_norm=2.0,
        diagnostic_h1_mode=None,
    ) == pytest.approx(0.5)


def test_no_h1_gradient_ratio_rejects_nonzero_selected_gradient():
    module = load_training_module()
    with pytest.raises(ValueError, match="exactly zero"):
        module.selected_h1_gradient_ratio(
            base_gradient_norm=4.0,
            selected_h1_gradient_norm=1e-9,
            diagnostic_h1_mode="no_h1",
        )


def test_component_gradient_cosine_is_aligned_and_handles_zero_component():
    module = load_training_module()
    assert module.gradient_tensor_cosine(
        [torch.tensor([1.0, 2.0]), None],
        [torch.tensor([2.0, 4.0]), None],
    ) == pytest.approx(1.0)
    assert module.gradient_tensor_cosine(
        [torch.tensor([1.0, 0.0])],
        [torch.tensor([0.0, 1.0])],
    ) == pytest.approx(0.0)
    assert (
        module.gradient_tensor_cosine(
            [torch.tensor([1.0])],
            [torch.tensor([0.0])],
        )
        is None
    )
    assert module.gradient_tensor_cosine(
        [torch.tensor([1.0 + 2.0j])],
        [torch.tensor([2.0 + 4.0j])],
    ) == pytest.approx(1.0)


def test_component_gradient_projection_preserves_sign_and_total_direction():
    module = load_training_module()
    assert module.gradient_tensor_projection(
        [torch.tensor([1.0, 1.0])],
        [torch.tensor([2.0, 0.0])],
    ) == pytest.approx(1.0)
    assert module.gradient_tensor_projection(
        [torch.tensor([-1.0, 0.0])],
        [torch.tensor([2.0, 0.0])],
    ) == pytest.approx(-1.0)
    assert module.gradient_tensor_projection(
        [torch.tensor([1.0 + 2.0j])],
        [torch.tensor([2.0 + 4.0j])],
    ) == pytest.approx(torch.sqrt(torch.tensor(5.0)).item())


def test_manifest_preparation_cannot_start_training():
    source = (
        PROJECT_ROOT / "scripts/prepare_paper_stage_i_no_h1.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "build_paper_optimizer",
        "optimizer.step(",
        "scheduler.step(",
        "save_paper_checkpoint",
        ".backward(",
    ):
        assert forbidden not in source


def test_unit_index_manifest_preparation_cannot_train_or_write_checkpoint():
    source = (
        PROJECT_ROOT / "scripts/prepare_paper_stage_i_unit_index.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "optimizer.step(",
        "scheduler.step(",
        "save_paper_checkpoint",
        ".backward(",
        "torch.save(",
    ):
        assert forbidden not in source
    assert "load_paper_checkpoint" in source
    assert "isolate_unit_cube_spacing_amplification" in source


def test_stored_training_logs_selected_decompositions_and_unit_diagnostic():
    source = (
        PROJECT_ROOT / "scripts/train_paper_reduced.py"
    ).read_text(encoding="utf-8")
    for required in (
        "diagnostic_unit_index_h1_raw",
        "unit_index_to_selected_h1_raw_ratio",
        'f"selected_h1_direction_{direction}"',
        'f"selected_h1_shell_{shell_index}"',
        'row["selected_h1_shell_inner_0_1"]',
        'row["selected_h1_shell_middle_2_5"]',
        'row["selected_h1_shell_outer_6_7"]',
    ):
        assert required in source


def test_stored_manifest_preparation_cannot_train_or_write_checkpoint():
    source = (
        PROJECT_ROOT / "scripts/prepare_paper_stage_i_stored_coordinate.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "torch.optim",
        "build_paper_optimizer",
        "optimizer.step(",
        "scheduler.step(",
        "save_paper_checkpoint",
        ".backward(",
        "torch.save(",
    ):
        assert forbidden not in source
    assert "load_paper_checkpoint" in source
    assert "stored_spherical_coordinate_H1_adaptation_diagnosis" in source
    assert "expected_classification.items()" in source


def test_final_stage_i_comparison_reuses_frozen_scoring_and_outputs_contract():
    source = (
        PROJECT_ROOT / "scripts/compare_paper_stage_i_extensions.py"
    ).read_text(encoding="utf-8")
    assert "state_category_scores" in source
    assert "model_only_saturation" in source
    for output in (
        "extension_comparison.csv",
        "extension_comparison.json",
        "extension_comparison.md",
        "table2_style.csv",
        "oracle_comparison.csv",
        "rollout_comparison.csv",
        "morphology_comparison.csv",
        "boundary_comparison.csv",
        "gradient_comparison.csv",
        "stage_i_decision.json",
        "stage_i_decision.md",
    ):
        assert output in source


def test_saved_final_stage_i_decision_uses_completed_run_c_gate():
    path = (
        PROJECT_ROOT
        / "outputs/paper_reduced100/stage_i/stage_i_decision.json"
    )
    if not path.exists():
        pytest.skip("local Stage I final summary is Git ignored")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["choice"] == "D"
    assert payload["name"] == "No extension rescues Full"
    assert all(payload["run_c_gate"].values())
    assert payload["extension_better_category_counts_vs_full"] == {
        "no_h1": 5,
        "stored_coordinate": 5,
        "unit_index": 5,
    }
    assert payload["extension_better_category_counts_vs_plain"] == {
        "no_h1": 1,
        "stored_coordinate": 0,
        "unit_index": 1,
    }
