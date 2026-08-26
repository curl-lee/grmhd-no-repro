from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_evaluation_module():
    path = PROJECT_ROOT / "scripts/evaluate_paper_stage_g.py"
    spec = importlib.util.spec_from_file_location(
        "evaluate_paper_stage_g_for_test",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stage_i_and_stage_g_manifests_resolve_the_same_pairing_identity():
    module = load_evaluation_module()
    shared = "1" * 64
    order = "2" * 64
    assert module.manifest_pairing_hashes(
        {
            "shared_initial_state": {"tensor_state_sha256": shared},
            "checksums": {"pair_order_file": order},
        }
    ) == (shared, order)
    assert module.manifest_pairing_hashes(
        {
            "checksums": {
                "shared_initial_state_tensor": shared,
                "pair_order_file": order,
            }
        }
    ) == (shared, order)


def test_evaluation_rejects_missing_or_malformed_pairing_hashes():
    module = load_evaluation_module()
    with pytest.raises(ValueError, match="shared initial state"):
        module.manifest_pairing_hashes(
            {"checksums": {"pair_order_file": "2" * 64}}
        )
    with pytest.raises(ValueError, match="pair order"):
        module.manifest_pairing_hashes(
            {
                "checksums": {
                    "shared_initial_state_tensor": "1" * 64,
                    "pair_order_file": "short",
                }
            }
        )


def test_no_h1_comparison_reuses_frozen_stage_g_scoring_functions():
    source = (
        PROJECT_ROOT / "scripts/compare_paper_stage_i_no_h1.py"
    ).read_text(encoding="utf-8")
    assert "state_category_scores" in source
    assert "model_only_saturation" in source
    assert "fixed_phi_theta_r" in source
    assert "equatorial_phi_r" in source
    assert "not Cartesian slices" in source
    assert "unit_index_h1.yaml" not in source
    assert "stored_coordinate_volume_h1.yaml" not in source


def test_unit_index_comparison_reuses_frozen_scoring_and_partial_outputs():
    source = (
        PROJECT_ROOT / "scripts/compare_paper_stage_i_unit_index.py"
    ).read_text(encoding="utf-8")
    for required in (
        "state_category_scores",
        "model_only_saturation",
        "fixed_phi_theta_r",
        "equatorial_phi_r",
        "not Cartesian slices",
        "partial_extension_table2.csv",
        "partial_rollout_comparison.csv",
        "partial_morphology_comparison.csv",
        "partial_gradient_comparison.csv",
        'table_row("canonical_oracle"',
        "final_stage_i_decision",
    ):
        assert required in source
    assert "stored_coordinate_volume_h1.yaml" not in source
