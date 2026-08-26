from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import yaml

from grmhd.paper_stage_g_evaluation import artifact_diagnostics


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/paper_reduced100/stage_l_collapse_attribution.yaml"
OUTPUT = ROOT / "outputs/paper_reduced100/stage_l"


def load_json(name: str):
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))


def canonical_sha256(value) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def test_stage_l_attribution_thresholds_are_frozen_before_analysis():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert config["classification"] == {
        "kind": "post_hoc_attribution",
        "read_only_model_evaluation": True,
        "training_allowed": False,
        "stage_k_decision_frozen": "C. TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE",
    }
    assert config["analysis"]["gt_selected_steps"] == [1, 3, 5, 10, 19]
    assert config["analysis"]["no_gt_selected_steps"] == [25, 50, 75, 100]
    assert config["retention_contract"]["severe_loss_threshold"] == 0.5
    assert config["retention_contract"]["severe_loss_comparator"] == "strict_less_than"
    assert config["retention_contract"]["selected_gt_step_aggregation"] == {
        "rule": "majority",
        "required_steps": 3,
        "total_steps": 5,
    }


def test_stage_l_detector_implementation_and_contract_hashes_are_current():
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    frozen = config["frozen_detector"]
    actual_function_hash = hashlib.sha256(
        inspect.getsource(artifact_diagnostics).encode("utf-8")
    ).hexdigest()
    actual_file_hash = hashlib.sha256(
        (ROOT / frozen["source_path"]).read_bytes()
    ).hexdigest()
    assert actual_function_hash == frozen["implementation_source_sha256"]
    assert actual_file_hash == frozen["source_file_sha256"]

    contract = load_json("detector_contract.json")
    assert contract["function_source_sha256"] == actual_function_hash
    assert contract["source_file_sha256"] == actual_file_hash
    assert contract["formula"]["flag"] == "ratio < 0.05"
    assert contract["input_contract"]["shell_aggregation"] == "none"
    assert contract["input_contract"]["time_aggregation"] == "none_per_step"
    assert contract["detector_config_sha256"] == canonical_sha256(frozen)
    claimed_contract_hash = contract.pop("contract_sha256")
    assert claimed_contract_hash == canonical_sha256(contract)


def test_stage_l_exactly_reproduces_frozen_stage_k_flags():
    reproduction = load_json("detector_reproduction.json")
    assert reproduction["status"] == "passed"
    expected = {
        "25": ["Bcc3:possible_field_collapse"],
        "50": ["Bcc3:possible_field_collapse"],
        "75": [
            "Bcc3:possible_field_collapse",
            "vel3:possible_field_collapse",
        ],
        "100": [
            "Bcc3:possible_field_collapse",
            "vel3:possible_field_collapse",
        ],
    }
    for step, flags in expected.items():
        record = reproduction["steps"][step]
        assert record["exact_all_flag_match"] is True
        assert record["exact_collapse_flag_match"] is True
        assert record["reproduced_collapse_flags"] == flags
        assert record["recorded_collapse_flags"] == flags


def test_stage_l_provenance_gate_is_model_only_and_complete():
    manifest = load_json("run_manifest.json")
    assert manifest["status"] in {"detector_gate_passed", "attribution_complete"}
    assert manifest["no_training"] is True
    assert manifest["no_backward"] is True
    assert manifest["no_optimizer_created"] is True
    assert manifest["no_scheduler_created"] is True
    if manifest["status"] == "detector_gate_passed":
        assert manifest["scope_not_started"] == [
            "variance_attribution",
            "shell_attribution",
            "spectrum_attribution",
        ]
    else:
        assert manifest["scope_not_started"] == []
        assert {
            "preprocessing_floor",
            "gt_variance_shell_radial_spectrum",
            "no_gt_variance_shell_radial_spectrum",
            "fno_localno_comparison",
            "channel_and_overall_attribution",
        } <= set(manifest["scope_completed"])
    assert manifest["stage_k_decision_unchanged"] == (
        "C. TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE"
    )
    provenance = manifest["provenance"]
    assert provenance["stage_l_config_sha256"] == hashlib.sha256(
        CONFIG.read_bytes()
    ).hexdigest()
    for name in ("dataset", "manifest", "preprocessing", "pair_order"):
        assert provenance[name]["actual_sha256"] == provenance[name]["expected_sha256"]
    for record in provenance["stage_d_artifacts"].values():
        assert record["actual_sha256"] == record["expected_sha256"]
    assert (
        provenance["stage_e_loss_contract"]["actual_sha256"]
        == provenance["stage_e_loss_contract"]["expected_sha256"]
    )
    assert (
        provenance["oracle_semantics"]["actual_sha256"]
        == provenance["oracle_semantics"]["expected_sha256"]
    )
    assert set(manifest["checkpoints"]) == {
        "fno_full",
        "fno_plain",
        "localno_plain",
    }
    for checkpoints in manifest["checkpoints"].values():
        assert set(checkpoints) == {"best_validation_l2", "last"}
        for record in checkpoints.values():
            assert record["strict_model_reload"] is True
            assert record["eval_mode"] is True
            assert record["optimizer_created"] is False
            assert record["scheduler_created"] is False
            assert len(record["model_state_file_sha256"]) == 64
            assert len(record["checkpoint_directory_sha256"]) == 64
            state_path = ROOT / record["path"] / "paper_state_dict.pt"
            assert hashlib.sha256(state_path.read_bytes()).hexdigest() == record[
                "model_state_file_sha256"
            ]
    for record in provenance["stage_k_rollouts"].values():
        assert hashlib.sha256((ROOT / record["path"]).read_bytes()).hexdigest() == record[
            "sha256"
        ]
