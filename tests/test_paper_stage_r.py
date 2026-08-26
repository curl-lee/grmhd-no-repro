from __future__ import annotations

import inspect
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
import pytest
import torch

from grmhd import CHANNELS
from grmhd.models import build_model
from grmhd.paper_config import (
    STAGE_K_MODEL_CONTRACT,
    build_paper_model,
    load_paper_experiment_config,
)
from grmhd.paper_data_processor import PaperDataProcessor
from grmhd.paper_losses import PlainL2Loss
from grmhd.paper_stage_g import load_epoch_pair_order, tensor_state_sha256 as pairing_tensor_sha256
from grmhd.paper_stage_g_evaluation import artifact_diagnostics
from grmhd.paper_stage_m import (
    evaluate_oracle_conditioned_structure_gate,
    oracle_conditioned_gate_schema,
)
from grmhd.paper_stage_r import (
    FROZEN_HISTORY,
    contract_source_has_forbidden_operation,
    expected_rollout_transform_counts,
    loss_equivalence,
    normalized_residual_target,
    reconstruct_normalized_state,
    residual_plain_l2,
    validate_contract_config,
)
from scripts.evaluate_paper_stage_g import (
    interpret_rollout_output,
    manifest_pairing_hashes,
)
from scripts.train_paper_reduced import load_p3_train_envelope


ROOT = Path(__file__).resolve().parents[1]
STAGE_R_CONFIG = ROOT / "configs/paper_reduced100/stage_r_localno_p3_residual_plain.yaml"
STAGE_O_CONFIG = ROOT / "configs/paper_reduced100/stage_o_localno_p3_plain.yaml"
INITIAL_STATE = ROOT / "outputs/paper_reduced100/stage_k/shared_localno_initial_state.pt"
PAIR_ORDER = ROOT / "outputs/paper_reduced100/stage_g/epoch_pair_order.json"
STAGE_R_OUTPUT = ROOT / "outputs/paper_reduced100/stage_r"


@pytest.fixture(scope="module")
def config():
    return load_paper_experiment_config(STAGE_R_CONFIG, project_root=ROOT)


@pytest.fixture
def states():
    z = torch.randn(1, 8, 2, 2, 2)
    target = z + 0.1 * torch.randn_like(z)
    predicted = 0.1 * torch.randn_like(z)
    return z, target, predicted


def test_residual_target_is_target_minus_input(states):
    z, target, _ = states
    torch.testing.assert_close(normalized_residual_target(z, target), target - z)


def test_identity_reconstruction_is_input_plus_residual(states):
    z, _, predicted = states
    torch.testing.assert_close(reconstruct_normalized_state(z, predicted), z + predicted)


def test_evaluator_interprets_stage_r_raw_output_as_residual(states):
    z, _, predicted = states
    reconstructed, residual = interpret_rollout_output(
        predicted, z, residual_contract=True
    )
    assert residual is predicted
    torch.testing.assert_close(reconstructed, z + predicted)


def test_evaluator_reads_stage_r_pairing_manifest():
    manifest = json.loads((ROOT / "outputs/paper_reduced100/stage_r/run_manifest.json").read_text())
    initial, order = manifest_pairing_hashes(manifest)
    assert initial == "7e10895cb283be8532e4c036e8a526c536a632cce880c500080ea1d14fd89311"
    assert order == "5e1e361ea75e01ffd667822d8bfe870d70c5bc69570d52cdf806e637b2df2b52"


def test_residual_and_reconstructed_plain_l2_are_equivalent(states):
    z, target, predicted = states
    result = loss_equivalence(predicted, z, target)
    assert result["passed"]


def test_zero_residual_is_exact_persistence(states):
    z, _, _ = states
    assert torch.equal(reconstruct_normalized_state(z, torch.zeros_like(z)), z)


def test_exact_true_residual_reconstructs_target(states):
    z, target, _ = states
    assert torch.equal(reconstruct_normalized_state(z, target - z), target)


def test_shell_channels_cannot_enter_residual_target(states):
    z, target, _ = states
    shells = torch.randn_like(z)
    residual = normalized_residual_target(z, target)
    assert residual.shape[1] == 8
    assert residual.shape != torch.cat((residual, shells), dim=1).shape


def test_processor_exposes_residual_target_and_reconstructs_state(config):
    processor = PaperDataProcessor.from_config(config)
    with h5py.File(ROOT / "data_proc/grmhd_regrid_inner_r200_64.h5", "r") as handle:
        source = torch.from_numpy(np.asarray(handle["snapshots"][11], dtype=np.float32)).unsqueeze(0)
        target = torch.from_numpy(np.asarray(handle["snapshots"][12], dtype=np.float32)).unsqueeze(0)
        times = np.asarray(handle["times"][...], dtype=np.float64)
    sample = processor.preprocess({"physical_input": source, "physical_target": target, "source_index": torch.tensor([11]), "target_index": torch.tensor([12]), "time": torch.tensor([times[11]]), "target_time": torch.tensor([times[12]])})
    fields = processor.last_batch
    assert fields is not None
    torch.testing.assert_close(sample["y"], fields["normalized_target"] - fields["normalized_input"])
    assert sample["x"].shape[1] == 16 and sample["y"].shape[1] == 8
    raw_residual = torch.zeros_like(sample["y"])
    reconstructed, returned = processor.postprocess(raw_residual, sample)
    assert torch.equal(reconstructed, fields["normalized_input"])
    assert returned["predicted_residual"] is raw_residual


def test_contract_output_has_only_eight_physical_channels(states):
    z, _, predicted = states
    assert reconstruct_normalized_state(z, predicted).shape[1] == len(CHANNELS)


@pytest.mark.parametrize(
    "override",
    (
        {"residual_scale": 0.5},
        {"learnable_scale": True},
        {"clipping": True},
        {"state_skip": "none"},
    ),
)
def test_contract_rejects_scaling_learning_clipping_or_missing_skip(config, override):
    values = config.as_dict()
    values["prediction"].update(override)
    with pytest.raises(ValueError, match="prediction contract"):
        validate_contract_config(values)


def test_contract_source_has_no_hidden_clamp_or_detach():
    assert contract_source_has_forbidden_operation() is False


def test_p3_statistics_are_frozen_train_only(config):
    assert config.uses_p3 and config.stage_r
    assert config.values["evaluation"]["rho_press_eval_clamp"] is False
    assert config.values["preprocessing"]["stats_checksum"] == (
        "aea6d3f91e54865deeb9b2b1142c49a633ae8b67223075273e6c0b88bf541948"
    )
    metadata = json.loads(
        (ROOT / config.values["preprocessing"]["prototype_metadata"]).read_text()
    )
    assert metadata["training_indices"] == list(range(11, 91))
    assert metadata["validation_indices_used_for_fit"] == []


def test_stage_r_training_diagnostics_load_frozen_p3_train_envelope(config):
    envelope = load_p3_train_envelope(ROOT, config)
    assert envelope is not None
    assert set(envelope["channels"]) == set(CHANNELS)


def test_architecture_names_shapes_and_parameter_count_match_stage_k(config):
    stage_r = build_paper_model(config)
    stage_k = build_model(
        "localno_differential_3d",
        in_channels=16,
        out_channels=8,
        n_modes=(8, 8, 8),
        hidden_channels=16,
        n_layers=4,
        default_in_shape=(64, 64, 64),
        positional_embedding=None,
        fin_diff_kernel_size=3,
        mix_derivatives=True,
        conv_padding_mode="periodic",
        use_channel_mlp=False,
        local_no_skip="linear",
        norm=None,
        enforce_hermitian_symmetry=True,
    )
    assert [
        (n, tuple(v.shape) if torch.is_tensor(v) else v)
        for n, v in stage_r.state_dict().items()
    ] == [
        (n, tuple(v.shape) if torch.is_tensor(v) else v)
        for n, v in stage_k.state_dict().items()
    ]
    assert sum(p.numel() for p in stage_r.parameters()) == 358296


def test_shared_initial_state_strict_parity(config):
    state = torch.load(INITIAL_STATE, map_location="cpu", weights_only=True)
    model = build_paper_model(config)
    result = model.load_state_dict(state, strict=True)
    assert result.missing_keys == [] and result.unexpected_keys == []
    assert pairing_tensor_sha256(model.state_dict()) == (
        "7e10895cb283be8532e4c036e8a526c536a632cce880c500080ea1d14fd89311"
    )


def test_pair_order_is_frozen_and_complete():
    payload, orders = load_epoch_pair_order(PAIR_ORDER)
    assert payload["validation_shuffle"] is False
    assert len(orders) == 30
    assert all(sorted(order) == list(range(79)) for order in orders)


def test_stage_r_required_small_result_artifacts_exist():
    required = {
        "run_manifest.json",
        "run_manifest.md",
        "residual_target_audit.csv",
        "residual_target_audit.json",
        "residual_target_audit.md",
        "preflight.json",
        "preflight.md",
        "smoke_summary.json",
        "smoke_summary.md",
        "training_summary.csv",
        "training_summary.json",
        "training_summary.md",
        "validation_metrics.csv",
        "validation_metrics.json",
        "residual_metrics.csv",
        "residual_metrics.json",
        "p3_persistence.csv",
        "p3_persistence.json",
        "stage_o_stage_r_comparison.csv",
        "stage_o_stage_r_comparison.json",
        "stage_o_stage_r_comparison.md",
        "rollout_gt.csv",
        "rollout_gt.json",
        "rollout_no_gt.csv",
        "rollout_no_gt.json",
        "legacy_detector.csv",
        "legacy_detector.json",
        "stage_m_v1_gate.csv",
        "stage_m_v1_gate.json",
        "stage_m_v1_gate.md",
        "stage_r_decision.json",
        "stage_r_decision.md",
    }
    assert required <= {path.name for path in STAGE_R_OUTPUT.iterdir() if path.is_file()}


def test_stage_r_completed_counts_reload_and_decision_are_frozen():
    smoke = json.loads((STAGE_R_OUTPUT / "smoke_summary.json").read_text())
    training = json.loads((STAGE_R_OUTPUT / "training_summary.json").read_text())
    decision = json.loads((STAGE_R_OUTPUT / "stage_r_decision.json").read_text())
    assert smoke["passed"] and smoke["microbatches"] == 158
    assert smoke["optimizer_updates"] == 40
    assert training["epochs"] == 30 and training["microbatches"] == 2370
    assert training["optimizer_updates"] == 600 and training["best_epoch"] == 9
    assert training["runtime"]["nonfinite_count"] == 0
    assert training["checkpoint_reload"]["best"]["prediction_parity"]
    assert training["checkpoint_reload"]["last"]["prediction_parity"]
    assert decision["choice"] == "C"
    assert decision["decision"] == "RESIDUAL_PILOT_COMPLETE_BUT_ROLLOUT_UNSTABLE"


def test_stage_r_rollout_counts_no_clamp_and_no_gt_semantics():
    gt = json.loads((STAGE_R_OUTPUT / "rollout_gt.json").read_text())
    no_gt = json.loads((STAGE_R_OUTPUT / "rollout_no_gt.json").read_text())
    assert len(gt["records"]) == 19 and len(no_gt["records"]) == 81
    assert gt["finite"] and gt["rho_press_positive"]
    assert no_gt["finite"] and no_gt["rho_press_positive"]
    assert no_gt["transform_count_delta"] == {
        "input_encode": 100,
        "target_encode": 19,
        "oracle_decode": 19,
        "prediction_decode": 100,
    }
    assert all(row["evaluation_bound_clamp_fraction"] == 0.0 for row in gt["records"])
    assert all(
        row["model_only_saturation"] == "omitted_without_ground_truth_by_protocol"
        and row["ground_truth_available"] is False
        for row in no_gt["records"]
    )


def test_stage_r_persistence_parity_and_residual_skill_result():
    persistence = json.loads((STAGE_R_OUTPUT / "p3_persistence.json").read_text())
    residual = json.loads((STAGE_R_OUTPUT / "residual_metrics.json").read_text())
    comparison = json.loads(
        (STAGE_R_OUTPUT / "stage_o_stage_r_comparison.json").read_text()
    )
    assert persistence["stage_o_pipeline_parity"] is True
    assert residual["zero_residual_relative_l2"] == 1.0
    assert residual["arithmetic_average_relative_l2"] > 1.0
    assert comparison["aggregate"]["stage_r_over_stage_o"] < 1.0
    assert comparison["aggregate"]["stage_r_over_persistence"] > 1.0


def test_strict_plain_l2_parity(states):
    z, target, predicted = states
    plain = PlainL2Loss()
    residual = residual_plain_l2(predicted, z, target, loss=plain)
    state = plain(z + predicted, target)
    torch.testing.assert_close(residual, state)


def test_stage_o_direct_path_still_loads_unchanged():
    direct = load_paper_experiment_config(STAGE_O_CONFIG, project_root=ROOT)
    assert direct.stage_o and not direct.stage_r
    assert direct.prediction_mode == "direct"
    assert direct.values["model"] == STAGE_K_MODEL_CONTRACT


def test_legacy_detector_remains_runnable(states):
    z, _, _ = states
    result = artifact_diagnostics(z, z, z, reference_kind="ground_truth")
    assert result["flags"] == []


def _gate(**overrides):
    values = {
        "legacy_detector": {"flags": []},
        "engineering_checks": {
            "finite": True,
            "rho_press_positive": True,
            "transform_counters": True,
            "checkpoint_provenance": True,
            "decoded_range": True,
            "Rout": True,
            "shape_device": True,
        },
        "floor_metrics": {
            "oracle_legacy_detector_triggered": False,
            "median_variance_retention": 1.0,
            "median_shell_radial_retention": 1.0,
            "median_high_k_retention": 1.0,
        },
        "severe_by_step": {1: {k: False for k in ("global_variance", "shell_radial_variance", "dynamic_span", "high_k_energy")}},
        "shell_transport_skill": 0.1,
        "radial_transport_skill": 0.1,
        "shell_sign_agreement": 1.0,
        "radial_sign_agreement": 1.0,
    }
    values.update(overrides)
    return evaluate_oracle_conditioned_structure_gate(**values)


def test_stage_m_v1_gate_one_uses_p3_floor():
    assert oracle_conditioned_gate_schema()["oracle_conditioned_gate_version"] == "stage_m_v1"
    gate = _gate(floor_metrics={"oracle_legacy_detector_triggered": True, "median_variance_retention": 0.1, "median_shell_radial_retention": 0.1, "median_high_k_retention": 0.1})
    assert gate["gate_1_floor_qualification"]["floor_limited"]


def test_stage_m_v1_gate_two_uses_reconstructed_state_vs_p3_oracle():
    severe = {step: {"global_variance": True, "shell_radial_variance": True, "dynamic_span": False, "high_k_energy": False} for step in (1, 3)}
    assert _gate(severe_by_step=severe)["gate_2_model_added_degradation"]["failed"]


def test_stage_m_v1_gate_three_compares_against_p3_persistence():
    assert _gate(shell_transport_skill=0.2)["gate_3_transport_skill"]["passed"]


def test_rollout_transform_counters_are_exact():
    assert expected_rollout_transform_counts(19) == {"prediction_decode": 19, "next_input_encode": 19}
    assert expected_rollout_transform_counts(100) == {"prediction_decode": 100, "next_input_encode": 100}


def test_stage_k_through_q_history_is_frozen():
    assert set(FROZEN_HISTORY) == {f"Stage {name}" for name in "KLMNOPQ"}
    assert FROZEN_HISTORY["Stage Q"] == "D. NO_VALID_ANCHOR_CANDIDATE"


def test_only_approved_checkpoint_binaries_are_tracked():
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    approved_checkpoints = {
        "artifacts/stage_ag/training/checkpoints/best.pt",
        "artifacts/stage_ag/training/checkpoints/last.pt",
    }
    tracked_checkpoints = {
        path for path in tracked if path.endswith((".pt", ".pth"))
    }
    assert tracked_checkpoints == approved_checkpoints
    assert not any(path.endswith((".h5", ".hdf5", ".athdf")) for path in tracked)


def test_validation_cannot_tune_or_fit(config):
    assert config.values["preprocessing"]["validation_used_for_fit"] is False
    assert config.values["provenance"]["validation_not_used_for_fit"] is True
    assert "validation" not in inspect.signature(normalized_residual_target).parameters
