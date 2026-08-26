from __future__ import annotations

import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
import pytest
import torch
import yaml

from neuralop.layers.differential_conv import FiniteDifferenceConvolution
from neuralop.layers.discrete_continuous_convolution import (
    EquidistantDiscreteContinuousConv2d,
)

from grmhd.paper_config import (
    PLAIN_LOSS_CONTRACT,
    STAGE_K_MODEL_CONTRACT,
    build_paper_model,
    build_paper_training_loss,
    load_paper_experiment_config,
)
from grmhd.paper_data_processor import PaperDataProcessor
from grmhd.paper_losses import PlainL2Loss
from grmhd.paper_stage_g_evaluation import artifact_diagnostics
from grmhd.paper_stage_m import (
    evaluate_oracle_conditioned_structure_gate,
    oracle_conditioned_gate_schema,
)
from grmhd.paper_stage_n import NO_SOFTCLIP, PrototypePreprocessor, prototype_specs
from grmhd.paper_stage_o import (
    EXPECTED_INITIAL_TENSOR_STATE_SHA256,
    EXPECTED_PARAMETER_COUNT,
    load_frozen_p3,
    validate_stage_k_pairing,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/paper_reduced100/stage_o_localno_p3_plain.yaml"
STAGE_K_CONFIG = ROOT / "configs/paper_reduced100/stage_k_localno_differential_plain.yaml"
INITIAL = ROOT / "outputs/paper_reduced100/stage_k/shared_localno_initial_state.pt"
PAIR_ORDER = ROOT / "outputs/paper_reduced100/stage_g/epoch_pair_order.json"


@pytest.fixture(scope="module")
def stage_o_config():
    return load_paper_experiment_config(CONFIG, project_root=ROOT)


def test_stage_o_p3_config_parses_with_explicit_classification(stage_o_config):
    assert stage_o_config.stage_o is True
    assert stage_o_config.stage_k is True
    assert stage_o_config.values["reproduction_metadata"]["classification"] == (
        "adapted_transform_model_pilot"
    )
    assert stage_o_config.values["preprocessing"]["canonical_replacement"] is False


def test_stage_o_p3_statistics_checksum_and_train_only_scope(stage_o_config):
    preprocessor = load_frozen_p3(stage_o_config)
    assert isinstance(preprocessor, PrototypePreprocessor)
    assert preprocessor.training_indices == tuple(range(11, 91))
    metadata = json.loads(
        (ROOT / "outputs/paper_reduced100/stage_n/prototypes/p3_combined_v1/normalizer.json").read_text()
    )
    assert metadata["validation_indices_used_for_fit"] == []
    assert metadata["source_hdf5_checksum"] == stage_o_config.values["provenance"][
        "dataset"
    ]["sha256"]


def test_stage_o_rejects_p3_config_with_canonical_statistics(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text())
    raw["preprocessing"]["stats_path"] = "outputs/paper_reduced100/stats/normalizer.npz"
    raw["preprocessing"]["stats_checksum"] = (
        "1645714424d95b02de2283fc16a8ef05f3864d15f11025580af13f3e67964001"
    )
    raw["provenance"]["preprocessing"] = raw["provenance"][
        "canonical_preprocessing"
    ]
    path = tmp_path / "stage_o_bad_stats.yaml"
    raw["extends"] = str(STAGE_K_CONFIG)
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="preprocessing"):
        load_paper_experiment_config(path, project_root=ROOT)


def test_canonical_stage_k_rejects_p3_statistics_mixing(tmp_path):
    raw = yaml.safe_load(STAGE_K_CONFIG.read_text())
    raw["preprocessing"]["stats_path"] = (
        "outputs/paper_reduced100/stage_n/prototypes/p3_combined_v1/normalizer.npz"
    )
    raw["preprocessing"]["stats_checksum"] = (
        "aea6d3f91e54865deeb9b2b1142c49a633ae8b67223075273e6c0b88bf541948"
    )
    raw["provenance"]["preprocessing"] = {
        "path": raw["preprocessing"]["stats_path"],
        "sha256": raw["preprocessing"]["stats_checksum"],
    }
    path = tmp_path / "canonical_bad_stats.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="canonical|Preprocessing|preprocessing"):
        load_paper_experiment_config(path, project_root=ROOT)


def test_stage_o_architecture_is_exact_stage_k(stage_o_config):
    assert stage_o_config.values["model"] == STAGE_K_MODEL_CONTRACT
    stage_k = load_paper_experiment_config(STAGE_K_CONFIG, project_root=ROOT)
    left = build_paper_model(stage_o_config)
    right = build_paper_model(stage_k)
    assert [(name, tuple(value.shape)) for name, value in left.named_parameters()] == [
        (name, tuple(value.shape)) for name, value in right.named_parameters()
    ]
    assert sum(p.numel() for p in left.parameters()) == EXPECTED_PARAMETER_COUNT


def test_stage_o_initial_state_and_pair_order_are_strict(stage_o_config):
    pairing = validate_stage_k_pairing(
        stage_o_config,
        initial_state_path=INITIAL,
        pair_order_path=PAIR_ORDER,
    )
    assert pairing["initial_state_tensor_sha256"] == EXPECTED_INITIAL_TENSOR_STATE_SHA256
    assert pairing["parameter_count"] == EXPECTED_PARAMETER_COUNT
    assert pairing["pairs_per_epoch"] == 79
    assert pairing["epochs_covered"] == 30


def test_stage_o_plain_l2_contract_is_unchanged(stage_o_config):
    assert stage_o_config.values["loss"] == PLAIN_LOSS_CONTRACT
    loss = build_paper_training_loss(stage_o_config)
    assert isinstance(loss, PlainL2Loss)
    prediction = torch.zeros(1, 8, 1, 1, 1)
    prediction[:, 0] = 1.0
    prediction[:, 1] = 2.0
    target = torch.zeros_like(prediction)
    assert torch.equal(loss(prediction, target), torch.tensor(5.0))


def test_stage_o_model_shape_and_module_contract(stage_o_config):
    model = build_paper_model(stage_o_config)
    assert stage_o_config.values["model"]["in_channels"] == 16
    assert stage_o_config.values["model"]["out_channels"] == 8
    assert sum(isinstance(m, EquidistantDiscreteContinuousConv2d) for m in model.modules()) == 0
    assert sum(isinstance(m, FiniteDifferenceConvolution) for m in model.modules()) > 0


def test_stage_o_p3_numpy_and_tensor_encode_decode_are_finite(stage_o_config):
    p3 = load_frozen_p3(stage_o_config)
    with h5py.File(ROOT / "data_proc/grmhd_regrid_inner_r200_64.h5", "r") as handle:
        raw = np.asarray(handle["snapshots"][11, :, :2, :2, :2], dtype=np.float32)
    encoded_numpy, decoded_numpy = p3.round_trip(raw, channel_axis=0)
    encoded_tensor = p3.encode(torch.from_numpy(raw), channel_axis=0)
    decoded_tensor = p3.decode(encoded_tensor, channel_axis=0)
    assert np.isfinite(encoded_numpy).all() and np.isfinite(decoded_numpy).all()
    assert torch.isfinite(encoded_tensor).all() and torch.isfinite(decoded_tensor).all()
    assert np.allclose(encoded_numpy, encoded_tensor.numpy(), rtol=2e-5, atol=2e-6)
    assert np.allclose(decoded_numpy, decoded_tensor.numpy(), rtol=2e-5, atol=2e-6)


def test_stage_o_p3_tensor_decode_has_numpy_equivalent_float32_finite_guard(
    stage_o_config,
):
    p3 = load_frozen_p3(stage_o_config)
    normalized = torch.full((1, 8, 2, 2, 2), 1.0e30, dtype=torch.float32)
    tensor_decoded = p3.decode(normalized, channel_axis=1)
    numpy_decoded = p3.decode(normalized[0].numpy(), channel_axis=0)
    assert torch.isfinite(tensor_decoded).all()
    assert np.isfinite(numpy_decoded).all()
    assert float(torch.abs(tensor_decoded).max()) <= torch.finfo(torch.float32).max


def test_stage_o_p3_tensor_encode_uses_numpy_equivalent_float64_intermediates(
    stage_o_config,
):
    p3 = load_frozen_p3(stage_o_config)
    physical = torch.zeros(1, 8, 2, 2, 2, dtype=torch.float32)
    physical[:, 0:3] = torch.finfo(torch.float32).max
    physical[:, 3:5] = 1.0
    physical[:, 7] = 1.0e30
    tensor_encoded = p3.encode(physical, channel_axis=1)
    numpy_encoded = p3.encode(physical[0].numpy(), channel_axis=0)
    assert torch.isfinite(tensor_encoded).all()
    assert np.isfinite(numpy_encoded).all()


def test_stage_o_shell_channels_are_input_only(stage_o_config):
    processor = PaperDataProcessor.from_config(stage_o_config)
    with h5py.File(ROOT / "data_proc/grmhd_regrid_inner_r200_64.h5", "r") as handle:
        source = torch.from_numpy(np.asarray(handle["snapshots"][11], dtype=np.float32)).unsqueeze(0)
        target = torch.from_numpy(np.asarray(handle["snapshots"][12], dtype=np.float32)).unsqueeze(0)
        times = np.asarray(handle["times"][...], dtype=np.float64)
    sample = processor.preprocess(
        {
            "physical_input": source,
            "physical_target": target,
            "source_index": torch.tensor([11]),
            "target_index": torch.tensor([12]),
            "time": torch.tensor([times[11]]),
            "target_time": torch.tensor([times[12]]),
        }
    )
    assert sample["x"].shape == (1, 16, 64, 64, 64)
    assert sample["y"].shape == (1, 8, 64, 64, 64)
    assert torch.equal(sample["x"][:, 8:], processor.shells)


def test_legacy_detector_remains_available():
    state = torch.arange(1 * 8 * 4 * 4 * 4, dtype=torch.float32).reshape(
        1, 8, 4, 4, 4
    )
    result = artifact_diagnostics(
        state, state, state, reference_kind="ground_truth"
    )
    assert result["flags"] == []
    assert set(result["channels"]) == {
        "Bcc1", "Bcc2", "Bcc3", "rho", "press", "vel1", "vel2", "vel3"
    }


def _gate(**overrides):
    arguments = {
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
        "severe_by_step": {
            1: {
                "global_variance": False,
                "shell_radial_variance": False,
                "dynamic_span": False,
                "high_k_energy": False,
            }
        },
        "shell_transport_skill": 0.1,
        "radial_transport_skill": 0.1,
        "shell_sign_agreement": 1.0,
        "radial_sign_agreement": 1.0,
    }
    arguments.update(overrides)
    return evaluate_oracle_conditioned_structure_gate(**arguments)


def test_stage_m_v1_schema_and_gate_zero():
    schema = oracle_conditioned_gate_schema()
    assert schema["oracle_conditioned_gate_version"] == "stage_m_v1"
    assert _gate()["gate_0_engineering_validity"]["passed"] is True


def test_stage_o_gate_one_uses_p3_oracle_floor():
    gate = _gate(
        floor_metrics={
            "oracle_legacy_detector_triggered": True,
            "median_variance_retention": 0.1,
            "median_shell_radial_retention": 0.1,
            "median_high_k_retention": 0.1,
        }
    )
    assert gate["gate_1_floor_qualification"]["floor_limited"] is True


def test_stage_o_gate_two_uses_model_vs_p3_oracle():
    severe = {
        step: {
            "global_variance": True,
            "shell_radial_variance": True,
            "dynamic_span": False,
            "high_k_energy": False,
        }
        for step in (1, 3)
    }
    assert _gate(severe_by_step=severe)["gate_2_model_added_degradation"]["failed"]


def test_stage_o_gate_three_uses_p3_persistence_transport():
    gate = _gate(shell_transport_skill=0.2, radial_transport_skill=0.0)
    assert gate["gate_3_transport_skill"]["passed"] is True


def test_stage_k_canonical_config_still_loads_without_stage_o():
    config = load_paper_experiment_config(STAGE_K_CONFIG, project_root=ROOT)
    assert config.stage_k is True and config.stage_o is False
    assert config.values["preprocessing"]["mode"] == "canonical_paper"


def test_stage_k_through_n_decisions_are_not_overwritten():
    k = json.loads((ROOT / "outputs/paper_reduced100/stage_k/stage_k_decision.json").read_text())
    m = json.loads((ROOT / "outputs/paper_reduced100/stage_m/stage_m_decision.json").read_text())
    n = json.loads((ROOT / "outputs/paper_reduced100/stage_n/stage_n_decision.json").read_text())
    assert k["choice"] == "C"
    assert k["decision"] == "TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE"
    assert m["localno_transport_decision"] == "4. MIXED_TRANSPORT_AND_PREPROCESSING_FAILURE"
    assert n["combined_prototype_decision"] == "1. COMBINED_PROTOTYPE_READY_FOR_SHORT_PILOT"


def test_stage_o_has_no_validation_tuning(stage_o_config):
    values = stage_o_config.values
    assert values["provenance"]["validation_not_used_for_fit"] is True
    assert values["preprocessing"]["validation_used_for_fit"] is False
    assert values["scheduler"]["actual_default_epochs"] == 30
    assert "early_stopping" not in values or not values.get("early_stopping")


def test_stage_o_rejects_architecture_and_disco_overrides(tmp_path):
    raw = yaml.safe_load(CONFIG.read_text())
    raw["extends"] = str(STAGE_K_CONFIG)
    raw["model"] = {"use_disco": True, "prediction_mode": "residual"}
    path = tmp_path / "stage_o_bad_model.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="exactly the frozen P3 overrides"):
        load_paper_experiment_config(path, project_root=ROOT)


def test_no_stage_o_checkpoint_is_tracked():
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    forbidden = (
        "outputs/paper_reduced100/stage_o/localno_p3_plain/best_validation_l2/paper_state_dict.pt",
        "outputs/paper_reduced100/stage_o/localno_p3_plain/selected_states.pt",
    )
    assert not any(path in tracked for path in forbidden)


def test_physical_rollout_encode_decode_counter(stage_o_config):
    processor = PaperDataProcessor.from_config(stage_o_config)
    with h5py.File(ROOT / "data_proc/grmhd_regrid_inner_r200_64.h5", "r") as handle:
        state = torch.from_numpy(np.asarray(handle["snapshots"][91], dtype=np.float32)).unsqueeze(0)
    before = dict(processor.transform_counts)
    encoded = processor.encode_rollout_input(state)
    decoded = processor.decode_prediction(
        encoded["normalized_input"], apply_evaluation_clamp=False
    )
    delta = {
        key: processor.transform_counts[key] - before[key]
        for key in before
    }
    assert delta == {
        "input_encode": 1,
        "target_encode": 0,
        "oracle_decode": 0,
        "prediction_decode": 1,
    }
    assert torch.isfinite(decoded.physical_prediction).all()


def test_p3_channel_policy_is_exact():
    spec = prototype_specs(NO_SOFTCLIP)["P3"]
    assert dict(zip(("Bcc1", "Bcc2", "Bcc3", "rho", "press", "vel1", "vel2", "vel3"), spec.channel_policies)) == {
        "Bcc1": "canonical",
        "Bcc2": "no_softclip",
        "Bcc3": "no_softclip",
        "rho": "canonical",
        "press": "canonical",
        "vel1": "canonical",
        "vel2": "canonical",
        "vel3": "no_softclip",
    }
