from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from grmhd import CHANNELS
from grmhd.dataset import sha256_file
from grmhd.paper_config import build_paper_model, load_paper_experiment_config
from grmhd.paper_stage_o import load_frozen_p3
from grmhd.paper_stage_p import (
    FROZEN_HISTORY,
    decoder_derivative_numpy,
    directional_gain,
    distribution_summary,
    driver_support,
    final_mechanism_decision,
    finite_difference_decoder_derivative,
    first_failure,
    mechanism_label,
    load_stage_o_model_only,
    normalized_direct_step,
    ood_diagnostics,
    project_train_envelope,
    relative_l2,
    reset_channels,
    roundtrip_only,
    teacher_forced_predictions,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/paper_reduced100/stage_o_localno_p3_plain.yaml"
EXPERIMENT = ROOT / "outputs/paper_reduced100/stage_o/localno_p3_plain"


@pytest.fixture(scope="module")
def config():
    return load_paper_experiment_config(CONFIG_PATH, project_root=ROOT)


@pytest.fixture(scope="module")
def p3(config):
    return load_frozen_p3(config)


@pytest.mark.parametrize("name,epoch", [("best_validation_l2", 23), ("last", 30)])
def test_stage_o_trajectory_checkpoint_strict_model_only_reload(config, name, epoch):
    model, loaded_epoch, metadata = load_stage_o_model_only(
        EXPERIMENT / name,
        config=config,
        model=build_paper_model(config),
        expected_config_checksum=sha256_file(CONFIG_PATH),
    )
    assert loaded_epoch == epoch
    assert metadata["epoch"] == epoch
    assert model.training is False


def test_train_envelope_source_has_no_validation_leakage(config):
    metadata = json.loads(
        (ROOT / "outputs/paper_reduced100/stage_n/prototypes/p3_combined_v1/normalizer.json").read_text()
    )
    assert metadata["training_indices"] == list(range(11, 91))
    assert config.values["preprocessing"]["validation_used_for_fit"] is False


def test_decoder_analytic_derivative_matches_finite_difference(p3):
    state = np.zeros((8, 2, 2, 2), dtype=np.float64)
    analytic = decoder_derivative_numpy(p3, state, channel_axis=0)
    finite = finite_difference_decoder_derivative(
        p3, state, epsilon=1e-5, channel_axis=0
    )
    for name in ("Bcc2", "Bcc3", "vel3"):
        channel = CHANNELS.index(name)
        assert np.allclose(analytic[channel], finite[channel], rtol=2e-4, atol=1e-8)


def test_normalized_output_ood_fraction():
    envelope = {"q001": -1.0, "q999": 1.0, "minimum": -2.0, "maximum": 2.0}
    result = ood_diagnostics(np.array([-3.0, -0.5, 0.5, 3.0]), envelope)
    assert result["fraction_outside_train_q001_q999"] == 0.5
    assert result["fraction_outside_train_min_max"] == 0.5
    assert result["maximum_normalized_excess"] == 2.0


def test_decode_reencode_discrepancy_is_measured(p3):
    state = np.zeros((8, 2, 2, 2), dtype=np.float32)
    state[0] = 8.0
    feedback = roundtrip_only(p3, state, applications=1)
    assert relative_l2(feedback, state) > 0
    assert np.max(np.abs(feedback - state)) > 0


def test_first_failure_ordering():
    records = [{"step": 1, "failed": False}, {"step": 2, "failed": True}]
    assert first_failure(records, "failed") == 2
    assert first_failure(records, "missing") is None


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(self, *, x):
        self.inputs.append(x.detach().clone())
        return x[:, :8] + 1.0


def test_teacher_forced_predictions_do_not_feedback():
    model = RecordingModel()
    states = [torch.zeros(1, 8, 2, 2, 2), torch.full((1, 8, 2, 2, 2), 5.0)]
    shells = torch.full((1, 8, 2, 2, 2), 9.0)
    teacher_forced_predictions(model, states, shells)
    assert torch.equal(model.inputs[1][:, :8], states[1])


def test_normalized_direct_does_not_call_decode():
    model = RecordingModel()
    state = torch.zeros(1, 8, 2, 2, 2)
    shells = torch.ones_like(state)
    output = normalized_direct_step(model, state, shells)
    assert torch.equal(output, torch.ones_like(state))


def test_roundtrip_only_does_not_require_model(p3):
    state = np.zeros((8, 1, 1, 1), dtype=np.float32)
    assert roundtrip_only(p3, state, applications=2).shape == state.shape


def test_projection_uses_supplied_train_bounds_only():
    state = torch.full((1, 8, 1, 1, 1), 10.0)
    envelope = {name: {"q001": -1.0, "q999": 2.0} for name in CHANNELS}
    projected = project_train_envelope(state, envelope, ("Bcc2",))
    assert float(projected[:, 1]) == 2.0
    assert float(projected[:, 2]) == 10.0


def test_channel_reset_only_replaces_selected_channel():
    prediction = torch.zeros(1, 8, 1, 1, 1)
    target = torch.arange(8.0).reshape(1, 8, 1, 1, 1)
    result = reset_channels(prediction, target, ("Bcc3",))
    assert float(result[:, 2]) == 2.0
    assert torch.count_nonzero(result) == 1


def test_shell_channels_remain_fixed():
    model = RecordingModel()
    shells = torch.randn(1, 8, 2, 2, 2)
    normalized_direct_step(model, torch.zeros_like(shells), shells)
    normalized_direct_step(model, torch.ones_like(shells), shells)
    assert torch.equal(model.inputs[0][:, 8:], model.inputs[1][:, 8:])


def test_local_gain_never_uses_backward():
    parameter = torch.nn.Parameter(torch.tensor(2.0))

    def mapping(value):
        return value * parameter

    state = torch.ones(1, 8, 1, 1, 1)
    result = directional_gain(mapping, state, torch.ones_like(state), epsilon=1e-3)
    assert result["used_backward"] is False
    assert parameter.grad is None
    assert result["gain"] == pytest.approx(2.0, rel=1e-4)


def test_local_gain_epsilon_consistency():
    state = torch.ones(1, 8, 1, 1, 1)
    direction = torch.ones_like(state)
    low = directional_gain(lambda x: 3 * x, state, direction, epsilon=1e-3)
    high = directional_gain(lambda x: 3 * x, state, direction, epsilon=1e-2)
    assert low["gain"] == pytest.approx(high["gain"], rel=1e-4)


def test_driver_primary_rule_requires_four_supported_observations():
    assert driver_support([0.5, 0.6, 0.7, 0.8]) == "supported_primary_driver"
    assert driver_support([0.5, 0.6, 0.7]) == "not_supported"


@pytest.mark.parametrize("mechanism", ["M1", "M2", "M3", "M4", "M5"])
def test_single_mechanism_toy_decisions(mechanism):
    mechanisms = {name: "not_supported" for name in ("M1", "M2", "M3", "M4", "M5")}
    mechanisms[mechanism] = mechanism_label(supported=2)
    assert final_mechanism_decision(mechanisms).startswith(mechanism[-1])


def test_mixed_mechanism_toy_decision():
    mechanisms = {"M1": "supported", "M2": "supported", "M3": "not_supported"}
    assert final_mechanism_decision(mechanisms) == "6. MIXED_CLOSED_LOOP_FAILURE"


def test_frozen_stage_k_o_decisions_cannot_be_overwritten():
    assert FROZEN_HISTORY["Stage K"].startswith("C.")
    assert FROZEN_HISTORY["Stage O"] == "C. P3_PILOT_COMPLETE_BUT_ROLLOUT_UNSTABLE"


def test_strict_reload_does_not_write_checkpoint(config):
    path = EXPERIMENT / "best_validation_l2/paper_state_dict.pt"
    before = (sha256_file(path), path.stat().st_mtime_ns)
    model, _, _ = load_stage_o_model_only(
        path.parent,
        config=config,
        model=build_paper_model(config),
        expected_config_checksum=sha256_file(CONFIG_PATH),
    )
    assert model.training is False
    assert (sha256_file(path), path.stat().st_mtime_ns) == before


def test_model_only_reload_constructs_no_optimizer_or_scheduler(config, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Stage P must not construct optimizer/scheduler state")

    monkeypatch.setattr(torch.optim, "Adam", forbidden)
    monkeypatch.setattr(torch.optim, "AdamW", forbidden)
    monkeypatch.setattr(torch.optim.lr_scheduler, "LambdaLR", forbidden)
    model, epoch, _ = load_stage_o_model_only(
        EXPERIMENT / "best_validation_l2",
        config=config,
        model=build_paper_model(config),
        expected_config_checksum=sha256_file(CONFIG_PATH),
    )
    assert epoch == 23
    assert model.training is False


def test_distribution_summary_quantiles_are_ordered():
    summary = distribution_summary(np.arange(1000.0))
    assert summary["minimum"] <= summary["q001"] < summary["q999"] <= summary["maximum"]
