from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import torch
import yaml

import grmhd.paper_stage_q as stage_q
from grmhd.paper_stage_m import oracle_conditioned_gate_schema
from grmhd.paper_stage_q import (
    ALPHA_GRID,
    CANDIDATE_ALPHAS,
    FROZEN_HISTORY,
    anchored_directional_gain,
    anchored_model_step,
    anchored_output,
    apply_anchored_map,
    expected_transform_counts,
    model_state_sha256,
    readiness_predicates,
    select_candidate_alpha,
    validate_rollout_steps,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/paper_reduced100/stage_q_residual_anchor_audit.yaml"


@pytest.fixture(scope="module")
def qconfig():
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def states():
    z = torch.linspace(-1.0, 1.0, 64).reshape(1, 8, 2, 2, 2)
    direct = 2.0 * z + 0.25
    return z, direct


def test_alpha_zero_is_exact_persistence(states):
    z, direct = states
    assert torch.equal(anchored_output(z, direct, 0.0), z)


def test_alpha_one_is_exact_direct(states):
    z, direct = states
    assert torch.equal(anchored_output(z, direct, 1.0), direct)


@pytest.mark.parametrize("alpha", ALPHA_GRID)
def test_affine_identity(states, alpha):
    z, direct = states
    expected = (1.0 - alpha) * z + alpha * direct
    assert torch.allclose(anchored_output(z, direct, alpha), expected)


@pytest.mark.parametrize("alpha", ALPHA_GRID)
def test_fixed_point_preservation(alpha):
    z = torch.randn(1, 8, 2, 2, 2)
    assert torch.equal(anchored_output(z, z, alpha), z)


@pytest.mark.parametrize("alpha", ALPHA_GRID)
def test_residual_scaling(states, alpha):
    z, direct = states
    output = anchored_output(z, direct, alpha)
    assert torch.allclose(output - z, alpha * (direct - z))


def test_pairwise_alpha_residual_ordering(states):
    z, direct = states
    norms = [torch.linalg.vector_norm(anchored_output(z, direct, a) - z) for a in ALPHA_GRID]
    assert all(left <= right for left, right in zip(norms, norms[1:]))


class RecordingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.inputs: list[torch.Tensor] = []

    def forward(self, *, x):
        self.inputs.append(x.detach().clone())
        return self.weight * x[:, :8]


def test_shell_channels_are_input_only_and_untouched():
    model = RecordingModel()
    z = torch.randn(1, 8, 2, 2, 2)
    shells = torch.randn_like(z)
    before = shells.clone()
    direct, output = anchored_model_step(model, z, shells, alpha=0.25)
    assert direct.shape == output.shape == z.shape
    assert torch.equal(shells, before)
    assert torch.equal(model.inputs[0][:, 8:], shells)


def test_output_contract_has_no_hidden_clamp_or_transform():
    source = inspect.getsource(anchored_output)
    for forbidden in ("clip(", "clamp(", "nan_to_num", "decoder", "encoder", "bounds"):
        assert forbidden not in source


def test_no_parameter_or_buffer_mutation(states):
    model = RecordingModel()
    z, _ = states
    shells = torch.zeros_like(z)
    before = model_state_sha256(model)
    anchored_model_step(model, z, shells, alpha=0.5)
    assert model_state_sha256(model) == before
    assert model.weight.grad is None


def _summary(alpha: float, *, passing: bool = True):
    scale = 1.0 if passing else 100.0
    return {
        "alpha": alpha,
        "engineering": {
            "all_finite": True,
            "rho_press_positive": True,
            "transform_counter_errors": 0,
            "new_nonfinite_decoder_derivatives": 0,
        },
        "normalized_q_ood_fraction_median": 0.01 * scale,
        "rout_failure_count": 1 if passing else 79,
        "normalized_average": 1.0 * scale,
        "residual_over_true_residual_median": 0.2,
        "residual_cosine_channel_medians": {name: 0.2 for name in stage_q.CHANNELS},
        "persistence_equivalent": False,
        "gate_2_severe_count": 1 if passing else 100,
        "shell_transport_skill_median": 0.1,
        "radial_transport_skill_median": 0.1,
        "per_channel_relative_l2": {name: 1.0 for name in stage_q.CHANNELS},
    }


def _controls():
    persistence = _summary(0.0)
    persistence["normalized_average"] = 1.0
    persistence["normalized_q_ood_fraction_median"] = 0.0
    persistence["rout_failure_count"] = 0
    persistence["gate_2_severe_count"] = 0
    persistence["shell_transport_skill_median"] = 0.0
    persistence["radial_transport_skill_median"] = 0.0
    direct = _summary(1.0, passing=False)
    direct["normalized_q_ood_fraction_median"] = 0.1
    direct["rout_failure_count"] = 79
    direct["gate_2_severe_count"] = 100
    return persistence, direct


def test_train_only_alpha_selection_has_no_validation_argument(qconfig):
    assert "validation" not in inspect.signature(select_candidate_alpha).parameters
    persistence, direct = _controls()
    candidate = _summary(0.25)
    ready = readiness_predicates(candidate, persistence=persistence, direct=direct, config=qconfig)
    result = select_candidate_alpha(
        {0.0: persistence, 0.25: candidate, 1.0: direct},
        {0.125: {"passed": False}, 0.25: ready, 0.5: {"passed": False}},
        config=qconfig,
    )
    assert result["candidate_alpha"] == 0.25
    assert result["validation_used"] is False


@pytest.mark.parametrize("group", tuple("ABCDEF"))
def test_readiness_groups_a_through_f_pass(group, qconfig):
    persistence, direct = _controls()
    result = readiness_predicates(
        _summary(0.25), persistence=persistence, direct=direct, config=qconfig
    )
    assert result["group_pass"][group]


def test_tie_break_selects_larger_alpha_within_one_percent(qconfig):
    summaries = {alpha: _summary(alpha) for alpha in (0.125, 0.25, 0.5)}
    summaries[0.125]["normalized_average"] = 1.000
    summaries[0.25]["normalized_average"] = 1.005
    summaries[0.5]["normalized_average"] = 1.009
    readiness = {alpha: {"passed": True} for alpha in summaries}
    result = select_candidate_alpha(summaries, readiness, config=qconfig)
    assert result["candidate_alpha"] == 0.5


def test_no_candidate_behavior(qconfig):
    summaries = {alpha: _summary(alpha) for alpha in CANDIDATE_ALPHAS}
    readiness = {alpha: {"passed": False} for alpha in CANDIDATE_ALPHAS}
    result = select_candidate_alpha(summaries, readiness, config=qconfig)
    assert result["candidate_alpha"] is None
    assert result["validation_used"] is False


@pytest.mark.parametrize("alpha", (0.0, 0.25, 1.0))
def test_local_gain_analytic_numeric_parity(alpha):
    state = torch.ones(1, 8, 1, 1, 1)
    direction = torch.full_like(state, 0.5)
    result = anchored_directional_gain(
        lambda value: 3.0 * value,
        state,
        direction,
        alpha=alpha,
        epsilon=1.0e-3,
    )
    assert result["analytic_numeric_relative_difference"] < 1.0e-4
    assert result["gain"] == pytest.approx(1.0 + 2.0 * alpha, rel=1.0e-4)
    assert result["used_backward"] is False


def test_local_gain_rejects_unfrozen_epsilon():
    state = torch.ones(1, 8, 1, 1, 1)
    with pytest.raises(ValueError, match="epsilon"):
        anchored_directional_gain(
            lambda value: value,
            state,
            state,
            alpha=0.25,
            epsilon=1.0e-4,
        )


def test_two_application_limit():
    state = torch.ones(1, 8, 1, 1, 1)
    outputs = apply_anchored_map(lambda value: 2 * value, state, alpha=0.25, applications=2)
    assert len(outputs) == 2
    with pytest.raises(ValueError, match="at most two"):
        apply_anchored_map(lambda value: value, state, alpha=0.25, applications=3)


def test_rollout_maximum_is_19():
    assert validate_rollout_steps(19) == 19
    with pytest.raises(ValueError, match="19"):
        validate_rollout_steps(20)


def test_exact_transform_counters():
    assert expected_transform_counts(19) == {
        "prediction_decode": 19,
        "next_input_encode": 19,
    }


def test_stage_m_v1_is_unchanged():
    schema = oracle_conditioned_gate_schema()
    assert schema["oracle_conditioned_gate_version"] == "stage_m_v1"
    assert schema["reporting_only"] is True


def test_stage_k_through_p_decisions_are_frozen():
    assert set(FROZEN_HISTORY) == {"Stage K", "Stage L", "Stage M", "Stage N", "Stage O", "Stage P"}
    assert FROZEN_HISTORY["Stage O"].startswith("C.")
    assert FROZEN_HISTORY["Stage P"] == "6. MIXED_CLOSED_LOOP_FAILURE"


def test_module_constructs_no_training_state_or_backward():
    source = inspect.getsource(stage_q)
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "lr_scheduler" not in source


def test_module_writes_no_checkpoint():
    source = inspect.getsource(stage_q)
    assert "torch.save" not in source
    assert "save_state_dict" not in source
