from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from grmhd.paper_preprocessing import PaperPreprocessor
from grmhd.paper_stage_g_evaluation import artifact_diagnostics
from grmhd.paper_stage_l_attribution import SPECTRAL_BANDS
from grmhd.paper_stage_m import (
    evaluate_oracle_conditioned_structure_gate,
    oracle_conditioned_gate_schema,
)
from grmhd.paper_stage_n import (
    CANONICAL,
    MINIMAL_INVERSE_CLAMP,
    NO_SOFTCLIP,
    PrototypePreprocessor,
    assert_train_only_indices,
    channel_prototype_decision,
    classify_readiness,
    combined_prototype_decision,
    compact_impulse,
    constant_perturbation,
    directional_mode,
    finite_difference_jvp,
    minimal_inverse_clamp_limit,
    prototype_specs,
    radial_mode,
    response_matrix,
    sha256_json,
    shell_localized_perturbation,
    tensor_state_sha256,
    two_application_probe,
)


ROOT = Path(__file__).resolve().parents[1]


def base() -> PaperPreprocessor:
    return PaperPreprocessor(
        epsilon=np.array([1e-3, 1e-3, 1e-2, 1e-8, 1e-10, 0, 0, 0]),
        median=np.zeros(8), scale=np.ones(8),
        training_indices=range(11, 91), source_hdf5_checksum="test",
        protocol_name="stage_n_test", thermal_channel="press", paper_adaptation=True,
        eos_conversion="disabled_unverified_gamma",
    )


def fields() -> np.ndarray:
    value = np.linspace(-0.1, 0.1, 8 * 4 * 4 * 4, dtype=np.float32).reshape(8, 4, 4, 4)
    value[3:5] = np.abs(value[3:5]) + 1e-4
    return value


class IdentityPhysical(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :8]


class KeywordConv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv3d(16, 8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def state() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.randn(1, 8, 4, 4, 8), torch.zeros(1, 8, 4, 4, 8)


def gate_kwargs() -> dict:
    return {
        "legacy_detector": {"flags": []},
        "engineering_checks": {name: True for name in ("finite", "rho_press_positive", "transform_counters", "checkpoint_provenance", "decoded_range", "Rout", "shape_device")},
        "floor_metrics": {"oracle_legacy_detector_triggered": False, "median_variance_retention": 0.9, "median_shell_radial_retention": 0.9, "median_high_k_retention": 0.9},
        "severe_by_step": {step: {"global_variance": False, "shell_radial_variance": False, "dynamic_span": False, "high_k_energy": False} for step in (1, 5)},
        "shell_transport_skill": 0.1, "radial_transport_skill": 0.2,
        "shell_sign_agreement": 0.8, "radial_sign_agreement": 0.8,
    }


def test_prototype_metadata_and_policies():
    specs = prototype_specs(MINIMAL_INVERSE_CLAMP)
    assert specs["P0"].channel_policies == (CANONICAL,) * 8
    assert specs["P1"].channel_policies[2] == specs["P1"].channel_policies[7] == NO_SOFTCLIP
    assert specs["P3"].as_dict()["authorized_for_training"] is False


def test_canonical_prototype_parity():
    raw = fields()
    prototype = PrototypePreprocessor(base(), prototype_specs()["P0"])
    np.testing.assert_array_equal(prototype.encode_numpy(raw), base().encode_numpy(raw))
    np.testing.assert_array_equal(prototype.round_trip(raw)[1], base().decode_numpy(base().encode_numpy(raw)))


def test_bcc3_vel3_no_softclip_is_finite_and_reversible():
    raw = fields()
    prototype = PrototypePreprocessor(base(), prototype_specs()["P1"])
    encoded, decoded = prototype.round_trip(raw)
    assert np.isfinite(encoded).all() and np.isfinite(decoded).all()
    np.testing.assert_allclose(decoded[[2, 7]], raw[[2, 7]], rtol=1e-6, atol=1e-7)


def test_no_softclip_avoids_illegal_exact_saturation_inverse():
    raw = fields()
    raw[2] *= 1e8
    prototype = PrototypePreprocessor(base(), prototype_specs()["P1"])
    encoded, decoded = prototype.round_trip(raw)
    assert np.max(np.abs(encoded[2])) > 6 and np.isfinite(decoded[2]).all()


def test_bcc2_minimal_clamp_is_strictly_inside_gamma_and_finite():
    limit = minimal_inverse_clamp_limit()
    assert limit < 6 and np.nextafter(np.float32(limit), np.float32(np.inf)) == np.float32(6)
    prototype = PrototypePreprocessor(base(), prototype_specs()["P2A"])
    encoded = np.zeros((8, 4, 4, 4), dtype=np.float32)
    encoded[1, 0] = 6
    assert np.isfinite(prototype.decode_numpy(encoded)).all()


def test_bcc2_no_softclip_is_finite():
    prototype = PrototypePreprocessor(base(), prototype_specs()["P2B"])
    encoded, decoded = prototype.round_trip(fields())
    assert np.isfinite(encoded).all() and np.isfinite(decoded).all()


def test_train_only_statistics_reject_validation_indices():
    assert assert_train_only_indices(range(11, 91))[0] == 11
    with pytest.raises(ValueError, match="train snapshots"):
        assert_train_only_indices(range(11, 92))


def test_prototype_wrapper_accepts_a_separately_validated_train_only_split():
    expanded = PaperPreprocessor(
        epsilon=np.array([1e-3, 1e-3, 1e-2, 1e-8, 1e-10, 0, 0, 0]),
        median=np.zeros(8), scale=np.ones(8),
        training_indices=range(0, 169), source_hdf5_checksum="expanded-test",
        protocol_name="stage_s_expanded_p3", thermal_channel="press",
        paper_adaptation=True, eos_conversion="disabled_unverified_gamma",
    )
    prototype = PrototypePreprocessor(expanded, prototype_specs(NO_SOFTCLIP)["P3"])
    assert prototype.training_indices == tuple(range(169))


def test_prototype_normalizer_is_isolated_from_canonical():
    canonical = base()
    prototype = PrototypePreprocessor(canonical, prototype_specs()["P1"])
    before = canonical.median.copy()
    metadata = prototype.normalizer_metadata()
    metadata["median"][0] = 10
    np.testing.assert_array_equal(canonical.median, before)


def test_prototype_config_hash_is_deterministic():
    value = prototype_specs()["P1"].as_dict()
    assert sha256_json(value) == sha256_json(json.loads(json.dumps(value)))


def test_readiness_classification_contract():
    summary = {"all_finite": True, "median_raw_to_oracle_relative_l2": 0.1, "median_exact_saturation_occupancy": 0.0, **{f"median_{key}": 0.9 for key in ("global_variance_retention", "shell_radial_variance_retention", "high_k_retention", "dynamic_span_retention")}}
    canonical = {**summary, "median_raw_to_oracle_relative_l2": 0.3}
    result = classify_readiness(target_summary=summary, canonical_target_summary=canonical, control_summaries=[(canonical, canonical)])
    assert result["ready"] and result["core_passes"] == 4


def test_combined_prototype_does_not_modify_canonical_spec():
    before = prototype_specs()["P0"]
    combined = prototype_specs(MINIMAL_INVERSE_CLAMP)["P3"]
    assert before.channel_policies == (CANONICAL,) * 8
    assert combined.channel_policies[1] == MINIMAL_INVERSE_CLAMP


def test_constant_perturbation_only_touches_selected_channel():
    x, _ = state()
    delta = constant_perturbation(x, 3, 1e-3)
    assert torch.count_nonzero(delta[:, 3]) == delta[:, 3].numel()
    assert torch.count_nonzero(delta[:, :3]) == 0


def test_positive_negative_impulse_symmetry():
    x, _ = state()
    positive = compact_impulse(x, 1, (2, 2, 4), 1e-2)
    negative = compact_impulse(x, 1, (2, 2, 4), -1e-2)
    torch.testing.assert_close(positive, -negative)


def test_shell_localized_perturbation_shape_and_zero_mean():
    x, _ = state()
    mask = torch.zeros(4, 4, 8, dtype=torch.bool)
    mask[..., :2] = True
    delta = shell_localized_perturbation(x, mask, 0, 1e-2)
    assert delta.shape == x.shape
    assert abs(float(delta[:, 0][mask.unsqueeze(0)].mean())) < 1e-6


@pytest.mark.parametrize("name", ["constant", "linear_log_r", "inner_localized", "outer_localized", "mid_frequency"])
def test_radial_mode_shape(name):
    assert radial_mode((4, 5, 8), name).shape == (4, 5, 8)


def test_directional_modes_reuse_frozen_frequency_bands():
    assert set(SPECTRAL_BANDS) == {"low_k", "mid_k", "high_k"}
    assert directional_mode((16, 16, 16), "theta", "high_k").shape == (16, 16, 16)


def test_channel_response_matrix_shape():
    x, _ = state()
    directions = [constant_perturbation(x, channel, 1e-2) for channel in range(8)]
    matrix = response_matrix(directions, directions)
    assert matrix.shape == (8, 8) and np.all(np.diag(matrix) == 1)


def test_finite_difference_epsilon_consistency_for_identity():
    x, shells = state()
    direction = constant_perturbation(x, 2, 1.0)
    model = IdentityPhysical().eval()
    small = finite_difference_jvp(model, x, shells, direction, 1e-3)
    large = finite_difference_jvp(model, x, shells, direction, 1e-2)
    torch.testing.assert_close(small, large, rtol=2e-4, atol=2e-4)


def test_two_application_probe_limit():
    x, shells = state()
    first, second = two_application_probe(IdentityPhysical(), x, shells)
    torch.testing.assert_close(first, second)
    with pytest.raises(ValueError, match="exactly two"):
        two_application_probe(IdentityPhysical(), x, shells, applications=3)


def test_operator_probe_changes_no_parameters_and_creates_no_gradients():
    model = KeywordConv().eval()
    x, shells = state()
    before = tensor_state_sha256(model)
    with torch.no_grad():
        two_application_probe(model, x, shells)
    assert tensor_state_sha256(model) == before
    assert all(parameter.grad is None for parameter in model.parameters())


def test_tensor_state_hash_ignores_non_tensor_state_metadata():
    class MetadataModule(IdentityPhysical):
        def state_dict(self, *args, **kwargs):
            return {"tensor": torch.ones(1), "metadata": {"shape": [1]}}

    assert tensor_state_sha256(MetadataModule()) == tensor_state_sha256(MetadataModule())


def test_operator_script_does_not_chain_data_processor_eval():
    source = (ROOT / "scripts/analyze_paper_stage_n_operator.py").read_text()
    assert "PaperDataProcessor.from_config(config).to(device).eval()" not in source
    assert "directional_rows, constant_rows," in source


def test_operator_script_has_no_optimizer_scheduler_or_backward_call():
    source = (ROOT / "scripts/analyze_paper_stage_n_operator.py").read_text()
    assert "torch.optim" not in source
    assert ".backward(" not in source
    assert "optimizer.step(" not in source
    assert "scheduler.step(" not in source


def test_candidate_gate_schema_and_reporting_only_default():
    schema = oracle_conditioned_gate_schema()
    result = evaluate_oracle_conditioned_structure_gate(**gate_kwargs())
    assert schema["oracle_conditioned_gate_version"] == result["oracle_conditioned_gate_version"] == "stage_m_v1"
    assert result["reporting_only"] and not result["training_blocking"]


def test_candidate_gate_cannot_become_training_blocker():
    with pytest.raises(ValueError, match="reporting-only"):
        evaluate_oracle_conditioned_structure_gate(**gate_kwargs(), training_blocking=True)


def test_candidate_gate_preserves_legacy_payload():
    kwargs = gate_kwargs()
    original = json.loads(json.dumps(kwargs["legacy_detector"]))
    result = evaluate_oracle_conditioned_structure_gate(**kwargs)
    assert result["legacy_detector"] == original and kwargs["legacy_detector"] == original


def test_legacy_evaluation_implementation_has_no_regression():
    digest = hashlib.sha256(inspect.getsource(artifact_diagnostics).encode()).hexdigest()
    assert digest == "3a9ea391b663df6b58a0c75942283658ee0214669d234e2c5a9594cc5f36e95d"


def test_stage_k_l_m_decisions_cannot_be_overwritten():
    k = json.loads((ROOT / "outputs/paper_reduced100/stage_k/stage_k_decision.json").read_text())
    l = json.loads((ROOT / "outputs/paper_reduced100/stage_l/stage_l_decision.json").read_text())
    m = json.loads((ROOT / "outputs/paper_reduced100/stage_m/stage_m_decision.json").read_text())
    assert (k["choice"], k["decision"]) == ("C", "TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE")
    assert l["overall_decision"] == "3. MIXED_OVERALL"
    assert m["localno_transport_decision"] == "4. MIXED_TRANSPORT_AND_PREPROCESSING_FAILURE"


def test_channel_and_combined_decision_rules():
    validation = {"all_finite": True, "median_raw_to_oracle_relative_l2": 0.1, **{f"median_{key}": 0.9 for key in ("global_variance_retention", "shell_radial_variance_retention", "high_k_retention", "dynamic_span_retention")}}
    assert channel_prototype_decision(train_ready=True, validation_summary=validation, canonical_validation_l2=0.3).startswith("A.")
    assert combined_prototype_decision({"Bcc2": "A. PROTOTYPE_READY", "Bcc3": "A. PROTOTYPE_READY", "vel3": "A. PROTOTYPE_READY"}, p3_finite=True).startswith("1.")
