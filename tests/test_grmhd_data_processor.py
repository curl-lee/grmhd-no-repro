from __future__ import annotations

import torch

from grmhd.data_processor import GRMHDDataProcessor
from grmhd.hybrid import HybridTargetStats, POSITIVE_CHANNELS, SIGNED_CHANNELS
from grmhd.normalizer import GRMHDNormalizer


def _normalizer() -> GRMHDNormalizer:
    return GRMHDNormalizer(
        median=torch.zeros(8).numpy(),
        scale=torch.ones(8).numpy(),
        epsilon=torch.tensor([1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 0, 0, 0]).numpy(),
        gamma=6.0,
    )


def _hybrid() -> HybridTargetStats:
    return HybridTargetStats(
        scale=(0.5, 0.6, 0.7, 0.2, 0.3, 0.8, 0.9, 1.0),
        alpha=(3.0,) * 8,
        epsilon=(0.0, 0.0, 0.0, 1e-6, 2e-6, 0.0, 0.0, 0.0),
        alpha_quantile=0.999,
        min_scale=1e-12,
        min_alpha=1e-3,
        sample_count_per_channel=1,
        seed=42,
        source_hdf5_checksum="synthetic",
        training_indices=(0, 1),
        pair_starts=(0,),
        stats_checksum="synthetic",
    )


def _state(offset: float = 0.0) -> torch.Tensor:
    state = torch.full((1, 8, 4, 3, 2), 0.01 + offset)
    state[:, 0:3] -= 0.02
    state[:, 3] = 1.0 + offset
    state[:, 4] = 0.1 + 0.1 * offset
    return state


def test_zero_hybrid_residual_is_exact_physical_persistence():
    physical = _state()
    processor = GRMHDDataProcessor(
        normalizer=_normalizer(), target_mode="hybrid", hybrid_stats=_hybrid()
    )
    processor.eval()
    sample = processor.preprocess(
        {"physical_input": physical, "physical_target": _state(0.01)}
    )
    assert sample is not None
    prediction, processed = processor.postprocess(torch.zeros_like(physical), sample)
    torch.testing.assert_close(prediction, physical, rtol=0, atol=0)
    torch.testing.assert_close(processed["physical_prediction"], physical, rtol=0, atol=0)


def test_two_step_rollout_reencodes_physical_prediction_once_per_step():
    trajectory = torch.stack((_state()[0], _state(0.01)[0], _state(0.02)[0]), dim=0).unsqueeze(0)
    processor = GRMHDDataProcessor(
        normalizer=_normalizer(), target_mode="hybrid", hybrid_stats=_hybrid()
    )
    processor.eval()
    sample = {"physical_trajectory": trajectory, "trajectory_state": trajectory[:, 0]}
    first = processor.preprocess(sample, step=0)
    assert first is not None
    _, first = processor.postprocess(torch.zeros_like(first["physical_input"]), first, step=0)
    first_physical = first["trajectory_state"].clone()
    second = processor.preprocess(first, step=1)
    assert second is not None
    torch.testing.assert_close(second["physical_input"], first_physical, rtol=0, atol=0)
    expected_encoded = processor.normalizer.encode_tensor(first_physical, channel_axis=1)
    torch.testing.assert_close(second["encoded_state"], expected_encoded)
    _, second = processor.postprocess(torch.zeros_like(second["physical_input"]), second, step=1)
    assert processor.preprocess(second, step=2) is None
    assert processor.transform_counts == {
        "physical_input_encode": 2,
        "physical_target_encode": 0,
        "state_decode": 0,
        "hybrid_reconstruct": 2,
    }


def test_hybrid_signed_addition_and_positive_log_ratio_reconstruction():
    physical = _state()
    stats = _hybrid()
    raw = torch.full_like(physical, 0.2)
    bounded = stats.bounded_target_prediction(raw)
    prediction = stats.reconstruct(physical, raw)
    scale = torch.tensor(stats.scale).reshape(1, 8, 1, 1, 1)
    epsilon = torch.tensor(stats.epsilon).reshape(1, 8, 1, 1, 1)
    torch.testing.assert_close(
        prediction[:, SIGNED_CHANNELS],
        physical[:, SIGNED_CHANNELS] + scale[:, SIGNED_CHANNELS] * bounded[:, SIGNED_CHANNELS],
    )
    torch.testing.assert_close(
        prediction[:, POSITIVE_CHANNELS],
        (physical[:, POSITIVE_CHANNELS] + epsilon[:, POSITIVE_CHANNELS])
        * torch.exp(scale[:, POSITIVE_CHANNELS] * bounded[:, POSITIVE_CHANNELS])
        - epsilon[:, POSITIVE_CHANNELS],
    )
    assert torch.all(prediction[:, 3:5] > 0)


def test_shells_are_model_inputs_only_and_train_eval_spaces_are_distinct():
    physical = _state()
    target = _state(0.01)
    shells = torch.randn(8, 4, 3, 2)
    processor = GRMHDDataProcessor(
        normalizer=_normalizer(),
        shells=shells,
        target_mode="hybrid",
        hybrid_stats=_hybrid(),
    )
    processor.train()
    training = processor.preprocess(
        {"physical_input": physical, "physical_target": target}
    )
    assert training is not None
    assert training["x"].shape[1] == 16
    assert training["y"].shape[1] == 8
    assert training["trajectory_state"].shape[1] == 8
    train_prediction, _ = processor.postprocess(torch.zeros_like(physical), training)
    assert train_prediction.shape[1] == 8
    assert torch.count_nonzero(train_prediction) == 0

    processor.eval()
    evaluation = processor.preprocess(
        {"physical_input": physical, "physical_target": target}
    )
    assert evaluation is not None
    eval_prediction, evaluation = processor.postprocess(torch.zeros_like(physical), evaluation)
    torch.testing.assert_close(eval_prediction, physical, rtol=0, atol=0)
    torch.testing.assert_close(evaluation["y"], target, rtol=0, atol=0)
    assert torch.isfinite(eval_prediction).all()
    assert torch.all(eval_prediction[:, 3:5] > 0)
