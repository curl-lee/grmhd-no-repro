from __future__ import annotations

import numpy as np
import torch

from grmhd.data_processor import GRMHDDataProcessor
from grmhd.hybrid import HybridTargetStats
from grmhd.normalizer import GRMHDNormalizer
from grmhd.trainers import GRMHDRolloutTrainer


class PointwiseModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.01))

    def forward(self, x, **kwargs):
        return self.weight * x[:, :8]


class DummyDataset:
    stride = 1


def _stats():
    return HybridTargetStats(
        scale=(0.1,) * 8,
        alpha=(3.0,) * 8,
        epsilon=(0, 0, 0, 1e-6, 1e-6, 0, 0, 0),
        alpha_quantile=0.999,
        min_scale=1e-12,
        min_alpha=1e-3,
        sample_count_per_channel=1,
        seed=42,
        source_hdf5_checksum="x",
        training_indices=(0, 1),
        pair_starts=(0,),
        stats_checksum="x",
    )


def test_rollout_bptt_retains_gradient_through_intermediate_prediction():
    normalizer = GRMHDNormalizer(
        np.zeros(8),
        np.ones(8),
        epsilon=np.asarray([1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 0, 0, 0]),
    )
    processor = GRMHDDataProcessor(
        normalizer=normalizer, target_mode="hybrid", hybrid_stats=_stats()
    )
    trainer = GRMHDRolloutTrainer(
        model=PointwiseModel(),
        n_epochs=1,
        device="cpu",
        data_processor=processor,
        rollout_dataset=DummyDataset(),
        loss_weights={"target": 1, "decoded": 1, "rollout": 1, "range": 0},
    )
    physical = torch.full((1, 8, 3, 3, 3), 0.01)
    physical[:, 3] = 1.0
    physical[:, 4] = 0.1
    predictions = trainer.rollout_from_physical(physical, steps=3)
    assert predictions[0][1].requires_grad
    assert predictions[1][1].grad_fn is not None
    predictions[-1][1].sum().backward()
    assert trainer.model.weight.grad is not None
    assert torch.isfinite(trainer.model.weight.grad)
    assert trainer.model.weight.grad != 0


def test_rollout_curriculum_matches_round3_schedule():
    normalizer = GRMHDNormalizer(np.zeros(8), np.ones(8))
    processor = GRMHDDataProcessor(
        normalizer=normalizer, target_mode="hybrid", hybrid_stats=_stats()
    )
    trainer = GRMHDRolloutTrainer(
        model=PointwiseModel(),
        n_epochs=10,
        data_processor=processor,
        rollout_dataset=DummyDataset(),
        loss_weights={},
    )
    expected = {0: 1, 2: 1, 3: 2, 7: 2, 8: 3, 12: 3}
    for epoch, rollout_k in expected.items():
        trainer.on_epoch_start(epoch)
        assert trainer.rollout_k == rollout_k
