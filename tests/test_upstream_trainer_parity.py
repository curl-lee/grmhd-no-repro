from __future__ import annotations

import numpy as np
import torch

from grmhd.normalizer import GRMHDNormalizer
from grmhd.parity import compare_upstream_and_manual_train_batch
from grmhd.upstream_adapters import UpstreamFNOConfig


def test_upstream_trainer_and_manual_batch_match_exactly():
    torch.manual_seed(7)
    physical_input = torch.randn(1, 8, 16, 16, 16) * 0.01
    physical_target = physical_input + torch.randn_like(physical_input) * 0.001
    physical_input[:, 3] = torch.exp(physical_input[:, 3])
    physical_input[:, 4] = 0.1 * torch.exp(physical_input[:, 4])
    physical_target[:, 3] = torch.exp(physical_target[:, 3])
    physical_target[:, 4] = 0.1 * torch.exp(physical_target[:, 4])
    normalizer = GRMHDNormalizer(
        np.zeros(8),
        np.ones(8),
        epsilon=np.asarray([1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 0, 0, 0]),
    )
    report = compare_upstream_and_manual_train_batch(
        batch={"physical_input": physical_input, "physical_target": physical_target},
        normalizer=normalizer,
        shells=None,
        downsample=1,
        model_config=UpstreamFNOConfig(in_channels=8),
    )
    assert report["status"] == "passed", report
    assert report["loss_reduction"] == "sum"
    assert report["parameter_update_max_abs_difference"] == 0.0
