from __future__ import annotations

from copy import deepcopy
import numpy as np
import torch

from grmhd.normalizer import GRMHDNormalizer
from grmhd.hybrid import HybridTargetStats
from grmhd.parity import compare_upstream_and_local_autoregression
from grmhd.upstream_adapters import UpstreamFNOConfig, build_upstream_fno


def test_upstream_and_local_physical_autoregression_match():
    torch.manual_seed(17)
    trajectory = torch.randn(1, 4, 8, 16, 16, 16) * 0.01
    trajectory[:, :, 3] = torch.exp(trajectory[:, :, 3])
    trajectory[:, :, 4] = 0.1 * torch.exp(trajectory[:, :, 4])
    normalizer = GRMHDNormalizer(
        np.zeros(8),
        np.ones(8),
        epsilon=np.asarray([1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 0, 0, 0]),
    )
    config = UpstreamFNOConfig(in_channels=8)
    state = deepcopy(build_upstream_fno(config).state_dict())
    report = compare_upstream_and_local_autoregression(
        sample={
            "physical_trajectory": trajectory,
            "trajectory_state": trajectory[:, 0],
        },
        normalizer=normalizer,
        shells=None,
        downsample=1,
        model_config=config,
        model_state_dict=state,
        max_steps=10,
    )
    assert report["status"] == "passed", report
    assert report["trajectory_length"] == 3
    assert report["maximum_trace_abs_difference"] == 0.0
    assert report["upstream_transform_counts"] == {
        "physical_input_encode": 3,
        "physical_target_encode": 0,
        "state_decode": 3,
        "hybrid_reconstruct": 0,
    }


def test_upstream_and_local_hybrid_autoregression_match():
    torch.manual_seed(23)
    trajectory = torch.randn(1, 3, 8, 16, 16, 16) * 0.005
    trajectory[:, :, 3] = torch.exp(trajectory[:, :, 3])
    trajectory[:, :, 4] = 0.1 * torch.exp(trajectory[:, :, 4])
    normalizer = GRMHDNormalizer(
        np.zeros(8), np.ones(8),
        epsilon=np.asarray([1e-3, 1e-3, 1e-3, 1e-6, 1e-6, 0, 0, 0]),
    )
    hybrid = HybridTargetStats(
        scale=(0.01,) * 8, alpha=(5.0,) * 8,
        epsilon=(0.0, 0.0, 0.0, 1e-8, 1e-8, 0.0, 0.0, 0.0),
        alpha_quantile=0.999, min_scale=1e-12, min_alpha=1e-3,
        sample_count_per_channel=1, seed=42, source_hdf5_checksum="synthetic",
        training_indices=(0, 1), pair_starts=(0,), stats_checksum="synthetic",
    )
    config = UpstreamFNOConfig(in_channels=8)
    state = deepcopy(build_upstream_fno(config).state_dict())
    report = compare_upstream_and_local_autoregression(
        sample={"physical_trajectory": trajectory, "trajectory_state": trajectory[:, 0]},
        normalizer=normalizer, shells=None, downsample=1, model_config=config,
        model_state_dict=state, max_steps=10, target_mode="hybrid",
        hybrid_stats=hybrid,
    )
    assert report["status"] == "passed", report
    assert report["trajectory_length"] == 2
    assert report["maximum_trace_abs_difference"] == 0.0
    assert report["upstream_transform_counts"]["hybrid_reconstruct"] == 2
