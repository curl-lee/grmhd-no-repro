from __future__ import annotations

import numpy as np
import pytest

from grmhd.paper_preprocessing import PaperPreprocessor
from scripts.audit_paper_preprocessing_loss import (
    decode_robust_z_channel,
    decode_soft_clipped_channel,
    error_metrics,
    finite_summary,
    spatial_clamp_statistics,
)


def make_preprocessor() -> PaperPreprocessor:
    return PaperPreprocessor(
        epsilon=np.asarray([1e-3, 1e-3, 1e-2, 1e-6, 1e-8, 0, 0, 0]),
        median=np.zeros(8),
        scale=np.ones(8),
        training_indices=(0, 1),
        source_hdf5_checksum="synthetic",
        protocol_name="paper_reduced100_press_spherical_ks",
    )


def test_finite_summary_and_masked_error_metrics():
    values = np.asarray([0.0, 1.0, 2.0, np.inf])
    summary = finite_summary(values)
    assert summary["count"] == 4
    assert summary["finite_count"] == 3
    assert summary["nonfinite_fraction"] == pytest.approx(0.25)
    assert summary["min"] == 0
    assert summary["max"] == 2
    assert summary["q0.5"] == 1

    truth = np.asarray([1.0, 2.0, 3.0, 4.0])
    prediction = np.asarray([1.0, 1.0, 5.0, 4.0])
    all_metrics = error_metrics(truth, prediction)
    assert all_metrics["mae"] == pytest.approx(0.75)
    assert all_metrics["rmse"] == pytest.approx(np.sqrt(5 / 4))
    assert all_metrics["relative_l2"] == pytest.approx(np.sqrt(5 / 30))
    masked = error_metrics(truth, prediction, np.asarray([False, True, True, False]))
    assert masked["count"] == 2
    assert masked["error_squared_sum"] == pytest.approx(5)
    empty = error_metrics(truth, prediction, np.zeros(4, dtype=bool))
    assert empty["count"] == 0
    assert empty["relative_l2"] is None


def test_canonical_clamp_differs_from_diagnostic_unclamped_decode():
    preprocessor = make_preprocessor()
    robust_z = np.asarray([-20.0, -1.0, 0.0, 1.0, 20.0], dtype=np.float64)
    physical = decode_robust_z_channel(robust_z, 5, preprocessor)
    soft64 = 6.0 * np.tanh(robust_z / 6.0)
    canonical = decode_soft_clipped_channel(
        soft64.astype(np.float32),
        5,
        preprocessor,
        gamma=6.0,
        inverse_clamp_fraction=0.99,
        output_dtype=np.float32,
        finite_guard=True,
    )
    diagnostic = decode_soft_clipped_channel(
        soft64,
        5,
        preprocessor,
        gamma=6.0,
        inverse_clamp_fraction=None,
        output_dtype=np.float64,
        finite_guard=False,
    )
    np.testing.assert_allclose(diagnostic, physical, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(canonical[1:4], physical[1:4], rtol=1e-5, atol=1e-6)
    assert canonical[0] > physical[0]
    assert canonical[-1] < physical[-1]


def test_spatial_clamp_statistics_tracks_sign_snapshot_shell_and_regions():
    positive = np.zeros((2, 2, 4, 4), dtype=bool)
    negative = np.zeros_like(positive)
    positive[0, 0, 1:3, 0:2] = True
    negative[1, 1, 0, 3] = True
    r = np.asarray([1.0, 2.0, 4.0, 8.0])
    shell_edges = np.geomspace(1.0, 8.0, 9)
    shell_indices = np.digitize(r, shell_edges[1:-1], right=False)
    report = spatial_clamp_statistics(
        positive,
        negative,
        snapshot_indices=(11, 12),
        times=np.arange(20, dtype=np.float64),
        r=r,
        theta=np.asarray([0.2, 1.4, 1.7, 2.9]),
        phi=np.asarray([0.1, 3.2]),
        shell_indices=shell_indices,
        shell_edges=shell_edges,
    )
    assert report["positive_count"] == 4
    assert report["negative_count"] == 1
    assert report["total_count"] == 5
    assert len(report["snapshot_records"]) == 2
    assert len(report["shell_records"]) == 8
    assert np.asarray(report["fixed_phi"]["total_fraction_map"]).shape == (4, 4)
    assert np.asarray(report["equatorial"]["total_fraction_map"]).shape == (2, 4)
    assert report["snapshot_concentration"]["top_1_snapshot_hit_share"] == pytest.approx(0.8)
    assert report["region_concentration"][
        "inner_two_log_shells"
    ]["clamp_hit_count"] >= 0
