from __future__ import annotations

import numpy as np
import torch

from grmhd.paper_stage_g_evaluation import (
    artifact_diagnostics,
    physical_state_statistics,
    radial_shell_indices,
    temporal_series_statistics,
    total_variation_and_high_k,
)


def test_stage_g_radial_statistics_and_outer_shells_are_explicit():
    r = np.geomspace(1.0, 8.0, 8)
    state = torch.arange(8 * 2 * 3 * 8, dtype=torch.float32).reshape(1, 8, 2, 3, 8)
    edges, indices = radial_shell_indices(r)
    assert len(edges) == 9
    assert sorted(set(indices.tolist())) == list(range(8))
    result = physical_state_statistics(state, r)
    assert result["coordinate_semantics"].startswith("native spherical Kerr-Schild")
    assert set(result["channels"]) == {
        "Bcc1", "Bcc2", "Bcc3", "rho", "press", "vel1", "vel2", "vel3"
    }
    assert len(result["channels"]["rho"]["outermost_two_shells"]) == 2


def test_stage_g_spectral_and_artifact_diagnostics_are_finite():
    generator = torch.Generator().manual_seed(9)
    reference = torch.randn(1, 8, 8, 8, 8, generator=generator)
    prediction = 0.9 * reference
    spectral = total_variation_and_high_k(prediction)
    assert all(
        0 <= record["high_k_energy_fraction"] <= 1
        for record in spectral["per_channel"].values()
    )
    artifacts = artifact_diagnostics(
        prediction, reference, reference, reference_kind="ground_truth"
    )
    assert artifacts["reference_kind"] == "ground_truth"
    assert artifacts["flags"] == []


def test_stage_g_temporal_statistics_have_fixed_lags_and_psd():
    t = np.arange(100, dtype=np.float64)
    series = np.stack(
        [np.sin(2 * np.pi * (channel + 1) * t / 100) for channel in range(8)],
        axis=1,
    )
    result = temporal_series_statistics(series)
    assert result["steps"] == 100
    assert set(result["channels"]["Bcc1"]["autocorrelation"]) == {"1", "5", "10", "19"}
    assert result["channels"]["Bcc1"]["dominant_temporal_frequency"] is not None
