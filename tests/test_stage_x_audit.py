import numpy as np

from grmhd.stage_x_audit import (
    classify_radial_identifiability,
    classify_temporal_fidelity,
    cosine_similarity,
    nearest_reference_indices,
    radial_region_masks,
    relative_l2,
    sample_reference_to_grid,
    shell_moments,
    spectral_energy_fractions,
)


def test_reference_restriction_and_metrics() -> None:
    reference = np.arange(8 * 8 * 8, dtype=np.float64).reshape(8, 8, 8)
    coordinate = np.arange(8, dtype=np.float64) + 0.5
    target = coordinate[[1, 3, 5, 7]]
    index = nearest_reference_indices(target, coordinate)
    sampled = sample_reference_to_grid(reference, index, index, index)
    expected = reference[np.ix_([1, 3, 5, 7], [1, 3, 5, 7], [1, 3, 5, 7])]
    assert np.array_equal(sampled, expected)
    assert relative_l2(sampled, expected) == 0.0
    assert np.isclose(cosine_similarity(sampled, expected), 1.0)


def test_shell_region_and_spectrum_contracts() -> None:
    r = np.geomspace(1.0, 100.0, 16)
    value = np.broadcast_to(r[None, None, :], (4, 5, 16))
    masks = radial_region_masks(r)
    assert set(masks) == {"inner", "middle", "outer"}
    assert np.all(np.sum(np.stack(list(masks.values())), axis=0) == 1)
    means, variances = shell_moments(value, r, shell_count=4)
    assert np.all(np.diff(means) > 0)
    assert np.all(variances >= 0)
    low, high = spectral_energy_fractions(np.ones((8, 8, 8)))
    assert np.isclose(low + high, 1.0)
    assert high == 0.0


def test_predeclared_classifications() -> None:
    assert classify_temporal_fidelity([0.1, 0.2], [0.98, 0.96]) == "GOOD"
    assert classify_temporal_fidelity([0.3, 0.4], [0.9, 0.85]) == "MODERATE"
    assert classify_temporal_fidelity([0.8, 0.9], [0.4, 0.5]) == "POOR"
    assert classify_radial_identifiability([1e-7, 2e-7], [0.1, 0.2]) == "PARTIAL"
