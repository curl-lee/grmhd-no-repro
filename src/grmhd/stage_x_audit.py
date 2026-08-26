"""Pure numerical contracts for the Stage X representation/regrid audit.

The helpers in this module do not train or mutate a model.  They make the
sampling-reference comparisons and the predeclared Stage X classifications
small enough to unit test independently of the raw ATHDF files.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


EPS = 1.0e-30


def relative_l2(value: np.ndarray, reference: np.ndarray) -> float:
    """Return ||value-reference||_2 / ||reference||_2."""

    left = np.asarray(value, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} != {right.shape}")
    numerator = float(np.sum(np.square(left - right)))
    denominator = float(np.sum(np.square(right)))
    return math.sqrt(numerator / max(denominator, EPS))


def cosine_similarity(value: np.ndarray, reference: np.ndarray) -> float:
    """Cosine similarity after flattening, with an explicit zero-vector rule."""

    left = np.asarray(value, dtype=np.float64).reshape(-1)
    right = np.asarray(reference, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} != {right.shape}")
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 and right_norm == 0.0:
        return 1.0
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def nearest_reference_indices(
    target: np.ndarray, reference: np.ndarray, *, periodic: bool = False,
    period: float | None = None,
) -> np.ndarray:
    """Indices of nearest reference cell centres for each target centre."""

    target = np.asarray(target, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if target.ndim != 1 or reference.ndim != 1 or not len(reference):
        raise ValueError("coordinate arrays must be non-empty and one-dimensional")
    distance = np.abs(target[:, None] - reference[None, :])
    if periodic:
        if period is None or period <= 0:
            raise ValueError("periodic indexing requires a positive period")
        distance = np.minimum(distance, float(period) - np.minimum(distance, float(period)))
    return np.argmin(distance, axis=1).astype(np.int64)


def sample_reference_to_grid(
    reference: np.ndarray, phi_index: np.ndarray, theta_index: np.ndarray,
    r_index: np.ndarray,
) -> np.ndarray:
    """Nearest-centre restriction of a (phi,theta,r) sampling reference."""

    value = np.asarray(reference)
    if value.ndim != 3:
        raise ValueError("reference must have axis order (phi,theta,r)")
    return value[np.ix_(phi_index, theta_index, r_index)]


def radial_region_masks(r: np.ndarray) -> dict[str, np.ndarray]:
    """Three equal-width regions in log(r), fixed before viewing field values."""

    radius = np.asarray(r, dtype=np.float64)
    if radius.ndim != 1 or np.any(radius <= 0) or np.any(np.diff(radius) <= 0):
        raise ValueError("r must be positive and strictly increasing")
    edges = np.linspace(np.log(radius[0]), np.log(radius[-1]), 4)
    log_r = np.log(radius)
    return {
        "inner": (log_r >= edges[0]) & (log_r < edges[1]),
        "middle": (log_r >= edges[1]) & (log_r < edges[2]),
        "outer": (log_r >= edges[2]) & (log_r <= edges[3]),
    }


def log_shell_indices(r: np.ndarray, shell_count: int = 8) -> tuple[np.ndarray, np.ndarray]:
    radius = np.asarray(r, dtype=np.float64)
    if shell_count <= 0 or radius.ndim != 1 or np.any(radius <= 0):
        raise ValueError("invalid radial shell request")
    edges = np.geomspace(float(radius[0]), float(radius[-1]), shell_count + 1)
    indices = np.digitize(radius, edges[1:-1], right=False)
    return indices.astype(np.int64), edges


def shell_moments(value: np.ndarray, r: np.ndarray, shell_count: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Mean and variance in fixed log-r shells for a scalar 3D field."""

    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or array.shape[-1] != len(r):
        raise ValueError("value must use (phi,theta,r) order")
    shell_index, _ = log_shell_indices(r, shell_count)
    means = np.empty(shell_count, dtype=np.float64)
    variances = np.empty(shell_count, dtype=np.float64)
    for shell in range(shell_count):
        selected = array[..., shell_index == shell]
        means[shell] = float(np.mean(selected))
        variances[shell] = float(np.var(selected))
    return means, variances


def radial_profile(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError("value must use (phi,theta,r) order")
    return np.mean(array, axis=(0, 1))


def spectral_energy_fractions(value: np.ndarray, cutoff: float = 0.5) -> tuple[float, float]:
    """Index-grid low/high Fourier energy fractions.

    This is deliberately labelled an index-grid diagnostic: no Cartesian or
    Kerr--Schild metric meaning is attached to it.
    """

    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 3 or not 0 < cutoff < 1:
        raise ValueError("expected a 3D array and cutoff in (0,1)")
    transformed = np.fft.rfftn(array, norm="ortho")
    energy = np.square(np.abs(transformed))
    axes = [np.fft.fftfreq(n) for n in array.shape[:-1]]
    axes.append(np.fft.rfftfreq(array.shape[-1]))
    normalized = [np.abs(axis) / 0.5 for axis in axes]
    radius = np.sqrt(
        normalized[0][:, None, None] ** 2
        + normalized[1][None, :, None] ** 2
        + normalized[2][None, None, :] ** 2
    )
    total = float(np.sum(energy))
    high = float(np.sum(energy[radius >= cutoff])) / max(total, EPS)
    return 1.0 - high, high


def classify_temporal_fidelity(
    relative_differences: Iterable[float], cosines: Iterable[float],
    *, good_relative_max: float = 0.25, good_cosine_min: float = 0.95,
    moderate_relative_max: float = 0.50, moderate_cosine_min: float = 0.80,
) -> str:
    """Classify by medians using thresholds frozen before raw diagnostics."""

    rel = np.asarray(list(relative_differences), dtype=np.float64)
    cos = np.asarray(list(cosines), dtype=np.float64)
    if not len(rel) or len(rel) != len(cos) or not np.isfinite(rel).all() or not np.isfinite(cos).all():
        raise ValueError("fidelity metrics must be non-empty, paired, and finite")
    rel_median = float(np.median(rel))
    cos_median = float(np.median(cos))
    if rel_median <= good_relative_max and cos_median >= good_cosine_min:
        return "GOOD"
    if rel_median <= moderate_relative_max and cos_median >= moderate_cosine_min:
        return "MODERATE"
    return "POOR"


def classify_radial_identifiability(
    shell_off_distances: Iterable[float], shell_on_distances: Iterable[float],
    *, off_max: float = 1.0e-5, on_min: float = 1.0e-3,
    has_continuous_coordinates: bool = False,
) -> str:
    """Classify whether frozen responses distinguish translated radial patterns."""

    off = np.asarray(list(shell_off_distances), dtype=np.float64)
    on = np.asarray(list(shell_on_distances), dtype=np.float64)
    if not len(off) or len(off) != len(on) or not np.isfinite(off).all() or not np.isfinite(on).all():
        raise ValueError("identifiability metrics must be non-empty, paired, and finite")
    if float(np.min(on)) >= on_min and not has_continuous_coordinates:
        return "PARTIAL"
    if float(np.min(on)) >= on_min and has_continuous_coordinates:
        return "GOOD"
    return "POOR"
