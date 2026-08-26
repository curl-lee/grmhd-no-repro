"""Pure contracts for the Stage Z spherical regrid-resolution experiment.

Stage Z changes only the target sampling resolution.  The helpers here keep
the information-gain decision, fixed physical-r shells, and spherical
coordinate-volume proxy explicit and independently testable.
"""

from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np


REPRODUCTION_SCOPE = "ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION"
REFERENCE_SEMANTICS = "HIGHER_RES_SAMPLING_REFERENCE"


def relative_improvement(baseline: float, candidate: float) -> float:
    """Return the fractional reduction of a positive error metric."""

    left = float(baseline)
    right = float(candidate)
    if not math.isfinite(left) or not math.isfinite(right) or left <= 0 or right < 0:
        raise ValueError("resolution errors must be finite with baseline > 0 and candidate >= 0")
    return (left - right) / left


def information_gain(
    baseline: Mapping[str, float], candidate: Mapping[str, float], *, threshold: float = 0.20
) -> dict[str, object]:
    """Apply the predeclared two-of-four Stage Z information-gain gate."""

    if not 0 <= float(threshold) < 1:
        raise ValueError("information-gain threshold must lie in [0, 1)")
    rel = relative_improvement(
        baseline["temporal_increment_relative_difference"],
        candidate["temporal_increment_relative_difference"],
    )
    shell = relative_improvement(
        baseline["shell_increment_relative_l2"],
        candidate["shell_increment_relative_l2"],
    )
    radial = relative_improvement(
        baseline["radial_increment_relative_l2"],
        candidate["radial_increment_relative_l2"],
    )
    cosine_delta = float(candidate["temporal_increment_cosine"]) - float(
        baseline["temporal_increment_cosine"]
    )
    conditions = {
        "temporal_increment_relative_difference_improves_ge_20pct": rel >= threshold,
        "temporal_increment_cosine_improves": cosine_delta > 0,
        "shell_increment_error_improves_ge_20pct": shell >= threshold,
        "radial_increment_error_improves_ge_20pct": radial >= threshold,
    }
    return {
        "threshold": float(threshold),
        "relative_improvements": {
            "temporal_increment_relative_difference": rel,
            "temporal_increment_cosine_absolute_delta": cosine_delta,
            "shell_increment_relative_l2": shell,
            "radial_increment_relative_l2": radial,
        },
        "conditions": conditions,
        "condition_count": int(sum(conditions.values())),
        "higher_res_data_information_gain": sum(conditions.values()) >= 2,
    }


def fixed_shell_indices(r: np.ndarray, physical_edges: Iterable[float]) -> np.ndarray:
    """Assign radial centres using frozen physical internal shell boundaries.

    The outer two supplied edges document the Stage-T shell contract.  Values
    slightly beyond those centre-derived endpoints remain in the first/last
    shell; only the seven internal physical boundaries determine membership.
    """

    radius = np.asarray(r, dtype=np.float64)
    edges = np.asarray(tuple(float(value) for value in physical_edges), dtype=np.float64)
    if radius.ndim != 1 or not len(radius) or np.any(radius <= 0) or np.any(np.diff(radius) <= 0):
        raise ValueError("r must be positive and strictly increasing")
    if edges.ndim != 1 or len(edges) < 2 or np.any(edges <= 0) or np.any(np.diff(edges) <= 0):
        raise ValueError("physical shell edges must be positive and strictly increasing")
    return np.digitize(radius, edges[1:-1], right=False).astype(np.int64)


def fixed_shell_tensor(
    r: np.ndarray, nphi: int, ntheta: int, physical_edges: Iterable[float],
    *, dtype: np.dtype = np.float32,
) -> np.ndarray:
    """Build one-hot shell channels with identical physical boundaries."""

    edges = tuple(float(value) for value in physical_edges)
    if min(int(nphi), int(ntheta)) <= 0:
        raise ValueError("angular grid sizes must be positive")
    index = fixed_shell_indices(r, edges)
    radial = np.eye(len(edges) - 1, dtype=dtype)[index].T
    result = np.broadcast_to(
        radial[:, None, None, :], (len(edges) - 1, int(nphi), int(ntheta), len(index))
    ).copy()
    if not np.all(result.sum(axis=0) == 1):
        raise RuntimeError("every target voxel must belong to exactly one shell")
    return result


def target_edges(
    r: np.ndarray, theta: np.ndarray, phi: np.ndarray, *, r_bounds: tuple[float, float],
    theta_bounds: tuple[float, float], phi_bounds: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct the frozen production target faces from its construction."""

    radius = np.asarray(r, dtype=np.float64)
    polar = np.asarray(theta, dtype=np.float64)
    azimuth = np.asarray(phi, dtype=np.float64)
    if min(len(radius), len(polar), len(azimuth)) <= 0:
        raise ValueError("coordinate arrays cannot be empty")
    return (
        np.geomspace(float(r_bounds[0]), float(r_bounds[1]), len(radius) + 1),
        np.linspace(float(theta_bounds[0]), float(theta_bounds[1]), len(polar) + 1),
        np.linspace(float(phi_bounds[0]), float(phi_bounds[1]), len(azimuth) + 1),
    )


def spherical_volume_proxy_weights(
    r: np.ndarray, theta: np.ndarray, phi: np.ndarray, *, r_bounds: tuple[float, float],
    theta_bounds: tuple[float, float], phi_bounds: tuple[float, float],
) -> np.ndarray:
    """Return r^2 sin(theta) dr dtheta dphi proxy weights in (phi,theta,r)."""

    radius = np.asarray(r, dtype=np.float64)
    polar = np.asarray(theta, dtype=np.float64)
    azimuth = np.asarray(phi, dtype=np.float64)
    r_edges, theta_edges, phi_edges = target_edges(
        radius, polar, azimuth, r_bounds=r_bounds,
        theta_bounds=theta_bounds, phi_bounds=phi_bounds,
    )
    dr = np.diff(r_edges)
    dtheta = np.diff(theta_edges)
    dphi = np.diff(phi_edges)
    weights = (
        dphi[:, None, None]
        * dtheta[None, :, None]
        * dr[None, None, :]
        * np.square(radius)[None, None, :]
        * np.sin(polar)[None, :, None]
    )
    if weights.shape != (len(azimuth), len(polar), len(radius)) or np.any(weights <= 0):
        raise FloatingPointError("invalid spherical coordinate-volume proxy weights")
    return weights


def weighted_relative_l2(value: np.ndarray, reference: np.ndarray, weights: np.ndarray) -> float:
    """Weighted relative L2; weights may omit leading batch/channel axes."""

    left = np.asarray(value, dtype=np.float64)
    right = np.asarray(reference, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if left.shape != right.shape or left.shape[-weight.ndim :] != weight.shape:
        raise ValueError("weighted relative L2 shapes do not align")
    numerator = float(np.sum(weight * np.square(left - right)))
    denominator = float(np.sum(weight * np.square(right)))
    return math.sqrt(numerator / max(denominator, 1.0e-300))
