"""Stage AC mathematical-contract feasibility checks."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def audit_radial_hat_feasibility(
    *,
    in_shape: Sequence[int],
    out_shape: Sequence[int],
    domain_length: Sequence[float],
    kernel_size: int = 5,
    eps: float = 1.0e-12,
) -> dict[str, Any]:
    """Evaluate the frozen point-sampled radial-hat normalization contract.

    This does not repair or reinterpret the requested discretization. It makes
    zero-support basis functions explicit before trainable code is constructed.
    """

    incoming = tuple(int(value) for value in in_shape)
    outgoing = tuple(int(value) for value in out_shape)
    lengths = tuple(float(value) for value in domain_length)
    if len(incoming) != 3 or len(outgoing) != 3 or len(lengths) != 3:
        raise ValueError("Stage AC feasibility audit requires three dimensions")
    if kernel_size < 2 or eps <= 0:
        raise ValueError("invalid radial-hat audit parameters")
    if any(value <= 0 for value in incoming + outgoing) or any(value <= 0 for value in lengths):
        raise ValueError("shape and domain lengths must be positive")
    if any(left < right or left % right for left, right in zip(incoming, outgoing, strict=True)):
        raise ValueError("out_shape must divide in_shape without upsampling")

    radius = max(
        length / count for length, count in zip(lengths, outgoing, strict=True)
    )
    local_shape = tuple(
        math.floor(2.0 * radius * count / length) + 1
        for count, length in zip(incoming, lengths, strict=True)
    )
    coordinates = [
        np.linspace(-radius, radius, count, dtype=np.float64) for count in local_shape
    ]
    mesh = np.meshgrid(*coordinates, indexing="ij")
    rho = np.sqrt(sum(value * value for value in mesh))
    q_weight = math.prod(lengths) / math.prod(incoming)
    width = radius / (kernel_size - 1)
    centers = np.linspace(0.0, radius, kernel_size, dtype=np.float64)
    basis = []
    for index, center in enumerate(centers):
        values = np.maximum(1.0 - np.abs(rho - center) / width, 0.0)
        values = np.where(rho <= radius, values, 0.0)
        normalization = float(q_weight * values.sum(dtype=np.float64))
        normalized_integral = (
            float(q_weight * np.sum(values / (normalization + eps), dtype=np.float64))
            if normalization > 0.0
            else 0.0
        )
        basis.append({
            "index": index,
            "center": float(center),
            "hat_width": float(width),
            "nonzero_stencil_entries": int(np.count_nonzero(values)),
            "raw_sum": float(values.sum(dtype=np.float64)),
            "Z_k": normalization,
            "normalized_quadrature_integral": normalized_integral,
            "normalizable": normalization > eps,
        })
    unique_supported_radii = np.unique(rho[rho <= radius]).tolist()
    normalizable = all(item["normalizable"] for item in basis)
    normalized = normalizable and all(
        abs(float(item["normalized_quadrature_integral"]) - 1.0) < 1.0e-5
        for item in basis
    )
    return {
        "schema_version": "stage-ac-radial-hat-feasibility-v1",
        "implementation_class": "ADAPTED_ISOTROPIC_RADIAL_PIECEWISE_LINEAR",
        "distance": "NORMALIZED_COMPUTATIONAL_INDEX_DISTANCE",
        "in_shape": list(incoming),
        "out_shape": list(outgoing),
        "domain_length": list(lengths),
        "kernel_size": kernel_size,
        "radius_cutoff": radius,
        "local_stencil_shape": list(local_shape),
        "quadrature_weight": q_weight,
        "eps": eps,
        "unique_supported_radii_within_cutoff": unique_supported_radii,
        "basis": basis,
        "all_basis_normalizable": normalizable,
        "quadrature_normalization_contract_pass": normalized,
        "zero_support_basis_indices": [
            item["index"] for item in basis if not item["normalizable"]
        ],
    }
