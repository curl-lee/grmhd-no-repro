#!/usr/bin/env python3
"""Materialize Stage AD implementation gates before any GRMHD training."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from grmhd.dataset import sha256_file
from grmhd.operators.disco3d import AdaptedRadialDISCO3d, radial_hat_support


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/stage_ad"
PINNED = "86a8bc7812a31b42c4f7895693cf4ac11521c066"


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True))


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def reference_correlation(
    operator: AdaptedRadialDISCO3d, x: torch.Tensor
) -> torch.Tensor:
    kernel = operator.dense_kernel()
    result = x.new_zeros((x.shape[0], operator.out_channels, *x.shape[-3:]))
    outputs_per_group = operator.out_channels // operator.groups
    half = tuple(size // 2 for size in operator.stencil_shape)
    for group in range(operator.groups):
        input_start = group * operator.groupsize
        output_start = group * outputs_per_group
        for local_output in range(outputs_per_group):
            output_channel = output_start + local_output
            for local_input in range(operator.groupsize):
                input_channel = input_start + local_input
                for di in range(operator.stencil_shape[0]):
                    for dj in range(operator.stencil_shape[1]):
                        for dk in range(operator.stencil_shape[2]):
                            offset = (di - half[0], dj - half[1], dk - half[2])
                            shifted = torch.roll(
                                x[:, input_channel],
                                shifts=tuple(-value for value in offset),
                                dims=(-3, -2, -1),
                            )
                            result[:, output_channel] += (
                                kernel[output_channel, local_input, di, dj, dk] * shifted
                            )
            if operator.bias is not None:
                result[:, output_channel] += operator.bias[output_channel]
    return result


def provenance() -> tuple[dict, dict, dict]:
    config_path = ROOT / "configs/stage_ad/disco3d_localno.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stage_t_path = ROOT / config["frozen_stage_t_config"]
    if sha256_file(stage_t_path) != config["frozen_stage_t_config_sha256"]:
        raise ValueError("Stage AD frozen Stage-T config checksum changed")
    stage_t = yaml.safe_load(stage_t_path.read_text(encoding="utf-8"))
    stage_s_path = ROOT / stage_t["frozen_stage_s_config"]
    if sha256_file(stage_s_path) != stage_t["frozen_stage_s_config_sha256"]:
        raise ValueError("Stage AD frozen Stage-S config checksum changed")
    stage_s = yaml.safe_load(stage_s_path.read_text(encoding="utf-8"))
    upstream = subprocess.check_output(
        ["git", "-C", str(ROOT / "external/neuraloperator"), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    upstream_status = subprocess.check_output(
        ["git", "-C", str(ROOT / "external/neuraloperator"), "status", "--short"],
        text=True,
    ).strip()
    if upstream != PINNED or upstream_status:
        raise ValueError("pinned upstream changed")
    checksums = {
        "stage_ad_config": sha256_file(config_path),
        "stage_t_config": sha256_file(stage_t_path),
        "stage_s_config": sha256_file(stage_s_path),
        "dataset": sha256_file(ROOT / stage_s["data"]["dataset"]),
        "normalizer": sha256_file(
            ROOT / stage_s["preprocessing"]["artifact"] / "normalizer.npz"
        ),
        "common_initial_state": sha256_file(
            ROOT / config["initialization"]["common_state"]
        ),
    }
    if checksums["dataset"] != stage_s["data"]["dataset_sha256"]:
        raise ValueError("frozen expanded64 dataset changed")
    if checksums["normalizer"] != stage_s["preprocessing"]["normalizer_sha256"]:
        raise ValueError("frozen Stage-S P3 normalizer changed")
    return config, stage_s, {"upstream_commit": upstream, "checksums": checksums}


def make_figures(support) -> None:
    figure_dir = OUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    radius = support.radius_cutoff
    values = np.linspace(0.0, radius, 401)
    plt.figure(figsize=(7, 4))
    for k, center in enumerate(support.centers.numpy()):
        hat = np.maximum(1.0 - np.abs(values - center) / support.width, 0.0)
        plt.plot(values / support.delta_cell, hat, label=f"k={k}")
    plt.xlabel(r"$\rho/\Delta$")
    plt.ylabel(r"$\psi_k(\rho)$")
    plt.legend(ncol=5, fontsize=8)
    plt.tight_layout()
    plt.savefig(figure_dir / "radial_basis_support.png", dpi=160)
    plt.close()

    rho = support.rho.numpy() / support.delta_cell
    inside = rho <= 3.0
    indices = np.indices(support.stencil_shape)
    plt.figure(figsize=(5, 5))
    plt.scatter(indices[1][inside], indices[0][inside], c=rho[inside], s=18)
    plt.gca().set_aspect("equal")
    plt.xlabel("local theta index")
    plt.ylabel("local phi index")
    plt.colorbar(label=r"$\rho/\Delta$")
    plt.tight_layout()
    plt.savefig(figure_dir / "local_7x7x7_support.png", dpi=160)
    plt.close()


def main() -> None:
    config, stage_s, frozen = provenance()
    operator_config = config["operator"]
    audits = {}
    for radius_cells in (1, 2, 3):
        support = radial_hat_support(
            grid_shape=operator_config["grid_shape"],
            domain_length=operator_config["domain_length"],
            radius_cells=radius_cells,
            basis_count=operator_config["basis_count"],
            eps=operator_config["normalization_epsilon"],
        )
        positive = (support.raw_basis > 0).sum(dim=(1, 2, 3)).tolist()
        audits[radius_cells] = {
            "radius_cells": radius_cells,
            "radius_cutoff": support.radius_cutoff,
            "stencil_shape": list(support.stencil_shape),
            "positive_points_by_basis": positive,
            "all_bases_active": all(value > 0 for value in positive),
        }
    if audits[1]["all_bases_active"] or audits[2]["all_bases_active"]:
        raise RuntimeError("Stage AD analytical minimal-support premise changed")
    if not audits[3]["all_bases_active"]:
        raise RuntimeError("Stage AD three-cell support does not activate all hats")

    support = radial_hat_support(
        grid_shape=operator_config["grid_shape"],
        domain_length=operator_config["domain_length"],
        radius_cells=3,
        basis_count=5,
        eps=operator_config["normalization_epsilon"],
    )
    if support.stencil_shape != (7, 7, 7):
        raise RuntimeError("Stage AD support did not resolve to 7x7x7")
    positive_mask = support.raw_basis > 0
    rows = []
    for k in range(5):
        positives = support.raw_basis[k][positive_mask[k]]
        rows.append({
            "k": k,
            "center": float(support.centers[k]),
            "width": support.width,
            "number_of_positive_stencil_points": int(positive_mask[k].sum()),
            "raw_weight_sum": float(support.raw_basis[k].sum()),
            "quadrature_weighted_Zk": float(support.normalizations[k]),
            "min_positive_value": float(positives.min()),
            "max_value": float(support.raw_basis[k].max()),
        })
    atomic_csv(OUT / "implementation/basis_support_audit.csv", rows)
    inside = support.rho <= support.radius_cutoff
    partition_error = float(torch.max(torch.abs(support.raw_basis.sum(0)[inside] - 1.0)))
    partition_pass = partition_error < 1e-6
    atomic_json(OUT / "implementation/basis_partition_audit.json", {
        "max_abs_error": partition_error,
        "tolerance": 1e-6,
        "pass": partition_pass,
        "support_points": int(inside.sum()),
    })
    normalized_integrals = (
        support.quadrature_weight * support.normalized_basis.sum(dim=(1, 2, 3))
    )
    normalization_error = torch.abs(normalized_integrals - 1.0)
    quadrature_pass = bool(torch.max(normalization_error) < 1e-6)
    atomic_json(OUT / "implementation/quadrature_audit.json", {
        "quadrature_weight": support.quadrature_weight,
        "normalizations_Zk": support.normalizations.tolist(),
        "normalized_integrals": normalized_integrals.tolist(),
        "absolute_errors": normalization_error.tolist(),
        "maximum_Z_over_minimum_Z": float(
            support.normalizations.max() / support.normalizations.min()
        ),
        "tolerance": 1e-6,
        "pass": quadrature_pass,
    })
    compact_pass = bool(torch.max(torch.abs(support.raw_basis[:, ~inside])) < 1e-12)
    atomic_json(OUT / "implementation/compact_support_audit.json", {
        "bounding_box_points": math.prod(support.stencil_shape),
        "points_with_rho_le_radius": int(inside.sum()),
        "points_outside_radius": int((~inside).sum()),
        "maximum_absolute_basis_outside": float(
            torch.max(torch.abs(support.raw_basis[:, ~inside]))
        ),
        "pass": compact_pass,
    })

    torch.manual_seed(101)
    operator = AdaptedRadialDISCO3d(
        2, 2, grid_shape=(7, 7, 7), domain_length=(2.0, 2.0, 2.0)
    ).double()
    x = torch.randn(1, 2, 7, 7, 7, dtype=torch.float64)
    optimized = operator(x)
    reference = reference_correlation(operator, x)
    difference = optimized - reference
    relative_l2 = float((
        torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(reference)
    ).detach())
    max_abs = float(torch.max(torch.abs(difference)).detach())
    dense_pass = relative_l2 < 1e-8 and max_abs < 1e-8
    atomic_json(OUT / "implementation/dense_reference_test.json", {
        "grid_shape": [7, 7, 7],
        "boundary": "circular",
        "convention": "cross-correlation",
        "relative_l2": relative_l2,
        "max_abs": max_abs,
        "relative_tolerance": 1e-8,
        "absolute_tolerance": 1e-8,
        "match": dense_pass,
    })

    orientation_x = torch.arange(5**3, dtype=torch.float64).reshape(1, 1, 5, 5, 5)
    asymmetric = torch.zeros(1, 1, 3, 3, 3, dtype=torch.float64)
    asymmetric[0, 0, 0, 1, 1] = 2.0
    asymmetric[0, 0, 2, 1, 1] = -0.5
    observed = F.conv3d(F.pad(orientation_x, (1, 1, 1, 1, 1, 1), mode="circular"), asymmetric)
    expected = 2.0 * torch.roll(orientation_x, 1, -3) - 0.5 * torch.roll(orientation_x, -1, -3)
    flipped = -0.5 * torch.roll(orientation_x, 1, -3) + 2.0 * torch.roll(orientation_x, -1, -3)
    orientation_error = float(torch.max(torch.abs(observed - expected)))
    orientation_pass = orientation_error == 0.0 and not torch.equal(observed, flipped)
    atomic_json(OUT / "implementation/kernel_orientation_test.json", {
        "implementation_convention": "cross-correlation",
        "explicit_reference_convention": "cross-correlation",
        "maximum_absolute_error": orientation_error,
        "distinguishes_flipped_kernel": not torch.equal(observed, flipped),
        "pass": orientation_pass,
    })

    grad_operator = AdaptedRadialDISCO3d(
        1, 1, grid_shape=(7, 7, 7), domain_length=(2.0, 2.0, 2.0)
    ).double()
    grad_x = torch.randn(1, 1, 7, 7, 7, dtype=torch.float64, requires_grad=True)
    grad_weight = grad_operator.weight.detach().clone().requires_grad_(True)
    grad_bias = grad_operator.bias.detach().clone().requires_grad_(True)
    gradcheck_pass = bool(torch.autograd.gradcheck(
        grad_operator.forward_with_parameters,
        (grad_x, grad_weight, grad_bias),
        eps=1e-6,
        atol=1e-5,
        rtol=1e-3,
        fast_mode=True,
    ))
    atomic_json(OUT / "implementation/gradcheck.json", {
        "dtype": "float64",
        "tested": ["input", "weight", "bias"],
        "eps": 1e-6,
        "atol": 1e-5,
        "rtol": 1e-3,
        "fast_mode": True,
        "pass": gradcheck_pass,
    })

    contract = f"""# Stage AD Repair Contract

- `REPRODUCTION_SCOPE = {config['reproduction_scope']}`
- `DISCO3D_IMPLEMENTATION = {operator_config['implementation']}`
- `EXACT_3D_DISCO_IMPLEMENTATION_FOUND = false`
- `EXACT_REPRODUCTION_BLOCKED = true`
- sole scientific repair: cutoff radius `1 cell -> 3 cells`
- basis count: 5; basis family unchanged
- computational domain length: {operator_config['domain_length']}
- regular model grid: {operator_config['grid_shape']}
- boundary: computational periodic adaptation
- dataset: `{stage_s['data']['dataset']}`
- frozen P3 normalizer: `{stage_s['preprocessing']['artifact']}`
- upstream: `{frozen['upstream_commit']}`; clean

No validation result was used to choose the radius.  R=3Δ is the smallest integer-cell
support that makes every one of the five pre-frozen radial hats discretely active.
"""
    atomic_text(OUT / "scope/repair_contract.md", contract)
    atomic_text(OUT / "implementation/radius_support_derivation.md", f"""# Radius Support Derivation

`delta_cell = max(L_i/N_i) = {support.delta_cell}` and
`radius_cutoff = 3 * delta_cell = {support.radius_cutoff}`.  The frozen local-shape formula
resolves to `{list(support.stencil_shape)}`.

Deterministic integer-cell audit:

```json
{json.dumps(audits, indent=2, sort_keys=True)}
```

Thus m=1 and m=2 leave at least one hat inactive, while m=3 activates all five.  No
larger radius was tested or authorized.
""")
    make_figures(support)

    gates = {
        "ALL_BASES_ACTIVE": audits[3]["all_bases_active"],
        "BASIS_PARTITION_OF_UNITY_PASS": partition_pass,
        "QUADRATURE_NORMALIZATION_PASS": quadrature_pass,
        "COMPACT_SUPPORT_PASS": compact_pass,
        "DENSE_REFERENCE_MATCH": dense_pass,
        "KERNEL_ORIENTATION_PASS": orientation_pass,
        "GRADCHECK_PASS": gradcheck_pass,
        "STENCIL_SHAPE": list(support.stencil_shape),
        "DISCO3D_EXECUTION_BACKEND": operator.execution_backend,
        "provenance": frozen,
    }
    atomic_json(OUT / "implementation/implementation_gates.json", gates)
    if not all(value for key, value in gates.items() if key.endswith("PASS") or key.endswith("MATCH") or key == "ALL_BASES_ACTIVE"):
        raise RuntimeError("Stage AD implementation gate failed")
    print(json.dumps(gates, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
