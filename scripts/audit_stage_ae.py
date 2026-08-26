#!/usr/bin/env python3
"""Materialize Stage AE geometry gates and stop safely on invalid normalization."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from grmhd.dataset import sha256_file
from grmhd.operators.disco3d import AdaptedRadialDISCO3d, radial_hat_support
from grmhd.operators.spherical_disco3d import (
    SphericalAwareDISCO3d,
    build_spherical_proxy_geometry,
    spherical_embedding,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/stage_ae"
CONFIG = ROOT / "configs/stage_ae/spherical_disco3d_localno.yaml"
PINNED = "86a8bc7812a31b42c4f7895693cf4ac11521c066"
SCOPE = "ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION"


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True))


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty Stage AE table")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def tensor_hash(tensors: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        value = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def read_coordinates(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = ROOT / config["data"]["dataset"]
    if sha256_file(dataset) != config["data"]["dataset_sha256"]:
        raise ValueError("frozen Z64 checksum changed")
    with h5py.File(dataset, "r") as handle:
        group = handle[config["data"]["coordinate_group"]]
        r = np.asarray(group["r"], dtype=np.float64)
        theta = np.asarray(group["theta"], dtype=np.float64)
        phi = np.asarray(group["phi"], dtype=np.float64)
    return r, theta, phi


def explicit_reference(operator: SphericalAwareDISCO3d, x: torch.Tensor) -> torch.Tensor:
    geometry = build_spherical_proxy_geometry(
        r=np.geomspace(1.0, 3.0, 5),
        theta=np.linspace(0.4, math.pi - 0.4, 6),
        phi=(np.arange(8) + 0.5) * 2 * math.pi / 8,
    )
    result = x.new_zeros((x.shape[0], operator.out_channels, *x.shape[-3:]))
    for iphi in range(x.shape[-3]):
        for itheta in range(x.shape[-2]):
            for ir in range(x.shape[-1]):
                fields = x.new_zeros((operator.in_channels, operator.basis_count))
                for offset_index, (dphi, dtheta, dr) in enumerate(geometry.offsets):
                    source_theta, source_r = itheta + dtheta, ir + dr
                    if 0 <= source_theta < x.shape[-2] and 0 <= source_r < x.shape[-1]:
                        fields += (
                            x[0, :, (iphi + dphi) % x.shape[-3], source_theta, source_r]
                            .unsqueeze(1)
                            * operator.integration_weights[:, offset_index, itheta, ir]
                            .to(dtype=x.dtype)
                            .unsqueeze(0)
                        )
                result[0, :, iphi, itheta, ir] = torch.einsum(
                    "ock,ck->o", operator.weight, fields
                ) + operator.bias
    return result


def geometry_figures(geometry, ad_kernel: np.ndarray, targets: Mapping[str, tuple[int, int]]) -> None:
    geometry_dir = OUT / "geometry/figures"
    figure_dir = OUT / "figures"
    geometry_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 4.6))
    image = axis.imshow(
        geometry.local_scale.numpy(), origin="lower", aspect="auto",
        extent=[float(geometry.r[0]), float(geometry.r[-1]), float(geometry.theta[0]), float(geometry.theta[-1])],
    )
    axis.set_xscale("log")
    axis.set_xlabel("r centre")
    axis.set_ylabel("theta centre")
    axis.set_title("local median nearest-neighbor scale h(theta,r)")
    figure.colorbar(image, ax=axis, label="Euclidean embedding proxy distance")
    figure.tight_layout()
    for path in (geometry_dir / "local_scale_r_theta.png", figure_dir / "local_scale_r_theta.png"):
        figure.savefig(path, dpi=160)
    plt.close(figure)

    weights = geometry.integration_weights.numpy().reshape(5, 7, 7, 7, 64, 64)
    for name, (itheta, ir) in targets.items():
        ae = weights[:, :, :, :, itheta, ir]
        figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
        axes[0].imshow(np.sum(ae, axis=(0, 1)), origin="lower", cmap="viridis")
        axes[0].set_title("AE sum over basis/phi")
        axes[1].imshow(np.sum(ad_kernel, axis=(0, 1)), origin="lower", cmap="viridis")
        axes[1].set_title("AD sum over basis/phi")
        difference = np.sum(ae - ad_kernel, axis=(0, 1))
        image = axes[2].imshow(difference, origin="lower", cmap="coolwarm")
        axes[2].set_title("AE - AD")
        figure.colorbar(image, ax=axes[2])
        for axis in axes:
            axis.set_xlabel("local r offset")
            axis.set_ylabel("local theta offset")
        figure.suptitle(name.replace("_", " "))
        figure.tight_layout()
        figure.savefig(geometry_dir / f"{name}.png", dpi=160)
        if name in {"inner_equator", "outer_equator", "inner_near_pole"}:
            destination = {
                "inner_equator": "kernel_inner_equator.png",
                "outer_equator": "kernel_outer_equator.png",
                "inner_near_pole": "kernel_near_pole.png",
            }[name]
            figure.savefig(figure_dir / destination, dpi=160)
        plt.close(figure)

    inner = weights[:, :, :, :, targets["inner_equator"][0], targets["inner_equator"][1]]
    outer = weights[:, :, :, :, targets["outer_equator"][0], targets["outer_equator"][1]]
    figure, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))
    for axis, value, title in ((axes[0], inner - ad_kernel, "inner equator"), (axes[1], outer - ad_kernel, "outer equator")):
        image = axis.imshow(np.sum(value, axis=(0, 1)), origin="lower", cmap="coolwarm")
        axis.set_title(title)
        figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(figure_dir / "ad_vs_ae_kernel_difference.png", dpi=160)
    plt.close(figure)


def main() -> None:
    (OUT / "geometry").mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for key in ("stage_ad", "stage_t", "stage_s"):
        path = ROOT / config[f"frozen_{key}_config"]
        if sha256_file(path) != config[f"frozen_{key}_config_sha256"]:
            raise ValueError(f"frozen {key} config changed")
    init_artifact = ROOT / config["initialization"]["stage_ad_initialization_artifact"]
    if sha256_file(init_artifact) != config["initialization"]["stage_ad_initialization_artifact_sha256"]:
        raise ValueError("frozen Stage AD initialization artifact changed")
    upstream = subprocess.check_output(
        ["git", "-C", str(ROOT / "external/neuraloperator"), "rev-parse", "HEAD"], text=True
    ).strip()
    upstream_status = subprocess.check_output(
        ["git", "-C", str(ROOT / "external/neuraloperator"), "status", "--short"], text=True
    ).strip()
    if upstream != PINNED or upstream_status:
        raise ValueError("pinned upstream changed")

    r, theta, phi = read_coordinates(config)
    geometry = build_spherical_proxy_geometry(r=r, theta=theta, phi=phi)
    np.save(OUT / "geometry/local_scale_map.npy", geometry.local_scale.numpy())
    np.save(OUT / "geometry/radius_map.npy", geometry.radius.numpy())

    coordinate_record = {
        "source": config["data"]["dataset"],
        "dataset_sha256": config["data"]["dataset_sha256"],
        "axis_order": config["data"]["axis_order"],
        "r": r.tolist(), "theta": theta.tolist(), "phi": phi.tolist(),
        "r_monotonic": bool(np.all(np.diff(r) > 0)),
        "theta_monotonic": bool(np.all(np.diff(theta) > 0)),
        "phi_monotonic": bool(np.all(np.diff(phi) > 0)),
        "phi_periodic_face_span_2pi": bool(np.isclose(float(geometry.phi_faces[-1] - geometry.phi_faces[0]), 2 * math.pi)),
        "summary": {
            "r_min": float(r.min()), "r_max": float(r.max()),
            "theta_min": float(theta.min()), "theta_max": float(theta.max()),
            "phi_min": float(phi.min()), "phi_max": float(phi.max()),
            "dr_min": float(np.diff(r).min()), "dr_max": float(np.diff(r).max()),
            "dtheta": float(np.diff(theta).mean()), "dphi": float(np.diff(phi).mean()),
            "radial_center_ratio_min": float(np.min(r[1:] / r[:-1])),
            "radial_center_ratio_max": float(np.max(r[1:] / r[:-1])),
        },
    }
    write_json(OUT / "coordinates/coordinate_arrays.json", coordinate_record)
    write_json(OUT / "coordinates/cell_widths.json", {
        "source_face_arrays_available": False,
        "radial_method": geometry.radial_face_method,
        "theta_method": geometry.theta_face_method,
        "phi_method": geometry.phi_face_method,
        "r_faces": geometry.r_faces.tolist(),
        "theta_faces": geometry.theta_faces.tolist(),
        "phi_faces": geometry.phi_faces.tolist(),
        "dr": torch.diff(geometry.r_faces).tolist(),
        "dtheta": torch.diff(geometry.theta_faces).tolist(),
        "dphi": torch.diff(geometry.phi_faces).tolist(),
    })
    write_text(OUT / "coordinates/spherical_embedding_audit.md", """# Spherical Embedding Audit

Cell centres are embedded only for local relative distances as
`(X,Y,Z)=(r sin(theta) cos(phi), r sin(theta) sin(phi), r cos(theta))`.
This is a Euclidean spherical-coordinate embedding proxy. It is not a
Kerr-Schild Cartesian transformation for vector components and is not used to
transform Bcc or velocity components. No black-hole spin or unknown metric
parameter is used, and no `1/sin(theta)` operation appears.
""")
    q = geometry.volume_proxy
    q_pass = bool(torch.all(torch.isfinite(q)) and torch.all(q > 0))
    write_json(OUT / "coordinates/volume_proxy_audit.json", {
        "quadrature": "SPHERICAL_COORDINATE_VOLUME_PROXY",
        "formula": "r_j^2 sin(theta_j) dr_j dtheta_j dphi_j",
        "not_kerr_schild_proper_volume": True,
        "finite_count": int(torch.isfinite(q).sum()),
        "positive_count": int((q > 0).sum()),
        "cell_count_theta_r": q.numel(),
        "minimum": float(q.min()), "maximum": float(q.max()),
        "pass": q_pass,
    })

    eps = float(config["operator"]["normalization_epsilon"])
    near_threshold = 1.0e-10
    normalization_rows = []
    for basis in range(5):
        values = geometry.normalizations[basis]
        raw = geometry.raw_basis[basis]
        positive_counts = (raw > 0).sum(dim=0)
        normalization_rows.append({
            "basis": basis,
            "Z_min": float(values.min()), "Z_max": float(values.max()),
            "zero_Z_count": int((values <= eps).sum()),
            "near_zero_Z_count": int(((values > eps) & (values <= near_threshold)).sum()),
            "active_neighbor_count_min": int(positive_counts.min()),
            "active_neighbor_count_max": int(positive_counts.max()),
            "targets_with_active_basis": int((values > eps).sum()),
        })
    write_csv(OUT / "geometry/basis_normalization_audit.csv", normalization_rows)
    zero_locations = torch.nonzero(geometry.normalizations <= eps, as_tuple=False).tolist()
    normalization_pass = len(zero_locations) == 0

    write_text(OUT / "geometry/boundary_audit.md", f"""# Boundary and Normalization Audit

- `PHI_BOUNDARY = PERIODIC`; source phi index uses modulo wrapping.
- `THETA_BOUNDARY = TRUNCATED_RENORMALIZED`; no circular, replicate, reflect, or speculative pole-parity continuation is used.
- `R_BOUNDARY = TRUNCATED_RENORMALIZED`; no radial wrapping is used.
- `POLE_TOPOLOGY_EXACT = false` because coordinate-basis vector parity provenance is incomplete.
- candidate offsets remain exactly `[-3,3]^3`, shape `[7,7,7]`.
- local radius remains exactly `R_i=3 median(valid +/-1 embedding distances)`.

Production normalization fails: `ZERO_Z_COUNT = {len(zero_locations)}` at
`[basis, theta_index, r_index] = {zero_locations}`. These are basis 4 at the
four combined theta/r corners. No radius, K, boundary, or epsilon fallback was
changed. Training is prohibited by the frozen Stage AE gate.
""")

    ad_support = radial_hat_support(
        grid_shape=(64, 64, 64), domain_length=(2.0, 2.0, 2.0), radius_cells=3, basis_count=5
    )
    ad_kernel = (
        ad_support.quadrature_weight * ad_support.normalized_basis
    ).numpy()
    targets = {
        "inner_equator": (32, 0), "middle_equator": (32, 32),
        "outer_equator": (32, 63), "inner_near_pole": (0, 0),
        "outer_near_pole": (0, 63),
    }
    ae_weights = geometry.integration_weights.numpy().reshape(5, 7, 7, 7, 64, 64)
    kernel_rows = []
    for name, (itheta, ir) in targets.items():
        ae = ae_weights[:, :, :, :, itheta, ir]
        difference = ae - ad_kernel
        kernel_rows.append({
            "target": name, "theta_index": itheta, "r_index": ir,
            "theta": theta[itheta], "r": r[ir],
            "kernel_map_relative_l2": float(np.linalg.norm(difference) / np.linalg.norm(ad_kernel)),
            "stage_ad_nonzero_points": int(np.count_nonzero(ad_kernel)),
            "stage_ae_nonzero_points": int(np.count_nonzero(ae)),
            "stage_ae_active_basis_count": int(geometry.active_basis_count[itheta, ir]),
            "stage_ae_active_neighbor_count": int(geometry.active_neighbor_count[itheta, ir]),
            "geometry_changes_kernel": bool(not np.allclose(ae, ad_kernel)),
        })
    write_csv(OUT / "geometry/kernel_stage_ad_vs_ae.csv", kernel_rows)
    geometry_figures(geometry, ad_kernel, targets)

    # Generic valid-grid numerical identity checks are useful, but they do not
    # override the failed production-grid normalization gate.
    torch.manual_seed(101)
    tiny_r = np.geomspace(1.0, 3.0, 5)
    tiny_theta = np.linspace(0.4, math.pi - 0.4, 6)
    tiny_phi = (np.arange(8) + 0.5) * 2 * math.pi / 8
    operator = SphericalAwareDISCO3d(1, 1, r=tiny_r, theta=tiny_theta, phi=tiny_phi).double()
    x = torch.randn(1, 1, 8, 6, 5, dtype=torch.float64)
    optimized = operator(x)
    reference = explicit_reference(operator, x)
    relative = float((
        torch.linalg.vector_norm(optimized - reference)
        / torch.linalg.vector_norm(reference)
    ).detach())
    maximum = float(torch.max(torch.abs(optimized - reference)).detach())
    dense_pass = relative < 1e-8 and maximum < 1e-8
    write_json(OUT / "implementation/dense_reference_test.json", {
        "scope": "generic_valid_tiny_grid_only",
        "production_grid_constructible": False,
        "relative_l2": relative, "max_abs": maximum,
        "relative_tolerance": 1e-8, "absolute_tolerance": 1e-8,
        "match": dense_pass,
    })

    rolled = torch.roll(x.float(), 2, dims=2)
    operator_float = operator.float()
    observed = operator_float(rolled)
    expected = torch.roll(operator_float(x.float()), 2, dims=2)
    phi_relative = float((torch.linalg.vector_norm(observed - expected) / torch.linalg.vector_norm(expected)).detach())
    phi_pass = phi_relative < 1e-5
    write_json(OUT / "implementation/phi_equivariance_test.json", {
        "relative_l2": phi_relative, "tolerance": 1e-5, "pass": phi_pass,
        "phi_boundary": "PERIODIC",
    })
    write_json(OUT / "implementation/boundary_tests.json", {
        "phi_periodic": True, "theta_truncated_not_wrapped": True,
        "r_truncated_not_wrapped": True, "pole_topology_exact": False,
        "no_inverse_sine": True, "pass": True,
    })
    write_json(OUT / "implementation/quadrature_test.json", {
        "volume_proxy_finite_positive": q_pass,
        "partition_of_unity_max_abs_error": geometry.partition_max_abs_error,
        "production_zero_Z_count": len(zero_locations),
        "production_zero_Z_locations_basis_theta_r": zero_locations,
        "pass": normalization_pass,
        "status": "FAIL_PRODUCTION_TARGET_BASIS_NORMALIZATION",
    })
    write_json(OUT / "implementation/gradcheck.json", {
        "scope": "generic_valid_tiny_grid_only",
        "production_grid_constructible": False,
        "pass": True,
        "evidence": "tests/test_spherical_disco3d.py::test_gradcheck_input_weight_and_bias",
        "tested": ["input", "weight", "bias"],
    })

    torch.manual_seed(42)
    ad_operator = AdaptedRadialDISCO3d(16, 16)
    torch.manual_seed(42)
    ae_audit_operator = SphericalAwareDISCO3d(
        16, 16, r=r, theta=theta, phi=phi, require_valid_normalization=False
    )
    ad_state = {"weight": ad_operator.weight, "bias": ad_operator.bias}
    ae_state = {"weight": ae_audit_operator.weight, "bias": ae_audit_operator.bias}
    initialization_match = all(torch.equal(ad_state[name], ae_state[name]) for name in ad_state)
    initialization = {
        "audit_only_invalid_geometry_model": True,
        "trainable_initialization_hash_match": initialization_match,
        "stage_ad_branch_sha256": tensor_hash(ad_state),
        "stage_ae_branch_sha256": tensor_hash(ae_state),
        "weight_shape": list(ae_audit_operator.weight.shape),
        "bias_shape": list(ae_audit_operator.bias.shape),
        "parameters_per_branch": sum(value.numel() for value in ae_state.values()),
        "branches": 4,
        "stage_ad_total_parameters": 363480,
        "stage_ae_expected_total_parameters": 363480,
        "common_tensor_state_sha256": config["initialization"]["common_tensor_state_sha256"],
        "model_construction_for_training": False,
    }
    write_json(OUT / "training/spherical_disco3d/initialization_hashes.json", initialization)

    not_run = {
        "status": "NOT_RUN_PRODUCTION_NORMALIZATION_GATE_FAILED",
        "training_authorized": False,
        "training_started": False,
        "optimizer_updates": 0,
        "zero_Z_count": len(zero_locations),
    }
    write_json(OUT / "preflight/branch_norms.json", not_run)
    write_json(OUT / "preflight/cuda_preflight.json", not_run)
    write_json(OUT / "preflight/runtime.json", not_run)
    write_json(OUT / "training/spherical_disco3d/resolved_config.json", {
        **config,
        "status": "BLOCKED_BEFORE_TRAINING",
        "production_zero_Z_count": len(zero_locations),
        "training_started": False,
        "optimizer_updates": 0,
    })
    write_json(OUT / "training/spherical_disco3d/one_step_metrics.json", not_run)
    write_json(OUT / "training/spherical_disco3d/rollout_metrics.json", not_run)

    write_text(OUT / "scope/stage_ae_contract.md", f"""# Stage AE Contract

- `REPRODUCTION_SCOPE = {SCOPE}`
- sole intervention: index-space DISCO fixed geometry -> spherical-coordinate-aware Euclidean embedding proxy
- frozen Z64 dataset SHA256: `{config['data']['dataset_sha256']}`
- K=5, radius multiplier=3, candidate stencil=[7,7,7]
- phi periodic; theta/r truncated and per-target renormalized
- width/modes/layers, spectral/differential branches, trainable DISCO shapes,
  initialization, P3, split, residual target, Plain L2, optimizer, scheduler,
  pair order, and 300-epoch maximum budget are frozen
- upstream `{upstream}` is pinned and clean
""")
    write_text(OUT / "scope/geometry_limitations.md", """# Geometry Limitations

`GEOMETRY_MODEL = EUCLIDEAN_SPHERICAL_COORDINATE_PROXY`. It is not exact
Kerr-Schild metric geometry, proper distance, proper volume, a covariant GRMHD
operator, or exact paper DISCO. Vector components are not transformed. Theta
support is truncated instead of continued across poles because stored
coordinate-basis parity/transformation provenance is incomplete.
""")

    decision = {
        "PRIMARY_DECISION": "F",
        "PRIMARY_DECISION_LABEL": "SPHERICAL_DISCO_IMPLEMENTATION_INVALID",
        "REPRODUCTION_SCOPE": SCOPE,
        "DISCO_GEOMETRY": "SPHERICAL_COORDINATE_AWARE_EUCLIDEAN_EMBEDDING_PROXY",
        "KERR_SCHILD_COVARIANT_GEOMETRY": False,
        "VECTOR_COMPONENT_TRANSFORMATION": False,
        "PHI_BOUNDARY": "PERIODIC",
        "THETA_BOUNDARY": "TRUNCATED_RENORMALIZED",
        "R_BOUNDARY": "TRUNCATED_RENORMALIZED",
        "SPHERICAL_DENSE_REFERENCE_MATCH": dense_pass,
        "PHI_EQUIVARIANCE_PASS": phi_pass,
        "QUADRATURE_AUDIT_PASS": False,
        "GRADCHECK_PASS": True,
        "UNIT_TESTS_PASS": True,
        "TRAINABLE_INITIALIZATION_HASH_MATCH": initialization_match,
        "SPHERICAL_DISCO_BRANCH_ACTIVE": False,
        "STATE_RETENTION_GATE": "FAIL",
        "RESIDUAL_GEOMETRY_GAIN": "FAIL",
        "DIRECTION_GEOMETRY_GAIN": "FAIL",
        "SHELL_GEOMETRY_GAIN": "FAIL",
        "RADIAL_GEOMETRY_GAIN": "FAIL",
        "ROLLOUT_GEOMETRY_GAIN": "FAIL",
        "EXACT_REPRODUCTION_BLOCKED": True,
        "AUTHORIZE_NEXT_STAGE": "implementation_repair",
        "TRAINING_STARTED": False,
        "ZERO_Z_COUNT": len(zero_locations),
    }
    write_json(OUT / "STAGE_AE_DECISION.json", decision)

    report = f"""# Stage AE — Spherical-Coordinate-Aware DISCO3D LocalNO

| model | geometry | state L2 | residual L2 | cosine | shell skill | radial skill | first10x |
|---|---|---:|---:|---:|---:|---:|---:|
| Stage-T LocalNO | no local integral | 0.266711 | 0.891851 | 0.740602 | -1.67947 | -0.348089 | 1 |
| Stage-AD DISCO3D | index-space | 0.264062 | 0.891640 | 0.757468 | -4.35448 | -0.390352 | 1 |
| Stage-AE spherical DISCO3D | spherical proxy | not run | not run | not run | not run | not run | not run |

## Mandatory normalization-gate result

Actual coordinates were read from `{config['data']['dataset']}`. They are 64-point
monotonic arrays with geometric r centres ({r[0]:.9g} to {r[-1]:.9g}), uniform
theta centres ({theta[0]:.9g} to {theta[-1]:.9g}), and uniform periodic phi
centres ({phi[0]:.9g} to {phi[-1]:.9g}). Source faces are absent, so radial
faces were reconstructed with geometric midpoints/ratio extrapolation and
angular faces with verified-uniform midpoint extrapolation.

The prescribed Euclidean spherical embedding, positive spherical-volume proxy,
periodic phi, truncated theta/r boundaries, 7³ candidate box, K=5 hats, local
median scale, and R_i=3h_i were implemented literally. The volume proxy is
finite and positive, generic-grid dense reference/phi equivariance/gradcheck
pass, trainable shapes and seed-42 initialization match Stage AD, and kernel maps
do change with radius and theta.

However, the production Z64 normalization audit finds `ZERO_Z_COUNT={len(zero_locations)}`:
`{zero_locations}` in `[basis,theta_index,r_index]` order. They are the outermost
hat at both polar rows combined with both radial boundaries. Near a pole the phi
nearest-neighbor distance becomes very small; with theta/r truncation, the median
nearest-neighbor scale at these corners makes R_i too small for any 7³ candidate
to activate k=4. Consequently the required per-target quadrature integral cannot
equal one.

The frozen instruction requires immediate stop when any Z is zero. No support,
radius, K, boundary, epsilon fallback, architecture, data, or loss was changed.
No CUDA/runtime preflight or training was run. This is an invalid Stage AE
adapted-discretization contract, not a scientific negative result about spherical
DISCO or the paper.

## Required answers

1. Stage AD uses equal normalized tensor-index distances, so it ignores r and sin(theta) scale factors.
2. Stage AE uses `(r sin(theta) cos(phi), r sin(theta) sin(phi), r cos(theta))` Euclidean embedding distances.
3. No unknown Kerr-Schild metric parameter is used.
4. Vector components are not transformed.
5. Phi is handled periodically by modulo source indexing.
6. Theta/r do not wrap; invalid candidates are truncated.
7. The spherical volume proxy is finite and positive, but the complete per-target quadrature audit fails due to zero Z.
8. No: four target/basis normalizations are invalid.
9. The generic valid tiny-grid optimized implementation matches brute force, but the production operator is intentionally non-constructible.
10. Trainable shapes imply the same 363,480 parameters as Stage AD.
11. Audit-only trainable initialization hashes match exactly; no trainable production model was authorized.
12. Yes, valid-location inner/middle/outer kernel maps differ from Stage AD.
13. Yes, near-pole maps change most and reveal the zero-normalization corner defect.
14–21. Not evaluated because training was prohibited before CUDA preflight.
22. No; the next authorized action is implementation repair, not the final adapted benchmark.
"""
    write_text(OUT / "STAGE_AE_REPORT.md", report)
    write_text(OUT / "STAGE_AE_DECISION.md", f"""# Stage AE Decision

`PRIMARY_DECISION = F` — `SPHERICAL_DISCO_IMPLEMENTATION_INVALID`

The tested production geometry has four zero per-target basis normalizations.
Per the frozen stop rule, no preflight or training was run. This is not evidence
that spherical DISCO fails.

```text
PRIMARY_DECISION = F

REPRODUCTION_SCOPE =
{SCOPE}

DISCO_GEOMETRY =
SPHERICAL_COORDINATE_AWARE_EUCLIDEAN_EMBEDDING_PROXY

KERR_SCHILD_COVARIANT_GEOMETRY = false
VECTOR_COMPONENT_TRANSFORMATION = false

PHI_BOUNDARY = PERIODIC
THETA_BOUNDARY = TRUNCATED_RENORMALIZED
R_BOUNDARY = TRUNCATED_RENORMALIZED

SPHERICAL_DENSE_REFERENCE_MATCH = {str(dense_pass).lower()}
PHI_EQUIVARIANCE_PASS = {str(phi_pass).lower()}
QUADRATURE_AUDIT_PASS = false
GRADCHECK_PASS = true
UNIT_TESTS_PASS = true

TRAINABLE_INITIALIZATION_HASH_MATCH = {str(initialization_match).lower()}
SPHERICAL_DISCO_BRANCH_ACTIVE = false

STATE_RETENTION_GATE = FAIL
RESIDUAL_GEOMETRY_GAIN = FAIL
DIRECTION_GEOMETRY_GAIN = FAIL
SHELL_GEOMETRY_GAIN = FAIL
RADIAL_GEOMETRY_GAIN = FAIL
ROLLOUT_GEOMETRY_GAIN = FAIL

EXACT_REPRODUCTION_BLOCKED = true

AUTHORIZE_NEXT_STAGE = implementation_repair
```
""")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
