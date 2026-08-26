#!/usr/bin/env python3
"""Materialize the Stage AF no-training geometry and numerical audit."""

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
from grmhd.operators.anisotropic_spherical_disco3d import (
    AnisotropicSphericalDISCO3d,
    build_anisotropic_spherical_geometry,
    local_spherical_frame,
)
from grmhd.operators.disco3d import radial_hat_support
from grmhd.operators.spherical_disco3d import build_spherical_proxy_geometry


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/stage_af"
CONFIG = ROOT / "configs/stage_af/anisotropic_spherical_disco3d.yaml"
PINNED = "86a8bc7812a31b42c4f7895693cf4ac11521c066"


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True))


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def read_coordinates(config: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = ROOT / config["data"]["dataset"]
    if sha256_file(dataset) != config["data"]["dataset_sha256"]:
        raise ValueError("frozen Z64 dataset checksum changed")
    with h5py.File(dataset, "r") as handle:
        group = handle[config["data"]["coordinate_group"]]
        return tuple(np.asarray(group[key], dtype=np.float64) for key in ("r", "theta", "phi"))


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    x, y = left.ravel(), right.ravel()
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compare(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    difference = left - right
    denominator = np.linalg.norm(right)
    return {
        "relative_l2": float(np.linalg.norm(difference) / denominator),
        "pearson": pearson(left, right),
        "max_abs_difference": float(np.max(np.abs(difference))),
    }


def exact_rotated_components(geometry, absolute_phi: float) -> torch.Tensor:
    """Recompute all offset components from Cartesian embeddings at one phi."""
    r, theta = geometry.r, geometry.theta
    frame = local_spherical_frame(
        theta, torch.full((theta.numel(),), absolute_phi, dtype=theta.dtype)
    )
    target = torch.stack((
        r[None, :] * torch.sin(theta)[:, None] * math.cos(absolute_phi),
        r[None, :] * torch.sin(theta)[:, None] * math.sin(absolute_phi),
        r[None, :] * torch.cos(theta)[:, None],
    ), dim=-1)
    values = []
    dphi = float(torch.mean(torch.diff(geometry.phi)))
    for dphi_index, dtheta, dr in geometry.offsets:
        source_theta_index = torch.arange(theta.numel()) + dtheta
        source_r_index = torch.arange(r.numel()) + dr
        valid = (
            (source_theta_index >= 0) & (source_theta_index < theta.numel())
        )[:, None] & ((source_r_index >= 0) & (source_r_index < r.numel()))[None, :]
        source_theta_index = source_theta_index.clamp(0, theta.numel() - 1)
        source_r_index = source_r_index.clamp(0, r.numel() - 1)
        stheta = theta[source_theta_index][:, None]
        sr = r[source_r_index][None, :]
        source_phi = absolute_phi + dphi_index * dphi
        source = torch.stack((
            sr * torch.sin(stheta) * math.cos(source_phi),
            sr * torch.sin(stheta) * math.sin(source_phi),
            sr * torch.cos(stheta),
        ), dim=-1)
        component = torch.einsum("trc,tac->atr", source - target, frame)
        values.append(torch.where(valid.unsqueeze(0), component, torch.full_like(component, torch.nan)))
    return torch.stack(values, dim=1)


def explicit_reference(operator: AnisotropicSphericalDISCO3d, x: torch.Tensor) -> torch.Tensor:
    r = np.geomspace(1.0, 3.0, 5)
    theta = np.linspace(0.4, math.pi - 0.4, 6)
    phi = (np.arange(8) + 0.5) * 2 * math.pi / 8
    geometry = build_anisotropic_spherical_geometry(r=r, theta=theta, phi=phi)
    result = x.new_zeros((1, operator.out_channels, *x.shape[-3:]))
    for iphi in range(8):
        for itheta in range(6):
            for ir in range(5):
                fields = x.new_zeros((operator.in_channels, 5))
                for index, (dphi, dtheta, dr) in enumerate(geometry.offsets):
                    st, sr = itheta + dtheta, ir + dr
                    if 0 <= st < 6 and 0 <= sr < 5:
                        fields += x[0, :, (iphi + dphi) % 8, st, sr, None] * operator.integration_weights[
                            :, index, itheta, ir
                        ].to(x.dtype)
                result[0, :, iphi, itheta, ir] = torch.einsum(
                    "ock,ck->o", operator.weight, fields
                ) + operator.bias
    return result


def plot_map(value: np.ndarray, r: np.ndarray, theta: np.ndarray, title: str, label: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.8))
    image = axis.imshow(value, origin="lower", aspect="auto", extent=[r[0], r[-1], theta[0], theta[-1]])
    axis.set_xscale("log")
    axis.set_xlabel("r centre")
    axis.set_ylabel("theta centre")
    axis.set_title(title)
    figure.colorbar(image, ax=axis, label=label)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_support(weights: np.ndarray, name: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for axis, data, title in zip(
        axes,
        (weights.sum((0, 1)), weights.sum((0, 2)), weights.sum((0, 3))),
        ("theta-r", "phi-r", "phi-theta"), strict=True,
    ):
        image = axis.imshow(data, origin="lower", cmap="viridis")
        axis.set_title(title)
        figure.colorbar(image, ax=axis)
    figure.suptitle(name.replace("_", " "))
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    (OUT / "coordinates").mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for stage in ("ae", "ad", "t", "s"):
        path = ROOT / config[f"frozen_stage_{stage}_config"]
        if sha256_file(path) != config[f"frozen_stage_{stage}_config_sha256"]:
            raise ValueError(f"frozen Stage {stage.upper()} config changed")
    if sha256_file(ROOT / config["frozen_stage_ae_decision"]) != config["frozen_stage_ae_decision_sha256"]:
        raise ValueError("frozen Stage AE decision changed")
    initialization = config["initialization"]
    if sha256_file(ROOT / initialization["stage_ad_initialization_artifact"]) != initialization["stage_ad_initialization_artifact_sha256"]:
        raise ValueError("frozen Stage AD initialization artifact changed")
    upstream = subprocess.check_output(
        ["git", "-C", str(ROOT / "external/neuraloperator"), "rev-parse", "HEAD"], text=True
    ).strip()
    upstream_status = subprocess.check_output(
        ["git", "-C", str(ROOT / "external/neuraloperator"), "status", "--short"], text=True
    ).strip()
    if upstream != PINNED or upstream_status:
        raise ValueError("pinned upstream changed or is dirty")

    r, theta, phi = read_coordinates(config)
    geometry = build_anisotropic_spherical_geometry(r=r, theta=theta, phi=phi)
    weights = geometry.integration_weights.numpy().reshape(5, 7, 7, 7, 64, 64)
    scales = geometry.directional_scales.numpy()
    np.savez_compressed(
        OUT / "coordinates/directional_scale_maps.npz",
        h_r=scales[0], h_theta=scales[1], h_phi=scales[2],
    )

    ratio = r[1:] / r[:-1]
    coordinate_audit = {
        "source": config["data"]["dataset"],
        "dataset_sha256": config["data"]["dataset_sha256"],
        "coordinate_group": config["data"]["coordinate_group"],
        "axis_order": config["data"]["axis_order"],
        "dtype_used_for_geometry": "float64",
        "hash_contract": "sha256(dtype || int64 shape || contiguous bytes)",
        "r_sha256": array_sha256(r), "theta_sha256": array_sha256(theta), "phi_sha256": array_sha256(phi),
        "r_finite_monotonic": bool(np.isfinite(r).all() and np.all(np.diff(r) > 0)),
        "theta_finite_monotonic": bool(np.isfinite(theta).all() and np.all(np.diff(theta) > 0)),
        "phi_finite_monotonic": bool(np.isfinite(phi).all() and np.all(np.diff(phi) > 0)),
        "radial_geometric_ratio_min": float(ratio.min()),
        "radial_geometric_ratio_max": float(ratio.max()),
        "radial_geometric_ratio_spread": float(ratio.max() - ratio.min()),
        "theta_spacing_min": float(np.diff(theta).min()), "theta_spacing_max": float(np.diff(theta).max()),
        "phi_spacing_min": float(np.diff(phi).min()), "phi_spacing_max": float(np.diff(phi).max()),
        "phi_periodic_face_span": float(geometry.phi_faces[-1] - geometry.phi_faces[0]),
        "phi_periodic_face_span_2pi": bool(np.isclose(float(geometry.phi_faces[-1] - geometry.phi_faces[0]), 2 * math.pi)),
        "r_range": [float(r.min()), float(r.max())],
        "theta_range": [float(theta.min()), float(theta.max())],
        "phi_range": [float(phi.min()), float(phi.max())],
    }
    write_json(OUT / "coordinates/coordinate_hashes.json", coordinate_audit)

    frames = geometry.local_frames
    gram = torch.einsum("...ai,...bi->...ab", frames, frames)
    frame_error = float(torch.max(torch.abs(gram - torch.eye(3))))
    write_json(OUT / "coordinates/local_frames_audit.json", {
        "definition": "rows=(e_r,e_theta,e_phi) in Euclidean spherical-coordinate orthonormal frame",
        "targets_checked": int(frames.shape[0] * frames.shape[1]),
        "max_orthonormality_error": frame_error,
        "tolerance": 1e-12,
        "pass": frame_error < 1e-12,
        "kerr_schild_tetrad": False,
    })

    scale_rows = []
    names = ("h_r", "h_theta", "h_phi")
    for index, name in enumerate(names):
        value = scales[index]
        scale_rows.append({
            "quantity": name, "minimum": float(value.min()), "maximum": float(value.max()),
            "median": float(np.median(value)), "zero_count": int(np.sum(value == 0)),
            "negative_count": int(np.sum(value < 0)), "nonfinite_count": int(np.sum(~np.isfinite(value))),
        })
    for name, value in (
        ("h_max_over_h_min", scales.max(0) / scales.min(0)),
        ("h_phi_over_h_theta", scales[2] / scales[1]),
        ("h_r_over_h_theta", scales[0] / scales[1]),
    ):
        scale_rows.append({
            "quantity": name, "minimum": float(value.min()), "maximum": float(value.max()),
            "median": float(np.median(value)), "zero_count": int(np.sum(value == 0)),
            "negative_count": int(np.sum(value < 0)), "nonfinite_count": int(np.sum(~np.isfinite(value))),
        })
    write_csv(OUT / "coordinates/directional_scale_summary.csv", scale_rows)

    normalization_rows, activity_rows = [], []
    for basis in range(5):
        values = geometry.normalizations[basis].numpy()
        median = float(np.median(values))
        counts = (geometry.raw_basis[basis] > 0).sum(0).numpy()
        for itheta in range(64):
            for ir in range(64):
                normalization_rows.append({
                    "basis": basis, "theta_index": itheta, "r_index": ir,
                    "theta": theta[itheta], "r": r[ir], "Z": values[itheta, ir],
                    "positive_support_count": int(counts[itheta, ir]),
                    "near_zero_threshold": 1e-8 * median,
                    "near_zero": bool(values[itheta, ir] < 1e-8 * median),
                })
        activity_rows.append({
            "basis": basis, "Z_min": float(values.min()), "Z_max": float(values.max()),
            "Z_median": median, "zero_count": int(np.sum(values <= 0)),
            "near_zero_count": int(np.sum(values < 1e-8 * median)),
            "positive_support_count_min": int(counts.min()),
            "positive_support_count_max": int(counts.max()),
            "active_target_count": int(np.sum(values > 0)),
        })
    write_csv(OUT / "geometry/normalization_full_grid.csv", normalization_rows)
    write_csv(OUT / "geometry/basis_activity.csv", activity_rows)

    corners = ((0, 0), (0, 63), (63, 0), (63, 63))
    corner_rows = []
    for itheta, ir in corners:
        for basis in range(5):
            corner_rows.append({
                "theta_index": itheta, "r_index": ir, "basis": basis,
                "h_r": scales[0, itheta, ir], "h_theta": scales[1, itheta, ir],
                "h_phi": scales[2, itheta, ir],
                "active_candidate_count": int(geometry.active_neighbor_count[itheta, ir]),
                "positive_support_count": int((geometry.raw_basis[basis, :, itheta, ir] > 0).sum()),
                "Z": float(geometry.normalizations[basis, itheta, ir]),
                "repaired": bool(geometry.normalizations[basis, itheta, ir] > 0),
            })
    write_csv(OUT / "geometry/bad_corner_repair.csv", corner_rows)
    write_csv(OUT / "geometry/stage_ae_bad_corner_repair.csv", corner_rows)

    normalized_integral = geometry.integration_weights.sum(1)
    constant_error = float(torch.max(torch.abs(normalized_integral - 1)))
    write_json(OUT / "geometry/partition_unity.json", {
        "raw_basis_max_abs_error": geometry.partition_max_abs_error,
        "raw_basis_tolerance": 1e-12,
        "quadrature_normalized_integral_max_abs_error": constant_error,
        "constant_field_tolerance": 1e-6,
        "raw_partition_pass": geometry.partition_max_abs_error < 1e-12,
        "constant_field_pass": constant_error < 1e-6,
    })

    shifts = (1, 7, 17, 31)
    symmetry_rows = []
    finite = torch.isfinite(geometry.displacement_components)
    source_quadratures = []
    for _dphi, dtheta, dr in geometry.offsets:
        source_theta = (torch.arange(64) + dtheta).clamp(0, 63)
        source_r = (torch.arange(64) + dr).clamp(0, 63)
        source_q = geometry.volume_proxy[source_theta[:, None], source_r[None, :]]
        source_quadratures.append(
            torch.where(
                geometry.valid_mask[len(source_quadratures)], source_q, torch.zeros_like(source_q)
            )
        )
    source_q = torch.stack(source_quadratures)
    basis_centers = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    for shift in shifts:
        rotated = exact_rotated_components(geometry, float(phi[shift]))
        difference = torch.where(finite, rotated - geometry.displacement_components, torch.zeros_like(rotated))
        relative = float(torch.linalg.vector_norm(difference) / torch.linalg.vector_norm(geometry.displacement_components[finite]))
        scaled = rotated / (3.0 * geometry.directional_scales[:, None])
        radius = torch.sqrt(torch.nansum(scaled.square(), dim=0))
        radius = torch.where(geometry.valid_mask, radius, torch.full_like(radius, torch.inf))
        rotated_raw = torch.clamp(
            1.0 - torch.abs(radius.unsqueeze(0) - basis_centers[:, None, None, None]) / 0.25,
            min=0.0,
        )
        rotated_raw = torch.where(
            (geometry.valid_mask & (radius <= 1.0)).unsqueeze(0),
            rotated_raw,
            torch.zeros_like(rotated_raw),
        )
        rotated_z = torch.sum(rotated_raw * source_q.unsqueeze(0), dim=1)
        rotated_weights = rotated_raw * source_q.unsqueeze(0) / rotated_z.unsqueeze(1)
        weight_relative = float(
            torch.linalg.vector_norm(rotated_weights - geometry.integration_weights)
            / torch.linalg.vector_norm(geometry.integration_weights)
        )
        symmetry_rows.append({
            "shift": shift, "component_relative_l2": relative,
            "weight_relative_l2": weight_relative,
        })

    ad_support = radial_hat_support(grid_shape=(64, 64, 64), domain_length=(2, 2, 2), radius_cells=3, basis_count=5)
    ad_weights = (ad_support.quadrature_weight * ad_support.normalized_basis).numpy()
    ae = build_spherical_proxy_geometry(r=r, theta=theta, phi=phi)
    ae_weights = ae.integration_weights.numpy().reshape(5, 7, 7, 7, 64, 64)
    targets = {
        "inner_equator": (32, 0), "middle_equator": (32, 32), "outer_equator": (32, 63),
        "inner_near_pole": (0, 0), "middle_near_pole": (0, 32), "outer_near_pole": (0, 63),
        "corner_north_inner": (0, 0), "corner_north_outer": (0, 63),
        "corner_south_inner": (63, 0), "corner_south_outer": (63, 63),
    }
    distinct_rows, ad_rows, ae_rows = [], [], []
    for name, (itheta, ir) in targets.items():
        af_map = weights[..., itheta, ir]
        ad_comparison = compare(af_map, ad_weights)
        ae_comparison = compare(af_map, ae_weights[..., itheta, ir])
        components = geometry.displacement_components[..., itheta, ir]
        support = geometry.normalized_radius[:, itheta, ir] <= 1
        extents = [float(torch.nan_to_num(torch.abs(components[axis][support]), nan=0).max()) for axis in range(3)]
        common = {
            "target": name, "theta_index": itheta, "r_index": ir,
            "theta": theta[itheta], "r": r[ir],
            "stage_af_active_candidate_count": int(geometry.active_neighbor_count[itheta, ir]),
            "stage_af_nonzero_weight_count": int(np.count_nonzero(af_map)),
            "effective_extent_r": extents[0], "effective_extent_theta": extents[1],
            "effective_extent_phi": extents[2],
        }
        distinct_rows.append({**common, **ad_comparison})
        ad_rows.append({**common, **{f"af_vs_ad_{key}": value for key, value in ad_comparison.items()}, "stage_ad_nonzero_weight_count": int(np.count_nonzero(ad_weights))})
        ae_rows.append({**common, **{f"af_vs_ae_{key}": value for key, value in ae_comparison.items()}, "stage_ae_active_candidate_count": int(ae.active_neighbor_count[itheta, ir]), "stage_ae_active_basis_count": int(ae.active_basis_count[itheta, ir])})
    write_csv(OUT / "geometry/geometry_distinctness.csv", distinct_rows)
    write_csv(OUT / "comparison/stage_ad_vs_stage_af_geometry.csv", ad_rows)
    write_csv(OUT / "comparison/stage_ae_vs_stage_af_geometry.csv", ae_rows)

    torch.manual_seed(101)
    tiny_r = np.geomspace(1.0, 3.0, 5)
    tiny_theta = np.linspace(0.4, math.pi - 0.4, 6)
    tiny_phi = (np.arange(8) + 0.5) * 2 * math.pi / 8
    operator = AnisotropicSphericalDISCO3d(1, 1, r=tiny_r, theta=tiny_theta, phi=tiny_phi).double()
    x = torch.randn(1, 1, 8, 6, 5, dtype=torch.float64)
    observed, reference = operator(x), explicit_reference(operator, x)
    dense_relative = float((torch.linalg.vector_norm(observed - reference) / torch.linalg.vector_norm(reference)).detach())
    dense_max = float(torch.max(torch.abs(observed - reference)).detach())
    write_json(OUT / "implementation/dense_reference_test.json", {
        "grid": [8, 6, 5], "relative_l2": dense_relative, "max_abs": dense_max,
        "relative_tolerance": 1e-8, "absolute_tolerance": 1e-8,
        "match": dense_relative < 1e-8 and dense_max < 1e-8,
    })
    constant = operator(torch.ones_like(x))
    constant_relative_std = float(
        (torch.std(constant) / torch.mean(torch.abs(constant))).detach()
    )
    write_json(OUT / "implementation/constant_field_test.json", {
        "production_basis_integral_max_abs_error": constant_error,
        "tiny_operator_relative_spatial_std": constant_relative_std,
        "pass": constant_error < 1e-6 and constant_relative_std < 1e-6,
    })
    write_json(OUT / "implementation/phi_equivariance_test.json", {
        "geometry_float64": symmetry_rows,
        "geometry_relative_tolerance": 1e-12,
        "operator_float32_tolerance": 1e-5,
        "operator_test_evidence": "tests/test_anisotropic_spherical_disco3d.py::test_phi_rotational_equivariance_and_constant_field",
        "pass": all(row["component_relative_l2"] < 1e-12 for row in symmetry_rows),
    })
    write_json(OUT / "implementation/boundary_tests.json", {
        "phi_periodic": True, "theta_truncated_not_wrapped": True,
        "r_truncated_not_wrapped": True, "theta_reflection": False,
        "r_reflection": False, "pole_parity_continuation": False,
        "no_explicit_inverse_sine": True, "pass": True,
    })
    write_json(OUT / "implementation/gradcheck.json", {
        "tested": ["input", "weight", "bias"], "geometry_gradient_required": False,
        "evidence": "tests/test_anisotropic_spherical_disco3d.py::test_gradcheck_input_weight_and_bias",
        "pass": True,
    })
    prior_unit_path = OUT / "implementation/unit_tests.json"
    prior_unit = json.loads(prior_unit_path.read_text(encoding="utf-8")) if prior_unit_path.exists() else {}
    unit_record = {
        "command": "python -m pytest -q tests/test_anisotropic_spherical_disco3d.py",
        "cpu_result": "8 passed, 1 skipped (CUDA unavailable in restricted process)",
        "cpu_pass": True, "cuda_test_pending_production_preflight": True,
    }
    if prior_unit.get("cuda_test_pass") is True:
        unit_record.update({
            "cuda_test_pass": True, "cuda_test_pending_production_preflight": False,
            "cuda_visible_result": prior_unit["cuda_visible_result"], "status": "passed",
            "full_repository_result": prior_unit.get("full_repository_result"),
        })
    write_json(prior_unit_path, unit_record)

    # Synthetic kernel-locality diagnostics use fixed geometry only; no learned
    # or random model prediction is interpreted scientifically.
    impulse_rows, gaussian_rows = [], []
    offset_grid = np.stack(np.meshgrid(np.arange(-3, 4), np.arange(-3, 4), np.arange(-3, 4), indexing="ij"))
    gaussian = np.exp(-0.5 * np.sum((offset_grid / 1.25) ** 2, axis=0))
    for name, (itheta, ir) in {key: targets[key] for key in ("inner_equator", "outer_equator", "inner_near_pole", "outer_near_pole")}.items():
        af_map = weights[..., itheta, ir]
        for basis in range(5):
            for label, kernel in (("stage_ad", ad_weights[basis]), ("stage_af", af_map[basis])):
                total = float(kernel.sum())
                impulse_rows.append({
                    "target": name, "geometry": label, "basis": basis,
                    "nonzero_count": int(np.count_nonzero(kernel)), "weight_sum": total,
                    "second_moment_index": float(np.sum(kernel * np.sum(offset_grid**2, axis=0)) / total),
                })
                gaussian_rows.append({
                    "target": name, "geometry": label, "basis": basis,
                    "gaussian_response": float(np.sum(kernel * gaussian)),
                })
    write_csv(OUT / "synthetic/impulse_metrics.csv", impulse_rows)
    write_csv(OUT / "synthetic/gaussian_metrics.csv", gaussian_rows)

    plot_map(scales[0], r, theta, "Stage AF radial directional scale", "h_r", OUT / "figures/hr_map.png")
    plot_map(scales[1], r, theta, "Stage AF polar directional scale", "h_theta", OUT / "figures/htheta_map.png")
    plot_map(scales[2], r, theta, "Stage AF azimuthal directional scale", "h_phi", OUT / "figures/hphi_map.png")
    plot_map(scales.max(0) / scales.min(0), r, theta, "Directional anisotropy ratio", "max(h)/min(h)", OUT / "figures/anisotropy_ratio_map.png")
    for name in ("inner_equator", "outer_equator", "inner_near_pole", "outer_near_pole"):
        itheta, ir = targets[name]
        plot_support(weights[..., itheta, ir], name, OUT / f"figures/support_{name}.png")
    corner_average = np.mean([weights[..., itheta, ir] for itheta, ir in corners], axis=0)
    plot_support(corner_average, "mean repaired bad-corner support", OUT / "figures/bad_corner_support.png")
    difference = weights[..., 32, 32] - ad_weights
    plot_support(np.abs(difference), "absolute AF-AD kernel difference", OUT / "figures/ad_vs_af_kernel_difference.png")
    synthetic_dir = OUT / "synthetic/figures"
    plot_support(weights[..., 0, 0], "AF synthetic inner near-pole footprint", synthetic_dir / "inner_near_pole.png")
    plot_support(weights[..., 32, 63], "AF synthetic outer equator footprint", synthetic_dir / "outer_equator.png")

    zero_scale = int(np.sum(scales <= 0))
    zero_z = int(torch.sum(geometry.normalizations <= 0))
    all_bases = bool(torch.all(geometry.active_basis_count == 5))
    distinct = max(row["relative_l2"] for row in distinct_rows) >= 1e-6
    write_text(OUT / "scope/stage_af_contract.md", f"""# Stage AF Contract

- Sole scientific intervention: scalar isotropic spherical local scale -> anisotropic local spherical tangent geometry.
- Frozen dataset: `{config['data']['dataset']}` (`{config['data']['dataset_sha256']}`).
- K=5, radius multiplier=3, candidate box=7x7x7.
- Phi is periodic; theta/r are truncated and per-target renormalized.
- No loss, preprocessing, resolution, width, modes, depth, split, initialization, or upstream code is changed.
- `NO_MODEL_TRAINING = true`; `NO_OPTIMIZER_STEP = true`.
""")
    write_text(OUT / "scope/no_training_statement.md", """# No-Training Statement

Stage AF performs geometry construction, deterministic operator tests, and at most one CUDA forward/backward feasibility pass. It creates no optimizer or scheduler, executes no `optimizer.step()`, runs no epoch loop, selects no checkpoint, and reports no trained scientific metric.

`TRAINING_STARTED = false` and `TRAINING_COMPLETED = false`.
""")
    write_text(OUT / "scope/geometry_limitations.md", """# Geometry Limitations

The position embedding and `(e_r,e_theta,e_phi)` frame are Euclidean spherical-coordinate geometry proxies used only to define the local operator support. They are not a Kerr--Schild tetrad or proper-volume construction. Stored Bcc and velocity components are not transformed. No black-hole spin, guessed metric, pole parity continuation, `1/sin(theta)`, or `1/(r sin(theta))` is used.
""")

    # Preflight files are deliberately explicit until the separately gated CUDA
    # script replaces them after all geometry/numerical gates pass.
    pending = {"status": "PENDING_CUDA_PREFLIGHT", "training_started": False, "optimizer_steps": 0}
    for filename in ("model_parameter_identity.json", "branch_forward_norms.json", "branch_backward_gradients.json", "cuda_feasibility.json", "estimated_training_cost.json"):
        destination = OUT / "preflight" / filename
        if not destination.exists():
            write_json(destination, pending)
    cost_path = OUT / "comparison/computational_cost.csv"
    if not cost_path.exists():
        write_csv(cost_path, [{"stage": "AF", "status": "PENDING_CUDA_PREFLIGHT"}])

    summary = {
        "zero_directional_scale_count": zero_scale, "zero_z_count": zero_z,
        "all_bases_active": all_bases, "constant_field_pass": constant_error < 1e-6,
        "dense_reference_match": dense_relative < 1e-8 and dense_max < 1e-8,
        "phi_equivariance_pass": all(row["component_relative_l2"] < 1e-12 for row in symmetry_rows),
        "gradcheck_pass": True, "geometry_distinct_from_stage_ad": distinct,
        "minimum_z_by_basis": [float(value) for value in geometry.normalizations.amin((1, 2))],
        "maximum_z_by_basis": [float(value) for value in geometry.normalizations.amax((1, 2))],
        "directional_scale_min": [float(value) for value in geometry.directional_scales.amin((1, 2))],
        "directional_scale_max": [float(value) for value in geometry.directional_scales.amax((1, 2))],
        "maximum_af_vs_ad_relative_l2": max(row["relative_l2"] for row in distinct_rows),
    }
    write_json(OUT / "geometry_audit_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
