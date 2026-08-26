#!/usr/bin/env python3
"""Run Stage U no-training source, grid, FD, boundary, and FFT audits."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping

import h5py
import matplotlib.pyplot as plt
import numpy as np
import yaml

from grmhd.dataset import sha256_file
from grmhd.stage_u_geometry import (
    coordinate_gradient,
    inferred_faces,
    padded_centered_derivative_1d,
    relative_l2,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/stage_u"
AXES = {"phi": 0, "theta": 1, "r": 2}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty audit table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def stats(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
    }


def load_contract() -> tuple[dict[str, Any], dict[str, Any]]:
    stage_t = yaml.safe_load((ROOT / "configs/stage_t/optimization_convergence.yaml").read_text())
    stage_s_path = ROOT / stage_t["frozen_stage_s_config"]
    if sha256_file(stage_s_path) != stage_t["frozen_stage_s_config_sha256"]:
        raise ValueError("Frozen Stage S configuration changed")
    stage_s = yaml.safe_load(stage_s_path.read_text())
    checks = {
        ROOT / stage_s["data"]["dataset"]: stage_s["data"]["dataset_sha256"],
        ROOT / stage_s["preprocessing"]["artifact"] / "normalizer.npz": stage_s["preprocessing"]["normalizer_sha256"],
        ROOT / stage_s["frozen_pairing"]["initial_state"]: stage_s["frozen_pairing"]["initial_state_file_sha256"],
    }
    for path, expected in checks.items():
        if sha256_file(path) != expected:
            raise ValueError(f"Frozen Stage U input changed: {path}")
    upstream = subprocess.check_output(
        ["git", "-C", str(ROOT / "external/neuraloperator"), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(ROOT / "external/neuraloperator"), "status", "--short"], text=True
    ).strip()
    if upstream != stage_s["frozen_pairing"]["upstream_commit"] or dirty:
        raise ValueError("Pinned upstream provenance changed")
    return stage_t, stage_s


def grid_audit(stage_s: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    dataset = ROOT / stage_s["data"]["dataset"]
    with h5py.File(dataset, "r") as handle:
        coords = {name: np.asarray(handle[f"coords/{name}"], dtype=np.float64) for name in AXES}
        stored = sorted(handle["coords"].keys())
        metadata = json.loads(handle.attrs["metadata_json"])
        shape = [int(value) for value in handle["snapshots"].shape]
    r, theta, phi = coords["r"], coords["theta"], coords["phi"]
    dr, dt, dp = np.diff(r), np.diff(theta), np.diff(phi)
    r_faces = inferred_faces(r)
    theta_faces = inferred_faces(theta)
    phi_faces = inferred_faces(phi, periodic_extent=2.0 * np.pi)
    rr = r.reshape(1, 1, -1)
    tt = theta.reshape(1, -1, 1)
    dr_cell = np.gradient(r).reshape(1, 1, -1)
    dtheta_cell = np.gradient(theta).reshape(1, -1, 1)
    dphi_cell = np.gradient(phi).reshape(-1, 1, 1)
    dl_r = np.broadcast_to(dr_cell, (64, 64, 64))
    dl_theta = np.broadcast_to(rr * dtheta_cell, (64, 64, 64))
    dl_phi = np.broadcast_to(rr * np.sin(tt) * dphi_cell, (64, 64, 64))
    thirds = {"inner": slice(0, 64 // 3), "middle": slice(64 // 3, 2 * 64 // 3), "outer": slice(2 * 64 // 3, 64)}
    proxies: dict[str, Any] = {}
    for name, values in (("dl_r", dl_r), ("dl_theta", dl_theta), ("dl_phi", dl_phi)):
        proxies[name] = {
            "global": stats(values),
            **{shell: stats(values[..., section]) for shell, section in thirds.items()},
        }
    delta_log = np.diff(np.log(r))
    ratio = r[1:] / r[:-1]
    payload = {
        "schema_version": "stage-u-grid-geometry-v1",
        "dataset": str(dataset.relative_to(ROOT)),
        "dataset_sha256": sha256_file(dataset),
        "axis_order": ["phi", "theta", "r"],
        "snapshot_shape": shape,
        "stored_coordinate_datasets": stored,
        "face_arrays": {
            "stored": False,
            "status": "not available; inferred from centers for audit only",
            "r": r_faces.tolist(), "theta": theta_faces.tolist(), "phi": phi_faces.tolist(),
        },
        "radial": {
            "count": len(r), "minimum": float(r.min()), "maximum": float(r.max()),
            "spacing": stats(dr), "center_ratio": stats(ratio), "delta_log_r": stats(delta_log),
            "geometric_or_log_grid": bool(np.allclose(delta_log, delta_log[0], rtol=1e-10, atol=1e-12)),
            "inferred_face_bounds": [float(r_faces[0]), float(r_faces[-1])],
        },
        "theta": {
            "count": len(theta), "minimum": float(theta.min()), "maximum": float(theta.max()),
            "spacing": stats(dt), "uniform": bool(np.allclose(dt, dt[0], rtol=1e-10, atol=1e-12)),
            "inferred_face_bounds": [float(theta_faces[0]), float(theta_faces[-1])],
        },
        "phi": {
            "count": len(phi), "minimum": float(phi.min()), "maximum": float(phi.max()),
            "spacing": stats(dp), "uniform": bool(np.allclose(dp, dp[0], rtol=1e-10, atol=1e-12)),
            "full_2pi_by_inferred_faces": bool(np.isclose(phi_faces[-1] - phi_faces[0], 2 * np.pi)),
            "seam_policy": "periodic cell centers; inferred faces at 0 and 2pi",
            "inferred_face_bounds": [float(phi_faces[0]), float(phi_faces[-1])],
        },
        "physical_scale_proxy_label": "SPHERICAL_COORDINATE_SCALE_PROXY",
        "proper_distance_warning": "Not a strict Kerr-Schild proper distance; no metric factors beyond r and r*sin(theta).",
        "physical_scale_proxies": proxies,
        "tensor_cell_scale_ratios": {
            "radial_max_over_min": float(dl_r.max() / dl_r.min()),
            "theta_max_over_min": float(dl_theta.max() / dl_theta.min()),
            "phi_max_over_min": float(dl_phi.max() / dl_phi.min()),
        },
        "hdf5_metadata": metadata,
    }
    write_json(OUT / "grid_geometry_audit.json", payload)
    return payload, coords


def source_audit(stage_t: Mapping[str, Any], stage_s: Mapping[str, Any], grid: Mapping[str, Any]) -> None:
    model = stage_s["model"]
    freeze = {
        "DATASET_FROZEN": True, "SPLIT_FROZEN": True, "PREPROCESSING_FROZEN": True,
        "TARGET_CONTRACT_FROZEN": stage_s["prediction"]["mode"] == "normalized_residual",
        "LOSS_FROZEN": stage_s["loss"]["name"] == "PlainL2Loss",
    }
    text = f"""# Stage U Operator Source Audit

## Frozen contract

```text
{chr(10).join(f'{key} = {str(value).lower()}' for key, value in freeze.items())}
```

- Dataset: `{stage_s['data']['dataset']}` (`{stage_s['data']['dataset_sha256']}`)
- P3 normalizer: `{stage_s['preprocessing']['artifact']}/normalizer.npz` (`{stage_s['preprocessing']['normalizer_sha256']}`)
- Split: train pair sources `0..167`, dropped `168->169`, validation pair sources `169..210`.
- Target/reconstruction: `z[t+1]-z[t]`, then `z[t]+model_residual`; loss is unmodified `PlainL2Loss`.
- Seed/width/modes/layers: `{stage_s['runtime']['seed']}` / `{model['hidden_channels']}` / `{model['n_modes']}` / `{model['n_layers']}`.
- Stage T optimization horizon: 1200 epochs, 50,400 updates, 3,150 warmup updates. Stage U pilots stop at epoch 150 but evaluate the same scheduler's first 6,300 updates.

## Spectral branch

- Evidence: `external/neuraloperator/neuralop/layers/spectral_convolution.py:429-449` obtains the last three spatial axes and calls `torch.fft.rfftn(..., dim=[-3,-2,-1])`. Model tensors are `(batch, channel, phi, theta, r)`, so FFT axes are `phi, theta, r`.
- Config requests `n_modes=[8,8,8]`. Source lines 400-415 convert the real-FFT last-axis storage to `[8,8,5]`; this corresponds to 8 total requested modes per axis.
- A standard Fourier basis makes a periodic-extension/translation-invariant index-space assumption on every transformed axis. This is appropriate at the phi seam, but is not a physical boundary model for theta or r.
- `domain_padding=None` because Stage T passes none and `LocalNO` source lines 291-300 disables it. No spectral domain padding/unpadding is active.

## Differential branch

- Evidence: `external/neuraloperator/neuralop/layers/differential_conv.py:6-101` explicitly documents a regular grid and implements a bias-free `Conv3d` followed by coefficient-sum centering and division by one `grid_width` scalar.
- Resolved kernel is `3x3x3`, groups=1 (`mix_derivatives=true`), four layers, each weight shape `(16,16,3,3,3)`.
- `conv_padding_mode=periodic` maps to PyTorch `circular`; it is applied identically on phi, theta, and r (`differential_conv.py:59-79`).
- `local_no_block.py:468-471` computes `grid_width_scaling_factor = 1/(x.shape[-1]/default_in_shape[0])`. At frozen 64^3, this is exactly `1.0`, derived only from the last tensor size, and reused for all three axes.
- It does not read stored `r/theta/phi`, does not know `dr` varies by a factor `{grid['tensor_cell_scale_ratios']['radial_max_over_min']:.3f}`, and does not use `1/r` or `1/(r sin(theta))`.

## Positional and boundary information

- `positional_embedding=null`, so `GridEmbeddingND` is disabled (`local_no.py:267-300,429-444`). The network receives no continuous r, theta, or phi coordinate channels.
- The extra eight inputs are one-hot radial shell memberships constructed from physical r in `scripts/train_stage_s.py:90-96`. They provide coarse radial region identity, not continuous coordinates, theta, or phi.
- Physical data policy: phi is periodic; theta spans pole-to-pole without a periodic seam; r is bounded and nonperiodic.
- Model policy: all three differential axes are circular. Therefore:

```text
PHI_PERIODIC = true
THETA_PERIODIC_IN_MODEL = true
R_PERIODIC_IN_MODEL = true
```

The last two are operator assumptions inconsistent with the stored spherical-domain boundaries. The FFT branch likewise uses periodic Fourier basis functions along all three tensor axes; this is reported as an assumption mismatch, not as a claim that FFT is intrinsically wrong.

## Provenance

- Project branch at audit: `main`.
- Pinned upstream: `{stage_s['frozen_pairing']['upstream_commit']}`, upstream worktree clean.
- `CARTESIAN_REMAP_NOT_AUTHORIZED`: Bcc1/2/3 and vel1/2/3 remain spherical Kerr-Schild coordinate-basis channel names.
"""
    (OUT / "operator_source_audit.md").parent.mkdir(parents=True, exist_ok=True)
    (OUT / "operator_source_audit.md").write_text(text, encoding="utf-8")


def analytic_fields(coords: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    phi = coords["phi"].reshape(-1, 1, 1)
    theta = coords["theta"].reshape(1, -1, 1)
    r = coords["r"].reshape(1, 1, -1)
    full = np.ones((64, 64, 64), dtype=np.float64)
    fields: dict[str, dict[str, Any]] = {}
    fields["radial_linear"] = {"value": full * r, "truth": {"phi": 0*full, "theta": 0*full, "r": full}}
    fields["radial_log"] = {"value": full * np.log(r), "truth": {"phi": 0*full, "theta": 0*full, "r": full/r}}
    fields["theta_cosine"] = {"value": full * np.cos(theta), "truth": {"phi": 0*full, "theta": -full*np.sin(theta), "r": 0*full}}
    fields["phi_sine"] = {"value": full * np.sin(phi), "truth": {"phi": full*np.cos(phi), "theta": 0*full, "r": 0*full}}
    mixed = np.sin(theta) * np.cos(phi) / r
    fields["mixed_spherical"] = {
        "value": mixed,
        "truth": {
            "phi": -np.sin(theta)*np.sin(phi)/r,
            "theta": np.cos(theta)*np.cos(phi)/r,
            "r": -np.sin(theta)*np.cos(phi)/(r*r),
        },
    }
    for name, center_index in (("gaussian_inner", 10), ("gaussian_middle", 31), ("gaussian_outer", 53)):
        sigma_index = 3.0
        index = np.arange(64, dtype=np.float64)
        radial = np.exp(-0.5*((index-center_index)/sigma_index)**2)
        value = full * radial.reshape(1,1,-1)
        truth_r = coordinate_gradient(value, coords["r"], axis=2)
        fields[name] = {"value": value, "truth": {"phi": 0*full, "theta": 0*full, "r": truth_r}}
    return fields


def derivative_audit(coords: Mapping[str, np.ndarray]) -> tuple[str, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    shell_rows: list[dict[str, Any]] = []
    fields = analytic_fields(coords)
    thirds = {"inner": slice(0,21), "middle": slice(21,42), "outer": slice(42,64)}
    descriptions = {}
    radial_profiles: dict[str, np.ndarray] = {}
    for field_name, spec in fields.items():
        value = spec["value"]
        descriptions[field_name] = {"shape": list(value.shape), "formula": field_name}
        for direction, axis in AXES.items():
            truth_coordinate = np.asarray(spec["truth"][direction])
            index = padded_centered_derivative_1d(
                value, np.arange(64, dtype=np.float64), axis=axis, padding="periodic"
            )
            if direction == "phi":
                coordinate = padded_centered_derivative_1d(value, coords[direction], axis=axis, padding="periodic")
            else:
                coordinate = coordinate_gradient(value, coords[direction], axis=axis)
            scale = np.ones_like(truth_coordinate)
            if direction == "theta":
                scale = 1.0 / coords["r"].reshape(1,1,-1)
            elif direction == "phi":
                scale = 1.0 / (np.sin(coords["theta"]).reshape(1,-1,1)*coords["r"].reshape(1,1,-1))
            truth_proxy = truth_coordinate * scale
            concepts = {
                "index_space_upstream_assumption": (index, truth_coordinate),
                "stored_coordinate_aware": (coordinate, truth_coordinate),
                "spherical_scale_proxy": (coordinate*scale, truth_proxy),
            }
            for concept, (prediction, expected) in concepts.items():
                expected_norm = float(np.linalg.norm(expected))
                difference = prediction - expected
                boundary = np.take(prediction, [0,-1], axis=axis)
                expected_boundary = np.take(expected, [0,-1], axis=axis)
                row = {
                    "field": field_name, "direction": direction, "concept": concept,
                    "expected_zero": expected_norm < 1e-20,
                    "normalized_l2_error": None if expected_norm < 1e-20 else relative_l2(prediction, expected),
                    "absolute_rms_error": float(np.sqrt(np.mean(difference*difference))),
                    "boundary_normalized_l2_error": None if np.linalg.norm(expected_boundary) < 1e-20 else relative_l2(boundary, expected_boundary),
                    "maximum_absolute_error": float(np.max(np.abs(difference))),
                }
                rows.append(row)
                for shell, section in thirds.items():
                    pred_shell, truth_shell = prediction[..., section], expected[..., section]
                    shell_rows.append({
                        "field": field_name, "direction": direction, "concept": concept, "shell": shell,
                        "normalized_l2_error": None if np.linalg.norm(truth_shell) < 1e-20 else relative_l2(pred_shell, truth_shell),
                        "absolute_rms_error": float(np.sqrt(np.mean((pred_shell-truth_shell)**2))),
                    })
            if direction == "r" and np.linalg.norm(truth_coordinate) > 0:
                radial_profiles[f"{field_name}_index"] = np.sqrt(np.mean((index-truth_coordinate)**2,axis=(0,1)))
                radial_profiles[f"{field_name}_coordinate"] = np.sqrt(np.mean((coordinate-truth_coordinate)**2,axis=(0,1)))
    relevant_index = [r["normalized_l2_error"] for r in rows if r["concept"] == "index_space_upstream_assumption" and r["normalized_l2_error"] is not None]
    median_index = float(np.median(relevant_index))
    classification = "STRONG" if median_index > 0.5 else ("MODERATE" if median_index > 0.1 else "LOW")
    write_json(OUT / "synthetic_fd/synthetic_fields.json", {
        "schema_version": "stage-u-synthetic-fields-v1", "grid_shape": [64,64,64],
        "axis_order": ["phi","theta","r"], "fields": descriptions,
        "concepts": {
            "A": "centered index derivative, unit scalar grid width, circular all axes",
            "B": "stored-coordinate derivative using actual r/theta/phi centers",
            "C": "B with scalar spherical-coordinate 1/r and 1/(r sin theta) proxies",
        },
        "INDEX_FD_GEOMETRY_ERROR": classification,
        "median_relevant_index_space_normalized_l2": median_index,
        "kerr_schild_warning": "Spherical proxy only; not a strict Kerr-Schild covariant derivative.",
    })
    write_csv(OUT / "synthetic_fd/derivative_errors.csv", rows)
    write_csv(OUT / "synthetic_fd/shell_derivative_errors.csv", shell_rows)
    fig, axis = plt.subplots(figsize=(8,5))
    rcoord = coords["r"]
    for name, values in radial_profiles.items():
        if "radial_log" in name or "mixed" in name:
            axis.loglog(rcoord, np.maximum(values,1e-16), label=name)
    axis.set(xlabel="stored r center", ylabel="radial derivative RMS absolute error", title="Stage U derivative error versus radius")
    axis.legend(fontsize=7); axis.grid(True, which="both", alpha=.3)
    fig.tight_layout(); (OUT / "synthetic_fd/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "synthetic_fd/figures/derivative_error_vs_radius.png", dpi=150); plt.close(fig)
    return classification, rows


def boundary_audit(coords: Mapping[str, np.ndarray]) -> tuple[str, dict[str, str]]:
    full = np.ones((64,64,64), dtype=np.float64)
    tests = {
        "phi_seam_sin": (full*np.sin(coords["phi"]).reshape(-1,1,1), full*np.cos(coords["phi"]).reshape(-1,1,1), "phi"),
        "theta_cosine": (full*np.cos(coords["theta"]).reshape(1,-1,1), -full*np.sin(coords["theta"]).reshape(1,-1,1), "theta"),
        "radial_log": (full*np.log(coords["r"]).reshape(1,1,-1), full/coords["r"].reshape(1,1,-1), "r"),
    }
    rows=[]
    for test,(value,truth,direction) in tests.items():
        axis=AXES[direction]
        for mode in ("periodic","reflect","replicate","zeros"):
            pred=padded_centered_derivative_1d(value,coords[direction],axis=axis,padding=mode)
            first=np.take(pred,[0],axis=axis); first_truth=np.take(truth,[0],axis=axis)
            last=np.take(pred,[-1],axis=axis); last_truth=np.take(truth,[-1],axis=axis)
            boundary=np.take(pred,[0,-1],axis=axis); boundary_truth=np.take(truth,[0,-1],axis=axis)
            rows.append({
                "test":test,"direction":direction,"padding":mode,
                "global_normalized_l2_error":relative_l2(pred,truth),
                "boundary_normalized_l2_error":relative_l2(boundary,boundary_truth),
                "lower_boundary_normalized_l2_error":relative_l2(first,first_truth),
                "upper_boundary_normalized_l2_error":relative_l2(last,last_truth),
                "maximum_absolute_error":float(np.max(np.abs(pred-truth))),
            })
    chosen={"phi":"periodic"}
    for direction,test in (("theta","theta_cosine"),("r","radial_log")):
        subset=[row for row in rows if row["test"]==test and row["padding"] in {"reflect","replicate","zeros"}]
        # Predeclare a minimax boundary rule so the high-amplitude inner radial
        # derivative cannot hide a disastrous relative error at the outer edge.
        chosen[direction]=min(
            subset,
            key=lambda row:max(
                row["lower_boundary_normalized_l2_error"],
                row["upper_boundary_normalized_l2_error"],
            ),
        )["padding"]
    periodic_theta=next(row for row in rows if row["test"]=="theta_cosine" and row["padding"]=="periodic")
    periodic_r=next(row for row in rows if row["test"]=="radial_log" and row["padding"]=="periodic")
    theta_bad=periodic_theta["boundary_normalized_l2_error"]>0.25
    r_bad=periodic_r["boundary_normalized_l2_error"]>0.25
    mismatch="THETA_AND_R" if theta_bad and r_bad else ("THETA_ONLY" if theta_bad else ("R_ONLY" if r_bad else "NONE"))
    write_csv(OUT / "boundary_audit/boundary_errors.csv",rows)
    write_json(OUT / "boundary_audit/decision.json",{"BOUNDARY_MISMATCH":mismatch,"synthetic_selected_padding":chosen})
    labels=[f"{r['direction']}:{r['padding']}" for r in rows]
    values=[r["boundary_normalized_l2_error"] for r in rows]
    fig,axis=plt.subplots(figsize=(11,5)); axis.bar(range(len(rows)),values); axis.set_yscale("log")
    axis.set_xticks(range(len(rows)),labels,rotation=60,ha="right"); axis.set_ylabel("boundary normalized L2 error")
    axis.set_title("Synthetic boundary policy audit"); fig.tight_layout()
    (OUT/"boundary_audit/figures").mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/"boundary_audit/figures/boundary_error.png",dpi=150); plt.close(fig)
    return mismatch,chosen


def fft_lowpass(signal: np.ndarray, keep: int = 8) -> np.ndarray:
    coeff=np.fft.rfft(signal)
    kept=np.zeros_like(coeff); kept[:keep//2+1]=coeff[:keep//2+1]
    return np.fft.irfft(kept,n=signal.size)


def spectral_audit(coords: Mapping[str,np.ndarray], grid: Mapping[str,Any]) -> str:
    n=64; index=np.arange(n); rows=[]
    signals={
        "pure_phi_sinusoid":np.sin(4*coords["phi"]),
        "pure_theta_sinusoid":np.sin(4*np.pi*(coords["theta"]-coords["theta"][0])/(coords["theta"][-1]-coords["theta"][0])),
        "pure_log_r_sinusoid":np.sin(4*np.pi*(np.log(coords["r"])-np.log(coords["r"][0]))/(np.log(coords["r"][-1])-np.log(coords["r"][0]))),
        "localized_radial_pulse":np.exp(-.5*((index-12)/2.5)**2),
        "localized_theta_pulse":np.exp(-.5*((index-12)/2.5)**2),
    }
    direction={"pure_phi_sinusoid":"phi","pure_theta_sinusoid":"theta","pure_log_r_sinusoid":"r","localized_radial_pulse":"r","localized_theta_pulse":"theta"}
    for name,signal in signals.items():
        response=fft_lowpass(signal)
        shifted=np.roll(signal,25); shifted_response=fft_lowpass(shifted)
        equivariance=relative_l2(shifted_response,np.roll(response,25))
        wrap_energy=float(np.sum(shifted[:5]**2)+np.sum(shifted[-5:]**2))/max(float(np.sum(shifted**2)),1e-30)
        rows.append({"test":name,"axis":direction[name],"kept_modes":8,"lowpass_relative_l2":relative_l2(response,signal),"circular_shift_equivariance_error":equivariance,"shifted_boundary_energy_fraction":wrap_energy})
    radial_scale=grid["tensor_cell_scale_ratios"]["radial_max_over_min"]
    response_mismatch=max(row["circular_shift_equivariance_error"] for row in rows)
    classification="STRONG" if radial_scale>10 and response_mismatch<1e-10 else ("MODERATE" if radial_scale>2 else "LOW")
    write_csv(OUT/"spectral_audit/spectral_response.csv",rows)
    write_json(OUT/"spectral_audit/decision.json",{
        "SPECTRAL_INDEX_SPACE_MISMATCH":classification,
        "interpretation":"Standard FFT is exactly circular-shift equivariant in tensor index while a fixed radial index shift changes the spherical-coordinate scale.",
        "radial_tensor_cell_scale_max_over_min":radial_scale,
        "operator_warning":"Assumption mismatch, not a claim that FFT is intrinsically wrong.",
    })
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    pulse=signals["localized_radial_pulse"]
    axes[0].plot(index,pulse,label="input");axes[0].plot(index,fft_lowpass(pulse),label="8-mode response");axes[0].legend();axes[0].set_title("Radial index pulse")
    dr=np.gradient(coords["r"]);axes[1].semilogy(coords["r"],dr);axes[1].set(xlabel="r",ylabel="local dr",title="Physical radial scale per tensor step")
    fig.tight_layout();(OUT/"spectral_audit/figures").mkdir(parents=True,exist_ok=True)
    fig.savefig(OUT/"spectral_audit/figures/spectral_response.png",dpi=150);plt.close(fig)
    return classification


def main() -> None:
    stage_t,stage_s=load_contract()
    grid,coords=grid_audit(stage_s)
    source_audit(stage_t,stage_s,grid)
    index_error,_=derivative_audit(coords)
    boundary,chosen=boundary_audit(coords)
    spectral=spectral_audit(coords,grid)
    write_json(OUT/"audit_summary.json",{
        "DATASET_FROZEN":True,"SPLIT_FROZEN":True,"PREPROCESSING_FROZEN":True,
        "TARGET_CONTRACT_FROZEN":True,"LOSS_FROZEN":True,
        "INDEX_FD_GEOMETRY_ERROR":index_error,"BOUNDARY_MISMATCH":boundary,
        "SPECTRAL_INDEX_SPACE_MISMATCH":spectral,"synthetic_selected_padding":chosen,
        "AUTHORIZE_U3":index_error in {"MODERATE","STRONG"} or boundary!="NONE",
        "CARTESIAN_REMAP_NOT_AUTHORIZED":True,
    })
    print(json.dumps(json.loads((OUT/"audit_summary.json").read_text()),indent=2))


if __name__=="__main__":
    main()
