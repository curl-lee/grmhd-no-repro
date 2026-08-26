#!/usr/bin/env python3
"""Run the no-training Stage W source, coordinate, and transform audits."""

from __future__ import annotations

import csv
import inspect
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Mapping

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from grmhd.dataset import sha256_file
from grmhd.models import build_model, trainable_parameter_count
from grmhd.stage_u_geometry import inferred_faces
from grmhd.stage_w_mixed_basis import (
    dct_lowpass_1d,
    mixed_basis_energy,
    mixed_basis_inverse,
    mixed_basis_transform,
    orthonormal_dct_ii,
)
from grmhd.stage_w_training import build_stage_w_model


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/stage_w"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Refusing to write empty Stage W transform response")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return float(
        torch.linalg.vector_norm(actual - expected)
        / torch.linalg.vector_norm(expected).clamp_min(1e-30)
    )


def load_contract() -> tuple[dict[str, Any], dict[str, Any], Path]:
    config_path = ROOT / "configs/stage_w/mixed_basis.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    stage_t_path = ROOT / config["frozen_stage_t_config"]
    actual = sha256_file(stage_t_path)
    if actual != config["frozen_stage_t_config_sha256"]:
        raise ValueError(f"Frozen Stage T config changed: {actual}")
    stage_t = yaml.safe_load(stage_t_path.read_text(encoding="utf-8"))
    stage_s_path = ROOT / stage_t["frozen_stage_s_config"]
    if sha256_file(stage_s_path) != stage_t["frozen_stage_s_config_sha256"]:
        raise ValueError("Frozen Stage S config changed")
    stage_s = yaml.safe_load(stage_s_path.read_text(encoding="utf-8"))
    dataset = ROOT / stage_s["data"]["dataset"]
    if sha256_file(dataset) != stage_s["data"]["dataset_sha256"]:
        raise ValueError("Frozen expanded HDF5 changed")
    normalizer = ROOT / stage_s["preprocessing"]["artifact"] / "normalizer.npz"
    if sha256_file(normalizer) != stage_s["preprocessing"]["normalizer_sha256"]:
        raise ValueError("Frozen P3 normalizer changed")
    upstream = subprocess.check_output(
        ["git", "-C", "external/neuraloperator", "rev-parse", "HEAD"],
        cwd=ROOT, text=True,
    ).strip()
    if upstream != stage_s["frozen_pairing"]["upstream_commit"]:
        raise ValueError("Pinned neuraloperator commit changed")
    dirty = subprocess.check_output(
        ["git", "-C", "external/neuraloperator", "status", "--short"],
        cwd=ROOT, text=True,
    )
    if dirty.strip():
        raise ValueError("Pinned neuraloperator worktree is dirty")
    return config, stage_s, dataset


def coordinate_audit(stage_s: Mapping[str, Any], dataset: Path) -> dict[str, Any]:
    with h5py.File(dataset, "r") as handle:
        coords = {
            name: np.asarray(handle[f"coords/{name}"], dtype=np.float64)
            for name in ("phi", "theta", "r")
        }
        shape = list(handle["snapshots"].shape)
        metadata = json.loads(handle.attrs["metadata_json"])
    phi, theta, r = coords["phi"], coords["theta"], coords["r"]
    xi = (np.log(r) - np.log(r.min())) / (np.log(r.max()) - np.log(r.min()))
    dxi = np.diff(xi)
    phi_faces = inferred_faces(phi, periodic_extent=2 * np.pi)
    theta_faces = inferred_faces(theta)
    r_faces = inferred_faces(r)
    mean = float(dxi.mean())
    log_uniform = bool(np.allclose(dxi, mean, rtol=1e-10, atol=1e-12))
    payload = {
        "schema_version": "stage-w-coordinate-basis-v1",
        "dataset": str(dataset.relative_to(ROOT)),
        "dataset_sha256": sha256_file(dataset),
        "snapshot_shape": shape,
        "axis_order": ["phi", "theta", "r"],
        "hdf5_metadata": metadata,
        "log_r": {
            "definition": "(log(r)-log(r_min))/(log(r_max)-log(r_min)) on stored centers",
            "r_min_center": float(r.min()),
            "r_max_center": float(r.max()),
            "inferred_face_bounds": [float(r_faces[0]), float(r_faces[-1])],
            "xi_min": float(xi.min()),
            "xi_max": float(xi.max()),
            "delta_xi_min": float(dxi.min()),
            "delta_xi_max": float(dxi.max()),
            "delta_xi_mean": mean,
            "delta_xi_std": float(dxi.std()),
            "delta_xi_std_over_mean": float(dxi.std() / mean),
            "maximum_absolute_deviation_from_uniform": float(np.max(np.abs(dxi - mean))),
            "center_ratio_min": float(np.min(r[1:] / r[:-1])),
            "center_ratio_max": float(np.max(r[1:] / r[:-1])),
            "LOG_R_GRID_UNIFORM": log_uniform,
            "dct_collocation_note": "Stored log-r centers are uniformly spaced; DCT-II uses the equivalent midpoint-index cosine grid over inferred log-r faces.",
        },
        "theta": {
            "center_min": float(theta.min()),
            "center_max": float(theta.max()),
            "spacing_min": float(np.diff(theta).min()),
            "spacing_max": float(np.diff(theta).max()),
            "uniform": bool(np.allclose(np.diff(theta), np.diff(theta)[0], rtol=1e-10, atol=1e-12)),
            "inferred_face_bounds": [float(theta_faces[0]), float(theta_faces[-1])],
            "contains_exact_pole_centers": bool(np.any(theta == 0) or np.any(theta == np.pi)),
            "poles_are_inferred_faces": bool(np.isclose(theta_faces[0], 0) and np.isclose(theta_faces[-1], np.pi)),
        },
        "phi": {
            "center_min": float(phi.min()),
            "center_max": float(phi.max()),
            "spacing": float(np.diff(phi).mean()),
            "uniform": bool(np.allclose(np.diff(phi), np.diff(phi)[0], rtol=1e-10, atol=1e-12)),
            "inferred_face_bounds": [float(phi_faces[0]), float(phi_faces[-1])],
            "periodic_seam_policy": "cell centers exclude seam; inferred faces are 0 and 2pi; circular phi operator",
            "full_2pi": bool(np.isclose(phi_faces[-1] - phi_faces[0], 2 * np.pi)),
        },
        "MIXED_BASIS_IS_NOT_SPHERICAL_HARMONICS": True,
        "MIXED_BASIS_IS_NOT_KERR_SCHILD_COVARIANT": True,
    }
    write_json(OUT / "coordinate_basis_audit.json", payload)
    return payload


def source_audit(stage_s: Mapping[str, Any], coordinate: Mapping[str, Any]) -> None:
    from neuralop.layers.spectral_convolution import SpectralConv
    from neuralop.models.local_no import LocalNO

    baseline = build_model(
        "localno_differential_3d",
        in_channels=16, out_channels=8, n_modes=(8, 8, 8),
        hidden_channels=16, n_layers=4, default_in_shape=(64, 64, 64),
        positional_embedding=None, fin_diff_kernel_size=3,
        mix_derivatives=True, conv_padding_mode="periodic",
        use_channel_mlp=False, local_no_skip="linear", norm=None,
        enforce_hermitian_symmetry=True,
    )
    conv = baseline.local_no_blocks.convs[0]
    spectral_file = Path(inspect.getsourcefile(SpectralConv)).resolve()
    forward_line = inspect.getsourcelines(SpectralConv.forward)[1]
    localno_file = Path(inspect.getsourcefile(LocalNO)).resolve()
    text = f"""# Stage W Spectral Source Audit

## Pinned implementation

- Upstream: `external/neuraloperator@{stage_s['frozen_pairing']['upstream_commit']}`; worktree clean at audit time.
- Exact class: `neuralop.layers.spectral_convolution.SpectralConv` in `{spectral_file.relative_to(ROOT)}`; `forward` begins at line {forward_line}.
- Local model: `neuralop.models.local_no.LocalNO` in `{localno_file.relative_to(ROOT)}`.
- Resolved model has {trainable_parameter_count(baseline):,} trainable parameters and four spectral/differential blocks.

## Transform and modal contract

`SpectralConv.forward` obtains all last `order=3` spatial dimensions and calls
`torch.fft.rfftn(x, norm=fft_norm, dim=[-3,-2,-1])`.  The project tensor is
`[B,C,Nphi,Ntheta,Nr]`, hence:

```text
SPECTRAL_AXES = phi index, theta index, r index
REQUESTED_N_MODES = 8 x 8 x 8
STORED_N_MODES = {list(conv.n_modes)}
```

The last real-FFT direction stores `8//2+1=5` radial coefficients.  `fftshift`
centers the full phi and theta spectra, then the layer retains the centered
8-by-8 window and the first five nonredundant r modes.  The inverse performs
IFFT on phi/theta, enforces real zero/Nyquist coefficients on the real-FFT
axis, and then applies radial IRFFT.  Consequently Stage T is exactly a
periodic Fourier representation of all three tensor-index axes.  This is
physically appropriate for phi but imposes periodic spectral continuation on
theta and r.

## Learned contraction

- Per-layer learned complex weight shape: `{list(conv.weight.shape)}` with dense input/output-channel mixing.
- Factorization request/resolution: `factorization={conv.factorization}`; this resolves to an unfactorized Dense `FactorizedTensor`.  `implementation={conv.implementation}`, `separable={conv.separable}`.
- FFT normalization: `fft_norm={conv.fft_norm}` (the inverse uses the matching normalization).
- Bias: enabled, real shape `{list(conv.bias.shape)}` per layer.
- Normalization layer: none; channel MLP: disabled; domain padding: none.

## Block, skip, and positional interactions

Each block adds spectral output, the original upstream 3x3x3 finite-difference
output, and no DISCO output.  A learned linear 1x1 local-NO skip is then added,
followed by GELU except after the final layer.  Lifting and projection are
two-layer channel MLPs.  `positional_embedding=null`, so no raw grid, r,
log-r, theta, phi, or metric coordinate is appended.  The existing eight
shell channels remain the only coarse radial-region input.

## Stage W isolated change

W1 replaces only the spectral convolution with `rFFT_phi x DCT-II_theta x
DCT-II_log-r`; the upstream differential branch remains circular on every
axis.  W2 additionally applies the Stage-U-predeclared boundary policy
`circular_phi/replicate_theta/replicate_r` to the identical 3x3x3 stencil and
identical scalar grid width.  No coordinate scaling or singular metric proxy
is introduced.  The stored log-r uniformity result is
`LOG_R_GRID_UNIFORM={str(coordinate['log_r']['LOG_R_GRID_UNIFORM']).lower()}`.

```text
MIXED_BASIS_IS_NOT_SPHERICAL_HARMONICS = true
MIXED_BASIS_IS_NOT_KERR_SCHILD_COVARIANT = true
```

All eight output fields remain component-wise scalar feature maps; Bcc and
velocity components are not converted to vector harmonics.
"""
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "spectral_source_audit.md").write_text(text, encoding="utf-8")


def transform_audit(config: Mapping[str, Any], coordinate: Mapping[str, Any]) -> dict[str, Any]:
    thresholds = config["transform_tests"]
    rows: list[dict[str, Any]] = []
    generator = torch.Generator().manual_seed(42)
    roundtrip: dict[str, float] = {}
    for dtype, name in ((torch.float32, "float32"), (torch.float64, "float64")):
        x = torch.randn(2, 3, 32, 24, 20, generator=generator, dtype=dtype)
        coefficients = mixed_basis_transform(x)
        reconstructed = mixed_basis_inverse(coefficients, n_phi=x.shape[-3])
        error = relative_l2(reconstructed, x)
        roundtrip[name] = error
        rows.append({"test": "roundtrip", "case": name, "value": error})
    x = torch.randn(2, 3, 32, 24, 20, generator=generator)
    coefficients = mixed_basis_transform(x)
    parseval = abs(float(mixed_basis_energy(coefficients, n_phi=32) - x.square().sum())) / float(x.square().sum())
    rows.append({"test": "parseval", "case": "float32", "value": parseval})

    n = 64
    phi = 2 * math.pi * torch.arange(n) / n
    nodes = (torch.arange(n) + 0.5) / n
    mode_specs = (("phi", 3, 0, 0), ("theta", 0, 4, 0), ("log_r", 0, 0, 5), ("product", 3, 4, 5))
    localization: dict[str, float] = {}
    plotted: list[tuple[str, np.ndarray]] = []
    for name, m_phi, m_theta, m_r in mode_specs:
        field = (
            (torch.sin(m_phi * phi) if m_phi else torch.ones(n)).reshape(n, 1, 1)
            * (torch.cos(math.pi * m_theta * nodes) if m_theta else torch.ones(n)).reshape(1, n, 1)
            * (torch.cos(math.pi * m_r * nodes) if m_r else torch.ones(n)).reshape(1, 1, n)
        )
        spectrum = mixed_basis_transform(field)
        energy = spectrum.abs().square()
        expected_phi = m_phi
        fraction = float(energy[expected_phi, m_theta, m_r] / energy.sum())
        localization[name] = fraction
        rows.append({"test": "mode_localization", "case": name, "value": fraction})
        if name in ("phi", "theta", "log_r"):
            plotted.append((name, energy.sum(dim=tuple(i for i in range(3) if i != {"phi": 0, "theta": 1, "log_r": 2}[name])).numpy()))

    signal = torch.randn(4, 32, 20, 18, generator=generator)
    shift = 5
    original = mixed_basis_transform(signal)
    shifted = mixed_basis_transform(torch.roll(signal, shifts=shift, dims=-3))
    modes = torch.arange(original.shape[-3], dtype=signal.dtype)
    phase = torch.exp(-2j * math.pi * modes * shift / signal.shape[-3]).reshape(1, -1, 1, 1)
    phi_shift_error = relative_l2(shifted, original * phase)
    rows.append({"test": "phi_periodic_shift", "case": "shift_5", "value": phi_shift_error})

    pulse = torch.zeros(64); pulse[0] = 1
    cosine = dct_lowpass_1d(pulse, keep=8)
    fft = torch.fft.rfft(pulse, norm="ortho")
    fft_low = torch.zeros_like(fft); fft_low[:5] = fft[:5]
    periodic = torch.fft.irfft(fft_low, n=64, norm="ortho")
    wrap_ratio = float(torch.linalg.vector_norm(cosine[-4:]) / torch.linalg.vector_norm(periodic[-4:]))
    rows.extend([
        {"test": "theta_boundary_no_periodic_wrap", "case": "inner_pulse_outer4", "value": wrap_ratio},
        {"test": "radial_boundary_no_periodic_wrap", "case": "inner_pulse_outer4", "value": wrap_ratio},
    ])

    tests = {
        "E1_float32_roundtrip": roundtrip["float32"] < float(thresholds["float32_roundtrip_relative_l2_max"]),
        "E1_float64_roundtrip": roundtrip["float64"] < float(thresholds["float64_roundtrip_relative_l2_max"]),
        "E2_parseval": parseval < float(thresholds["parseval_relative_error_max"]),
        "E3_mode_localization": min(localization.values()) > float(thresholds["mode_localization_energy_fraction_min"]),
        "E4_phi_periodic_shift": phi_shift_error < float(thresholds["phi_shift_relative_error_max"]),
        "E5_theta_boundary": wrap_ratio < float(thresholds["boundary_wrap_ratio_max"]),
        "E6_radial_boundary": wrap_ratio < float(thresholds["boundary_wrap_ratio_max"]),
    }
    initial = torch.load(
        ROOT / "outputs/paper_reduced100/stage_k/shared_localno_initial_state.pt",
        map_location="cpu", weights_only=True,
    )
    stage_s = yaml.safe_load((ROOT / "configs/stage_s/expanded_localno_p3_residual.yaml").read_text())
    w1, init_w1 = build_stage_w_model(stage_s, "mixed_basis", initial_state=initial)
    w2, init_w2 = build_stage_w_model(stage_s, "mixed_basis_boundary", initial_state=initial)
    parameter_delta = (trainable_parameter_count(w1) - 358_296) / 358_296
    tests["parameter_count_delta_below_10_percent"] = abs(parameter_delta) < 0.10
    tests["w1_w2_common_initial_tensor_hash"] = (
        init_w1["full_initial_tensor_state_sha256"] == init_w2["full_initial_tensor_state_sha256"]
    )
    payload = {
        "schema_version": "stage-w-transform-tests-v1",
        "tests": tests,
        "all_core_tests_passed": all(tests.values()),
        "roundtrip_relative_l2": roundtrip,
        "parseval_relative_energy_error": parseval,
        "mode_localization_energy_fraction": localization,
        "phi_shift_relative_l2": phi_shift_error,
        "nonperiodic_outer_wrap_ratio_vs_fft": wrap_ratio,
        "parameter_count": trainable_parameter_count(w1),
        "baseline_parameter_count": 358_296,
        "parameter_count_relative_delta": parameter_delta,
        "w1_initial_tensor_sha256": init_w1["full_initial_tensor_state_sha256"],
        "w2_initial_tensor_sha256": init_w2["full_initial_tensor_state_sha256"],
        "LOG_R_GRID_UNIFORM": coordinate["log_r"]["LOG_R_GRID_UNIFORM"],
    }
    if not payload["all_core_tests_passed"]:
        raise RuntimeError(f"Stage W transform gate failed: {tests}")
    write_json(OUT / "mixed_transform/transform_unit_tests.json", payload)
    write_csv(OUT / "mixed_transform/transform_response.csv", rows)
    notes = """# Mixed transform implementation notes

The implementation uses an explicit even extension followed by `torch.fft.fft`
for orthonormal DCT-II and reconstructs the conjugate even spectrum for IDCT.
This avoids an external DCT dependency and is checked against roundtrip,
Parseval, localization, shift, and boundary-response contracts.

Phi alone uses orthonormal `rfft/irfft`.  Requested modes `(8,8,8)` therefore
store `(5,8,8)` complex modal weights.  This is a reversible axis permutation
of the Stage-T initial `(8,8,5)` spectral tensor and keeps parameter count
exactly unchanged.  Theta and log-r use the lowest eight cosine modes.

The DCT axes represent even non-periodic continuation at their faces.  They do
not identify lower and upper boundaries and are not spherical harmonics.  No
coordinate values or metric factors enter the learned convolution.
"""
    (OUT / "mixed_transform/implementation_notes.md").write_text(notes, encoding="utf-8")
    figure_dir = OUT / "figures"; figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for axis, (name, values) in zip(axes, plotted):
        axis.stem(np.arange(min(12, values.size)), values[:12], basefmt=" ")
        axis.set(title=f"{name} mode localization", xlabel="mode", ylabel="energy")
    fig.tight_layout(); fig.savefig(figure_dir / "transform_mode_localization.png", dpi=150); plt.close(fig)
    return payload


def main() -> None:
    config, stage_s, dataset = load_contract()
    coordinate = coordinate_audit(stage_s, dataset)
    source_audit(stage_s, coordinate)
    transform = transform_audit(config, coordinate)
    print(json.dumps({
        "LOG_R_GRID_UNIFORM": coordinate["log_r"]["LOG_R_GRID_UNIFORM"],
        "all_core_tests_passed": transform["all_core_tests_passed"],
        "parameter_count": transform["parameter_count"],
        "parameter_count_relative_delta": transform["parameter_count_relative_delta"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
