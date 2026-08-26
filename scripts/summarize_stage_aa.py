#!/usr/bin/env python3
"""Finalize the Stage AA GPU-gate failure without entering model execution."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from grmhd import CHANNELS
from grmhd.dataset import sha256_file
from grmhd.paper_stage_n import NO_SOFTCLIP, PrototypePreprocessor, prototype_specs


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/stage_aa"
SCOPE = "ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION"
SELECTED_STEPS = (1, 2, 3, 5, 10, 19, 25, 42, 50, 75, 100)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def atomic_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty Stage AA table {path}")
    fields = list(rows[0])
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_data() -> dict[str, Any]:
    stage_z = yaml.safe_load((ROOT / "configs/stage_z/higher_resolution.yaml").read_text(encoding="utf-8"))
    expected = {
        64: json.loads((ROOT / "artifacts/stage_z/preprocessing/p3_64.json").read_text(encoding="utf-8"))["dataset_sha256"],
        96: json.loads((ROOT / "artifacts/stage_z/regrid/z96_manifest.json").read_text(encoding="utf-8"))["sha256"],
        128: json.loads((ROOT / "artifacts/stage_z/regrid/z128_manifest.json").read_text(encoding="utf-8"))["sha256"],
    }
    observed = {}
    for resolution in (64, 96, 128):
        path = ROOT / stage_z["data"]["resolutions"][resolution]
        observed[resolution] = sha256_file(path)
        if observed[resolution] != expected[resolution]:
            raise ValueError(f"Z{resolution} checksum changed after Stage Z")
    path96 = ROOT / stage_z["data"]["resolutions"][96]
    nan_count = inf_count = nonpositive_rho = nonpositive_press = 0
    with h5py.File(path96, "r") as handle:
        snapshots = handle["snapshots"]
        if tuple(snapshots.shape) != (212, 8, 96, 96, 96):
            raise ValueError("Z96 shape changed")
        if snapshots.dtype != np.dtype(np.float32):
            raise ValueError("Z96 dtype changed")
        axis_order = str(handle["metadata"].attrs["axis_order"])
        if axis_order != "N,C,Nphi,Ntheta,Nr":
            raise ValueError("Z96 axis order changed")
        for index in range(212):
            values = np.asarray(snapshots[index], dtype=np.float32)
            nan_count += int(np.isnan(values).sum())
            inf_count += int(np.isinf(values).sum())
            nonpositive_rho += int(np.count_nonzero(values[3] <= 0))
            nonpositive_press += int(np.count_nonzero(values[4] <= 0))
    if nan_count or inf_count or nonpositive_rho or nonpositive_press:
        raise FloatingPointError("Z96 finite/positivity validation failed")

    p3_summary = json.loads(
        (ROOT / "artifacts/stage_z/preprocessing/p3_96.json").read_text(encoding="utf-8")
    )
    normalizer = ROOT / p3_summary["artifact_directory"]
    if sha256_file(normalizer / "normalizer.npz") != p3_summary["normalizer_sha256"]:
        raise ValueError("Z96 P3 normalizer checksum changed")
    if p3_summary["fit_snapshot_indices"] != list(range(169)):
        raise ValueError("Z96 P3 fit indices changed")
    if p3_summary["validation_indices_used_for_fit"]:
        raise ValueError("Z96 P3 unexpectedly used validation")
    processor = PrototypePreprocessor.load(
        normalizer, spec=prototype_specs(NO_SOFTCLIP)["P3"]
    )
    numerator = np.zeros(8, dtype=np.float64)
    denominator = np.zeros(8, dtype=np.float64)
    with h5py.File(path96, "r") as handle:
        for index in range(169, 212):
            raw = np.asarray(handle["snapshots"][index], dtype=np.float32)
            decoded = processor.decode_numpy(
                processor.encode_numpy(raw, channel_axis=0), channel_axis=0
            )
            difference = decoded.astype(np.float64) - raw.astype(np.float64)
            numerator += np.sum(np.square(difference), axis=(1, 2, 3))
            denominator += np.sum(np.square(raw, dtype=np.float64), axis=(1, 2, 3))
    oracle = np.sqrt(numerator / np.maximum(denominator, 1.0e-300))
    stage_z_oracle = json.loads(
        (ROOT / "artifacts/stage_z/preprocessing/oracle_floor_summary.json").read_text(encoding="utf-8")
    )["resolutions"]["96"]["per_channel"]
    expected_oracle = np.asarray([stage_z_oracle[name] for name in CHANNELS], dtype=np.float64)
    if not np.allclose(oracle, expected_oracle, rtol=1.0e-12, atol=1.0e-15):
        raise ValueError("Z96 P3 oracle floor no longer matches Stage Z")
    return {
        "schema_version": "stage-aa-z96-validation-v1",
        "reproduction_scope": SCOPE,
        "status": "passed_data_and_preprocessing_only",
        "model_execution_authorized": False,
        "dataset_sha256": {str(key): value for key, value in observed.items()},
        "dataset_expected_sha256": {str(key): value for key, value in expected.items()},
        "z96": {
            "shape": [212, 8, 96, 96, 96], "dtype": "float32",
            "axis_order": "N,C,Nphi,Ntheta,Nr", "finite": True,
            "rho_positive": True, "press_positive": True,
        },
        "p3": {
            "normalizer_sha256": p3_summary["normalizer_sha256"],
            "fit_snapshot_indices": list(range(169)),
            "validation_indices_used_for_fit": [],
            "oracle_floor_recomputed": dict(zip(CHANNELS, oracle.tolist(), strict=True)),
            "oracle_floor_matches_stage_z": True,
            "priority_channels": ["Bcc2", "Bcc3", "vel3", "rho", "press"],
        },
    }


def annotated_pair_plot(path: Path, ylabel: str, z64: float, *, zero: bool = False) -> None:
    fig, axis = plt.subplots(figsize=(7, 4.2))
    axis.scatter([64], [z64], s=60, label="Z64 Stage-T epoch 150")
    if zero:
        axis.axhline(0, color="black", alpha=0.5, linewidth=0.8)
    axis.set(xlim=(55, 105), xticks=[64, 96], xlabel="spherical target resolution", ylabel=ylabel)
    axis.text(96, z64, "not available\nGPU gate FAIL", ha="center", va="bottom")
    axis.grid(True, alpha=0.3); axis.legend(); fig.tight_layout()
    fig.savefig(path, dpi=160); plt.close(fig)


def main() -> None:
    gate = json.loads((OUT / "gpu/gpu_gate.json").read_text(encoding="utf-8"))
    if gate["GPU_SCIENTIFIC_GATE"] != "FAIL":
        raise RuntimeError("This blocked-path summary must not run after GPU recovery")
    validation = validate_data()
    atomic_json(OUT / "data/z96_validation.json", validation)
    preflight = {
        "schema_version": "stage-aa-z96-cuda-preflight-v1",
        "reproduction_scope": SCOPE,
        "status": "not_entered_gpu_scientific_gate_fail",
        "GPU_SCIENTIFIC_GATE": "FAIL",
        "input_shape": None, "output_shape": None,
        "forward_finite": None, "loss_finite": None, "gradients_finite": None,
        "nonzero_gradient_count": None, "peak_allocated_vram": None,
        "peak_reserved_vram": None, "forward_time": None, "backward_time": None,
        "optimizer_step_executed": False,
        "activation_checkpointing_used": False,
        "Z96_TRAINING_RESOURCE_FEASIBLE": False,
        "cpu_fallback": False,
        "reason": "mandatory GPU scientific gate failed before model preflight",
    }
    atomic_json(OUT / "preflight/z96_cuda_preflight.json", preflight)

    stage_z_config = yaml.safe_load((ROOT / "configs/stage_z/higher_resolution.yaml").read_text(encoding="utf-8"))
    atomic_json(OUT / "training/z96/resolved_config.json", {
        "schema_version": "stage-aa-frozen-z96-training-contract-v1",
        "reproduction_scope": SCOPE,
        "status": "not_started_gpu_gate_fail",
        "training_started": False,
        "epochs_completed": 0,
        "optimizer_updates": 0,
        "dataset": stage_z_config["data"]["resolutions"][96],
        "p3": "artifacts/stage_z/preprocessing/p3_96.json",
        "model": stage_z_config["model"],
        "training": stage_z_config["training"],
        "gpu_gate": "artifacts/stage_aa/gpu/gpu_gate.json",
    })
    atomic_json(OUT / "training/z96/status.json", {
        "status": "not_started_gpu_gate_fail", "epochs_completed": 0,
        "optimizer_updates": 0, "checkpoints": [], "one_step_metrics": False,
        "rollout_metrics": False, "cpu_fallback": False,
    })

    z64 = json.loads(
        (ROOT / "artifacts/stage_t/full_long/checkpoint_metrics_epoch_0150.json").read_text(encoding="utf-8")
    )
    rollout64 = json.loads(
        (ROOT / "artifacts/stage_t/full_long/rollout/epoch_0150_rollout.json").read_text(encoding="utf-8")
    )
    persistence = json.loads(
        (ROOT / "artifacts/stage_z/comparison/persistence_by_resolution.json").read_text(encoding="utf-8")
    )["resolutions"]
    fidelity = json.loads(
        (ROOT / "artifacts/stage_z/regrid/information_gain.json").read_text(encoding="utf-8")
    )
    f64, f96 = fidelity["metrics"]["64"], fidelity["metrics"]["96"]
    z64_model = z64["model"]
    z64_values = {
        "model state L2": z64_model["normalized_relative_l2"]["arithmetic_average"],
        "persistence ratio": z64["model_error_over_persistence_error"],
        "residual L2": z64_model["residual"]["arithmetic_average_relative_l2"],
        "residual cosine": z64_model["residual"]["global_cosine"],
        "shell skill": z64["transport"]["shell_skill_median"],
        "radial skill": z64["transport"]["radial_skill_median"],
        "first10x": rollout64["FIRST_10X_PHYSICAL_RANGE_STEP"],
    }
    comparison = [
        {"metric": "increment fidelity error", "Z64": f64["temporal_increment_relative_difference"], "Z96": f96["temporal_increment_relative_difference"], "relative_change": (f64["temporal_increment_relative_difference"] - f96["temporal_increment_relative_difference"]) / f64["temporal_increment_relative_difference"], "kind": "DATA_INFORMATION_GAIN"},
        {"metric": "increment cosine", "Z64": f64["temporal_increment_cosine"], "Z96": f96["temporal_increment_cosine"], "relative_change": (f96["temporal_increment_cosine"] - f64["temporal_increment_cosine"]) / abs(f64["temporal_increment_cosine"]), "kind": "DATA_INFORMATION_GAIN"},
        {"metric": "persistence L2", "Z64": persistence["64"]["normalized_state_relative_l2"]["arithmetic_average"], "Z96": persistence["96"]["normalized_state_relative_l2"]["arithmetic_average"], "relative_change": (persistence["64"]["normalized_state_relative_l2"]["arithmetic_average"] - persistence["96"]["normalized_state_relative_l2"]["arithmetic_average"]) / persistence["64"]["normalized_state_relative_l2"]["arithmetic_average"], "kind": "SAME_RESOLUTION_BASELINE"},
    ]
    for metric, value in z64_values.items():
        comparison.append({
            "metric": metric, "Z64": value, "Z96": None,
            "relative_change": None, "kind": "LEARNED_DYNAMICS_GAIN_UNAVAILABLE",
        })
    atomic_csv(OUT / "comparison/z64_vs_z96.csv", comparison)

    temporal = read_csv(ROOT / "artifacts/stage_z/regrid/temporal_increment_fidelity.csv")
    channel_rows = []
    for name in CHANNELS:
        by_resolution = {}
        for resolution in (64, 96):
            selected = [row for row in temporal if row["channel"] == name and int(row["resolution"]) == resolution]
            by_resolution[resolution] = {
                "error": float(np.median([float(row["residual_relative_difference"]) for row in selected])),
                "cosine": float(np.median([float(row["residual_cosine"]) for row in selected])),
            }
        channel_rows.append({
            "channel": name,
            "fidelity_error_64": by_resolution[64]["error"],
            "fidelity_error_96": by_resolution[96]["error"],
            "fidelity_error_gain": (by_resolution[64]["error"] - by_resolution[96]["error"]) / by_resolution[64]["error"],
            "fidelity_cosine_64": by_resolution[64]["cosine"],
            "fidelity_cosine_96": by_resolution[96]["cosine"],
            "z64_residual_l2": z64_model["residual"]["per_channel_relative_l2"][name],
            "z64_residual_cosine": z64_model["residual"]["per_channel_cosine"][name],
            "z64_shell_skill": z64["transport"]["per_channel"][name]["shell_skill_median"],
            "z64_radial_skill": z64["transport"]["per_channel"][name]["radial_skill_median"],
            "z96_model_status": "not_available_gpu_gate_fail",
            "z96_residual_l2": None, "z96_residual_cosine": None,
            "z96_shell_skill": None, "z96_radial_skill": None,
            "residual_gain": None, "shell_skill_change": None, "radial_skill_change": None,
        })
    atomic_csv(OUT / "comparison/per_channel_comparison.csv", channel_rows)
    atomic_csv(OUT / "comparison/fidelity_vs_model_gain.csv", [{
        "channel": row["channel"],
        "fidelity_error_gain": row["fidelity_error_gain"],
        "model_residual_gain": None,
        "shell_skill_change": None,
        "radial_skill_change": None,
        "association_status": "not_computable_gpu_gate_fail",
    } for row in channel_rows])

    regional_rows = []
    for region in ("inner", "middle", "outer"):
        left = f64[f"temporal_increment_loss_{region}"]
        right = f96[f"temporal_increment_loss_{region}"]
        regional_rows.append({
            "region": region, "sampling_error_64": left, "sampling_error_96": right,
            "sampling_gain": (left - right) / left,
            "model_residual_error_64": None, "model_residual_error_96": None,
            "model_gain": None, "status": "model_regional_attribution_not_available_gpu_gate_fail",
        })
    atomic_csv(OUT / "comparison/regional_comparison.csv", regional_rows)

    rollout_rows = []
    by_step = {int(row["step"]): row for row in rollout64["records"]}
    for step in SELECTED_STEPS:
        left = by_step[step]
        transport = left.get("transport")
        rollout_rows.extend([
            {
                "resolution": 64, "step": step, "status": "stage_t_epoch150_reference",
                "normalized_gt_error": left.get("normalized_relative_l2_average"),
                "physical_gt_error": left.get("physical_relative_l2_average"),
                "residual_cosine": left.get("residual_cosine"),
                "shell_skill": transport.get("shell_skill_median") if transport else None,
                "radial_skill": transport.get("radial_skill_median") if transport else None,
                "finite": left["finite"], "rho_press_positive": left["rho_press_positive"],
                "physical_range_explosion": left["physical_range_explosion"],
            },
            {
                "resolution": 96, "step": step, "status": "not_available_gpu_gate_fail",
                "normalized_gt_error": None, "physical_gt_error": None,
                "residual_cosine": None, "shell_skill": None, "radial_skill": None,
                "finite": None, "rho_press_positive": None, "physical_range_explosion": None,
            },
        ])
    atomic_csv(OUT / "comparison/rollout_comparison.csv", rollout_rows)

    figures = OUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(channel_rows))
    axis.bar(x, [row["fidelity_error_gain"] for row in channel_rows], label="sampling fidelity gain")
    axis.set_xticks(x, [row["channel"] for row in channel_rows], rotation=35)
    axis.set_ylabel("relative gain")
    axis.text(3.5, 0.02, "model gain unavailable: GPU gate FAIL", ha="center")
    axis.grid(True, axis="y", alpha=0.3); axis.legend(); fig.tight_layout()
    fig.savefig(figures / "fidelity_vs_model_gain.png", dpi=160); plt.close(fig)
    annotated_pair_plot(figures / "state_l2_64_vs_96.png", "model state relative L2", z64_values["model state L2"])
    annotated_pair_plot(figures / "residual_l2_64_vs_96.png", "model residual relative L2", z64_values["residual L2"])
    annotated_pair_plot(figures / "shell_skill_64_vs_96.png", "shell skill", z64_values["shell skill"], zero=True)
    annotated_pair_plot(figures / "radial_skill_64_vs_96.png", "radial skill", z64_values["radial skill"], zero=True)
    gt = [row for row in rollout64["records"] if row["ground_truth_available"]]
    fig, axis = plt.subplots(figsize=(7, 4.2))
    axis.plot([row["step"] for row in gt], [row["normalized_relative_l2_average"] for row in gt], label="Z64 epoch 150")
    axis.set(xlabel="rollout step", ylabel="normalized GT error", yscale="log")
    axis.text(22, 1, "Z96 unavailable: GPU gate FAIL", ha="center")
    axis.grid(True, alpha=0.3); axis.legend(); fig.tight_layout()
    fig.savefig(figures / "rollout_error_64_vs_96.png", dpi=160); plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4.2))
    axis.step([row["step"] for row in rollout64["records"]], [int(row["physical_range_explosion"]) for row in rollout64["records"]], where="post", label="Z64 first-10x flag")
    axis.set(xlabel="rollout step", ylabel="physical range >=10x train", yticks=[0, 1])
    axis.text(52, 0.5, "Z96 unavailable: GPU gate FAIL", ha="center")
    axis.grid(True, alpha=0.3); axis.legend(); fig.tight_layout()
    fig.savefig(figures / "physical_range_64_vs_96.png", dpi=160); plt.close(fig)

    report = f"""# Stage AA — GPU Recovery and Z96 Controlled Model Completion

`REPRODUCTION_SCOPE = {SCOPE}`

| quantity | Z64 | Z96 |
|---|---:|---:|
| increment rel diff | {f64['temporal_increment_relative_difference']:.6f} | {f96['temporal_increment_relative_difference']:.6f} |
| increment cosine | {f64['temporal_increment_cosine']:.6f} | {f96['temporal_increment_cosine']:.6f} |
| persistence L2 | {persistence['64']['normalized_state_relative_l2']['arithmetic_average']:.6f} | {persistence['96']['normalized_state_relative_l2']['arithmetic_average']:.6f} |
| model state L2 | {z64_values['model state L2']:.6f} | not available |
| residual L2 | {z64_values['residual L2']:.6f} | not available |
| residual cosine | {z64_values['residual cosine']:.6f} | not available |
| shell skill | {z64_values['shell skill']:.6f} | not available |
| radial skill | {z64_values['radial skill']:.6f} | not available |
| first10x | {z64_values['first10x']} | not available |

## GPU recovery result

The GPU scientific gate did not recover. WSL `nvidia-smi` returns `GPU access
blocked by the operating system`; `/dev/dxg` is absent. CUDA-enabled PyTorch
`{gate['torch']['torch_version']}` is installed with compiled CUDA
`{gate['torch']['compiled_cuda']}`, but reports no CUDA device. WSL CUDA/NVML
stub libraries are present. Windows-side `nvidia-smi.exe` could not be queried
because WSL interop itself failed with `UtilBindVsockAnyPort`, so host driver
visibility remains unknown rather than assumed.

This locates the directly observed failure at `GPU_FAILURE_LAYER =
WSL_GPU_BRIDGE`. No package reinstall, driver change, project/model change, or
CPU fallback was attempted.

## Frozen artifact verification

Z64/Z96/Z128 full SHA256 values match the Stage-Z manifests. Z96 remains
`(212,8,96,96,96)` float32 with axis order `(N,C,Nphi,Ntheta,Nr)`, all finite,
and strictly positive rho/press. The Z96 P3 normalizer checksum, train-only fit
indices 0..168, and validation oracle floors exactly reproduce Stage Z.

The Z64 reference was read from the frozen Stage-T epoch-150 metrics and
rollout. Same-resolution persistence was read from the Stage-Z artifact. No
numbers in the comparison table were substituted for missing Z96 model output.

## Direct answers

1. **Failure layer?** Direct evidence identifies the WSL GPU bridge; host driver
   state is independently unknown because Windows interop also fails.
2. **GPU scientific gate recovered?** No, FAIL.
3. **Can Z96 train with the frozen architecture?** Not testable in this WSL
   instance; model preflight was not entered.
4. **Does Z96 beat same-resolution persistence better than Z64?** Unavailable.
5. **Residual L2 improved?** Unavailable.
6. **Residual cosine improved?** Unavailable.
7. **Shell skill improved?** Unavailable.
8. **Radial skill improved?** Unavailable.
9. **Was rollout step-1 failure delayed?** Unavailable.
10. **Do fidelity-sensitive channels gain most?** Association/Spearman metrics
    are not computable without Z96 model results.
11. **Did middle/inner sampling gain become dynamics gain?** Unavailable.
12. **Does the P3 rho/press physical-tail problem remain independent?** The
    preprocessing oracle remains unchanged; model-tail behavior cannot be tested.
13. **Is 64^3 the dominant learned-dynamics bottleneck?** Still unresolved.
14. **Select 64^3 or 96^3?** UNRESOLVED; selection is forbidden before model comparison.
15. **Authorize Z128?** No.
16. **Enter unified benchmark?** No; first restore WSL CUDA and complete Z96.

## Final labels

```text
PRIMARY_DECISION = F
PRIMARY_DECISION_LABEL = GPU_ENVIRONMENT_UNRESOLVED

GPU_FAILURE_LAYER = WSL_GPU_BRIDGE
GPU_SCIENTIFIC_GATE = FAIL

HIGHER_RES_DATA_INFORMATION_GAIN = true
Z96_TRAINING_RESOURCE_FEASIBLE = false

SELECTED_ADAPTED_RESOLUTION = UNRESOLVED
AUTHORIZE_Z128_PILOT = false

EXACT_REPRODUCTION_BLOCKED = true
REPRODUCTION_SCOPE =
ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION

AUTHORIZE_NEXT_STAGE = resource_recovery
```
"""
    atomic_text(OUT / "STAGE_AA_REPORT.md", report)
    atomic_text(OUT / "STAGE_AA_DECISION.md", """# Stage AA decision

`PRIMARY_DECISION = F — GPU_ENVIRONMENT_UNRESOLVED`

The mandatory GPU scientific gate failed before Z96 model preflight because
the WSL GPU bridge exposes no `/dev/dxg`. No training, optimizer update, CPU
fallback, driver/package change, or scientific-configuration change occurred.

```text
GPU_FAILURE_LAYER = WSL_GPU_BRIDGE
GPU_SCIENTIFIC_GATE = FAIL
HIGHER_RES_DATA_INFORMATION_GAIN = true
Z96_TRAINING_RESOURCE_FEASIBLE = false
SELECTED_ADAPTED_RESOLUTION = UNRESOLVED
AUTHORIZE_Z128_PILOT = false
EXACT_REPRODUCTION_BLOCKED = true
REPRODUCTION_SCOPE = ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION
AUTHORIZE_NEXT_STAGE = resource_recovery
```
""")
    print(json.dumps({
        "PRIMARY_DECISION": "F",
        "GPU_FAILURE_LAYER": gate["GPU_FAILURE_LAYER"],
        "GPU_SCIENTIFIC_GATE": gate["GPU_SCIENTIFIC_GATE"],
        "Z96_TRAINING_RESOURCE_FEASIBLE": False,
        "SELECTED_ADAPTED_RESOLUTION": "UNRESOLVED",
    }, indent=2))


if __name__ == "__main__":
    main()
