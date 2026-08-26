#!/usr/bin/env python3
"""Summarize completed Stage Z data gates and an explicit GPU resource block."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from grmhd import CHANNELS
from grmhd.dataset import sha256_file
from grmhd.stage_s_training import natural_epoch_orders, order_sha256
from grmhd.stage_z import REPRODUCTION_SCOPE
from train_stage_z import build_model_at_resolution


ROOT = Path(__file__).resolve().parents[1]
SELECTED_STEPS = (1, 2, 3, 5, 10, 19, 25, 42, 50, 75, 100)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty Stage Z comparison: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def finite_or_none(value: Any) -> float | None:
    return None if value is None else float(value)


def plot_information(figures: Path, gate: Mapping[str, Any]) -> None:
    figures.mkdir(parents=True, exist_ok=True)
    resolutions = [64, 96, 128]
    metrics = gate["metrics"]
    rel = [metrics[str(value)]["temporal_increment_relative_difference"] for value in resolutions]
    cos = [metrics[str(value)]["temporal_increment_cosine"] for value in resolutions]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax2 = ax1.twinx()
    ax1.plot(resolutions, rel, "o-", color="tab:red", label="relative difference")
    ax2.plot(resolutions, cos, "s-", color="tab:blue", label="cosine")
    ax1.set(xlabel="spherical target resolution", ylabel="median increment relative difference")
    ax2.set_ylabel("median increment cosine")
    ax1.grid(True, alpha=0.3)
    lines = ax1.lines + ax2.lines
    ax1.legend(lines, [line.get_label() for line in lines], loc="center right")
    fig.tight_layout()
    fig.savefig(figures / "temporal_increment_fidelity_vs_resolution.png", dpi=160)
    plt.close(fig)


def blocked_model_plot(
    path: Path, ylabel: str, z64_value: float, *, horizontal_zero: bool = False,
) -> None:
    fig, axis = plt.subplots(figsize=(7, 4.2))
    axis.scatter([64], [z64_value], s=55, label="Z64 Stage-T epoch 150")
    if horizontal_zero:
        axis.axhline(0, color="black", linewidth=0.8, alpha=0.5)
    axis.set(xlim=(55, 137), xticks=[64, 96, 128], xlabel="spherical target resolution", ylabel=ylabel)
    axis.text(96, z64_value, "Z96/Z128 unavailable:\nGPU preflight blocked", ha="center", va="bottom")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/stage_z/higher_resolution.yaml"))
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/stage_z"))
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    gate = json.loads((args.out_dir / "regrid/information_gain.json").read_text(encoding="utf-8"))
    oracle = json.loads((args.out_dir / "preprocessing/oracle_floor_summary.json").read_text(encoding="utf-8"))
    gpu = json.loads((args.out_dir / "training/z96/gpu_preflight.json").read_text(encoding="utf-8"))
    if gpu["training_resource_feasible"] or gpu["status"] != "blocked_gpu_unavailable":
        raise ValueError("summarizer is only for the observed Stage Z GPU-blocked path")
    z64 = json.loads((ROOT / "artifacts/stage_t/full_long/checkpoint_metrics_epoch_0150.json").read_text(encoding="utf-8"))
    rollout64 = json.loads((ROOT / "artifacts/stage_t/full_long/rollout/epoch_0150_rollout.json").read_text(encoding="utf-8"))
    training64 = json.loads((ROOT / "artifacts/stage_t/full_long/training_summary.json").read_text(encoding="utf-8"))
    persistence = json.loads(
        (args.out_dir / "comparison/persistence_by_resolution.json").read_text(encoding="utf-8")
    )["resolutions"]
    z64_model = z64["model"]
    fidelity = gate["metrics"]

    orders = natural_epoch_orders(range(168), epochs=150, seed=42)
    flat_order = [value for order in orders for value in order]
    stage_s = yaml.safe_load(
        (ROOT / "configs/stage_s/expanded_localno_p3_residual.yaml").read_text(encoding="utf-8")
    )
    initial_checks = {
        resolution: build_model_at_resolution(stage_s, config, resolution)
        for resolution in (64, 96, 128)
    }
    initial_control = {
        "schema_version": "stage-z-initial-and-order-control-v1",
        "reproduction_scope": REPRODUCTION_SCOPE,
        "shared_initial_tensor_sha256": config["model"]["shared_initial_tensor_sha256"],
        "shared_initial_file_sha256": sha256_file(ROOT / config["model"]["shared_initial_state"]),
        "parameter_count_by_resolution": {
            str(resolution): sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            for resolution, (model, _) in initial_checks.items()
        },
        "strict_cpu_initial_load_by_resolution": {"64": True, "96": True, "128": True},
        "trainable_tensor_hash_by_resolution": {
            str(resolution): digest for resolution, (_, digest) in initial_checks.items()
        },
        "pair_order_sha256": order_sha256(flat_order),
        "epoch_order_sha256": [order_sha256(order) for order in orders],
        "same_order_for_all_resolutions": True,
        "epochs": 150,
        "pairs_per_epoch": 168,
        "optimizer_updates_per_epoch": 42,
        "optimizer_updates": 6300,
    }
    atomic_json(args.out_dir / "training/initial_weight_and_order_control.json", initial_control)

    for resolution in (96, 128):
        status = "blocked_gpu_preflight" if resolution == 96 else "not_attempted_after_z96_gpu_block"
        resolved = {
            "schema_version": "stage-z-planned-training-contract-v1",
            "reproduction_scope": REPRODUCTION_SCOPE,
            "resolution": resolution,
            "status": status,
            "training_started": False,
            "epochs_completed": 0,
            "dataset": config["data"]["resolutions"][resolution],
            "preprocessing": f"artifacts/stage_z/preprocessing/p3_{resolution}.json",
            "model": config["model"],
            "training": config["training"],
            "shared_initial_tensor_sha256": initial_control["shared_initial_tensor_sha256"],
            "pair_order_sha256": initial_control["pair_order_sha256"],
            "gpu_blocker": "artifacts/stage_z/training/z96/gpu_preflight.json",
            "cpu_fallback": False,
        }
        atomic_json(args.out_dir / f"training/z{resolution}/resolved_config.json", resolved)
        atomic_json(args.out_dir / f"training/z{resolution}/status.json", {
            "resolution": resolution, "status": status, "training_started": False,
            "epochs_completed": 0, "checkpoints_created": [],
            "one_step_metrics_available": False, "rollout_metrics_available": False,
        })

    model_rows = []
    for resolution in (64, 96, 128):
        base = {
            "reproduction_scope": REPRODUCTION_SCOPE,
            "resolution": resolution,
            "increment_fidelity_class": gate["classification"][str(resolution)],
            "temporal_increment_relative_difference": fidelity[str(resolution)]["temporal_increment_relative_difference"],
            "temporal_increment_cosine": fidelity[str(resolution)]["temporal_increment_cosine"],
            "information_reference": "HIGHER_RES_SAMPLING_REFERENCE",
            "same_resolution_persistence_state_l2": persistence[str(resolution)]["normalized_state_relative_l2"]["arithmetic_average"],
            "same_resolution_persistence_physical_l2": persistence[str(resolution)]["physical_relative_l2"]["arithmetic_average"],
        }
        if resolution == 64:
            base.update({
                "model_status": "Stage-T epoch-150 reusable comparable reference",
                "state_normalized_l2": z64_model["normalized_relative_l2"]["arithmetic_average"],
                "persistence_ratio": z64["model_error_over_persistence_error"],
                "residual_relative_l2": z64_model["residual"]["arithmetic_average_relative_l2"],
                "residual_global_cosine": z64_model["residual"]["global_cosine"],
                "shell_skill": z64["transport"]["shell_skill_median"],
                "radial_skill": z64["transport"]["radial_skill_median"],
                "first_10x_physical_range_step": rollout64["FIRST_10X_PHYSICAL_RANGE_STEP"],
                "optimizer_updates": 6300,
            })
        else:
            base.update({
                "model_status": "blocked_gpu_preflight",
                "state_normalized_l2": None, "persistence_ratio": None,
                "residual_relative_l2": None, "residual_global_cosine": None,
                "shell_skill": None, "radial_skill": None,
                "first_10x_physical_range_step": None, "optimizer_updates": 0,
            })
        model_rows.append(base)
    atomic_csv(args.out_dir / "comparison/resolution_model_comparison.csv", model_rows)

    temporal_rows = read_csv(args.out_dir / "regrid/temporal_increment_fidelity.csv")
    channel_rows = []
    for resolution in (64, 96, 128):
        for name in CHANNELS:
            selected = [row for row in temporal_rows if int(row["resolution"]) == resolution and row["channel"] == name]
            row = {
                "reproduction_scope": REPRODUCTION_SCOPE, "resolution": resolution, "channel": name,
                "increment_relative_difference_median": float(np.median([float(item["residual_relative_difference"]) for item in selected])),
                "increment_cosine_median": float(np.median([float(item["residual_cosine"]) for item in selected])),
                "oracle_floor_physical_relative_l2": oracle["resolutions"][str(resolution)]["per_channel"][name],
                "same_resolution_persistence_state_l2": persistence[str(resolution)]["normalized_state_relative_l2"]["per_channel"][name],
                "same_resolution_persistence_physical_l2": persistence[str(resolution)]["physical_relative_l2"]["per_channel"][name],
                "model_status": "available_stage_t_epoch150" if resolution == 64 else "blocked_gpu_preflight",
                "state_relative_l2": z64_model["normalized_relative_l2"]["per_channel"][name] if resolution == 64 else None,
                "residual_relative_l2": z64_model["residual"]["per_channel_relative_l2"][name] if resolution == 64 else None,
                "residual_cosine": z64_model["residual"]["per_channel_cosine"][name] if resolution == 64 else None,
            }
            channel_rows.append(row)
    atomic_csv(args.out_dir / "comparison/per_channel_comparison.csv", channel_rows)

    source_shell = read_csv(args.out_dir / "regrid/shell_increment_fidelity.csv")
    shell_rows = []
    for resolution in (64, 96, 128):
        for name in CHANNELS:
            for shell in range(8):
                selected = [
                    row for row in source_shell
                    if int(row["resolution"]) == resolution and row["channel"] == name and int(row["shell"]) == shell
                ]
                shell_rows.append({
                    "reproduction_scope": REPRODUCTION_SCOPE,
                    "resolution": resolution, "channel": name, "shell": shell,
                    "median_absolute_increment_difference": float(np.median([
                        float(row["absolute_difference"]) for row in selected
                    ])),
                    "median_shell_vector_relative_l2": float(np.median([
                        float(row["shell_vector_relative_l2"]) for row in selected
                    ])),
                    "model_shell_skill": (
                        z64["transport"]["per_channel"][name]["shell_skill_median"]
                        if resolution == 64 else None
                    ),
                    "model_status": "available_stage_t_epoch150" if resolution == 64 else "blocked_gpu_preflight",
                })
    atomic_csv(args.out_dir / "comparison/per_shell_comparison.csv", shell_rows)

    rollout_by_step = {int(row["step"]): row for row in rollout64["records"]}
    rollout_rows = []
    for resolution in (64, 96, 128):
        for step in SELECTED_STEPS:
            if resolution == 64:
                record = rollout_by_step[step]
                transport = record.get("transport")
                rollout_rows.append({
                    "reproduction_scope": REPRODUCTION_SCOPE, "resolution": resolution, "step": step,
                    "status": "available_stage_t_epoch150", "finite": record["finite"],
                    "rho_press_positive": record["rho_press_positive"], "above_Rout": record["above_Rout"],
                    "normalized_state_l2": finite_or_none(record.get("normalized_relative_l2_average")),
                    "residual_cosine": finite_or_none(record.get("residual_cosine")),
                    "shell_skill": finite_or_none(transport.get("shell_skill_median") if transport else None),
                    "radial_skill": finite_or_none(transport.get("radial_skill_median") if transport else None),
                    "physical_range_explosion": record["physical_range_explosion"],
                })
            else:
                rollout_rows.append({
                    "reproduction_scope": REPRODUCTION_SCOPE, "resolution": resolution, "step": step,
                    "status": "blocked_gpu_preflight", "finite": None,
                    "rho_press_positive": None, "above_Rout": None, "normalized_state_l2": None,
                    "residual_cosine": None, "shell_skill": None, "radial_skill": None,
                    "physical_range_explosion": None,
                })
    atomic_csv(args.out_dir / "comparison/rollout_comparison.csv", rollout_rows)

    proxy_rows = [{
        "reproduction_scope": REPRODUCTION_SCOPE,
        "resolution": resolution,
        "metric_semantics": "SPHERICAL_COORDINATE_VOLUME_PROXY",
        "formula": "r^2 sin(theta) dr dtheta dphi; not strict Kerr-Schild proper volume",
        "model_weighted_error": None,
        "status": "not_recomputable_without_GPU_model_evaluation",
        "same_resolution_persistence_weighted_error_average": persistence[str(resolution)]["spherical_coordinate_volume_proxy_relative_l2"]["arithmetic_average"],
        "same_resolution_persistence_weighted_error_global": persistence[str(resolution)]["spherical_coordinate_volume_proxy_relative_l2"]["global"],
        "oracle_floor_unweighted_average": oracle["resolutions"][str(resolution)]["physical_relative_l2_arithmetic_average"],
    } for resolution in (64, 96, 128)]
    atomic_csv(args.out_dir / "comparison/physical_proxy_comparison.csv", proxy_rows)

    figures = args.out_dir / "figures"
    plot_information(figures, gate)
    blocked_model_plot(
        figures / "residual_cosine_vs_resolution.png", "one-step residual global cosine",
        z64_model["residual"]["global_cosine"], horizontal_zero=True,
    )
    blocked_model_plot(
        figures / "shell_skill_vs_resolution.png", "one-step shell skill",
        z64["transport"]["shell_skill_median"], horizontal_zero=True,
    )
    blocked_model_plot(
        figures / "radial_skill_vs_resolution.png", "one-step radial skill",
        z64["transport"]["radial_skill_median"], horizontal_zero=True,
    )
    records = rollout64["records"]
    fig, axis = plt.subplots(figsize=(7, 4.2))
    gt = [row for row in records if row["ground_truth_available"]]
    axis.plot([row["step"] for row in gt], [row["normalized_relative_l2_average"] for row in gt], label="Z64 epoch 150")
    axis.set(xlabel="closed-loop step", ylabel="normalized state relative L2", yscale="log")
    axis.text(22, 1, "Z96/Z128 blocked before training", ha="center")
    axis.grid(True, alpha=0.3); axis.legend(); fig.tight_layout()
    fig.savefig(figures / "rollout_error_vs_step.png", dpi=160); plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4.2))
    axis.step([row["step"] for row in records], [int(row["physical_range_explosion"]) for row in records], where="post", label="Z64 first-10x flag")
    axis.set(xlabel="closed-loop step", ylabel="physical range >=10x train", yticks=[0, 1])
    axis.text(52, 0.5, "Z96/Z128 blocked before training", ha="center")
    axis.grid(True, alpha=0.3); axis.legend(); fig.tight_layout()
    fig.savefig(figures / "physical_range_vs_step.png", dpi=160); plt.close(fig)

    rel64 = fidelity["64"]["temporal_increment_relative_difference"]
    rel96 = fidelity["96"]["temporal_increment_relative_difference"]
    inner_improvement = 1 - fidelity["96"]["temporal_increment_loss_inner"] / fidelity["64"]["temporal_increment_loss_inner"]
    middle_improvement = 1 - fidelity["96"]["temporal_increment_loss_middle"] / fidelity["64"]["temporal_increment_loss_middle"]
    outer_improvement = 1 - fidelity["96"]["temporal_increment_loss_outer"] / fidelity["64"]["temporal_increment_loss_outer"]
    report = f"""# Stage Z — Higher-Resolution Adapted Spherical-KS Reproduction and Regrid-Loss Test

`REPRODUCTION_SCOPE = {REPRODUCTION_SCOPE}`

Stage Z is not an exact paper reproduction, a paper-faithful LocalNO, or an
exact volumetric 3D DISCO reproduction. Exact reproduction remains blocked by
Stage Y. The data/information half completed; the model half was stopped at the
mandatory CUDA gate without CPU fallback.

## Frozen controls and generated datasets

- All 212 raw snapshots, chronological split, eight stored spherical-component
  fields, nearest-leaf sampling, residual contract, Stage-T architecture/loss,
  optimizer, seed, and pair order are frozen.
- Z64 regression: all 40 arrays at snapshots 0/50/100/150/211 are bitwise
  identical; `Z64_REGRID_REGRESSION_PASS = true`.
- Z64/Z96/Z128 SHA256: `3582a5c4...b50da`, `292d3fe6...1564`,
  `fe62e9c2...9b1d`. Z128 is only `HIGHER_RES_SAMPLING_REFERENCE`.
- All generated HDF5 arrays are float32 `(N,C,Nphi,Ntheta,Nr)`, finite, with
  strictly positive rho/press. Direct raw sampling equals decompressed HDF5
  bitwise on 120 checked cross-resolution arrays (80 at Z96/Z128).
- Uncompressed float32 estimates are 6,002,049,024 bytes for Z96 and
  14,227,079,168 bytes for Z128; the pre-generation project filesystem had
  about 935 GiB free and RAM had about 13 GiB available. Chunked generation
  never loaded a complete trajectory into RAM.
- Physical-r internal shell boundaries are identical across resolutions; they
  are not re-fit to equal index counts.
- Common trainable tensor hash: `{initial_control['shared_initial_tensor_sha256']}`;
  358,296 parameters at all resolutions; common pair-order hash:
  `{initial_control['pair_order_sha256']}`.

## Information-fidelity result

| resolution | class | median increment rel. difference | median cosine | shell error | radial error |
|---:|---|---:|---:|---:|---:|
| 64 | POOR | {rel64:.6f} | {fidelity['64']['temporal_increment_cosine']:.6f} | {fidelity['64']['shell_increment_relative_l2']:.6f} | {fidelity['64']['radial_increment_relative_l2']:.6f} |
| 96 | MODERATE | {rel96:.6f} | {fidelity['96']['temporal_increment_cosine']:.6f} | {fidelity['96']['shell_increment_relative_l2']:.6f} | {fidelity['96']['radial_increment_relative_l2']:.6f} |
| 128 | GOOD/reference | 0 | 1 | 0 | 0 |

Z96 improves relative difference by {gate['relative_improvements']['temporal_increment_relative_difference']:.2%},
shell error by {gate['relative_improvements']['shell_increment_relative_l2']:.2%},
radial error by {gate['relative_improvements']['radial_increment_relative_l2']:.2%},
and cosine by {gate['relative_improvements']['temporal_increment_cosine_absolute_delta']:.6f}.
All four predeclared conditions pass, so `HIGHER_RES_DATA_INFORMATION_GAIN = true`.
Regional relative improvements are inner {inner_improvement:.2%}, middle
{middle_improvement:.2%}, outer {outer_improvement:.2%}; middle improves most,
while outer remains the least rescued.

Per-channel, Bcc2 and rho have the largest relative-difference reductions
(about 25.4% and 23.3%). Bcc1 is essentially unchanged/slightly worse; vel2
improves only about 2.8%. Z96 remains MODERATE rather than close enough to call
the Z128 sampling reference reproduced.

## Resolution-specific P3 and oracle

P3 transform family/policies are identical, while numerical epsilon/median/MAD
statistics were independently fit from train snapshots 0..168 for each
resolution. Validation was never used for fit. Normalizer hashes are Z64
`c2a36e...ffa6`, Z96 `63c5dd...127`, Z128 `aa737d...d4a`.

For the priority channels Bcc2/Bcc3/vel3/rho/press, validation encode/decode
relative floors remain approximately 1e-8--3e-7 at every resolution. Thus
higher resolution did not reintroduce their preprocessing floor. The large
canonical softclip floors on Bcc1 and vel2 remain a frozen P3 limitation.

Same-resolution persistence was evaluated on all 42 validation pairs: normalized
state L2 is {persistence['64']['normalized_state_relative_l2']['arithmetic_average']:.6f}
/ {persistence['96']['normalized_state_relative_l2']['arithmetic_average']:.6f}
/ {persistence['128']['normalized_state_relative_l2']['arithmetic_average']:.6f}
for Z64/Z96/Z128. Physical L2 is
{persistence['64']['physical_relative_l2']['arithmetic_average']:.6f} /
{persistence['96']['physical_relative_l2']['arithmetic_average']:.6f} /
{persistence['128']['physical_relative_l2']['arithmetic_average']:.6f}.
The spherical-coordinate volume-proxy persistence error is recorded separately;
it is not strict Kerr-Schild proper-volume error and does not replace the selector.

## GPU gate and model status

`nvidia-smi` failed with `GPU access blocked by the operating system`;
PyTorch `{gpu['torch_version']}` / CUDA `{gpu['torch_cuda_version']}` reported
`cuda_available=false`, device count 0. Z96 forward/backward therefore could
not begin. No CPU fallback, optimizer step, epoch, checkpoint, one-step model
evaluation, volume-proxy model error, or rollout was produced. Z128 preflight
was not attempted after the prerequisite Z96 CUDA failure.

The reusable Z64 Stage-T epoch-150 reference is contract-compatible (same
split/P3 family/model/loss/scheduler prefix, 6,300 updates): state L2
{z64_model['normalized_relative_l2']['arithmetic_average']:.6f}, residual L2
{z64_model['residual']['arithmetic_average_relative_l2']:.6f}, residual cosine
{z64_model['residual']['global_cosine']:.6f}, shell/radial skill
{z64['transport']['shell_skill_median']:.6f}/{z64['transport']['radial_skill_median']:.6f},
and first-10x physical range step 1. It cannot answer the cross-resolution
learned-dynamics question alone.

## Direct answers

1. **Does Z96 preserve more increment than Z64?** Yes: all four frozen
   information gates improve and classification moves POOR -> MODERATE.
2. **Is Z96 already close to Z128?** No under the frozen fidelity thresholds;
   its median relative difference is still {rel96:.3f}. Z128 is a sampling
   reference, not truth.
3. **Most resolution-sensitive channels?** Bcc2 and rho by relative-difference
   reduction; vel3/vel1/press also improve materially. Bcc1 and vel2 change least.
4. **Most improved region?** Middle ({middle_improvement:.1%}), then inner
   ({inner_improvement:.1%}); outer improves only {outer_improvement:.1%}.
5. **Does higher resolution lower learned residual L2?** Not available: GPU block.
6. **Does learned residual cosine improve?** Not available: GPU block.
7. **Does shell skill improve?** Sampling shell fidelity improves; learned shell
   skill is not available.
8. **Does radial skill improve?** Sampling radial fidelity improves; learned
   radial skill is not available.
9. **Is first-10x delayed beyond step 1?** Not available for Z96/Z128.
10. **Does rho/press decoder-tail improve?** Their oracle floor remains tiny and
    stable; model decoder-tail behavior is not available.
11. **Does information gain become learned-dynamics gain?** Unresolved because
    no high-resolution model could run.
12. **Is 64^3 the main adapted bottleneck?** It is a demonstrated information
    bottleneck, but dominance over operator/objective limitations is unresolved.
13. **Use 64, 96, or 128?** `UNRESOLVED_PENDING_GPU_MODEL_COMPARISON`; Z96 is the
    first resource-prudent candidate because it passes the information gate,
    but it cannot be scientifically selected without its controlled model run.
14. **Proceed to unified Persistence/FNO/CNN/LocalNO benchmark?** Authorized only
    as a resource-feasible adapted suite after GPU access is restored and the
    Z96 controlled model comparison resolves the selected resolution.

## Final labels

```text
PRIMARY_DECISION = E
PRIMARY_DECISION_LABEL = HIGH_RES_MODEL_RESOURCE_LIMITED

REPRODUCTION_SCOPE =
ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION

Z64_REGRID_REGRESSION_PASS = true
HIGHER_RES_DATA_INFORMATION_GAIN = true

TEMPORAL_INCREMENT_FIDELITY_64 = POOR
TEMPORAL_INCREMENT_FIDELITY_96 = MODERATE
TEMPORAL_INCREMENT_FIDELITY_128 = GOOD_REFERENCE_SELF

Z96_TRAINING_RESOURCE_FEASIBLE = false
Z128_TRAINING_RESOURCE_FEASIBLE = false
SELECTED_ADAPTED_RESOLUTION = UNRESOLVED_PENDING_GPU_MODEL_COMPARISON

EXACT_REPRODUCTION_BLOCKED = true
AUTHORIZE_NEXT_STAGE = resource_feasible_adapted_baseline_suite
```
"""
    atomic_text(args.out_dir / "STAGE_Z_REPORT.md", report)
    atomic_text(args.out_dir / "STAGE_Z_DECISION.md", """# Stage Z decision

`PRIMARY_DECISION = E — HIGH_RES_MODEL_RESOURCE_LIMITED`

Z96 passes all four predeclared information-gain conditions relative to Z64,
but WSL currently exposes no CUDA device, so scientifically comparable Z96 and
Z128 model training/evaluation cannot run. No CPU fallback was used. Therefore
Stage Z establishes that 64^3 loses temporal information, but cannot determine
whether that loss is the dominant learned-dynamics bottleneck.

```text
REPRODUCTION_SCOPE = ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION
Z64_REGRID_REGRESSION_PASS = true
HIGHER_RES_DATA_INFORMATION_GAIN = true
SELECTED_ADAPTED_RESOLUTION = UNRESOLVED_PENDING_GPU_MODEL_COMPARISON
EXACT_REPRODUCTION_BLOCKED = true
AUTHORIZE_NEXT_STAGE = resource_feasible_adapted_baseline_suite
```
""")
    print(json.dumps({
        "PRIMARY_DECISION": "E",
        "HIGHER_RES_DATA_INFORMATION_GAIN": True,
        "Z96_TRAINING_RESOURCE_FEASIBLE": False,
        "SELECTED_ADAPTED_RESOLUTION": "UNRESOLVED_PENDING_GPU_MODEL_COMPARISON",
    }, indent=2))


if __name__ == "__main__":
    main()
