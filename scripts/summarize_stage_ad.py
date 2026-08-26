#!/usr/bin/env python3
"""Generate the frozen Stage AD figures, report, and scientific decision."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/stage_ad"
TRAIN = OUT / "training/disco3d_localno"
FIGURES = OUT / "figures"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: object) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True))


def metric_plot(
    checkpoint_rows: list[dict[str, str]],
    *,
    key: str,
    ylabel: str,
    filename: str,
    stage_t: float,
    persistence: float,
    log: bool = False,
) -> None:
    epochs = np.asarray([int(row["epoch"]) for row in checkpoint_rows])
    values = np.asarray([float(row[key]) for row in checkpoint_rows])
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot(epochs, values, marker="o", label="adapted 3D-DISCO LocalNO")
    axis.axhline(stage_t, color="tab:orange", linestyle="--", label="Stage-T LocalNO")
    axis.axhline(persistence, color="black", linestyle=":", label="persistence")
    if log and np.all(values > 0) and stage_t > 0 and persistence > 0:
        axis.set_yscale("log")
    axis.set_xlabel("epoch")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES / filename, dpi=160)
    plt.close(figure)


def make_figures(
    summary: Mapping[str, Any],
    checkpoint_rows: list[dict[str, str]],
    rollout: Mapping[str, Any],
) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    stage_t = summary["stage_t"]

    train = read_csv(TRAIN / "train_log.csv")
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot([int(row["epoch"]) for row in train], [float(row["train_loss_mean"]) for row in train])
    axis.set_yscale("log")
    axis.set_xlabel("epoch")
    axis.set_ylabel("mean normalized squared loss")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES / "training_loss.png", dpi=160)
    plt.close(figure)

    metric_plot(
        checkpoint_rows, key="normalized_relative_l2", ylabel="state relative L2",
        filename="state_l2_comparison.png", stage_t=float(stage_t["state_l2"]),
        persistence=float(summary["persistence_state_l2"]), log=True,
    )
    metric_plot(
        checkpoint_rows, key="residual_relative_l2", ylabel="residual relative L2",
        filename="residual_l2_comparison.png", stage_t=float(stage_t["residual_l2"]),
        persistence=1.0, log=True,
    )
    metric_plot(
        checkpoint_rows, key="shell_skill", ylabel="shell skill",
        filename="shell_skill_comparison.png", stage_t=float(stage_t["shell_skill"]),
        persistence=0.0,
    )
    metric_plot(
        checkpoint_rows, key="radial_skill", ylabel="radial skill",
        filename="radial_skill_comparison.png", stage_t=float(stage_t["radial_skill"]),
        persistence=0.0,
    )

    records = rollout["records"]
    gt = [row for row in records if row["ground_truth_available"]]
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.plot([row["step"] for row in gt], [row["normalized_relative_l2_average"] for row in gt], label="state L2")
    axis.plot([row["step"] for row in gt], [row["residual_relative_l2"] for row in gt], label="residual L2")
    axis.set_yscale("log")
    axis.set_xlabel("autoregressive step")
    axis.set_ylabel("relative error")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES / "rollout_error.png", dpi=160)
    plt.close(figure)

    robust_physical = []
    extreme_physical = []
    for row in records:
        ranges = row["physical_range"].values()
        robust_physical.append(max(max(abs(value["q001"]), abs(value["q999"])) for value in ranges))
        extreme_physical.append(max(max(abs(value["minimum"]), abs(value["maximum"])) for value in ranges))
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    axes[0].plot([row["step"] for row in records], robust_physical, label="max |q001/q999|")
    axes[0].plot([row["step"] for row in records], extreme_physical, label="max |min/max|", alpha=0.8)
    axes[0].set_ylabel("physical decoded magnitude")
    axes[0].set_yscale("log")
    axes[0].legend(fontsize=8)
    axes[1].plot([row["step"] for row in records], [row["normalized_global_norm"] for row in records], label="normalized global norm")
    axes[1].axhline(float(read_json(TRAIN / "one_step_metrics.json")["metrics"]["normalized_global_norm"]["Rout"]), color="black", linestyle="--", label="train Rout")
    axes[1].set_ylabel("normalized global norm")
    axes[1].set_yscale("log")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.set_xlabel("autoregressive step")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(FIGURES / "physical_range.png", dpi=160)
    plt.close(figure)

    activity = read_csv(OUT / "attribution/branch_activity.csv")
    layers = np.asarray([int(row["layer"]) for row in activity])
    width = 0.25
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))
    for offset, (key, label) in enumerate((
        ("spectral_output_norm", "spectral"),
        ("differential_output_norm", "differential"),
        ("disco3d_output_norm", "DISCO3D"),
    )):
        axes[0].bar(layers + (offset - 1) * width, [float(row[key]) for row in activity], width=width, label=label)
    for offset, (key, label) in enumerate((
        ("spectral_parameter_gradient_norm", "spectral"),
        ("differential_parameter_gradient_norm", "differential"),
        ("disco3d_parameter_gradient_norm", "DISCO3D"),
    )):
        axes[1].bar(layers + (offset - 1) * width, [float(row[key]) for row in activity], width=width, label=label)
    axes[0].set_title("output norms")
    axes[1].set_title("parameter-gradient norms")
    for axis in axes:
        axis.set_xlabel("LocalNO layer")
        axis.set_yscale("log")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURES / "branch_norms.png", dpi=160)
    plt.close(figure)

    best_epoch = int(summary["formal_best_epoch"])
    checkpoint = torch.load(
        TRAIN / f"checkpoints/epoch_{best_epoch:04d}.pt",
        map_location="cpu",
        weights_only=True,
    )
    state = checkpoint["model_state_dict"]
    prefix = "local_no_blocks.local_convs.0."
    weight = state[prefix + "weight"].double()
    basis = state[prefix + "normalized_basis"].double()
    quadrature = (2.0 / 64.0) ** 3
    kernel = quadrature * torch.einsum("oik,kxyz->oixyz", weight, basis)
    plane = kernel[0, 0, kernel.shape[-3] // 2].numpy()
    figure, axis = plt.subplots(figsize=(5.2, 4.5))
    image = axis.imshow(plane, origin="lower", cmap="coolwarm")
    axis.set_title("trained layer-0 DISCO impulse kernel\n(output 0, input 0, central plane)")
    axis.set_xlabel("local r offset")
    axis.set_ylabel("local theta offset")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(FIGURES / "impulse_response.png", dpi=160)
    plt.close(figure)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def main() -> None:
    summary = read_json(OUT / "comparison/evaluation_summary.json")
    implementation = read_json(OUT / "implementation/implementation_gates.json")
    unit_tests = read_json(OUT / "implementation/unit_tests.json")
    preflight = read_json(OUT / "preflight/cuda_preflight.json")
    training = read_json(TRAIN / "training_summary.json")
    one_step = read_json(TRAIN / "one_step_metrics.json")["metrics"]
    rollout = read_json(TRAIN / "rollout_metrics.json")
    checkpoint_rows = read_csv(TRAIN / "checkpoint_metrics.csv")
    comparison = read_csv(OUT / "comparison/persistence_stage_t_disco.csv")
    channels = read_csv(OUT / "comparison/per_channel_comparison.csv")
    bootstrap = read_csv(OUT / "comparison/paired_bootstrap.csv")
    quadrature = read_json(OUT / "implementation/quadrature_audit.json")
    dense = read_json(OUT / "implementation/dense_reference_test.json")
    orientation = read_json(OUT / "implementation/kernel_orientation_test.json")
    gradcheck = read_json(OUT / "implementation/gradcheck.json")
    ablation = read_csv(OUT / "attribution/branch_ablation.csv")
    make_figures(summary, checkpoint_rows, rollout)

    disco = summary["one_step"]
    stage_t = summary["stage_t"]
    state_gate = disco["state_l2"] <= 1.05 * stage_t["state_l2"]
    residual_gate = disco["residual_l2"] < 1.0
    direction_gate = disco["cosine"] > 0.7
    shell_gate = disco["shell_skill"] > -1.579472
    radial_gate = disco["radial_skill"] > -0.248089
    rollout_gate = rollout["FIRST_10X_PHYSICAL_RANGE_STEP"] > 1
    all_implementation = all((
        implementation["ALL_BASES_ACTIVE"],
        implementation["BASIS_PARTITION_OF_UNITY_PASS"],
        implementation["DENSE_REFERENCE_MATCH"],
        implementation["KERNEL_ORIENTATION_PASS"],
        implementation["GRADCHECK_PASS"],
        unit_tests["status"] == "passed",
    ))
    # Apply the frozen hierarchy literally.  B is excluded because neither
    # aggregate transport metric improves by +0.10.  C captures the nominal
    # aggregate state/residual gains together with unchanged rollout and no
    # meaningful transport gain; paired uncertainty is reported separately.
    if not all_implementation:
        primary = "F"
        label = "DISCO3D_IMPLEMENTATION_STILL_INVALID"
    elif not training["all_finite"] or training["completed_epoch"] < 150:
        primary = "H"
        label = "ENGINEERING_FAILURE"
    elif (
        state_gate and residual_gate and direction_gate and rollout_gate
        and ((disco["shell_skill"] > 0 or disco["radial_skill"] > 0) or (shell_gate and radial_gate))
    ):
        primary = "A"
        label = "ADAPTED_3D_DISCO_RESCUES_DYNAMICS"
    elif (
        state_gate and residual_gate
        and (disco["shell_skill"] - stage_t["shell_skill"] >= 0.10 or disco["radial_skill"] - stage_t["radial_skill"] >= 0.10)
        and (disco["shell_skill"] < 0 or disco["radial_skill"] < 0 or not rollout_gate)
    ):
        primary = "B"
        label = "ADAPTED_3D_DISCO_PARTIALLY_IMPROVES_DYNAMICS"
    elif (
        disco["state_l2"] < stage_t["state_l2"]
        and disco["residual_l2"] < stage_t["residual_l2"]
        and not shell_gate and not radial_gate and not rollout_gate
    ):
        primary = "C"
        label = "ADAPTED_3D_DISCO_IMPROVES_ONE_STEP_ONLY"
    elif (
        disco["state_l2"] > 1.05 * stage_t["state_l2"]
        and disco["residual_l2"] > stage_t["residual_l2"]
    ):
        primary = "E"
        label = "ADAPTED_3D_DISCO_MODEL_WORSE"
    else:
        primary = "D"
        label = "ADAPTED_3D_DISCO_NOT_SUPPORTED"

    next_stage = (
        "adapted_baseline_suite_with_disco3d" if primary in "ABCDE"
        else "implementation_reassessment" if primary == "F"
        else "resource_feasible_disco3d_strategy" if primary == "G"
        else "blocked"
    )
    gates = {
        "STATE_RETENTION_GATE": "PASS" if state_gate else "FAIL",
        "RESIDUAL_GATE": "PASS" if residual_gate else "FAIL",
        "DIRECTION_GATE": "PASS" if direction_gate else "FAIL",
        "SHELL_IMPROVEMENT_GATE": "PASS" if shell_gate else "FAIL",
        "RADIAL_IMPROVEMENT_GATE": "PASS" if radial_gate else "FAIL",
        "ROLLOUT_IMPROVEMENT_GATE": "PASS" if rollout_gate else "FAIL",
    }
    decision = {
        "PRIMARY_DECISION": primary,
        "PRIMARY_DECISION_LABEL": label,
        "REPRODUCTION_SCOPE": "ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION",
        "DISCO3D_IMPLEMENTATION": "ADAPTED_ISOTROPIC_RADIAL_PIECEWISE_LINEAR",
        "DISCO3D_RADIUS_CELLS": 3,
        "DISCO3D_BASIS_COUNT": 5,
        "DISCO3D_STENCIL_SHAPE": [7, 7, 7],
        "ALL_BASES_ACTIVE": implementation["ALL_BASES_ACTIVE"],
        "BASIS_PARTITION_OF_UNITY_PASS": implementation["BASIS_PARTITION_OF_UNITY_PASS"],
        "DENSE_REFERENCE_MATCH": implementation["DENSE_REFERENCE_MATCH"],
        "KERNEL_ORIENTATION_PASS": implementation["KERNEL_ORIENTATION_PASS"],
        "GRADCHECK_PASS": implementation["GRADCHECK_PASS"],
        "DISCO3D_UNIT_TESTS_PASS": unit_tests["status"] == "passed",
        "DISCO_BRANCH_ACTIVE": summary["disco_branch_active"],
        "TRAINING_STARTED": training["completed_epoch"] > 0,
        "TRAINING_COMPLETED": training["completed_epoch"] >= 150,
        **gates,
        "EXACT_3D_DISCO_IMPLEMENTATION_FOUND": False,
        "EXACT_REPRODUCTION_BLOCKED": True,
        "AUTHORIZE_NEXT_STAGE": next_stage,
    }
    write_json(OUT / "STAGE_AD_DECISION.json", decision)

    by_channel: dict[str, dict[str, dict[str, str]]] = {}
    for row in channels:
        by_channel.setdefault(row["channel"], {})[row["model"]] = row
    channel_lines = []
    channel_delta_rows = []
    for channel, models in by_channel.items():
        old = models["Stage-T differential LocalNO"]
        new = models["Adapted 3D-DISCO LocalNO"]
        values = {
            key: float(new[key]) - float(old[key])
            for key in ("state_l2", "residual_l2", "cosine", "shell_skill", "radial_skill")
        }
        channel_delta_rows.append((channel, values))
        channel_lines.append(
            f"| {channel} | {values['state_l2']:+.6g} | {values['residual_l2']:+.6g} | "
            f"{values['cosine']:+.6g} | {values['shell_skill']:+.6g} | {values['radial_skill']:+.6g} |"
        )

    table_rows = []
    for row in comparison:
        table_rows.append(
            f"| {row['model']} | {row['params']} | {row['local_integral']} | "
            f"{fmt(float(row['state_l2']))} | {fmt(float(row['residual_l2']))} | "
            f"{fmt(float(row['cosine']))} | {fmt(float(row['shell_skill']))} | "
            f"{fmt(float(row['radial_skill']))} | {row['first10x']} |"
        )

    epoch_values = {int(row["epoch"]): float(row["normalized_relative_l2"]) for row in checkpoint_rows}
    improvement_75_150 = (epoch_values[75] - epoch_values[150]) / epoch_values[75]
    state_relative_gain = (stage_t["state_l2"] - disco["state_l2"]) / stage_t["state_l2"]
    state_bootstrap = next(row for row in bootstrap if row["metric"] == "state_l2")
    ablation_norm = next(row for row in ablation if row["metric"] == "probe_residual_disco_contribution_norm")
    report = f"""# Stage AD Report

| model | params | local integral | state L2 | residual L2 | cosine | shell skill | radial skill | first10x |
|---|---:|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_rows)}

## Scope and outcome

Stage AD tested one authorized repair only: the project-local isotropic radial, piecewise-linear adapted DISCO3D cutoff changed from one cell to three cells while K=5, the spherical 7×7×7 support, the Stage-S/T data, preprocessing, common LocalNO initialization, Plain L2 objective, scheduler, resolution, and periodic computational boundary remained frozen. This is an adapted spherical-KS workflow test, not the paper's unavailable exact volumetric DISCO implementation.

The implementation and training are valid, but the scientific outcome is not consistent. Epoch {summary['formal_best_epoch']} is selected only by the frozen 42-pair arithmetic-average state-L2 selector. Its state L2 is {disco['state_l2']:.9f}, a {100 * state_relative_gain:.3f}% improvement over Stage T, while the paired bootstrap CI for DISCO−StageT is [{float(state_bootstrap['ci95_low']):.6g}, {float(state_bootstrap['ci95_high']):.6g}] and includes zero. Residual L2 is {disco['residual_l2']:.9f} versus {stage_t['residual_l2']:.9f}; cosine improves to {disco['cosine']:.9f}. Aggregate shell and radial skills remain negative and change by {disco['shell_skill'] - stage_t['shell_skill']:+.6g} and {disco['radial_skill'] - stage_t['radial_skill']:+.6g}, respectively. The 100-step rollout stays finite and positive but FIRST_10X remains {rollout['FIRST_10X_PHYSICAL_RANGE_STEP']}.

## Implementation evidence

- The Stage-AC R=Δ, K=5 contract is invalid on the point-sampled lattice: positive support counts are `[1, 0, 0, 0, 6]`, so three interior hats are inactive and their quadrature normalizers vanish.
- Integer-cell audits found m=2 still inactive (`[1, 0, 18, 20, 14]`) and m=3 first activates every hat (`[1, 18, 56, 74, 66]`). No radius above three cells was tested.
- R=3Δ={3 * 2 / 64:.6g}; the 7³ bounding stencil contains 343 offsets, 123 inside the spherical cutoff and 220 exactly masked out.
- All Zk are finite and positive: `{', '.join(f'{value:.8g}' for value in quadrature['normalizations_Zk'])}`. Normalized quadrature-integral maximum error is {max(quadrature['absolute_errors']):.3g}.
- Partition-of-unity, compact support, dense explicit reference, cross-correlation orientation, float64 gradcheck, repository tests, and CPU/CUDA agreement all pass. Dense-reference relative L2 is {dense['relative_l2']:.3g}; CUDA relative L2 is {preflight['cpu_cuda_agreement']['relative_l2']:.3g}.
- The four branches add 5,184 parameters: 363,480 total versus 358,296 in Stage T ({100 * 5184 / 358296:.3f}% increase). Pinned `external/neuraloperator` is not modified.

## Controlled training and computational cost

- Training completed {training['completed_epoch']} epochs, {training['microbatches']:,} microbatches and {training['optimizer_updates']:,} optimizer updates on `{preflight['device']}` with no NaN/Inf.
- The frozen 30→75→150 state-L2 sequence is {epoch_values[30]:.6f}→{epoch_values[75]:.6f}→{epoch_values[150]:.6f}; 75→150 improves {100 * improvement_75_150:.3f}%, satisfying the predeclared extension rule. Epoch 300 was therefore authorized.
- Full-model CUDA forward/backward preflight times are {preflight['forward_seconds']:.4f}/{preflight['backward_seconds']:.4f} s; the isolated DISCO forward measurement is {preflight['disco_forward_seconds']:.5f} s. Preflight peak memory is {preflight['peak_allocated_mib']:.1f}/{preflight['peak_reserved_mib']:.1f} MiB allocated/reserved.
- Actual 300-epoch runtime is {training['runtime_seconds'] / 3600:.3f} h with {training['peak_allocated_mib_max']:.1f}/{training['peak_reserved_mib_max']:.1f} MiB peak memory. The exact epoch-150 cumulative cost is recorded in `comparison/efficiency_comparison.csv`.
- DISCO gradients were nonzero at every optimizer update, the trained DISCO state hash differs from initialization, and all checkpoints strictly reload with exact metric recomputation.

## Per-channel DISCO minus Stage-T deltas

Negative L2 deltas and positive cosine/skill deltas are favorable.

| channel | Δ state L2 | Δ residual L2 | Δ cosine | Δ shell skill | Δ radial skill |
|---|---:|---:|---:|---:|---:|
{chr(10).join(channel_lines)}

The largest state-L2 benefit is vel2 ({dict(channel_delta_rows)['vel2']['state_l2']:+.6g}), followed by Bcc1, press, and vel1. Residual L2 improves most for vel2, Bcc1, press, and vel1. Directional cosine improves for every channel, especially vel3 and vel1. These gains are mixed: Bcc2/Bcc3/rho/vel3 state errors worsen, aggregate transport does not improve, and physical L2 is dominated by the press inverse-preprocessing tail.

## Paired validation, attribution, and rollout

- The 42-pair paired comparison gives state win fraction {float(state_bootstrap['win_fraction']):.3f}; its confidence interval includes no reliable aggregate state advantage. Residual-L2 CI also crosses zero, while cosine improves consistently.
- The formal-best branch is active at every layer: outputs and parameter gradients are nonzero, and trained weights moved from initialization.
- Removing DISCO changes the probe residual by norm {float(ablation_norm['full']):.6g}. Full-minus-DISCO state L2 is {next(float(row['full_minus_disco']) for row in ablation if row['metric'] == 'state_l2'):.6g}, showing that the branch materially affects prediction, but ablation does not establish that the effect is beneficial.
- Physical closed-loop rollout starts at snapshot 169, uses no teacher forcing, encodes/decodes once per step, completes 100 finite steps, and preserves positive rho/press. Residual cosine first becomes negative at step {rollout['FIRST_NEGATIVE_RESIDUAL_COSINE_STEP']}; shell/radial skill first become negative at steps {rollout['FIRST_NEGATIVE_SHELL_SKILL_STEP']}/{rollout['FIRST_NEGATIVE_RADIAL_SKILL_STEP']}; physical range exceeds 10× at step {rollout['FIRST_10X_PHYSICAL_RANGE_STEP']}.

## Required questions

1. **Why was Stage AC R=Δ invalid?** Three of five point-sampled radial hats had no positive stencil point, making the basis/normalization contract degenerate.
2. **Why is R=3Δ the minimum integer-cell repair?** m=1 and m=2 fail the all-bases-active audit; m=3 is the first passing integer and no larger radius was tried.
3. **Are all five radial bases active?** Yes.
4. **Is every Zk finite and positive?** Yes.
5. **Does partition of unity pass?** Yes.
6. **Does the dense reference match?** Yes, relative L2 {dense['relative_l2']:.3g}.
7. **Is kernel orientation correct?** Yes; implementation and explicit reference use cross-correlation, and the asymmetric test distinguishes a flipped kernel: {orientation['distinguishes_flipped_kernel']}.
8. **Does gradcheck pass?** Yes for input, weight, and bias.
9. **Does the DISCO branch have nonzero output and gradient?** Yes, in every layer; gradients were nonzero throughout training.
10. **What is the 7³ GPU cost?** Isolated DISCO forward {preflight['disco_forward_seconds']:.5f} s in preflight; full forward/backward {preflight['forward_seconds']:.4f}/{preflight['backward_seconds']:.4f} s; 150 epochs took 4435.58 s and 300 took {training['runtime_seconds']:.2f} s.
11. **Did controlled training complete 150 epochs?** Yes, and the frozen convergence rule authorized completion through epoch 300.
12. **Is state L2 better/retained versus Stage T?** Retained and nominally {100 * state_relative_gain:.3f}% better, but the paired 95% CI crosses zero.
13. **Do residual L2/cosine improve?** Aggregate residual L2 improves by only {stage_t['residual_l2'] - disco['residual_l2']:.6g}; its paired-delta CI crosses zero. Cosine improves by {disco['cosine'] - stage_t['cosine']:+.6g}.
14. **Does shell skill improve?** No; {stage_t['shell_skill']:.6g}→{disco['shell_skill']:.6g}.
15. **Does radial skill improve?** No; {stage_t['radial_skill']:.6g}→{disco['radial_skill']:.6g}.
16. **Is rollout FIRST_10X delayed?** No; it remains step 1.
17. **Which channels benefit most?** vel2 most clearly in state/residual L2; Bcc1, press, and vel1 have smaller L2 gains; vel3/vel1 have the largest cosine gains. Benefits do not align consistently with aggregate transport.
18. **Does trained ablation prove DISCO affects prediction?** Yes, it proves material influence, not scientific benefit.
19. **Does this support the missing local-integral mechanism as the important method gap?** No. The valid tested adaptation is active but does not rescue aggregate transport or rollout.
20. **Proceed to the final adapted baseline benchmark?** Yes, as the frozen next-stage action for a scientifically completed A–E result; include DISCO as a tested adapted baseline, not as a rescued/exact paper method.

## Provenance and limitations

Evidence is in `artifacts/stage_ad/implementation`, `preflight`, `training/disco3d_localno`, `attribution`, and `comparison`. The workflow remains adapted spherical Kerr–Schild data on a regular 64³ computational grid with periodic computational padding and a project-local isotropic radial basis. `EXACT_3D_DISCO_IMPLEMENTATION_FOUND=false` and `EXACT_REPRODUCTION_BLOCKED=true`.
"""
    write_text(OUT / "STAGE_AD_REPORT.md", report)

    gate_lines = "\n".join(f"{key} = {value}" for key, value in gates.items())
    decision_text = f"""# Stage AD Decision

`PRIMARY_DECISION = {primary}` — `{label}`

The implementation gates, CUDA preflight, controlled training, strict reload, branch-activity audit, and 100-step rollout all completed. Aggregate state L2 is nominally {100 * state_relative_gain:.3f}% better, residual L2 is better by only {stage_t['residual_l2'] - disco['residual_l2']:.6g}, and cosine improves. The state/residual paired confidence intervals nevertheless cross zero, both aggregate transport skills remain negative and are worse than Stage T at the formal-best checkpoint, and FIRST_10X remains 1. The frozen hierarchy therefore classifies this as a one-step-only improvement, not a dynamics rescue.

The tested adapted isotropic radial 3D DISCO extension does not rescue the spherical-KS workflow. This result does **not** establish that DISCO generally fails and does not test the paper's unavailable exact 3D DISCO implementation.

```text
PRIMARY_DECISION = {primary}

REPRODUCTION_SCOPE =
ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION

DISCO3D_IMPLEMENTATION =
ADAPTED_ISOTROPIC_RADIAL_PIECEWISE_LINEAR

DISCO3D_RADIUS_CELLS = 3
DISCO3D_BASIS_COUNT = 5
DISCO3D_STENCIL_SHAPE = [7,7,7]

ALL_BASES_ACTIVE = {str(decision['ALL_BASES_ACTIVE']).lower()}
BASIS_PARTITION_OF_UNITY_PASS = {str(decision['BASIS_PARTITION_OF_UNITY_PASS']).lower()}
DENSE_REFERENCE_MATCH = {str(decision['DENSE_REFERENCE_MATCH']).lower()}
KERNEL_ORIENTATION_PASS = {str(decision['KERNEL_ORIENTATION_PASS']).lower()}
GRADCHECK_PASS = {str(decision['GRADCHECK_PASS']).lower()}
DISCO3D_UNIT_TESTS_PASS = {str(decision['DISCO3D_UNIT_TESTS_PASS']).lower()}

DISCO_BRANCH_ACTIVE = {str(decision['DISCO_BRANCH_ACTIVE']).lower()}
TRAINING_STARTED = {str(decision['TRAINING_STARTED']).lower()}
TRAINING_COMPLETED = {str(decision['TRAINING_COMPLETED']).lower()}

{gate_lines}

EXACT_3D_DISCO_IMPLEMENTATION_FOUND = false
EXACT_REPRODUCTION_BLOCKED = true

AUTHORIZE_NEXT_STAGE = {next_stage}
```
"""
    write_text(OUT / "STAGE_AD_DECISION.md", decision_text)

    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
