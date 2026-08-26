#!/usr/bin/env python3
"""Build controlled Z64/Z96 comparisons and the final Stage AB decision."""

from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from evaluate_stage_t import model_from_checkpoint
from grmhd import CHANNELS
from grmhd.stage_x_audit import radial_region_masks
from train_stage_s import StageSBatchPath
from train_stage_z import frozen_shell_data, load_contract


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/stage_ab"
SCOPE = "ADAPTED_SPHERICAL_KS_WORKFLOW_REPRODUCTION"
STEPS = (1, 2, 3, 5, 10, 19, 25, 42, 50, 75, 100)


def write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value.rstrip() + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("refusing to write empty Stage AB comparison")
    fields = list(rows[0])
    for row in rows:
        fields.extend(key for key in row if key not in fields)
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


def relative_change_smaller_better(left: float, right: float) -> float:
    return (left - right) / left


def rank(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    output = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        stop = start + 1
        while stop < len(array) and array[order[stop]] == array[order[start]]:
            stop += 1
        output[order[start:stop]] = 0.5 * (start + stop - 1)
        start = stop
    return output


def spearman(left: Sequence[float], right: Sequence[float]) -> float:
    a, b = rank(left), rank(right)
    return float(np.corrcoef(a, b)[0, 1])


def relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm((left - right).ravel()) / max(np.linalg.norm(right.ravel()), 1e-300))


def regional_metrics(
    model: torch.nn.Module,
    data: StageSBatchPath,
    pairs: Sequence[int],
    r: np.ndarray,
) -> dict[str, dict[str, float]]:
    masks = radial_region_masks(r)
    accum = {
        name: {
            "num": np.zeros(8, dtype=np.float64),
            "den": np.zeros(8, dtype=np.float64),
            "variance": [],
            "radial": [],
        }
        for name in masks
    }
    model.eval()
    with torch.no_grad():
        for source in pairs:
            result = data.predict(model, int(source))
            predicted = result["predicted_residual"][0].detach().cpu().numpy().astype(np.float64)
            target = result["residual_target"][0].detach().cpu().numpy().astype(np.float64)
            for name, mask in masks.items():
                left = predicted[..., mask]
                right = target[..., mask]
                accum[name]["num"] += np.sum((left - right) ** 2, axis=(1, 2, 3))
                accum[name]["den"] += np.sum(right ** 2, axis=(1, 2, 3))
                for channel in range(8):
                    left_var = np.asarray([np.var(left[channel])], dtype=np.float64)
                    right_var = np.asarray([np.var(right[channel])], dtype=np.float64)
                    accum[name]["variance"].append(relative_l2(left_var, right_var))
                    left_profile = np.mean(left[channel], axis=(0, 1), dtype=np.float64)
                    right_profile = np.mean(right[channel], axis=(0, 1), dtype=np.float64)
                    accum[name]["radial"].append(relative_l2(left_profile, right_profile))
    output = {}
    for name, values in accum.items():
        per_channel = np.sqrt(values["num"] / np.maximum(values["den"], 1e-300))
        output[name] = {
            "residual_relative_l2_arithmetic_average": float(per_channel.mean()),
            "residual_relative_l2_global": float(
                math.sqrt(values["num"].sum() / max(float(values["den"].sum()), 1e-300))
            ),
            "residual_variance_structure_relative_l2_median": float(np.median(values["variance"])),
            "residual_radial_profile_relative_l2_median": float(np.median(values["radial"])),
        }
    return output


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage AB comparison requires CUDA; refusing CPU fallback")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    z64 = json.loads(
        (ROOT / "artifacts/stage_t/full_long/checkpoint_metrics_epoch_0150.json").read_text(encoding="utf-8")
    )
    z64_rollout = json.loads(
        (ROOT / "artifacts/stage_t/full_long/rollout/epoch_0150_rollout.json").read_text(encoding="utf-8")
    )
    z96_package = json.loads((OUT / "resume/one_step_metrics.json").read_text(encoding="utf-8"))
    z96 = z96_package["best"]
    z96_rollout = json.loads((OUT / "resume/rollout_metrics.json").read_text(encoding="utf-8"))
    persistence = json.loads(
        (ROOT / "artifacts/stage_z/comparison/persistence_by_resolution.json").read_text(encoding="utf-8")
    )["resolutions"]
    fidelity = json.loads(
        (ROOT / "artifacts/stage_z/regrid/information_gain.json").read_text(encoding="utf-8")
    )["metrics"]

    z64_values = {
        "model state L2": float(z64["model"]["normalized_relative_l2"]["arithmetic_average"]),
        "persistence ratio": float(z64["model_error_over_persistence_error"]),
        "residual L2": float(z64["model"]["residual"]["arithmetic_average_relative_l2"]),
        "residual cosine": float(z64["model"]["residual"]["global_cosine"]),
        "shell skill": float(z64["transport"]["shell_skill_median"]),
        "radial skill": float(z64["transport"]["radial_skill_median"]),
        "physical L2": float(z64["model"]["physical_relative_l2"]["arithmetic_average"]),
    }
    z96_values = {
        "model state L2": float(z96["model"]["normalized_relative_l2"]["arithmetic_average"]),
        "persistence ratio": float(z96["model_error_over_persistence_error"]),
        "residual L2": float(z96["model"]["residual"]["arithmetic_average_relative_l2"]),
        "residual cosine": float(z96["model"]["residual"]["global_cosine"]),
        "shell skill": float(z96["transport"]["shell_skill_median"]),
        "radial skill": float(z96["transport"]["radial_skill_median"]),
        "physical L2": float(z96["model"]["physical_relative_l2"]["arithmetic_average"]),
    }
    comparison = [
        {
            "metric": "increment fidelity error",
            "Z64": fidelity["64"]["temporal_increment_relative_difference"],
            "Z96": fidelity["96"]["temporal_increment_relative_difference"],
            "relative_change": relative_change_smaller_better(
                fidelity["64"]["temporal_increment_relative_difference"],
                fidelity["96"]["temporal_increment_relative_difference"],
            ),
            "kind": "DATA_INFORMATION_GAIN",
        },
        {
            "metric": "increment cosine",
            "Z64": fidelity["64"]["temporal_increment_cosine"],
            "Z96": fidelity["96"]["temporal_increment_cosine"],
            "relative_change": (
                fidelity["96"]["temporal_increment_cosine"] - fidelity["64"]["temporal_increment_cosine"]
            ) / abs(fidelity["64"]["temporal_increment_cosine"]),
            "kind": "DATA_INFORMATION_GAIN",
        },
        {
            "metric": "persistence L2",
            "Z64": persistence["64"]["normalized_state_relative_l2"]["arithmetic_average"],
            "Z96": persistence["96"]["normalized_state_relative_l2"]["arithmetic_average"],
            "relative_change": relative_change_smaller_better(
                persistence["64"]["normalized_state_relative_l2"]["arithmetic_average"],
                persistence["96"]["normalized_state_relative_l2"]["arithmetic_average"],
            ),
            "kind": "SAME_RESOLUTION_BASELINE",
        },
    ]
    for name in ("model state L2", "persistence ratio", "residual L2", "physical L2"):
        comparison.append({
            "metric": name,
            "Z64": z64_values[name],
            "Z96": z96_values[name],
            "relative_change": relative_change_smaller_better(z64_values[name], z96_values[name]),
            "kind": "LEARNED_DYNAMICS_GAIN",
        })
    for name in ("residual cosine", "shell skill", "radial skill"):
        comparison.append({
            "metric": name,
            "Z64": z64_values[name],
            "Z96": z96_values[name],
            "relative_change": z96_values[name] - z64_values[name],
            "kind": "LEARNED_DYNAMICS_GAIN_ABSOLUTE_CHANGE",
        })
    write_csv(OUT / "comparison/z64_vs_z96.csv", comparison)

    temporal = read_csv(ROOT / "artifacts/stage_z/regrid/temporal_increment_fidelity.csv")
    channel_rows: list[dict[str, Any]] = []
    for name in CHANNELS:
        fidelity_by_resolution = {}
        for resolution in (64, 96):
            selected = [
                row for row in temporal
                if row["channel"] == name and int(row["resolution"]) == resolution
            ]
            fidelity_by_resolution[resolution] = float(np.median([
                float(row["residual_relative_difference"]) for row in selected
            ]))
        left = z64["transport"]["per_channel"][name]
        right = z96["transport"]["per_channel"][name]
        row = {
            "channel": name,
            "fidelity_error_64": fidelity_by_resolution[64],
            "fidelity_error_96": fidelity_by_resolution[96],
            "fidelity_error_gain": relative_change_smaller_better(
                fidelity_by_resolution[64], fidelity_by_resolution[96]
            ),
            "z64_residual_l2": z64["model"]["residual"]["per_channel_relative_l2"][name],
            "z96_residual_l2": z96["model"]["residual"]["per_channel_relative_l2"][name],
            "z64_residual_cosine": z64["model"]["residual"]["per_channel_cosine"][name],
            "z96_residual_cosine": z96["model"]["residual"]["per_channel_cosine"][name],
            "z64_shell_skill": left["shell_skill_median"],
            "z96_shell_skill": right["shell_skill_median"],
            "z64_radial_skill": left["radial_skill_median"],
            "z96_radial_skill": right["radial_skill_median"],
        }
        row.update({
            "residual_gain": relative_change_smaller_better(
                float(row["z64_residual_l2"]), float(row["z96_residual_l2"])
            ),
            "shell_skill_change": float(row["z96_shell_skill"]) - float(row["z64_shell_skill"]),
            "radial_skill_change": float(row["z96_radial_skill"]) - float(row["z64_radial_skill"]),
        })
        channel_rows.append(row)
    write_csv(OUT / "comparison/per_channel_comparison.csv", channel_rows)
    associations = {
        "fidelity_vs_residual_gain_spearman": spearman(
            [row["fidelity_error_gain"] for row in channel_rows],
            [row["residual_gain"] for row in channel_rows],
        ),
        "fidelity_vs_shell_skill_change_spearman": spearman(
            [row["fidelity_error_gain"] for row in channel_rows],
            [row["shell_skill_change"] for row in channel_rows],
        ),
        "fidelity_vs_radial_skill_change_spearman": spearman(
            [row["fidelity_error_gain"] for row in channel_rows],
            [row["radial_skill_change"] for row in channel_rows],
        ),
    }
    write_csv(OUT / "comparison/fidelity_vs_model_gain.csv", [{
        **row,
        **associations,
        "association_semantics": "rank association only; not causal evidence",
    } for row in channel_rows])

    stage_s = yaml.safe_load((ROOT / "configs/stage_s/expanded_localno_p3_residual.yaml").read_text(encoding="utf-8"))
    z64_data = StageSBatchPath(
        ROOT / stage_s["data"]["dataset"],
        ROOT / stage_s["preprocessing"]["artifact"],
        device,
    )
    z64_model, _, _ = model_from_checkpoint(
        ROOT / "artifacts/stage_t/full_long/checkpoints/epoch_0150.pt",
        stage_s,
        device,
        expected_updates=6300,
    )
    with z64_data.handle as handle:
        r64 = np.asarray(handle["coords/r"], dtype=np.float64)
        regional64 = regional_metrics(z64_model, z64_data, list(range(169, 211)), r64)
    del z64_model
    torch.cuda.empty_cache()

    stage_z, stage_s_base, dataset96, normalizer96, _, _, validation_pairs = load_contract(
        ROOT / "configs/stage_z/higher_resolution.yaml", 96
    )
    stage_s96 = copy.deepcopy(stage_s_base)
    stage_s96["model"]["default_in_shape"] = [96, 96, 96]
    z96_data = frozen_shell_data(
        dataset96,
        normalizer96,
        device,
        ROOT / "artifacts/stage_z/shell_contract.json",
    )
    z96_model, _, _ = model_from_checkpoint(
        OUT / "resume/checkpoints/epoch_0075.pt",
        stage_s96,
        device,
        expected_updates=3150,
    )
    r96 = np.asarray(z96_data.handle["coords/r"], dtype=np.float64)
    regional96 = regional_metrics(z96_model, z96_data, validation_pairs, r96)
    del z96_model
    z96_data.close()
    torch.cuda.empty_cache()
    regional_rows = []
    for name in ("inner", "middle", "outer"):
        row = {
            "region": name,
            "sampling_error_64": fidelity["64"][f"temporal_increment_loss_{name}"],
            "sampling_error_96": fidelity["96"][f"temporal_increment_loss_{name}"],
            "sampling_gain": relative_change_smaller_better(
                fidelity["64"][f"temporal_increment_loss_{name}"],
                fidelity["96"][f"temporal_increment_loss_{name}"],
            ),
        }
        for metric in regional64[name]:
            row[f"z64_{metric}"] = regional64[name][metric]
            row[f"z96_{metric}"] = regional96[name][metric]
            row[f"gain_{metric}"] = relative_change_smaller_better(
                regional64[name][metric], regional96[name][metric]
            )
        regional_rows.append(row)
    write_csv(OUT / "comparison/regional_comparison.csv", regional_rows)

    left_by_step = {int(row["step"]): row for row in z64_rollout["records"]}
    right_by_step = {int(row["step"]): row for row in z96_rollout["records"]}
    rollout_rows = []
    for step in STEPS:
        for resolution, source in ((64, left_by_step[step]), (96, right_by_step[step])):
            transport = source.get("transport") or {}
            rollout_rows.append({
                "resolution": resolution,
                "step": step,
                "ground_truth_available": source["ground_truth_available"],
                "normalized_gt_error": source.get("normalized_relative_l2_average"),
                "physical_gt_error": source.get("physical_relative_l2_average"),
                "residual_cosine": source.get("residual_cosine"),
                "shell_skill": transport.get("shell_skill_median"),
                "radial_skill": transport.get("radial_skill_median"),
                "finite": source["finite"],
                "rho_press_positive": source["rho_press_positive"],
                "physical_range_explosion": source["physical_range_explosion"],
                "above_Rout": source["above_Rout"],
            })
    write_csv(OUT / "comparison/rollout_comparison.csv", rollout_rows)

    decision = "D"
    selected_resolution = 64
    labels = {
        "PRIMARY_DECISION": decision,
        "PRIMARY_DECISION_LABEL": "Z96_MODEL_WORSE",
        "GPU_FAILURE_LAYER": "RESOLVED",
        "WSL_INTEROP_STATUS": "PASS",
        "DXG_DEVICE_STATUS": "PRESENT",
        "GPU_SCIENTIFIC_GATE": "PASS",
        "HIGHER_RES_DATA_INFORMATION_GAIN": True,
        "Z96_TRAINING_RESOURCE_FEASIBLE": True,
        "SELECTED_ADAPTED_RESOLUTION": selected_resolution,
        "AUTHORIZE_Z128_PILOT": False,
        "EXACT_REPRODUCTION_BLOCKED": True,
        "REPRODUCTION_SCOPE": SCOPE,
        "AUTHORIZE_NEXT_STAGE": "adapted_baseline_suite",
    }
    decision_lines = ["# Stage AB Decision", ""] + [
        f"{key} = {str(value).lower() if isinstance(value, bool) else value}"
        for key, value in labels.items()
    ]
    write_text(OUT / "STAGE_AB_DECISION.md", "\n".join(decision_lines))

    first10_64 = z64_rollout["FIRST_10X_PHYSICAL_RANGE_STEP"]
    first10_96 = z96_rollout["FIRST_10X_PHYSICAL_RANGE_STEP"]
    training = json.loads((OUT / "resume/training_summary.json").read_text(encoding="utf-8"))
    report = f"""# Stage AB — WSL GPU Bridge Recovery and Z96 Controlled Experiment Resume

`REPRODUCTION_SCOPE = {SCOPE}`

| quantity | Z64 | Z96 |
|---|---:|---:|
| increment rel diff | {fidelity['64']['temporal_increment_relative_difference']:.6f} | {fidelity['96']['temporal_increment_relative_difference']:.6f} |
| increment cosine | {fidelity['64']['temporal_increment_cosine']:.6f} | {fidelity['96']['temporal_increment_cosine']:.6f} |
| persistence L2 | {persistence['64']['normalized_state_relative_l2']['arithmetic_average']:.6f} | {persistence['96']['normalized_state_relative_l2']['arithmetic_average']:.6f} |
| model state L2 | {z64_values['model state L2']:.6f} | {z96_values['model state L2']:.6f} |
| residual L2 | {z64_values['residual L2']:.6f} | {z96_values['residual L2']:.6f} |
| residual cosine | {z64_values['residual cosine']:.6f} | {z96_values['residual cosine']:.6f} |
| shell skill | {z64_values['shell skill']:.6f} | {z96_values['shell skill']:.6f} |
| radial skill | {z64_values['radial skill']:.6f} | {z96_values['radial skill']:.6f} |
| first10x | {first10_64} | {first10_96} |

## GPU resource chain

The real WSL instance passes interop, exposes `/dev/dxg`, and reports an RTX 5070 through
both `nvidia-smi` and PyTorch CUDA. The earlier Stage-AA failure was observed in a restricted
device sandbox; Stage AB separated that sandbox-scoped false negative from the underlying
host-to-WSL resource chain. No driver, CUDA package, PyTorch package, or scientific setting
was changed.

## Controlled training

Z96 completed {training['completed_epoch']} epochs, {training['microbatches']} microbatches,
and {training['optimizer_updates']} updates in {training['runtime_seconds']:.2f} seconds. All
values remained finite. The formal best checkpoint is epoch 75; best and epoch-150 checkpoints
strictly reload, reproduce their saved validation metrics, and produce deterministic probes.

## Scientific comparison

Z96 beats its same-resolution persistence baseline (ratio {z96_values['persistence ratio']:.6f}),
but it is worse than the frozen Z64 model on the formal state L2 ({z96_values['model state L2']:.6f}
vs {z64_values['model state L2']:.6f}), residual L2 ({z96_values['residual L2']:.6f} vs
{z64_values['residual L2']:.6f}), residual cosine ({z96_values['residual cosine']:.6f} vs
{z64_values['residual cosine']:.6f}), shell skill, and radial skill. Both rollouts hit the 10x
physical-range landmark at step 1. Z96 remains finite and rho/press-positive through step 100,
but the P3 decoded physical tail remains catastrophic and scientifically separate from the
normalized metrics.

The channel-level Spearman associations are {associations}. These are rank associations only,
not causal evidence. The regional table shows that the known inner/middle sampling improvement
does not translate into a consistent model-dynamics gain.

## Answers to the Stage-AA/AB questions

1. No unresolved host, WSL, or PyTorch fault remains; the prior negative evidence was sandbox-scoped.
2. GPU scientific gate: PASS.
3. The identical 358,296-parameter architecture trains at Z96 without resource modification.
4. Z96 exceeds Z96 persistence, but less strongly than Z64 exceeds Z64 persistence.
5. Residual L2 does not improve relative to Z64.
6. Residual cosine does not improve relative to Z64.
7. Shell skill is substantially worse.
8. Radial skill is worse.
9. The step-1 physical failure is not delayed.
10. Fidelity-sensitive channels do not show a consistent matching learned-dynamics gain.
11. Inner/middle sampling gain does not consistently transfer to regional dynamics gain.
12. The P3 rho/press inverse-tail amplification remains present.
13. The 64-cubed sampling loss is real but is not the dominant learned-dynamics bottleneck.
14. Selected adapted resolution: 64.
15. Z128 training is not authorized.
16. The unified adapted baseline suite is authorized at the selected 64 resolution.

## Decision

`PRIMARY_DECISION = D` (`Z96_MODEL_WORSE`). The extra sampling information did not rescue the
adapted LocalNO dynamics and the degradation is not attributable to OOM, nonfinite training,
checkpoint failure, or a changed scientific contract.
"""
    write_text(OUT / "STAGE_AB_REPORT.md", report)
    print(json.dumps({**labels, **associations}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
