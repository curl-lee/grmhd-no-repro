#!/usr/bin/env python
"""Compare Stage I no-H1 with the frozen Stage G Full and Plain pilots."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("MPLCONFIGDIR", "/tmp/grmhd-matplotlib")

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from compare_paper_stage_g import (
    GT_STEPS,
    NO_GT_STEPS,
    load_selected,
    model_only_saturation,
    state_category_scores,
    table_row,
)
from grmhd import CHANNELS
from grmhd.paper_config import load_paper_experiment_config
from grmhd.paper_data_processor import PaperDataProcessor


def mean_csv(path: Path, field: str) -> float | None:
    values = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = row.get(field)
            if value not in (None, ""):
                values.append(float(value))
    return float(np.mean(values)) if values else None


def effective_gradient_summary(path: Path) -> dict[str, float | None]:
    base = []
    other = []
    clip = []
    update = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            total = float(row["total_gradient_norm_before_clip"])
            scale = float(row.get("clip_scale") or min(1.0, 1.0 / max(total, 1e-30)))
            clip.append(scale)
            update.append(float(row["parameter_update_norm"]))
            if row.get("base_only_gradient_norm") not in (None, ""):
                base.append(float(row["base_only_gradient_norm"]) * scale)
            if row.get("other_prior_gradient_norm") not in (None, ""):
                other.append(float(row["other_prior_gradient_norm"]) * scale)
    return {
        "effective_base_gradient_norm_mean": (
            float(np.mean(base)) if base else None
        ),
        "effective_other_prior_gradient_norm_mean": (
            float(np.mean(other)) if other else None
        ),
        "clip_scale_mean": float(np.mean(clip)),
        "parameter_update_norm_mean": float(np.mean(update)),
    }


def selected_rollout(payload: Mapping[str, Any], step: int) -> Mapping[str, Any]:
    section = "gt_rollout" if step <= 19 else "no_gt_rollout"
    return payload[section]["selected"][str(step)]


def render_morphology(
    *,
    output_dir: Path,
    raw_targets: Mapping[int, torch.Tensor],
    oracle_targets: Mapping[int, torch.Tensor],
    persistence: torch.Tensor,
    states: Mapping[str, Mapping[int, torch.Tensor]],
    phi: np.ndarray,
    theta: np.ndarray,
    r: np.ndarray,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    fixed_phi_index = len(phi) // 2
    equatorial_theta_index = int(np.argmin(np.abs(theta - np.pi / 2)))
    for step in (*GT_STEPS, *NO_GT_STEPS):
        panels = (
            [
                ("raw truth", raw_targets[step]),
                ("canonical oracle", oracle_targets[step]),
                ("persistence", persistence),
            ]
            if step in GT_STEPS
            else [("persistence", persistence)]
        )
        panels.extend(
            [
                ("Stage G Full", states["full"][step]),
                ("Stage G Plain", states["plain"][step]),
                ("Stage I no-H1", states["no_h1"][step]),
            ]
        )
        for slice_name in ("fixed_phi_theta_r", "equatorial_phi_r"):
            figure, axes = plt.subplots(
                len(CHANNELS),
                len(panels),
                figsize=(4.1 * len(panels), 2.8 * len(CHANNELS)),
                constrained_layout=True,
            )
            for channel, channel_name in enumerate(CHANNELS):
                slices = []
                for _, state in panels:
                    values = state[0, channel].detach().cpu().numpy()
                    slices.append(
                        values[fixed_phi_index]
                        if slice_name == "fixed_phi_theta_r"
                        else values[:, equatorial_theta_index, :]
                    )
                combined = np.concatenate([value.reshape(-1) for value in slices])
                if channel_name in {"rho", "press"}:
                    vmin, vmax = np.quantile(combined, [0.01, 0.99])
                    cmap = "viridis"
                else:
                    maximum = float(np.quantile(np.abs(combined), 0.99))
                    vmin, vmax = -maximum, maximum
                    cmap = "coolwarm"
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                    vmin = float(combined.min())
                    vmax = float(combined.max() + 1e-12)
                for column, ((label, _), values) in enumerate(zip(panels, slices)):
                    axis = axes[channel, column]
                    extent = (
                        [
                            float(r.min()),
                            float(r.max()),
                            float(theta.min()),
                            float(theta.max()),
                        ]
                        if slice_name == "fixed_phi_theta_r"
                        else [
                            float(r.min()),
                            float(r.max()),
                            float(phi.min()),
                            float(phi.max()),
                        ]
                    )
                    image = axis.imshow(
                        values,
                        origin="lower",
                        aspect="auto",
                        extent=extent,
                        cmap=cmap,
                        vmin=vmin,
                        vmax=vmax,
                    )
                    if channel == 0:
                        axis.set_title(label)
                    if column == 0:
                        axis.set_ylabel(
                            f"{channel_name}\n"
                            + (
                                "theta"
                                if slice_name == "fixed_phi_theta_r"
                                else "phi"
                            )
                        )
                    if channel == len(CHANNELS) - 1:
                        axis.set_xlabel("r")
                    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
            availability = (
                "raw/oracle shown; preprocessing floor and clamp masks reported separately"
                if step in GT_STEPS
                else "no ground truth after step 19; raw/oracle intentionally omitted"
            )
            figure.suptitle(
                f"Stage I no-H1 comparison, step {step}: "
                f"{slice_name.replace('_', ' ')}\n{availability}; "
                "spherical Kerr-Schild stored components, not Cartesian slices",
                fontsize=12,
            )
            path = output_dir / f"no_h1_step_{step:03d}_{slice_name}.png"
            figure.savefig(path, dpi=120)
            plt.close(figure)
            files.append(str(path))
    return files


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    stage_g = root / "outputs/paper_reduced100/stage_g"
    stage_i = root / "outputs/paper_reduced100/stage_i"
    no_h1_dir = stage_i / "no_h1"
    dirs = {
        "full": stage_g / "pilot30_full_fno",
        "plain": stage_g / "pilot30_plain_l2",
        "no_h1": no_h1_dir,
    }
    metrics = {
        name: json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        for name, path in dirs.items()
    }
    evaluations = {
        name: json.loads(
            (path / "evaluation_summary.json").read_text(encoding="utf-8")
        )
        for name, path in dirs.items()
    }
    gt = {
        name: json.loads((path / "gt_rollout.json").read_text(encoding="utf-8"))
        for name, path in dirs.items()
    }
    no_gt = {
        name: json.loads(
            (path / "no_gt_rollout.json").read_text(encoding="utf-8")
        )
        for name, path in dirs.items()
    }
    states = {
        name: load_selected(path / "selected_states.pt")
        for name, path in dirs.items()
    }

    config = load_paper_experiment_config(
        root / "configs/paper_reduced100/full_fno_proxy.yaml",
        project_root=root,
    )
    processor = PaperDataProcessor.from_config(config)
    dataset_path = config.resolve_path(config.values["protocol"]["dataset"])
    raw_targets = {}
    oracle_targets = {}
    with h5py.File(dataset_path, "r") as handle:
        phi = np.asarray(handle["coords/phi"][...], dtype=np.float64)
        theta = np.asarray(handle["coords/theta"][...], dtype=np.float64)
        r = np.asarray(handle["coords/r"][...], dtype=np.float64)
        initial = torch.from_numpy(
            np.asarray(handle["snapshots"][91], dtype=np.float32)
        ).unsqueeze(0)
        for step in GT_STEPS:
            raw = torch.from_numpy(
                np.asarray(handle["snapshots"][91 + step], dtype=np.float32)
            ).unsqueeze(0)
            raw_targets[step] = raw
            oracle_targets[step] = processor.preprocessor.decode(
                processor.preprocessor.encode(raw, channel_axis=1),
                channel_axis=1,
            )
    persistence = processor.preprocessor.decode(
        processor.preprocessor.encode(initial, channel_axis=1),
        channel_axis=1,
    )

    category_records: dict[str, list[dict[str, float]]] = {
        name: [] for name in dirs
    }
    for name in dirs:
        for step in GT_STEPS:
            scores = state_category_scores(
                prediction=states[name][step],
                oracle=oracle_targets[step],
                r=r,
            )
            scores["step"] = float(step)
            scores["model_only_saturation"] = model_only_saturation(
                gt[name]["records"][step - 1]
            )
            category_records[name].append(scores)
    categories: dict[str, dict[str, Any]] = {}
    for category in (
        "center_morphology",
        "polar_morphology",
        "magnetic_texture",
        "radial_statistics",
        "outer_shell_statistics",
        "model_only_saturation",
    ):
        values = {
            name: float(
                np.mean([record[category] for record in category_records[name]])
            )
            for name in dirs
        }
        values["best"] = min(dirs, key=lambda name: values[name])
        categories[category] = values
    artifact_values = {
        name: int(
            sum(
                len(no_gt[name]["records"][step - 20]["artifacts"]["flags"])
                for step in NO_GT_STEPS
            )
        )
        for name in dirs
    }
    categories["step50_100_artifacts"] = {
        **artifact_values,
        "best": min(dirs, key=lambda name: artifact_values[name]),
    }

    validation_metrics = {
        name: evaluations[name]["best"]["validation"]["model"]["metrics"][
            "E_norm"
        ]
        for name in dirs
    }
    persistence_metric = evaluations["full"]["best"]["validation"][
        "persistence"
    ]["metrics"]["E_norm"]
    table = [
        table_row("persistence", persistence_metric),
        table_row("stage_g_full", validation_metrics["full"]),
        table_row("stage_g_plain", validation_metrics["plain"]),
        table_row("stage_i_no_h1", validation_metrics["no_h1"]),
    ]
    csv_path = stage_i / "no_h1_comparison.csv"
    fields = [
        "model",
        *CHANNELS,
        "arithmetic_average",
        "global_normalized_relative_l2",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)

    gradient_scale = {
        name: effective_gradient_summary(dirs[name] / "optimizer_log.csv")
        for name in dirs
    }
    train_component_means = {
        field: mean_csv(no_h1_dir / "train_log.csv", field)
        for field in (
            "base_fidelity_weighted",
            "other_prior_weighted",
            "loss.components.roi_weighted",
            "loss.components.bounds_weighted",
            "loss.components.envelope_weighted",
            "loss.components.dissipation_weighted",
            "diagnostic_current_upstream_h1_raw",
        )
    }
    clipping = {
        name: float(metrics[name]["runtime"]["clipping_fraction"])
        for name in dirs
    }
    gt_comparison = {
        str(step): {
            name: {
                "normalized_average": selected_rollout(
                    evaluations[name], step
                )["E_norm"],
                "prediction_global_norm": selected_rollout(
                    evaluations[name], step
                )["prediction_global_norm"],
                "artifact_flag_count": selected_rollout(
                    evaluations[name], step
                )["artifact_flag_count"],
            }
            for name in dirs
        }
        for step in GT_STEPS
    }
    no_gt_comparison = {
        str(step): {
            name: {
                "prediction_global_norm": selected_rollout(
                    evaluations[name], step
                )["prediction_global_norm"],
                "artifact_flag_count": selected_rollout(
                    evaluations[name], step
                )["artifact_flag_count"],
                "finite": selected_rollout(evaluations[name], step)["finite"],
                "rho_press_positive": selected_rollout(
                    evaluations[name], step
                )["rho_press_positive"],
            }
            for name in dirs
        }
        for step in NO_GT_STEPS
    }
    no_h1_better_than_full = [
        name
        for name, values in categories.items()
        if values["no_h1"] < values["full"]
    ]
    no_h1_at_least_plain = [
        name
        for name, values in categories.items()
        if values["no_h1"] <= values["plain"]
    ]
    result = {
        "schema_version": "paper-stage-i-no-h1-comparison-v1",
        "status": "completed",
        "classification": {
            "reproduction_level": "diagnostic_extension",
            "paper_faithful_full": False,
            "comparison_parent": "stage_g_paper_adapted_full",
            "no_h1_is_plain_l2": False,
            "unit_index_training_run": False,
            "stored_coordinate_training_run": False,
        },
        "table_2_style_validation": table,
        "oracle_aware": {
            name: {
                key: evaluations[name]["best"]["validation"]["model"]["metrics"][
                    key
                ]
                for key in (
                    "E_norm",
                    "E_model_oracle",
                    "E_model_raw",
                    "E_oracle_raw",
                )
            }
            for name in dirs
        },
        "training": {
            "clipping_fraction": clipping,
            "gradient_scale": gradient_scale,
            "no_h1_component_means": train_component_means,
            "best_epoch": {
                name: metrics[name]["best_epoch"] for name in dirs
            },
            "runtime_seconds": {
                name: metrics[name]["runtime"]["wall_seconds"] for name in dirs
            },
        },
        "gt_rollout": gt_comparison,
        "no_gt_rollout": no_gt_comparison,
        "morphology_and_boundary_categories": categories,
        "no_h1_better_than_stage_g_full_categories": no_h1_better_than_full,
        "no_h1_at_least_stage_g_plain_categories": no_h1_at_least_plain,
        "questions": {
            "clipping_reduced": clipping["no_h1"] < clipping["full"],
            "effective_base_gradient_scale": gradient_scale,
            "validation_one_step_improved_vs_full": (
                validation_metrics["no_h1"]["arithmetic_average"]
                < validation_metrics["full"]["arithmetic_average"]
            ),
            "gt_steps_improved_vs_full": {
                str(step): (
                    gt_comparison[str(step)]["no_h1"]["normalized_average"]
                    < gt_comparison[str(step)]["full"]["normalized_average"]
                )
                for step in GT_STEPS
            },
            "step50_100_norm_lower_than_full": {
                str(step): (
                    no_gt_comparison[str(step)]["no_h1"][
                        "prediction_global_norm"
                    ]
                    < no_gt_comparison[str(step)]["full"][
                        "prediction_global_norm"
                    ]
                )
                for step in NO_GT_STEPS
            },
            "morphology_improved_categories": no_h1_better_than_full,
            "radial_or_outer_improved": any(
                name in no_h1_better_than_full
                for name in ("radial_statistics", "outer_shell_statistics")
            ),
            "artifact_flags_reduced": (
                artifact_values["no_h1"] < artifact_values["full"]
            ),
            "no_h1_close_to_or_better_than_plain": {
                "validation_ratio": (
                    validation_metrics["no_h1"]["arithmetic_average"]
                    / validation_metrics["plain"]["arithmetic_average"]
                ),
                "categories_at_least_plain": no_h1_at_least_plain,
            },
            "other_full_priors_without_h1": train_component_means,
        },
    }
    figures = render_morphology(
        output_dir=stage_i / "figures",
        raw_targets=raw_targets,
        oracle_targets=oracle_targets,
        persistence=persistence,
        states=states,
        phi=phi,
        theta=theta,
        r=r,
    )
    result["morphology_figures"] = figures
    (stage_i / "no_h1_comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Stage I Run A no-H1 comparison",
        "",
        "This compares the frozen Stage G paper-adapted Full and Plain pilots "
        "with the matched Stage I no-H1 diagnostic extension. no-H1 retains "
        "ROI, bounds, envelope, dissipation, channel weights, preprocessing, "
        "optimizer, scheduler, initial state, and pair order; it is not Plain L2.",
        "",
        "## Validation",
        "",
        "| model | average | global |",
        "|---|---:|---:|",
    ]
    for row in table:
        lines.append(
            f"| {row['model']} | {row['arithmetic_average']:.6g} | "
            f"{row['global_normalized_relative_l2']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Morphology and boundary",
            "",
            "| category | Full | Plain | no-H1 | best |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for name, values in categories.items():
        lines.append(
            f"| {name} | {values['full']:.6g} | {values['plain']:.6g} | "
            f"{values['no_h1']:.6g} | {values['best']} |"
        )
    lines.extend(
        [
            "",
            f"- Full clipping: `{clipping['full']:.6g}`",
            f"- no-H1 clipping: `{clipping['no_h1']:.6g}`",
            "- Unit-index/stored-coordinate training executed: `false/false`",
        ]
    )
    (stage_i / "no_h1_comparison.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
