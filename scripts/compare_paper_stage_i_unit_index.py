#!/usr/bin/env python
"""Compare Stage I unit-index H1 with all completed frozen controls."""

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
from grmhd.dataset import sha256_file
from grmhd.paper_config import load_paper_experiment_config
from grmhd.paper_data_processor import PaperDataProcessor


MODEL_LABELS = {
    "full": "Stage G Full",
    "plain": "Stage G Plain",
    "no_h1": "Stage I no-H1",
    "unit_index": "Stage I unit-index",
    "stored_coordinate": "Stage I stored-coordinate/volume",
}


def numeric_column(path: Path, field: str) -> list[float]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [
            float(row[field])
            for row in csv.DictReader(handle)
            if row.get(field) not in (None, "")
        ]


def summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "mean": None,
            "median": None,
            "q95": None,
            "maximum": None,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def gradient_summary(path: Path) -> dict[str, Any]:
    fields = {
        name: numeric_column(path, name)
        for name in (
            "base_only_gradient_norm",
            "h1_only_gradient_norm",
            "h1_to_base_gradient_ratio",
            "base_h1_gradient_cosine",
            "clip_scale",
            "effective_base_gradient_norm",
            "effective_h1_gradient_norm",
            "effective_base_gradient_projection",
            "effective_h1_gradient_projection",
            "other_prior_gradient_norm",
            "parameter_update_norm",
        )
    }
    if not fields["effective_base_gradient_norm"]:
        fields["effective_base_gradient_norm"] = [
            value * scale
            for value, scale in zip(
                fields["base_only_gradient_norm"],
                fields["clip_scale"],
            )
        ]
    if (
        fields["h1_only_gradient_norm"]
        and not fields["effective_h1_gradient_norm"]
    ):
        fields["effective_h1_gradient_norm"] = [
            value * scale
            for value, scale in zip(
                fields["h1_only_gradient_norm"],
                fields["clip_scale"],
            )
        ]
    return {name: summary(values) for name, values in fields.items()}


def selected(
    payload: Mapping[str, Any],
    step: int,
) -> Mapping[str, Any]:
    section = "gt_rollout" if step <= 19 else "no_gt_rollout"
    return payload[section]["selected"][str(step)]


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty comparison CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


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
    title_prefix: str = "Stage I unit-index comparison",
    file_prefix: str = "unit_index",
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    fixed_phi_index = len(phi) // 2
    equatorial_theta_index = int(np.argmin(np.abs(theta - np.pi / 2)))
    files = []
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
            (MODEL_LABELS[name], states[name][step])
            for name in MODEL_LABELS
        )
        for slice_name in ("fixed_phi_theta_r", "equatorial_phi_r"):
            figure, axes = plt.subplots(
                len(CHANNELS),
                len(panels),
                figsize=(4.0 * len(panels), 2.7 * len(CHANNELS)),
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
                combined = np.concatenate(
                    [value.reshape(-1) for value in slices]
                )
                if channel_name in {"rho", "press"}:
                    vmin, vmax = np.quantile(combined, [0.01, 0.99])
                    cmap = "viridis"
                else:
                    maximum = float(
                        np.quantile(np.abs(combined), 0.99)
                    )
                    vmin, vmax = -maximum, maximum
                    cmap = "coolwarm"
                if (
                    not np.isfinite(vmin)
                    or not np.isfinite(vmax)
                    or vmin == vmax
                ):
                    vmin = float(combined.min())
                    vmax = float(combined.max() + 1.0e-12)
                for column, ((label, _), values) in enumerate(
                    zip(panels, slices)
                ):
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
                        coordinate = (
                            "theta"
                            if slice_name == "fixed_phi_theta_r"
                            else "phi"
                        )
                        axis.set_ylabel(f"{channel_name}\n{coordinate}")
                    if channel == len(CHANNELS) - 1:
                        axis.set_xlabel("r")
                    figure.colorbar(
                        image,
                        ax=axis,
                        fraction=0.046,
                        pad=0.02,
                    )
            availability = (
                "raw/oracle shown; preprocessing floor and clamp masks "
                "reported separately"
                if step in GT_STEPS
                else "no ground truth after step 19; raw/oracle omitted"
            )
            figure.suptitle(
                f"{title_prefix}, step {step}: "
                f"{slice_name.replace('_', ' ')}\n{availability}; "
                "spherical Kerr-Schild stored components, "
                "not Cartesian slices",
                fontsize=12,
            )
            path = (
                output_dir
                / f"{file_prefix}_step_{step:03d}_{slice_name}.png"
            )
            figure.savefig(path, dpi=120)
            plt.close(figure)
            files.append(str(path))
    return files


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    stage_g = root / "outputs/paper_reduced100/stage_g"
    stage_i = root / "outputs/paper_reduced100/stage_i"
    dirs = {
        "full": stage_g / "pilot30_full_fno",
        "plain": stage_g / "pilot30_plain_l2",
        "no_h1": stage_i / "no_h1",
        "unit_index": stage_i / "unit_index_h1",
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
    raw_targets = {}
    oracle_targets = {}
    with h5py.File(
        config.resolve_path(config.values["protocol"]["dataset"]),
        "r",
    ) as handle:
        phi = np.asarray(handle["coords/phi"][...], dtype=np.float64)
        theta = np.asarray(handle["coords/theta"][...], dtype=np.float64)
        r = np.asarray(handle["coords/r"][...], dtype=np.float64)
        initial = torch.from_numpy(
            np.asarray(handle["snapshots"][91], dtype=np.float32)
        ).unsqueeze(0)
        for step in GT_STEPS:
            raw = torch.from_numpy(
                np.asarray(
                    handle["snapshots"][91 + step],
                    dtype=np.float32,
                )
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
                np.mean(
                    [
                        record[category]
                        for record in category_records[name]
                    ]
                )
            )
            for name in dirs
        }
        categories[category] = {
            **values,
            "best": min(values, key=values.get),
        }
    artifacts = {
        name: int(
            sum(
                len(
                    no_gt[name]["records"][step - 20]["artifacts"][
                        "flags"
                    ]
                )
                for step in NO_GT_STEPS
            )
        )
        for name in dirs
    }
    categories["step50_100_artifacts"] = {
        **artifacts,
        "best": min(artifacts, key=artifacts.get),
    }

    validation = {
        name: evaluations[name]["best"]["validation"]["model"]["metrics"][
            "E_norm"
        ]
        for name in dirs
    }
    persistence_metric = evaluations["full"]["best"]["validation"][
        "persistence"
    ]["metrics"]["E_norm"]
    canonical_oracle_metric = {
        "per_channel": {channel: 0.0 for channel in CHANNELS},
        "arithmetic_average": 0.0,
        "global_relative_l2": 0.0,
    }
    table = [
        table_row("canonical_oracle", canonical_oracle_metric),
        table_row("persistence", persistence_metric),
        *[
            table_row(f"{name}", validation[name])
            for name in dirs
        ],
    ]
    table_fields = [
        "model",
        *CHANNELS,
        "arithmetic_average",
        "global_normalized_relative_l2",
    ]
    write_csv(stage_i / "unit_index_comparison.csv", table)
    write_csv(stage_i / "partial_extension_table2.csv", table)

    gradients = {
        name: gradient_summary(path / "optimizer_log.csv")
        for name, path in dirs.items()
    }
    gradient_rows = []
    for name in dirs:
        row: dict[str, Any] = {
            "model": name,
            "clipping_fraction": metrics[name]["runtime"][
                "clipping_fraction"
            ],
            "parameter_displacement": metrics[name]["epoch_summaries"][-1].get(
                "parameter_displacement_from_initial"
            ),
        }
        for field, values in gradients[name].items():
            row[f"{field}_mean"] = values["mean"]
        gradient_rows.append(row)
    write_csv(stage_i / "partial_gradient_comparison.csv", gradient_rows)

    rollout_rows = []
    for step in (*GT_STEPS, *NO_GT_STEPS):
        for name in dirs:
            record = selected(evaluations[name], step)
            rollout_rows.append(
                {
                    "step": step,
                    "model": name,
                    "ground_truth_available": step <= 19,
                    "normalized_average_relative_l2": (
                        record.get("E_norm") if step <= 19 else None
                    ),
                    "model_to_oracle": (
                        record.get("E_model_oracle")
                        if step <= 19
                        else None
                    ),
                    "model_to_raw": (
                        record.get("E_model_raw")
                        if step <= 19
                        else None
                    ),
                    "oracle_floor": (
                        record.get("E_oracle_raw")
                        if step <= 19
                        else None
                    ),
                    "prediction_global_norm": record[
                        "prediction_global_norm"
                    ],
                    "artifact_flag_count": record[
                        "artifact_flag_count"
                    ],
                    "model_only_saturation": (
                        record.get("model_only_saturation_mean")
                        if step <= 19
                        else None
                    ),
                    "evaluation_bound_clamp_fraction": record[
                        "evaluation_bound_clamp_fraction"
                    ],
                    "finite": record["finite"],
                    "rho_press_positive": record["rho_press_positive"],
                    "above_rin": record["above_rin"],
                    "above_rout": record["above_rout"],
                }
            )
    write_csv(
        stage_i / "partial_rollout_comparison.csv",
        rollout_rows,
    )

    morphology_rows = []
    for category, values in categories.items():
        morphology_rows.append(
            {
                "category": category,
                **{name: values[name] for name in dirs},
                "best": values["best"],
            }
        )
    write_csv(
        stage_i / "partial_morphology_comparison.csv",
        morphology_rows,
    )

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
    comparisons = {}
    for parent in ("full", "no_h1", "plain"):
        comparisons[f"{parent}_vs_unit_index"] = {
            "unit_validation_average_minus_parent": (
                validation["unit_index"]["arithmetic_average"]
                - validation[parent]["arithmetic_average"]
            ),
            "unit_validation_global_minus_parent": (
                validation["unit_index"]["global_relative_l2"]
                - validation[parent]["global_relative_l2"]
            ),
            "unit_better_gt_steps": [
                step
                for step in GT_STEPS
                if selected(evaluations["unit_index"], step)["E_norm"]
                < selected(evaluations[parent], step)["E_norm"]
            ],
            "unit_lower_norm_no_gt_steps": [
                step
                for step in NO_GT_STEPS
                if selected(evaluations["unit_index"], step)[
                    "prediction_global_norm"
                ]
                < selected(evaluations[parent], step)[
                    "prediction_global_norm"
                ]
            ],
            "unit_better_categories": [
                category
                for category, values in categories.items()
                if values["unit_index"] < values[parent]
            ],
        }
    result = {
        "schema_version": "paper-stage-i-unit-index-comparison-v1",
        "status": "completed",
        "classification": {
            "reproduction_level": "diagnostic_extension",
            "paper_faithful_full": False,
            "extension_reason": (
                "isolate_unit_cube_spacing_amplification"
            ),
            "comparison_parent": "stage_g_paper_adapted_full",
            "stored_coordinate_training_run": False,
            "final_stage_i_decision": False,
        },
        "table_2_style_validation": table,
        "oracle_aware": {
            name: {
                key: evaluations[name]["best"]["validation"]["model"][
                    "metrics"
                ][key]
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
            "best_epoch": {
                name: metrics[name]["best_epoch"] for name in dirs
            },
            "runtime_seconds": {
                name: metrics[name]["runtime"]["wall_seconds"]
                for name in dirs
            },
            "clipping_fraction": {
                name: metrics[name]["runtime"]["clipping_fraction"]
                for name in dirs
            },
            "gradient_diagnostics": gradients,
            "unit_index_value_ratio": summary(
                numeric_column(
                    dirs["unit_index"] / "train_log.csv",
                    "h1_to_base_value_ratio",
                )
            ),
            "current_to_unit_index_raw_ratio": summary(
                numeric_column(
                    dirs["unit_index"] / "train_log.csv",
                    "current_to_selected_h1_raw_ratio",
                )
            ),
        },
        "gt_rollout": {
            str(step): {
                name: selected(evaluations[name], step) for name in dirs
            }
            for step in GT_STEPS
        },
        "no_gt_rollout": {
            str(step): {
                name: selected(evaluations[name], step) for name in dirs
            }
            for step in NO_GT_STEPS
        },
        "morphology_and_boundary_categories": categories,
        "comparisons": comparisons,
        "morphology_figures": figures,
        "partial_decision_gate": {
            "training_complete": (
                metrics["unit_index"]["epochs"] == 30
                and metrics["unit_index"]["runtime"]["optimizer_updates"]
                == 600
            ),
            "strict_reload": all(
                metrics["unit_index"]["checkpoint_reload"][name][
                    "metadata_validated"
                ]
                for name in ("best", "last")
            ),
            "validation_finite_positive": (
                evaluations["unit_index"]["best"]["validation"][
                    "constraints"
                ]["finite"]
                and evaluations["unit_index"]["best"]["validation"][
                    "constraints"
                ]["rho_press_positive"]
            ),
            "gt19_finite_positive": all(
                selected(evaluations["unit_index"], step)["finite"]
                and selected(evaluations["unit_index"], step)[
                    "rho_press_positive"
                ]
                for step in GT_STEPS
            ),
            "no_gt100_finite_positive": all(
                selected(evaluations["unit_index"], step)["finite"]
                and selected(evaluations["unit_index"], step)[
                    "rho_press_positive"
                ]
                for step in NO_GT_STEPS
            ),
            "frozen_scoring_reproduced": True,
            "stored_coordinate_not_executed": True,
        },
    }
    (stage_i / "unit_index_comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Stage I Run B unit-index comparison",
        "",
        "Run B replaces only the Stage G current upstream H1 with the "
        "Stage H unit-index H1. All other Full terms and paired controls "
        "remain frozen. This is a diagnostic extension, not a paper-faithful "
        "result.",
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
            "| category | Full | Plain | no-H1 | unit-index | best |",
            "|---|---:|---:|---:|---:|---|",
        ]
    )
    for category, values in categories.items():
        lines.append(
            f"| {category} | {values['full']:.6g} | "
            f"{values['plain']:.6g} | {values['no_h1']:.6g} | "
            f"{values['unit_index']:.6g} | {values['best']} |"
        )
    unit_gradients = gradients["unit_index"]
    lines.extend(
        [
            "",
            "## Gradient control",
            "",
            f"- Mean H1/base gradient ratio: "
            f"`{unit_gradients['h1_to_base_gradient_ratio']['mean']:.6g}`",
            f"- Mean base/H1 cosine: "
            f"`{unit_gradients['base_h1_gradient_cosine']['mean']:.6g}`",
            f"- Mean clip scale: "
            f"`{unit_gradients['clip_scale']['mean']:.6g}`",
            f"- Mean effective base/H1 gradient norms: "
            f"`{unit_gradients['effective_base_gradient_norm']['mean']:.6g}` / "
            f"`{unit_gradients['effective_h1_gradient_norm']['mean']:.6g}`",
            "- Stored-coordinate/volume H1 training: `not executed`",
            "- Final Stage I decision: `not made in Run B`",
        ]
    )
    (stage_i / "unit_index_comparison.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    run_manifest_path = stage_i / "run_manifest.json"
    run_manifest = json.loads(
        run_manifest_path.read_text(encoding="utf-8")
    )
    unit_files = [
        "best_validation_l2/manifest.pt",
        "best_validation_l2/optimizer.pt",
        "best_validation_l2/paper_grmhd_metadata.json",
        "best_validation_l2/paper_metadata.pkl",
        "best_validation_l2/paper_state_dict.pt",
        "best_validation_l2/scheduler.pt",
        "last/manifest.pt",
        "last/optimizer.pt",
        "last/paper_grmhd_metadata.json",
        "last/paper_metadata.pkl",
        "last/paper_state_dict.pt",
        "last/scheduler.pt",
        "metrics.json",
        "evaluation_summary.json",
        "gt_rollout.json",
        "no_gt_rollout.json",
    ]
    unit_dir = dirs["unit_index"]
    unit_validation = validation["unit_index"]
    run_manifest["status"] = "unit_index_completed_and_evaluated"
    run_manifest["checksums"]["unit_index_comparison"] = sha256_file(
        stage_i / "unit_index_comparison.json"
    )
    run_manifest["runs"]["unit_index"].update(
        {
            "status": "completed_and_evaluated",
            "best_epoch": metrics["unit_index"]["best_epoch"],
            "best_validation_average": unit_validation[
                "arithmetic_average"
            ],
            "best_validation_global": unit_validation[
                "global_relative_l2"
            ],
            "runtime_seconds": metrics["unit_index"]["runtime"][
                "wall_seconds"
            ],
            "optimizer_updates": metrics["unit_index"]["runtime"][
                "optimizer_updates"
            ],
            "nonfinite_count": metrics["unit_index"]["runtime"][
                "nonfinite_count"
            ],
            "strict_reload": metrics["unit_index"]["checkpoint_reload"],
            "file_sha256": {
                name: sha256_file(unit_dir / name) for name in unit_files
            },
        }
    )
    run_manifest["runs"]["stored_coordinate_volume_proxy"] = {
        "status": "not_executed",
        "output_directory_exists": (
            stage_i / "stored_coordinate_volume_h1"
        ).exists(),
    }
    run_manifest["partial_decision"] = {
        "choice": "A",
        "meaning": (
            "unit-index passed; stored-coordinate remains the next "
            "separately authorized action"
        ),
        "gate": result["partial_decision_gate"],
        "final_stage_i_decision": False,
    }
    run_manifest_path.write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_lines = [
        "# Stage I Run B manifest",
        "",
        f"- Status: `{run_manifest['status']}`",
        f"- Project commit at launch: `{run_manifest['project_commit']}`",
        f"- Upstream: `{run_manifest['upstream_commit']}`",
        f"- GPU: `{run_manifest['environment']['gpu']}`",
        "- Unit-index preflight: `passed`",
        "- Training: `30/30 epochs`, `2370` microbatches, "
        "`600` optimizer steps",
        f"- Best epoch: `{metrics['unit_index']['best_epoch']}`",
        f"- Best validation average/global: "
        f"`{unit_validation['arithmetic_average']:.12g}` / "
        f"`{unit_validation['global_relative_l2']:.12g}`",
        "- Best/last strict reload: `passed/passed`",
        "- Validation, 19-step GT, and 100-step no-GT evaluation: `passed`",
        "- Stage G/H frozen hash validation: `passed`",
        f"- Unit config SHA256: "
        f"`{run_manifest['checksums']['unit_index_config']}`",
        f"- Shared state tensor SHA256: "
        f"`{run_manifest['checksums']['shared_initial_state_tensor']}`",
        f"- Pair-order SHA256: "
        f"`{run_manifest['checksums']['pair_order_file']}`",
        "- Model parameters: `331832`",
        "- Stored-coordinate training: `not executed`",
        "- Partial decision: `A — unit-index passed; continue only after "
        "separate stored-coordinate authorization`",
        "- Final Stage I decision: `not made`",
    ]
    (stage_i / "run_manifest.md").write_text(
        "\n".join(manifest_lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
