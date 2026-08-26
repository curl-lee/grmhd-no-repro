#!/usr/bin/env python
"""Compare matched Stage G Full/Plain pilots and render spherical-coordinate slices."""

from __future__ import annotations

import argparse
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

from grmhd import CHANNELS
from grmhd.paper_config import load_paper_experiment_config
from grmhd.paper_data_processor import PaperDataProcessor
from grmhd.paper_stage_g_evaluation import (
    physical_state_statistics,
    total_variation_and_high_k,
)


GT_STEPS = (1, 5, 10, 19)
NO_GT_STEPS = (50, 100)


def numeric_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "q95": float(np.quantile(array, 0.95)),
        "maximum": float(array.max()),
    }


def relative_l2(prediction: np.ndarray, reference: np.ndarray) -> float:
    return float(
        np.linalg.norm(prediction - reference)
        / max(np.linalg.norm(reference), 1e-30)
    )


def state_category_scores(
    *,
    prediction: torch.Tensor,
    oracle: torch.Tensor,
    r: np.ndarray,
) -> dict[str, float]:
    pred = prediction[0].detach().float().cpu().numpy().astype(np.float64)
    ref = oracle[0].detach().float().cpu().numpy().astype(np.float64)
    nr = pred.shape[-1]
    ntheta = pred.shape[-2]
    center = slice(0, max(1, nr // 4))
    polar_width = max(1, ntheta // 8)
    polar_indices = np.r_[0:polar_width, ntheta - polar_width : ntheta]
    center_errors = [
        relative_l2(pred[channel, ..., center], ref[channel, ..., center])
        for channel in range(len(CHANNELS))
    ]
    polar_errors = [
        relative_l2(
            pred[channel, :, polar_indices, :],
            ref[channel, :, polar_indices, :],
        )
        for channel in range(len(CHANNELS))
    ]
    pred_spectral = total_variation_and_high_k(prediction)["per_channel"]
    ref_spectral = total_variation_and_high_k(oracle)["per_channel"]
    magnetic_texture = []
    for name in CHANNELS[:3]:
        for key in ("total_variation_mean", "high_k_energy_fraction"):
            magnetic_texture.append(
                abs(pred_spectral[name][key] - ref_spectral[name][key])
                / max(abs(ref_spectral[name][key]), 1e-30)
            )
    pred_stats = physical_state_statistics(prediction, r)
    ref_stats = physical_state_statistics(oracle, r)
    radial_errors = []
    outer_errors = []
    for name in CHANNELS:
        pred_channel = pred_stats["channels"][name]
        ref_channel = ref_stats["channels"][name]
        radial_errors.append(
            relative_l2(
                np.asarray(pred_channel["radial_mean"]),
                np.asarray(ref_channel["radial_mean"]),
            )
        )
        pred_outer = np.asarray(
            [
                value
                for shell in pred_channel["outermost_two_shells"]
                for value in (shell["mean"], shell["std"])
            ]
        )
        ref_outer = np.asarray(
            [
                value
                for shell in ref_channel["outermost_two_shells"]
                for value in (shell["mean"], shell["std"])
            ]
        )
        outer_errors.append(relative_l2(pred_outer, ref_outer))
    return {
        "center_morphology": float(np.mean(center_errors)),
        "polar_morphology": float(np.mean(polar_errors)),
        "magnetic_texture": float(np.mean(magnetic_texture)),
        "radial_statistics": float(np.mean(radial_errors)),
        "outer_shell_statistics": float(np.mean(outer_errors)),
    }


def model_only_saturation(record: Mapping[str, Any]) -> float:
    return float(
        np.mean(
            [
                value["model_only_fraction"]
                for value in record["saturation"]["channels"].values()
            ]
        )
    )


def load_selected(path: Path) -> dict[int, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return {int(step): state for step, state in payload["selected_steps"].items()}


def table_row(name: str, metric: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": name,
        **{channel: metric["per_channel"][channel] for channel in CHANNELS},
        "arithmetic_average": metric["arithmetic_average"],
        "global_normalized_relative_l2": metric["global_relative_l2"],
    }


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fields = ["model", *CHANNELS, "arithmetic_average", "global_normalized_relative_l2"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def render_morphology(
    *,
    output_dir: Path,
    raw_targets: Mapping[int, torch.Tensor],
    oracle_targets: Mapping[int, torch.Tensor],
    persistence: torch.Tensor,
    full_states: Mapping[int, torch.Tensor],
    plain_states: Mapping[int, torch.Tensor],
    phi: np.ndarray,
    theta: np.ndarray,
    r: np.ndarray,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = []
    fixed_phi_index = len(phi) // 2
    equatorial_theta_index = int(np.argmin(np.abs(theta - np.pi / 2)))
    for step in (*GT_STEPS, *NO_GT_STEPS):
        if step in GT_STEPS:
            panels = [
                ("raw truth", raw_targets[step]),
                ("canonical oracle", oracle_targets[step]),
                ("persistence", persistence),
                ("Full", full_states[step]),
                ("Plain", plain_states[step]),
            ]
        else:
            panels = [
                ("persistence", persistence),
                ("Full", full_states[step]),
                ("Plain", plain_states[step]),
            ]
        for slice_name in ("fixed_phi_theta_r", "equatorial_phi_r"):
            figure, axes = plt.subplots(
                len(CHANNELS),
                len(panels),
                figsize=(4.3 * len(panels), 3.0 * len(CHANNELS)),
                constrained_layout=True,
            )
            for channel, channel_name in enumerate(CHANNELS):
                slices = []
                for _, state in panels:
                    values = state[0, channel].detach().cpu().numpy()
                    if slice_name == "fixed_phi_theta_r":
                        slices.append(values[fixed_phi_index])
                    else:
                        slices.append(values[:, equatorial_theta_index, :])
                combined = np.concatenate([values.reshape(-1) for values in slices])
                if channel_name in {"rho", "press"}:
                    vmin, vmax = np.quantile(combined, [0.01, 0.99])
                    cmap = "viridis"
                else:
                    maximum = float(np.quantile(np.abs(combined), 0.99))
                    vmin, vmax = -maximum, maximum
                    cmap = "coolwarm"
                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                    vmin, vmax = float(combined.min()), float(combined.max() + 1e-12)
                for column, ((label, _), values) in enumerate(zip(panels, slices)):
                    axis = axes[channel, column]
                    extent = (
                        [float(r.min()), float(r.max()), float(theta.min()), float(theta.max())]
                        if slice_name == "fixed_phi_theta_r"
                        else [float(r.min()), float(r.max()), float(phi.min()), float(phi.max())]
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
                            + ("theta" if slice_name == "fixed_phi_theta_r" else "phi")
                        )
                    if channel == len(CHANNELS) - 1:
                        axis.set_xlabel("r")
                    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
            availability = (
                "raw/oracle available"
                if step in GT_STEPS
                else "no ground truth after step 19; raw/oracle intentionally omitted"
            )
            figure.suptitle(
                f"Stage G step {step}: {slice_name.replace('_', ' ')}\n"
                f"{availability}; spherical Kerr-Schild stored components; "
                "Bcc3/vel3 include preprocessing oracle floor",
                fontsize=13,
            )
            path = output_dir / f"morphology_step_{step:03d}_{slice_name}.png"
            figure.savefig(path, dpi=120)
            plt.close(figure)
            files.append(str(path))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-g-dir",
        type=Path,
        default=Path("outputs/paper_reduced100/stage_g"),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    stage_g = args.stage_g_dir if args.stage_g_dir.is_absolute() else root / args.stage_g_dir
    full_dir = stage_g / "pilot30_full_fno"
    plain_dir = stage_g / "pilot30_plain_l2"
    manifest = json.loads((stage_g / "run_manifest.json").read_text(encoding="utf-8"))
    full_metrics = json.loads((full_dir / "metrics.json").read_text(encoding="utf-8"))
    plain_metrics = json.loads((plain_dir / "metrics.json").read_text(encoding="utf-8"))
    full_eval = json.loads((full_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
    plain_eval = json.loads((plain_dir / "evaluation_summary.json").read_text(encoding="utf-8"))
    full_gt = json.loads((full_dir / "gt_rollout.json").read_text(encoding="utf-8"))
    plain_gt = json.loads((plain_dir / "gt_rollout.json").read_text(encoding="utf-8"))
    full_no_gt = json.loads((full_dir / "no_gt_rollout.json").read_text(encoding="utf-8"))
    plain_no_gt = json.loads((plain_dir / "no_gt_rollout.json").read_text(encoding="utf-8"))
    full_states = load_selected(full_dir / "selected_states.pt")
    plain_states = load_selected(plain_dir / "selected_states.pt")

    config = load_paper_experiment_config(
        root / "configs/paper_reduced100/full_fno_proxy.yaml", project_root=root
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
                processor.preprocessor.encode(raw, channel_axis=1), channel_axis=1
            )
    persistence = processor.preprocessor.decode(
        processor.preprocessor.encode(initial, channel_axis=1), channel_axis=1
    )

    category_records = {"full": [], "plain": []}
    for mode, states, gt_records in (
        ("full", full_states, full_gt["records"]),
        ("plain", plain_states, plain_gt["records"]),
    ):
        for step in GT_STEPS:
            scores = state_category_scores(
                prediction=states[step], oracle=oracle_targets[step], r=r
            )
            scores["step"] = step
            scores["model_only_saturation"] = model_only_saturation(
                gt_records[step - 1]
            )
            category_records[mode].append(scores)
    category_scores = {}
    for category in (
        "center_morphology",
        "polar_morphology",
        "magnetic_texture",
        "radial_statistics",
        "outer_shell_statistics",
        "model_only_saturation",
    ):
        category_scores[category] = {
            mode: float(np.mean([record[category] for record in records]))
            for mode, records in category_records.items()
        }
    category_scores["step50_100_artifacts"] = {
        "full": float(
            sum(
                len(full_no_gt["records"][step - 20]["artifacts"]["flags"])
                for step in NO_GT_STEPS
            )
        ),
        "plain": float(
            sum(
                len(plain_no_gt["records"][step - 20]["artifacts"]["flags"])
                for step in NO_GT_STEPS
            )
        ),
    }
    for values in category_scores.values():
        values["better"] = (
            "full"
            if values["full"] < values["plain"]
            else "plain"
            if values["plain"] < values["full"]
            else "tie"
        )
    full_better_categories = [
        name for name, values in category_scores.items() if values["better"] == "full"
    ]

    oracle_metric = {
        "per_channel": {channel: 0.0 for channel in CHANNELS},
        "arithmetic_average": 0.0,
        "global_relative_l2": 0.0,
    }
    persistence_metric = full_eval["best"]["validation"]["persistence"]["metrics"][
        "E_norm"
    ]
    full_norm = full_eval["best"]["validation"]["model"]["metrics"]["E_norm"]
    plain_norm = plain_eval["best"]["validation"]["model"]["metrics"]["E_norm"]
    table = [
        table_row("canonical_oracle", oracle_metric),
        table_row("persistence", persistence_metric),
        table_row("full_fno_proxy", full_norm),
        table_row("plain_l2_fno", plain_norm),
    ]
    write_csv(stage_g / "pilot_comparison.csv", table)

    full_h1_values = [
        float(value) for value in full_metrics["h1_monitoring"]["value_ratios"]
    ]
    full_h1_gradients = [
        float(value) for value in full_metrics["h1_monitoring"]["gradient_ratios"]
    ]
    result = {
        "schema_version": "paper-stage-g-controlled-comparison-v1",
        "status": "completed",
        "scope": {
            "protocol": "reduced100",
            "coordinates": "spherical Kerr-Schild",
            "thermal": "press adaptation",
            "model": "FNO proxy, not 3D DISCO LocalNO",
            "training": "30-epoch resource-scaled pilot",
            "paper_comparison": "not directly comparable to paper Table 2",
        },
        "controlled_pairing": {
            "shared_initial_state_sha256": manifest["shared_initial_state"][
                "tensor_state_sha256"
            ],
            "pair_order_sha256": manifest["checksums"]["pair_order_file"],
            "parameter_count": manifest["model"]["parameter_count"],
            "unexpected_config_differences": manifest["paired_config_audit"][
                "unexpected_difference_count"
            ],
        },
        "table_2_style": table,
        "training": {
            "full": {
                "best_epoch": full_eval["best"]["reload"]["epoch"],
                "last_epoch": full_eval["last"]["reload"]["epoch"],
                "best_validation_average": full_norm["arithmetic_average"],
                "last_validation_average": full_eval["last"]["validation"]["model"][
                    "metrics"
                ]["E_norm"]["arithmetic_average"],
                "overfit_gap": full_eval["last"]["validation"]["model"]["metrics"][
                    "E_norm"
                ]["arithmetic_average"]
                - full_norm["arithmetic_average"],
                "runtime": full_metrics["runtime"],
                "h1_to_base_value_ratio": numeric_summary(full_h1_values),
                "h1_to_base_gradient_ratio": numeric_summary(full_h1_gradients),
            },
            "plain": {
                "best_epoch": plain_eval["best"]["reload"]["epoch"],
                "last_epoch": plain_eval["last"]["reload"]["epoch"],
                "best_validation_average": plain_norm["arithmetic_average"],
                "last_validation_average": plain_eval["last"]["validation"]["model"][
                    "metrics"
                ]["E_norm"]["arithmetic_average"],
                "overfit_gap": plain_eval["last"]["validation"]["model"]["metrics"][
                    "E_norm"
                ]["arithmetic_average"]
                - plain_norm["arithmetic_average"],
                "runtime": plain_metrics["runtime"],
                "disabled_components": plain_metrics["disabled_components"],
            },
            "validation_curves": {
                "full": [
                    record[
                        "validation_normalized_arithmetic_average_relative_l2"
                    ]
                    for record in full_metrics["epoch_summaries"]
                ],
                "plain": [
                    record[
                        "validation_normalized_arithmetic_average_relative_l2"
                    ]
                    for record in plain_metrics["epoch_summaries"]
                ],
            },
        },
        "oracle_aware": {
            mode: {
                name: payload["best"]["validation"]["model"]["metrics"][name]
                for name in (
                    "E_norm",
                    "E_model_oracle",
                    "E_model_raw",
                    "E_oracle_raw",
                )
            }
            for mode, payload in (("full", full_eval), ("plain", plain_eval))
        },
        "gt_rollout_selected": {
            mode: payload["gt_rollout"]["selected"]
            for mode, payload in (("full", full_eval), ("plain", plain_eval))
        },
        "no_gt_rollout_selected": {
            mode: payload["no_gt_rollout"]["selected"]
            for mode, payload in (("full", full_eval), ("plain", plain_eval))
        },
        "morphology_and_boundary_categories": category_scores,
        "full_better_categories": full_better_categories,
    }
    conditions = {
        "both_completed_30_epochs": full_metrics["epochs"] == plain_metrics["epochs"] == 30,
        "checkpoints_reload": all(
            payload[name]["reload"]["strict_model_reload"]
            for payload in (full_eval, plain_eval)
            for name in ("best", "last")
        ),
        "no_nonfinite": (
            full_metrics["runtime"]["nonfinite_count"]
            == plain_metrics["runtime"]["nonfinite_count"]
            == 0
        ),
        "rho_press_positive": all(
            payload[key]["rho_press_positive"]
            for payload in (full_eval, plain_eval)
            for key in ("gt_rollout", "no_gt_rollout")
        ),
        "outputs_nonconstant": all(
            float(state.std()) > 0 for state in (full_states[100], plain_states[100])
        ),
        "nonzero_updates": (
            full_metrics["runtime"]["optimizer_updated_model"]
            and plain_metrics["runtime"]["optimizer_updated_model"]
        ),
        "step19_not_catastrophic": all(
            payload["gt_rollout"]["selected"]["19"]["finite"]
            and payload["gt_rollout"]["selected"]["19"]["E_norm"] < 10
            for payload in (full_eval, plain_eval)
        ),
        "step50_finite": all(
            payload["no_gt_rollout"]["selected"]["50"]["finite"]
            for payload in (full_eval, plain_eval)
        ),
        "full_better_in_at_least_two_categories": len(full_better_categories) >= 2,
        "full_one_step_not_order_of_magnitude_worse": (
            full_norm["arithmetic_average"]
            < 10 * plain_norm["arithmetic_average"]
        ),
    }
    if all(conditions.values()):
        decision = "A"
        decision_text = "进入论文消融"
    elif all(
        conditions[key]
        for key in (
            "both_completed_30_epochs",
            "checkpoints_reload",
            "no_nonfinite",
            "rho_press_positive",
            "outputs_nonconstant",
            "nonzero_updates",
            "step19_not_catastrophic",
            "step50_finite",
        )
    ):
        decision = "B"
        decision_text = "先诊断 H1 adaptation"
    else:
        decision = "C"
        decision_text = "基础训练失败，暂停"
    result["decision"] = {
        "choice": decision,
        "text": decision_text,
        "conditions": conditions,
    }
    figure_files = render_morphology(
        output_dir=stage_g / "figures",
        raw_targets=raw_targets,
        oracle_targets=oracle_targets,
        persistence=persistence,
        full_states=full_states,
        plain_states=plain_states,
        phi=phi,
        theta=theta,
        r=r,
    )
    result["morphology_figures"] = figure_files
    (stage_g / "pilot_comparison.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Stage G Full versus Plain controlled comparison",
        "",
        "This is a reduced100, spherical Kerr-Schild, press-adapted, FNO-proxy, "
        "30-epoch resource-scaled pilot. It is not directly comparable to paper Table 2.",
        "",
        "## Table-2-style normalized validation",
        "",
        "| model | "
        + " | ".join(CHANNELS)
        + " | average | global |",
        "|---|" + "---:|" * (len(CHANNELS) + 2),
    ]
    for row in table:
        lines.append(
            "| "
            + str(row["model"])
            + " | "
            + " | ".join(f"{row[channel]:.6g}" for channel in CHANNELS)
            + f" | {row['arithmetic_average']:.6g}"
            + f" | {row['global_normalized_relative_l2']:.6g} |"
        )
    lines.extend(
        [
            "",
            "## Morphology and boundary categories",
            "",
            "| category | Full | Plain | better |",
            "|---|---:|---:|---|",
        ]
    )
    for name, values in category_scores.items():
        lines.append(
            f"| {name} | {values['full']:.6g} | {values['plain']:.6g} | "
            f"{values['better']} |"
        )
    lines.extend(
        [
            "",
            f"- Full-better categories: `{', '.join(full_better_categories) or 'none'}`",
            f"- Decision: `{decision}. {decision_text}`",
            "",
            "The comparison does not claim exact paper reproduction or direct numerical "
            "equivalence to the paper's 1200-epoch 3D DISCO LocalNO experiments.",
        ]
    )
    (stage_g / "pilot_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
