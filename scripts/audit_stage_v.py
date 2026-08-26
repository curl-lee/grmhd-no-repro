#!/usr/bin/env python3
"""Run Stage V no-training alignment, gradient, and train-only OOD audits."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch
import yaml

from grmhd import CHANNELS
from grmhd.paper_stage_g_evaluation import radial_shell_indices
from grmhd.paper_stage_m import radial_profile_vector, transport_metrics, variance_vector
from grmhd.stage_v_objectives import (
    classify_gradient_conflict,
    flattened_gradient,
    gradient_cosine,
    gradient_norm_ratio,
    objective_components,
    radial_profile,
    relative_transport_loss,
    shell_variance,
)

from evaluate_stage_s import encoded_channel
from evaluate_stage_t import model_from_checkpoint
from train_stage_s import StageSBatchPath, build_frozen_model
from train_stage_t import verify_frozen_contract


ROOT = Path(__file__).resolve().parents[1]
PRIMARY = ("Bcc2", "Bcc3", "vel3")
FOCUS = ("Bcc2", "Bcc3", "vel3", "rho", "press")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def safe_rel(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left.ravel()) / max(float(np.linalg.norm(right.ravel())), 1e-300))


def safe_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left.ravel()) * np.linalg.norm(right.ravel()))
    return float(np.dot(left.ravel(), right.ravel()) / max(denominator, 1e-300))


def pair_metrics(
    model: torch.nn.Module,
    data: StageSBatchPath,
    source: int,
    shell_index: np.ndarray,
    split: str,
) -> dict[str, Any]:
    with torch.no_grad():
        result = data.predict(model, source)
        physical_prediction = data.preprocessor.decode_tensor(result["z_prediction"], channel_axis=1)
        physical_input = data.preprocessor.decode_tensor(result["z_input"], channel_axis=1)
        physical_oracle = data.preprocessor.decode_tensor(result["z_target"], channel_axis=1)
    arrays = {
        name: value[0].detach().cpu().numpy().astype(np.float64)
        for name, value in {
            **result,
            "physical_prediction": physical_prediction,
            "physical_input": physical_input,
            "physical_oracle": physical_oracle,
        }.items()
        if torch.is_tensor(value) and value.ndim == 5 and value.shape[1] == 8
    }
    residual_error = arrays["predicted_residual"] - arrays["residual_target"]
    state_error = arrays["z_prediction"] - arrays["z_target"]
    physical_error = arrays["physical_prediction"] - arrays["raw_target"]
    row: dict[str, Any] = {
        "source_snapshot": source,
        "target_snapshot": source + 1,
        "split": split,
        "plain_residual_loss": float(np.mean(np.square(residual_error), axis=(1, 2, 3)).sum()),
        "state_relative_l2": float(np.mean([
            safe_rel(state_error[channel], arrays["z_target"][channel]) for channel in range(8)
        ])),
        "residual_relative_l2": float(np.mean([
            safe_rel(residual_error[channel], arrays["residual_target"][channel]) for channel in range(8)
        ])),
        "residual_global_relative_l2": safe_rel(residual_error, arrays["residual_target"]),
        "residual_cosine": safe_cosine(arrays["predicted_residual"], arrays["residual_target"]),
        "physical_relative_l2": float(np.mean([
            safe_rel(physical_error[channel], arrays["raw_target"][channel]) for channel in range(8)
        ])),
        "decoder_extreme_tail_fraction": float(np.mean(np.abs(arrays["z_prediction"]) > 5.94)),
    }
    row["residual_cosine_error"] = 1.0 - row["residual_cosine"]
    physical_sse = np.sum(np.square(physical_error), axis=(1, 2, 3))
    row["rho_press_physical_tail_contribution"] = float(
        physical_sse[3:5].sum() / max(float(physical_sse.sum()), 1e-300)
    )
    shell_errors: list[float] = []
    radial_errors: list[float] = []
    for channel, name in enumerate(CHANNELS):
        channel_residual_error = residual_error[channel]
        row[f"{name}_plain_loss"] = float(np.mean(np.square(channel_residual_error)))
        row[f"{name}_state_relative_l2"] = safe_rel(state_error[channel], arrays["z_target"][channel])
        row[f"{name}_residual_relative_l2"] = safe_rel(channel_residual_error, arrays["residual_target"][channel])
        row[f"{name}_residual_cosine"] = safe_cosine(
            arrays["predicted_residual"][channel], arrays["residual_target"][channel]
        )
        row[f"{name}_physical_relative_l2"] = safe_rel(physical_error[channel], arrays["raw_target"][channel])
        shell = transport_metrics(
            variance_vector(arrays["physical_input"][channel], shell_index),
            variance_vector(arrays["physical_oracle"][channel], shell_index),
            variance_vector(arrays["physical_prediction"][channel], shell_index),
            epsilon=1e-30,
            sign_zero_tolerance=1e-12,
        )
        radial = transport_metrics(
            radial_profile_vector(arrays["physical_input"][channel]),
            radial_profile_vector(arrays["physical_oracle"][channel]),
            radial_profile_vector(arrays["physical_prediction"][channel]),
            epsilon=1e-30,
            sign_zero_tolerance=1e-12,
        )
        row[f"{name}_shell_error"] = shell["relative_error"]
        row[f"{name}_radial_error"] = radial["relative_error"]
        if name in PRIMARY:
            if shell["relative_error"] is not None:
                shell_errors.append(float(shell["relative_error"]))
            if radial["relative_error"] is not None:
                radial_errors.append(float(radial["relative_error"]))
    row["shell_error"] = float(np.median(shell_errors))
    row["radial_error"] = float(np.median(radial_errors))
    return row


def correlation_rows(pair_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], bool]:
    output: list[dict[str, Any]] = []
    metrics = (
        "state_relative_l2",
        "residual_relative_l2",
        "residual_cosine_error",
        "shell_error",
        "radial_error",
        "physical_relative_l2",
        "rho_press_physical_tail_contribution",
    )
    for split in ("train", "validation", "combined"):
        selected = list(pair_rows) if split == "combined" else [row for row in pair_rows if row["split"] == split]
        for metric in metrics:
            x = np.asarray([float(row["plain_residual_loss"]) for row in selected])
            y = np.asarray([float(row[metric]) for row in selected])
            pearson = pearsonr(x, y)
            spearman = spearmanr(x, y)
            output.append({
                "split": split,
                "scope": "aggregate",
                "channel": "all",
                "x": "plain_residual_loss",
                "y": metric,
                "pair_count": len(selected),
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_r": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
            })
        for channel in FOCUS:
            for metric in ("state_relative_l2", "residual_relative_l2", "shell_error", "radial_error"):
                x = np.asarray([float(row[f"{channel}_plain_loss"]) for row in selected])
                y = np.asarray([float(row[f"{channel}_{metric}"]) for row in selected])
                pearson = pearsonr(x, y)
                spearman = spearmanr(x, y)
                output.append({
                    "split": split,
                    "scope": "per_channel",
                    "channel": channel,
                    "x": f"{channel}_plain_loss",
                    "y": metric,
                    "pair_count": len(selected),
                    "pearson_r": float(pearson.statistic),
                    "pearson_p": float(pearson.pvalue),
                    "spearman_r": float(spearman.statistic),
                    "spearman_p": float(spearman.pvalue),
                })
    lookup = {(row["split"], row["y"]): float(row["spearman_r"]) for row in output if row["scope"] == "aggregate"}
    misaligned = any(
        abs(lookup[(split, "residual_relative_l2")]) >= 0.50
        and abs(lookup[(split, "shell_error")]) < 0.30
        and abs(lookup[(split, "radial_error")]) < 0.30
        for split in ("validation", "combined")
    )
    return output, misaligned


def module_masks(names: Sequence[str]) -> dict[str, list[bool]]:
    output: dict[str, list[bool]] = {"global": [True] * len(names)}
    categories = {
        "spectral_branch": lambda name: "local_no_blocks.convs." in name,
        "differential_branch": lambda name: "local_no_blocks.differential." in name,
        "lifting": lambda name: name.startswith("lifting."),
        "projection": lambda name: name.startswith("projection."),
    }
    for label, predicate in categories.items():
        output[label] = [predicate(name) for name in names]
    for layer in range(4):
        pattern = re.compile(rf"local_no_blocks\.(?:convs|differential|local_no_skips)\.{layer}(?:\.|$)")
        output[f"layer_{layer}"] = [bool(pattern.search(name)) for name in names]
    return output


def subset_vector(
    gradients: Sequence[torch.Tensor | None],
    parameters: Sequence[torch.nn.Parameter],
    mask: Sequence[bool],
) -> torch.Tensor:
    return flattened_gradient(
        [gradient for gradient, keep in zip(gradients, mask, strict=True) if keep],
        [parameter for parameter, keep in zip(parameters, mask, strict=True) if keep],
    )


def gradient_audit(
    *,
    model: torch.nn.Module,
    data: StageSBatchPath,
    sources: Sequence[int],
    shell_index: np.ndarray,
    epsilon: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    names = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    masks = module_masks(names)
    global_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    module_rows: list[dict[str, Any]] = []
    shell_tensor = torch.as_tensor(shell_index, device=data.device, dtype=torch.long)
    model.eval()
    for position, source in enumerate(sources):
        result = data.predict(model, int(source))
        components = objective_components(
            predicted_residual=result["predicted_residual"],
            residual_target=result["residual_target"],
            normalized_input=result["z_input"],
            normalized_target=result["z_target"],
            preprocessor=data.preprocessor,
            shell_index=shell_tensor,
            epsilon=epsilon,
        )
        objective_order = ("plain", "direction", "shell", "radial", "transport")
        gradients: dict[str, tuple[torch.Tensor | None, ...]] = {}
        for index, objective in enumerate(objective_order):
            gradients[objective] = torch.autograd.grad(
                components[objective],
                parameters,
                # The same forward graph is also used below for frozen
                # channel-attribution gradients.
                retain_graph=True,
                allow_unused=True,
            )
        for objective in objective_order[1:]:
            for module, mask in masks.items():
                plain_vector = subset_vector(gradients["plain"], parameters, mask)
                aux_vector = subset_vector(gradients[objective], parameters, mask)
                row = {
                    "pair_source": int(source),
                    "subset_position": position,
                    "scope": "aggregate",
                    "channel": "all",
                    "objective": objective,
                    "module": module,
                    "cosine_with_plain": gradient_cosine(plain_vector, aux_vector),
                    "aux_over_plain_norm_ratio": gradient_norm_ratio(aux_vector, plain_vector),
                    "plain_gradient_norm": float(torch.linalg.vector_norm(plain_vector.float()).detach().cpu()),
                    "aux_gradient_norm": float(torch.linalg.vector_norm(aux_vector.float()).detach().cpu()),
                }
                module_rows.append(row)
                if module == "global":
                    global_rows.append(dict(row))
        # Channel attribution compares each channel's own Plain gradient with
        # its own combined shell/radial transport gradient on the same frozen
        # pair subset.  It does not participate in lambda selection.
        shell_input = shell_variance(result["z_input"], shell_tensor)
        shell_target = shell_variance(result["z_target"], shell_tensor)
        shell_prediction = shell_variance(result["z_prediction"], shell_tensor)
        radial_input = radial_profile(result["z_input"])
        radial_target = radial_profile(result["z_target"])
        radial_prediction = radial_profile(result["z_prediction"])
        for channel_name in FOCUS:
            channel = CHANNELS.index(channel_name)
            plain_channel = (
                result["predicted_residual"][:, channel]
                - result["residual_target"][:, channel]
            ).square().mean()
            shell_channel = relative_transport_loss(
                shell_input[:, channel : channel + 1],
                shell_target[:, channel : channel + 1],
                shell_prediction[:, channel : channel + 1],
                epsilon=epsilon,
            )
            radial_channel = relative_transport_loss(
                radial_input[:, channel : channel + 1],
                radial_target[:, channel : channel + 1],
                radial_prediction[:, channel : channel + 1],
                epsilon=epsilon,
            )
            transport_channel = 0.5 * (shell_channel + radial_channel)
            plain_gradient = torch.autograd.grad(
                plain_channel, parameters, retain_graph=True, allow_unused=True
            )
            transport_gradient = torch.autograd.grad(
                transport_channel,
                parameters,
                retain_graph=channel_name != FOCUS[-1],
                allow_unused=True,
            )
            plain_vector = flattened_gradient(plain_gradient, parameters)
            transport_vector = flattened_gradient(transport_gradient, parameters)
            channel_rows.append({
                "pair_source": int(source),
                "subset_position": position,
                "scope": "per_channel",
                "channel": channel_name,
                "objective": "transport",
                "module": "global",
                "cosine_with_plain": gradient_cosine(plain_vector, transport_vector),
                "aux_over_plain_norm_ratio": gradient_norm_ratio(transport_vector, plain_vector),
                "plain_gradient_norm": float(torch.linalg.vector_norm(plain_vector.float()).detach().cpu()),
                "aux_gradient_norm": float(torch.linalg.vector_norm(transport_vector.float()).detach().cpu()),
            })
        del result, components, gradients
    transport_cosines = [
        row["cosine_with_plain"] for row in global_rows if row["objective"] in ("shell", "radial", "transport")
    ]
    classification = classify_gradient_conflict(transport_cosines)
    ratios = {
        objective: [
            float(row["aux_over_plain_norm_ratio"])
            for row in global_rows
            if row["objective"] == objective and row["aux_over_plain_norm_ratio"] is not None
        ]
        for objective in ("direction", "shell", "radial", "transport")
    }
    medians = {name: float(np.median(values)) for name, values in ratios.items()}
    weights = {
        "schema_version": "stage-v-train-only-gradient-scaling-v1",
        "checkpoint": "shared_initial_state",
        "train_only": True,
        "validation_used_for_weight_selection": False,
        "pair_subset": [int(value) for value in sources],
        "pair_subset_count": len(sources),
        "gradient_norm_ratio_medians_aux_over_plain": medians,
        "direction": {
            "target_fraction_single": 0.10,
            "lambda_single": 0.10 / medians["direction"],
            "target_fraction_combined": 0.05,
            "lambda_combined": 0.05 / medians["direction"],
        },
        "transport": {
            "definition": "0.5*(normalized_shell_variance_relative_squared_error+normalized_radial_profile_relative_squared_error)",
            "target_fraction_single": 0.10,
            "lambda_single": 0.10 / medians["transport"],
            "target_fraction_combined": 0.05,
            "lambda_combined": 0.05 / medians["transport"],
        },
        "OBJECTIVE_GRADIENT_CONFLICT": classification,
        "classification_rule": "STRONG median<=0 or >=50% nonpositive; MODERATE median<0.25 or >=25% nonpositive; WEAK median<0.5; else NONE",
    }
    return global_rows + channel_rows, module_rows, weights


def snapshot_features(
    data: StageSBatchPath, snapshot: int, shell_index: np.ndarray
) -> tuple[list[str], np.ndarray]:
    raw = np.asarray(data.snapshots[snapshot], dtype=np.float32)
    names: list[str] = []
    values: list[float] = []
    for channel, channel_name in enumerate(CHANNELS):
        normalized = encoded_channel(raw[channel], channel, data.preprocessor)
        median = float(np.median(normalized))
        for label, value in (
            ("median", median),
            ("mad", float(np.median(np.abs(normalized - median)))),
            ("q01", float(np.quantile(normalized, 0.01))),
            ("q99", float(np.quantile(normalized, 0.99))),
            ("std", float(np.std(normalized))),
        ):
            names.append(f"{channel_name}:{label}")
            values.append(value)
        for shell in range(8):
            selected = normalized[..., shell_index == shell]
            names.extend((f"{channel_name}:shell_{shell + 1}_mean", f"{channel_name}:shell_{shell + 1}_variance"))
            values.extend((float(np.mean(selected)), float(np.var(selected))))
    return names, np.asarray(values, dtype=np.float64)


def ood_audit(data: StageSBatchPath, shell_index: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    matrix = []
    feature_names: list[str] | None = None
    for snapshot in range(212):
        names, values = snapshot_features(data, snapshot, shell_index)
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise RuntimeError("Stage V OOD feature order changed")
        matrix.append(values)
        if snapshot % 20 == 0:
            print(json.dumps({"ood_feature_snapshot": snapshot}), flush=True)
    features = np.stack(matrix)
    train = features[:169]
    reference_median = np.median(train, axis=0)
    reference_mad = np.median(np.abs(train - reference_median), axis=0)
    epsilon = np.maximum(1e-12, 1e-6 * np.maximum(np.abs(reference_median), 1.0))
    standardized = (features - reference_median) / (reference_mad + epsilon)
    scores = np.sqrt(np.mean(np.square(standardized), axis=1))
    rows = [
        {
            "snapshot": snapshot,
            "time": float(data.times[snapshot]),
            "split": "train" if snapshot <= 168 else "validation",
            "ood_score": float(scores[snapshot]),
        }
        for snapshot in range(212)
    ]
    train_q95, train_q99 = np.quantile(scores[:169], [0.95, 0.99])
    early = float(np.median(scores[169:183]))
    middle = float(np.median(scores[183:197]))
    late = float(np.median(scores[197:211]))
    validation_median = float(np.median(scores[169:211]))
    late_over_early = late / max(early, 1e-300)
    if validation_median > train_q99 and late_over_early > 1.10:
        shift = "STRONG"
    elif validation_median > train_q95 or late_over_early > 1.10:
        shift = "MODERATE"
    else:
        shift = "LOW"
    reference = {
        "schema_version": "stage-v-train-only-ood-reference-v1",
        "space": "P3_normalized_snapshot_state",
        "train_snapshot_indices": list(range(169)),
        "validation_used_for_fit": False,
        "feature_count": len(feature_names or []),
        "feature_names": feature_names,
        "train_feature_median": reference_median.tolist(),
        "train_feature_mad": reference_mad.tolist(),
        "epsilon": epsilon.tolist(),
        "score_definition": "sqrt(mean(((x-train_median)/(train_MAD+epsilon))^2))",
        "train_score_q95": float(train_q95),
        "train_score_q99": float(train_q99),
        "validation_score_median": validation_median,
        "validation_third_medians": {"early": early, "middle": middle, "late": late},
        "late_over_early": late_over_early,
        "SHIFT_FINDING": shift,
        "shift_rule": "STRONG validation_median>train_q99 and late/early>1.10; MODERATE either validation_median>train_q95 or late/early>1.10; else LOW",
    }
    return reference, rows, shift


def make_audit_figures(
    root: Path,
    pair_rows: Sequence[Mapping[str, Any]],
    gradient_rows: Sequence[Mapping[str, Any]],
    ood_rows: Sequence[Mapping[str, Any]],
) -> None:
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    validation = [row for row in pair_rows if row["split"] == "validation"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for axis, metric in zip(axes, ("residual_relative_l2", "shell_error", "radial_error"), strict=True):
        axis.scatter([row["plain_residual_loss"] for row in validation], [row[metric] for row in validation], s=16)
        axis.set_xlabel("Plain residual loss")
        axis.set_ylabel(metric.replace("_", " "))
    fig.tight_layout()
    fig.savefig(figures / "objective_metric_correlation.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    labels = ("direction", "shell", "radial", "transport")
    values = [[
        row["cosine_with_plain"] for row in gradient_rows
        if row["objective"] == label and row.get("scope", "aggregate") == "aggregate"
    ] for label in labels]
    axis.boxplot(values, tick_labels=labels, showfliers=True)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylabel("gradient cosine with Plain")
    fig.tight_layout()
    fig.savefig(figures / "gradient_cosine.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    axis.plot([row["snapshot"] for row in ood_rows], [row["ood_score"] for row in ood_rows])
    axis.axvline(168.5, color="red", linestyle="--", label="train/validation boundary")
    axis.set_xlabel("snapshot")
    axis.set_ylabel("train-derived OOD score")
    axis.legend()
    fig.tight_layout()
    fig.savefig(figures / "ood_vs_time.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/stage_v/objective_alignment.yaml"))
    parser.add_argument("--root", type=Path, default=Path("artifacts/stage_v"))
    parser.add_argument("--phase", choices=("pairwise", "gradient", "ood", "all"), default="all")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    stage_t_path = ROOT / config["frozen_stage_t_config"]
    stage_t, stage_s, freeze, train_pairs, _, validation_pairs = verify_frozen_contract(stage_t_path)
    if not torch.cuda.is_available():
        raise RuntimeError("Stage V audit requires CUDA; refusing silent CPU fallback")
    device = torch.device("cuda:0")
    if stage_s["runtime"]["required_device_substring"] not in torch.cuda.get_device_name(device):
        raise RuntimeError("Stage V audit GPU identity differs from the frozen contract")
    data = StageSBatchPath(ROOT / stage_s["data"]["dataset"], ROOT / stage_s["preprocessing"]["artifact"], device)
    r = np.asarray(data.handle["coords/r"], dtype=np.float64)
    _, shell_index = radial_shell_indices(r, int(stage_s["representation"]["shell_count"]))
    pair_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    ood_rows: list[dict[str, Any]] = []

    if args.phase in ("pairwise", "all"):
        checkpoint = ROOT / config["baseline"]["checkpoint"]
        model, _, _ = model_from_checkpoint(
            checkpoint, stage_s, device, expected_updates=int(config["baseline"]["optimizer_updates"])
        )
        for split, pairs in (("train", train_pairs), ("validation", validation_pairs)):
            for index, source in enumerate(pairs):
                pair_rows.append(pair_metrics(model, data, int(source), shell_index, split))
                if index % 20 == 0:
                    print(json.dumps({"pairwise_split": split, "completed": index + 1, "total": len(pairs)}), flush=True)
        correlations, misaligned = correlation_rows(pair_rows)
        write_csv(args.root / "alignment/pair_metrics.csv", pair_rows)
        write_csv(args.root / "alignment/objective_metric_correlations.csv", correlations)
        write_json(args.root / "alignment/objective_metric_alignment.json", {
            "PLAIN_L2_TRANSPORT_MISALIGNMENT": misaligned,
            "rule": "validation or combined abs Spearman(Plain,residual L2)>=0.50 while abs Spearman with shell and radial errors are both <0.30",
        })
        del model
        torch.cuda.empty_cache()

    if args.phase in ("gradient", "all"):
        model = build_frozen_model(stage_s)
        initial = torch.load(ROOT / stage_s["frozen_pairing"]["initial_state"], map_location="cpu", weights_only=True)
        model.load_state_dict(initial, strict=True)
        model.to(device)
        gradient_rows, module_rows, weights = gradient_audit(
            model=model,
            data=data,
            sources=[int(value) for value in config["gradient_audit"]["train_only_pair_subset"]],
            shell_index=shell_index,
            epsilon=float(config["gradient_audit"]["epsilon"]),
        )
        write_csv(args.root / "alignment/gradient_alignment.csv", gradient_rows)
        write_csv(args.root / "alignment/gradient_alignment_by_module.csv", module_rows)
        write_json(args.root / "loss_weights/train_only_gradient_scaling.json", weights)
        del model
        torch.cuda.empty_cache()

    if args.phase in ("ood", "all"):
        reference, ood_rows, _ = ood_audit(data, shell_index)
        write_json(args.root / "shift/train_reference_features.json", reference)
        write_csv(args.root / "shift/snapshot_ood_scores.csv", ood_rows)

    # Permit individually resumed phases to render when prior tables exist.
    if not pair_rows and (args.root / "alignment/pair_metrics.csv").exists():
        with (args.root / "alignment/pair_metrics.csv").open(newline="", encoding="utf-8") as handle:
            pair_rows = list(csv.DictReader(handle))
    if not gradient_rows and (args.root / "alignment/gradient_alignment.csv").exists():
        with (args.root / "alignment/gradient_alignment.csv").open(newline="", encoding="utf-8") as handle:
            gradient_rows = list(csv.DictReader(handle))
    if not ood_rows and (args.root / "shift/snapshot_ood_scores.csv").exists():
        with (args.root / "shift/snapshot_ood_scores.csv").open(newline="", encoding="utf-8") as handle:
            ood_rows = list(csv.DictReader(handle))
    if pair_rows and gradient_rows and ood_rows:
        make_audit_figures(args.root, pair_rows, gradient_rows, ood_rows)
    data.close()
    print(json.dumps({"stage": "V", "phase": args.phase, "status": "audit_complete"}), flush=True)


if __name__ == "__main__":
    main()
