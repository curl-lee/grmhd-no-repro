#!/usr/bin/env python
"""Descriptive Stage G log correlations for the post-hoc Stage H audit."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr

from grmhd.paper_stage_h import (
    OUTPUT_ROOT,
    freeze_or_validate_inputs,
    project_root_from_file,
    write_csv,
)


def correlation(
    name: str,
    first: Sequence[float],
    second: Sequence[float],
    *,
    source: str,
    limitation: str = "",
) -> dict[str, Any]:
    if len(first) != len(second):
        raise ValueError("correlation series lengths differ")
    if len(first) < 3:
        return {
            "comparison": name,
            "n": len(first),
            "pearson_r": None,
            "pearson_p": None,
            "spearman_rho": None,
            "spearman_p": None,
            "status": "not_computed",
            "source": source,
            "limitation": limitation or "fewer than three aligned observations",
        }
    x = np.asarray(first, dtype=np.float64)
    y = np.asarray(second, dtype=np.float64)
    if np.std(x) == 0 or np.std(y) == 0:
        return {
            "comparison": name,
            "n": len(first),
            "pearson_r": None,
            "pearson_p": None,
            "spearman_rho": None,
            "spearman_p": None,
            "status": "undefined_constant_series",
            "source": source,
            "limitation": limitation or "at least one series is constant",
        }
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    return {
        "comparison": name,
        "n": len(first),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "status": "descriptive_only",
        "source": source,
        "limitation": limitation,
    }


def read_optimizer_log(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    root = project_root_from_file()
    output_root = root / OUTPUT_ROOT
    freeze_or_validate_inputs(root, output_root)
    full_dir = root / "outputs/paper_reduced100/stage_g/pilot30_full_fno"
    epochs = []
    for epoch in range(1, 31):
        payload = json.loads(
            (full_dir / f"validation_epoch_{epoch:03d}.json").read_text(
                encoding="utf-8"
            )
        )
        training = payload["training"]
        epochs.append(
            {
                "epoch_one_based": epoch,
                "epoch_zero_based": int(training["epoch"]),
                "h1_base_value_mean": float(
                    training["h1_to_base_value_ratio"]["mean"]
                ),
                "h1_base_gradient_mean": float(
                    training["h1_to_base_gradient_ratio"]["mean"]
                ),
                "clipping_fraction": float(training["clipping_fraction"]),
                "validation_average_l2": float(
                    training[
                        "validation_normalized_arithmetic_average_relative_l2"
                    ]
                ),
                "validation_global_l2": float(
                    training["validation_normalized_global_relative_l2"]
                ),
            }
        )
    optimizer = read_optimizer_log(full_dir / "optimizer_log.csv")
    correlations = [
        correlation(
            "h1_base_value_vs_validation_average_l2",
            [row["h1_base_value_mean"] for row in epochs],
            [row["validation_average_l2"] for row in epochs],
            source="30 Full epoch summaries",
            limitation="n=30; descriptive post-hoc association, not causal",
        ),
        correlation(
            "h1_base_gradient_vs_validation_average_l2",
            [row["h1_base_gradient_mean"] for row in epochs],
            [row["validation_average_l2"] for row in epochs],
            source="30 Full epoch summaries",
            limitation="n=30; descriptive post-hoc association, not causal",
        ),
        correlation(
            "h1_base_value_vs_validation_global_l2",
            [row["h1_base_value_mean"] for row in epochs],
            [row["validation_global_l2"] for row in epochs],
            source="30 Full epoch summaries",
            limitation="n=30; descriptive post-hoc association, not causal",
        ),
        correlation(
            "h1_base_value_vs_clipping_fraction",
            [row["h1_base_value_mean"] for row in epochs],
            [row["clipping_fraction"] for row in epochs],
            source="30 Full epoch summaries",
            limitation="clipping fraction is exactly one in every epoch",
        ),
        correlation(
            "preclip_total_gradient_vs_parameter_update_norm",
            [float(row["total_gradient_norm_before_clip"]) for row in optimizer],
            [float(row["parameter_update_norm"]) for row in optimizer],
            source="600 Full optimizer-step log rows",
            limitation=(
                "learning-rate schedule and clipping are uncontrolled confounders; "
                "descriptive only"
            ),
        ),
        correlation(
            "h1_base_gradient_ratio_vs_parameter_update_norm",
            [float(row["h1_to_base_gradient_ratio"]) for row in optimizer],
            [float(row["parameter_update_norm"]) for row in optimizer],
            source="600 Full optimizer-step log rows",
            limitation=(
                "learning-rate schedule and clipping are uncontrolled confounders; "
                "descriptive only"
            ),
        ),
        correlation(
            "h1_base_value_vs_outer_shell_error",
            [],
            [],
            source="Stage G saved logs",
            limitation=(
                "Stage G did not save an epoch-aligned outer-shell error series; "
                "the final comparison has only checkpoint-level aggregate scores"
            ),
        ),
        correlation(
            "h1_base_value_vs_artifact_flags",
            [],
            [],
            source="Stage G saved logs",
            limitation=(
                "artifact flags are available only for selected final rollout steps, "
                "not all 30 epochs"
            ),
        ),
        correlation(
            "inner_shell_h1_vs_morphology_score",
            [],
            [],
            source="Stage H fixed checkpoint audit and Stage G comparison",
            limitation=(
                "no epoch-aligned morphology series exists; two final model scores "
                "are insufficient for correlation"
            ),
        ),
    ]
    write_csv(output_root / "h1_training_log_correlation.csv", correlations)
    lines = [
        "# Stage H training-log correlations",
        "",
        "All coefficients are descriptive post-hoc statistics. They do not identify",
        "causal effects and were not used to fit or tune any training parameter.",
        "",
        "| Comparison | n | Pearson r | Spearman rho | Status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in correlations:
        pearson = "NA" if row["pearson_r"] is None else f"{row['pearson_r']:.6g}"
        spearman = (
            "NA" if row["spearman_rho"] is None else f"{row['spearman_rho']:.6g}"
        )
        lines.append(
            f"| {row['comparison']} | {row['n']} | {pearson} | "
            f"{spearman} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "Unavailable epoch-aligned outer-shell, artifact, and morphology series",
            "are reported as not computed rather than inferred from final-checkpoint",
            "summaries. Full clipping fraction is constant at 1.0, so its correlation",
            "with H1/base is mathematically undefined.",
            "",
        ]
    )
    (output_root / "h1_training_log_correlation.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    print({"status": "completed", "correlations": correlations})


if __name__ == "__main__":
    main()
