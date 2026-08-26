#!/usr/bin/env python
"""Create a controlled engineering comparison of the two Stage F smoke runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping


def _global(metrics: Mapping[str, Any], name: str) -> float:
    return float(metrics["one_step"]["oracle_aware"]["metrics"][name]["global_relative_l2"])


def smoke_row(metrics: Mapping[str, Any]) -> dict[str, Any]:
    reloads = metrics["checkpoint_reload"]
    return {
        "mode": metrics["mode"],
        "status": metrics["status"],
        "model_parameter_count": metrics["parameter_count"],
        "initial_state_hash": metrics["initial_state_hash"],
        "protocol_checksum": metrics["protocol_checksum"],
        "epochs": metrics["epochs"],
        "train_batches": metrics["train_batches"],
        "runtime_seconds": metrics["runtime"]["wall_seconds"],
        "total_wall_seconds": metrics["runtime"]["total_wall_seconds"],
        "samples_per_second": metrics["runtime"]["samples_per_second"],
        "device": metrics["runtime"]["device"],
        "peak_allocated_mib": metrics["runtime"]["peak_allocated_mib"],
        "peak_reserved_mib": metrics["runtime"]["peak_reserved_mib"],
        "validation_normalized_global_relative_l2": _global(metrics, "E_norm"),
        "model_to_oracle_global_relative_l2": _global(metrics, "E_model_oracle"),
        "model_to_raw_global_relative_l2": _global(metrics, "E_model_raw"),
        "oracle_floor_global_relative_l2": _global(metrics, "E_oracle_raw"),
        "rollout3_finite": metrics["rollout3"]["finite"],
        "rollout3_rho_press_positive": metrics["rollout3"]["rho_press_positive"],
        "rollout3_no_double_transform": metrics["rollout3"]["no_double_transform"],
        "best_checkpoint_reload": all(
            reloads["best"][key]
            for key in (
                "strict_model_reload",
                "optimizer_reload",
                "scheduler_reload",
                "prediction_parity",
                "metadata_validated",
            )
        ),
        "last_checkpoint_reload": all(
            reloads["last"][key]
            for key in (
                "strict_model_reload",
                "optimizer_reload",
                "scheduler_reload",
                "prediction_parity",
                "metadata_validated",
            )
        ),
        "clipping_fraction": metrics["runtime"]["clipping_fraction"],
        "h1_value_ratio_max": (
            max(metrics["h1_monitoring"]["value_ratios"])
            if metrics["h1_monitoring"]["value_ratios"]
            else None
        ),
        "h1_gradient_ratio_max": (
            max(metrics["h1_monitoring"]["gradient_ratios"])
            if metrics["h1_monitoring"]["gradient_ratios"]
            else None
        ),
        "enabled_loss_components": (
            ["base", "h1", "roi", "bounds", "envelope", "dissipation"]
            if metrics["mode"] == "full"
            else ["plain_l2"]
        ),
        "disabled_loss_components": metrics["disabled_components"],
    }


def build_comparison(full: Mapping[str, Any], plain: Mapping[str, Any]) -> dict[str, Any]:
    rows = [smoke_row(full), smoke_row(plain)]
    controlled = {
        "parameter_count_equal": rows[0]["model_parameter_count"]
        == rows[1]["model_parameter_count"],
        "initial_state_hash_equal": rows[0]["initial_state_hash"]
        == rows[1]["initial_state_hash"],
        "protocol_checksum_equal": rows[0]["protocol_checksum"]
        == rows[1]["protocol_checksum"],
        "epochs_equal": rows[0]["epochs"] == rows[1]["epochs"],
        "train_batches_equal": rows[0]["train_batches"] == rows[1]["train_batches"],
    }
    return {
        "schema_version": "paper-stage-f-smoke-comparison-v1",
        "status": "passed" if all(controlled.values()) else "failed",
        "engineering_smoke_only": True,
        "scientific_comparison": False,
        "interpretation": (
            "The two-epoch smoke validates controlled pipeline execution only; "
            "it cannot establish that Full is better or worse than Plain."
        ),
        "controlled_identity": controlled,
        "runs": rows,
    }


def write_comparison(payload: Mapping[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "smoke_comparison.json").write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = list(payload["runs"])
    csv_rows = []
    for row in rows:
        csv_rows.append(
            {
                key: json.dumps(value) if isinstance(value, list) else value
                for key, value in row.items()
            }
        )
    with (output_dir / "smoke_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    lines = [
        "# Stage F controlled smoke comparison",
        "",
        "This is a two-epoch engineering smoke comparison. It does not establish scientific",
        "superiority or inferiority of either loss.",
        "",
        "| field | Full | Plain |",
        "| --- | ---: | ---: |",
    ]
    for key in (
        "status",
        "model_parameter_count",
        "initial_state_hash",
        "epochs",
        "train_batches",
        "runtime_seconds",
        "peak_allocated_mib",
        "validation_normalized_global_relative_l2",
        "model_to_oracle_global_relative_l2",
        "model_to_raw_global_relative_l2",
        "oracle_floor_global_relative_l2",
        "rollout3_finite",
        "rollout3_rho_press_positive",
        "best_checkpoint_reload",
        "last_checkpoint_reload",
        "clipping_fraction",
        "h1_value_ratio_max",
        "h1_gradient_ratio_max",
        "enabled_loss_components",
        "disabled_loss_components",
    ):
        lines.append(f"| {key} | `{rows[0][key]}` | `{rows[1][key]}` |")
    (output_dir / "smoke_comparison.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage-f-dir",
        type=Path,
        default=Path("outputs/paper_reduced100/stage_f"),
    )
    args = parser.parse_args()
    root = args.stage_f_dir
    full = json.loads((root / "smoke_full_fno/metrics.json").read_text(encoding="utf-8"))
    plain = json.loads((root / "smoke_plain_l2/metrics.json").read_text(encoding="utf-8"))
    payload = build_comparison(full, plain)
    write_comparison(payload, root)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
