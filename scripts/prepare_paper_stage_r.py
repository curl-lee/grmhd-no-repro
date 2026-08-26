#!/usr/bin/env python
"""Verify Stage R provenance and audit the frozen residual targets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any

import h5py
import numpy as np
import torch

from grmhd import CHANNELS
from grmhd.dataset import sha256_file
from grmhd.paper_config import load_paper_experiment_config
from grmhd.paper_data_processor import PaperDataProcessor
from grmhd.paper_stage_o import validate_stage_k_pairing
from grmhd.paper_stage_r import FROZEN_HISTORY, validate_contract_config


QUANTILES = (0.001, 0.01, 0.5, 0.99, 0.999)


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fft_masks(shape: tuple[int, int, int]) -> tuple[np.ndarray, ...]:
    frequencies = np.meshgrid(
        np.fft.fftfreq(shape[0]),
        np.fft.fftfreq(shape[1]),
        np.fft.rfftfreq(shape[2]),
        indexing="ij",
    )
    radius = np.sqrt(sum(value * value for value in frequencies))
    maximum = float(radius.max())
    return (
        radius <= maximum / 3.0,
        (radius > maximum / 3.0) & (radius <= 2.0 * maximum / 3.0),
        radius > 2.0 * maximum / 3.0,
    )


def audit_split(
    *,
    handle: h5py.File,
    preprocessor: Any,
    shells: np.ndarray,
    source_indices: list[int],
    split: str,
) -> dict[str, Any]:
    shape = tuple(int(value) for value in handle["snapshots"].shape[2:])
    voxels = math.prod(shape)
    fft_masks = _fft_masks(shape)
    shell_masks = [shells[index].astype(bool) for index in range(8)]
    pair_rows: list[dict[str, Any]] = []
    adjacent_global: list[float] = []
    adjacent_channel = {name: [] for name in CHANNELS}
    previous: np.ndarray | None = None
    radial_sum = np.zeros((8, shape[2]), dtype=np.float64)
    radial_count = np.zeros((8, shape[2]), dtype=np.int64)
    shell_sum = np.zeros((8, 8), dtype=np.float64)
    shell_square = np.zeros((8, 8), dtype=np.float64)
    shell_count = np.zeros((8, 8), dtype=np.int64)
    frequency_energy = np.zeros((8, 3), dtype=np.float64)
    channel_ratio_values = {name: [] for name in CHANNELS}
    below_count = np.zeros(8, dtype=np.int64)
    total_count = np.zeros(8, dtype=np.int64)
    with tempfile.TemporaryDirectory(prefix=f"stage_r_{split}_") as temporary:
        store_path = Path(temporary) / "residuals.float32"
        store = np.memmap(
            store_path,
            mode="w+",
            dtype=np.float32,
            shape=(len(source_indices), 8, voxels),
        )
        for pair_index, source_index in enumerate(source_indices):
            raw_input = np.asarray(handle["snapshots"][source_index], dtype=np.float32)
            raw_target = np.asarray(handle["snapshots"][source_index + 1], dtype=np.float32)
            z_input = preprocessor.encode_numpy(raw_input, channel_axis=0).astype(
                np.float32, copy=False
            )
            z_target = preprocessor.encode_numpy(raw_target, channel_axis=0).astype(
                np.float32, copy=False
            )
            residual = z_target - z_input
            store[pair_index] = residual.reshape(8, -1)
            residual64 = residual.astype(np.float64)
            state64 = z_input.astype(np.float64)
            residual_norm = float(np.linalg.norm(residual64.reshape(-1)))
            state_norm = float(np.linalg.norm(state64.reshape(-1)))
            row = {
                "split": split,
                "source_snapshot": source_index,
                "target_snapshot": source_index + 1,
                "residual_norm": residual_norm,
                "state_norm": state_norm,
                "residual_state_norm_ratio": residual_norm / max(state_norm, 1.0e-30),
            }
            if previous is not None:
                denominator = np.linalg.norm(previous.reshape(-1)) * residual_norm
                cosine = float(
                    np.sum(previous.astype(np.float64) * residual64)
                    / max(denominator, 1.0e-30)
                )
                adjacent_global.append(cosine)
                row["previous_pair_residual_cosine"] = cosine
                for channel, name in enumerate(CHANNELS):
                    left = previous[channel].astype(np.float64)
                    right = residual64[channel]
                    channel_cosine = float(
                        np.sum(left * right)
                        / max(np.linalg.norm(left) * np.linalg.norm(right), 1.0e-30)
                    )
                    adjacent_channel[name].append(channel_cosine)
            else:
                row["previous_pair_residual_cosine"] = None
            pair_rows.append(row)
            previous = residual.copy()
            for channel in range(8):
                channel_residual = residual64[channel]
                channel_ratio_values[CHANNELS[channel]].append(
                    float(
                        np.linalg.norm(channel_residual)
                        / max(np.linalg.norm(state64[channel]), 1.0e-30)
                    )
                )
                radial_sum[channel] += channel_residual.sum(axis=(0, 1))
                radial_count[channel] += shape[0] * shape[1]
                tolerance = 8.0 * np.finfo(np.float32).eps * np.maximum(
                    np.abs(state64[channel]), 1.0
                )
                below_count[channel] += int(
                    np.count_nonzero(np.abs(channel_residual) <= tolerance)
                )
                total_count[channel] += channel_residual.size
                for shell_index, mask in enumerate(shell_masks):
                    values = channel_residual[mask]
                    shell_sum[channel, shell_index] += float(values.sum())
                    shell_square[channel, shell_index] += float(np.square(values).sum())
                    shell_count[channel, shell_index] += values.size
                spectrum = np.fft.rfftn(channel_residual)
                power = np.square(np.abs(spectrum))
                for band, mask in enumerate(fft_masks):
                    frequency_energy[channel, band] += float(power[mask].sum())
        store.flush()
        rows = []
        for channel, name in enumerate(CHANNELS):
            values = np.asarray(store[:, channel]).reshape(-1)
            quantiles = np.quantile(values, QUANTILES)
            channel_ratios = channel_ratio_values[name]
            total_energy = max(float(frequency_energy[channel].sum()), 1.0e-30)
            shell_variance = []
            for shell_index in range(8):
                count = max(int(shell_count[channel, shell_index]), 1)
                mean = shell_sum[channel, shell_index] / count
                variance = shell_square[channel, shell_index] / count - mean * mean
                shell_variance.append(max(float(variance), 0.0))
            rows.append(
                {
                    "split": split,
                    "channel": name,
                    "count": int(values.size),
                    "mean": float(np.mean(values, dtype=np.float64)),
                    "std": float(np.std(values, dtype=np.float64)),
                    "rms": float(np.sqrt(np.mean(np.square(values), dtype=np.float64))),
                    "q001": float(quantiles[0]),
                    "q01": float(quantiles[1]),
                    "q50": float(quantiles[2]),
                    "q99": float(quantiles[3]),
                    "q999": float(quantiles[4]),
                    "maximum_magnitude": float(np.max(np.abs(values))),
                    "residual_state_norm_ratio_mean": float(np.mean(channel_ratios)),
                    "residual_state_norm_ratio_median": float(np.median(channel_ratios)),
                    "adjacent_pair_residual_cosine_mean": (
                        None
                        if not adjacent_channel[name]
                        else float(np.mean(adjacent_channel[name]))
                    ),
                    "fraction_below_numerical_tolerance": float(
                        below_count[channel] / max(total_count[channel], 1)
                    ),
                    "shell_residual_variance": shell_variance,
                    "radial_residual_profile": (
                        radial_sum[channel] / np.maximum(radial_count[channel], 1)
                    ).tolist(),
                    "low_k_energy_fraction": float(frequency_energy[channel, 0] / total_energy),
                    "mid_k_energy_fraction": float(frequency_energy[channel, 1] / total_energy),
                    "high_k_energy_fraction": float(frequency_energy[channel, 2] / total_energy),
                }
            )
        del store
    return {
        "split": split,
        "pairs": len(source_indices),
        "source_indices": source_indices,
        "target_indices": [value + 1 for value in source_indices],
        "rows": rows,
        "pair_rows": pair_rows,
        "adjacent_pair_residual_cosine_global": {
            "mean": float(np.mean(adjacent_global)) if adjacent_global else None,
            "median": float(np.median(adjacent_global)) if adjacent_global else None,
            "count": len(adjacent_global),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/paper_reduced100/stage_r_localno_p3_residual_plain.yaml"),
    )
    parser.add_argument(
        "--initial-state",
        type=Path,
        default=Path("outputs/paper_reduced100/stage_k/shared_localno_initial_state.pt"),
    )
    parser.add_argument(
        "--pair-order",
        type=Path,
        default=Path("outputs/paper_reduced100/stage_g/epoch_pair_order.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper_reduced100/stage_r"),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    if git(root, "branch", "--show-current") != "main":
        raise RuntimeError("Stage R preparation is running on the wrong branch")
    if git(root, "-C", "external/neuraloperator", "status", "--short"):
        raise RuntimeError("Pinned upstream worktree is dirty")
    config_path = args.config if args.config.is_absolute() else root / args.config
    initial_path = args.initial_state if args.initial_state.is_absolute() else root / args.initial_state
    pair_order_path = args.pair_order if args.pair_order.is_absolute() else root / args.pair_order
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    config = load_paper_experiment_config(config_path, project_root=root)
    validate_contract_config(config.values)
    pairing = validate_stage_k_pairing(
        config,
        initial_state_path=initial_path,
        pair_order_path=pair_order_path,
        epochs=30,
    )
    processor = PaperDataProcessor.from_config(config)
    h5_path = config.resolve_path(config.values["protocol"]["dataset"])
    with h5py.File(h5_path, "r") as handle:
        shape = list(handle["snapshots"].shape)
        channels = [value.decode("utf-8") for value in handle["channels"][...]]
        train = audit_split(
            handle=handle,
            preprocessor=processor.preprocessor,
            shells=processor.shells.squeeze(0).cpu().numpy(),
            source_indices=list(range(11, 90)),
            split="train",
        )
        # The contract/config is frozen before this confirmatory read.
        validation = audit_split(
            handle=handle,
            preprocessor=processor.preprocessor,
            shells=processor.shells.squeeze(0).cpu().numpy(),
            source_indices=list(range(91, 110)),
            split="validation_confirmatory",
        )
    audit = {
        "schema_version": "paper-stage-r-residual-target-audit-v1",
        "prediction_mode": "normalized_residual",
        "target_definition": "delta_z_true = z_t1 - z_t",
        "reconstruction": "z_hat_t1 = z_t + r_theta",
        "persistence": "predicted_residual_equals_zero",
        "residual_scale": 1.0,
        "clipping": False,
        "train_only_contract_freeze": True,
        "validation_used_to_modify_contract": False,
        "train": train,
        "validation_confirmatory": validation,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "residual_target_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    flat_rows = []
    for record in (train, validation):
        for row in record["rows"]:
            flat_rows.append(
                {
                    key: json.dumps(value) if isinstance(value, list) else value
                    for key, value in row.items()
                }
            )
    write_csv(output / "residual_target_audit.csv", flat_rows)
    (output / "residual_target_audit.md").write_text(
        "\n".join(
            [
                "# Stage R residual-target audit",
                "",
                "- Contract: `delta_z_true = z_t1 - z_t`; reconstruction uses an identity skip.",
                "- Persistence is exactly the zero predicted residual.",
                "- Train calibration source: 79 transitions 11->12 through 89->90.",
                "- Validation: 19 confirmatory transitions, not used to alter the contract.",
                "- Residual scale: `1.0`; clipping/renormalization: disabled.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    values = config.values
    manifest = {
        "schema_version": "paper-stage-r-run-manifest-v1",
        "status": "provenance_contract_and_residual_audit_passed",
        "classification": "adapted_residual_contract_model_pilot",
        "project": {
            "branch": git(root, "branch", "--show-current"),
            "commit": git(root, "rev-parse", "HEAD"),
            "stage_q_base": "8bc4ca4",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "numpy": np.__version__,
        },
        "upstream_commit": git(root, "-C", "external/neuraloperator", "rev-parse", "HEAD"),
        "data": {"shape": shape, "channels": channels},
        "checksums": {
            "dataset": values["provenance"]["dataset"]["sha256"],
            "manifest": values["provenance"]["manifest"]["sha256"],
            "p3_statistics": values["preprocessing"]["stats_checksum"],
            "p3_config": values["preprocessing"]["prototype_config_checksum"],
            "shells": values["provenance"]["artifacts"]["shells"]["sha256"],
            "stage_k_architecture": values["provenance"]["stage_o_frozen"]["stage_k_config"]["sha256"],
            "stage_k_initial_state_file": values["provenance"]["stage_o_frozen"]["stage_k_initial_state_file"]["sha256"],
            "pair_order": pairing["pair_order_sha256"],
            "stage_m_gate": values["provenance"]["stage_o_frozen"]["stage_m_gate"]["sha256"],
            **{name: record["sha256"] for name, record in values["provenance"]["stage_r_frozen"].items()},
            "stage_r_config": sha256_file(config_path),
            "residual_target_audit": sha256_file(output / "residual_target_audit.json"),
        },
        "pairing": pairing,
        "contract": dict(values["prediction"]),
        "loss": dict(values["loss"]),
        "data_split": {
            "train_snapshots": [11, 91],
            "train_pairs": 79,
            "validation_snapshots": [91, 111],
            "validation_pairs": 19,
            "dropped_transition": [90, 91],
            "test": None,
        },
        "training_contract": {
            "epochs": 30,
            "batch_size": 1,
            "gradient_accumulation": 4,
            "optimizer": "Adam",
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-4,
            "warmup_epochs": 2,
            "cosine_min_learning_rate": 1.0e-6,
            "gradient_clip_norm": 1.0,
            "mixed_precision": False,
            "early_stopping": False,
            "expected_microbatches": 2370,
            "expected_optimizer_updates": 600,
        },
        "evaluation": dict(values["evaluation"]),
        "historical_status_unchanged": FROZEN_HISTORY,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "run_manifest.md").write_text(
        "\n".join(
            [
                "# Stage R run manifest",
                "",
                "- Provenance, P3 train-only statistics, architecture, initial state, and pair order: passed.",
                "- Only factor changed from Stage O: normalized residual target plus identity reconstruction.",
                f"- Shared initial tensor hash: `{pairing['initial_state_tensor_sha256']}`.",
                f"- Parameters: `{pairing['parameter_count']}`.",
                "- Next gate: real 64^3 RTX 5070 residual-contract preflight.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": manifest["status"], "pairing": pairing}, indent=2))


if __name__ == "__main__":
    main()
