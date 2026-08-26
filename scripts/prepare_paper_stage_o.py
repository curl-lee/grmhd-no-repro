#!/usr/bin/env python
"""Verify frozen Stage O provenance/P3 parity and write the run manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch

from grmhd.dataset import sha256_file
from grmhd.paper_config import load_paper_experiment_config
from grmhd.paper_stage_o import reproduce_p3_roundtrip, validate_stage_k_pairing


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/paper_reduced100/stage_o_localno_p3_plain.yaml"),
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
        default=Path("outputs/paper_reduced100/stage_o"),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.project_root.resolve()
    if git(root, "branch", "--show-current") != "main":
        raise RuntimeError("Stage O preparation is running on the wrong branch")
    upstream = git(root, "-C", "external/neuraloperator", "rev-parse", "HEAD")
    upstream_status = git(root, "-C", "external/neuraloperator", "status", "--short")
    if upstream != "86a8bc7812a31b42c4f7895693cf4ac11521c066" or upstream_status:
        raise RuntimeError("Pinned upstream changed or is dirty")

    config_path = args.config if args.config.is_absolute() else root / args.config
    initial_path = (
        args.initial_state if args.initial_state.is_absolute() else root / args.initial_state
    )
    order_path = args.pair_order if args.pair_order.is_absolute() else root / args.pair_order
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    config = load_paper_experiment_config(config_path, project_root=root)
    pairing = validate_stage_k_pairing(
        config,
        initial_state_path=initial_path,
        pair_order_path=order_path,
        epochs=30,
    )
    parity = reproduce_p3_roundtrip(config)
    values = config.values
    frozen = values["provenance"]["stage_o_frozen"]
    manifest = {
        "schema_version": "paper-stage-o-run-manifest-v1",
        "status": "provenance_and_p3_parity_passed",
        "classification": "adapted_transform_model_pilot",
        "project": {
            "branch": git(root, "branch", "--show-current"),
            "commit": git(root, "rev-parse", "HEAD"),
            "stage_n_base": "2c762a47f7235b6b608a00f780d8f2cc18df6627",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "numpy": np.__version__,
        },
        "upstream_commit": upstream,
        "preprocessing": {
            "mode": "stage_n_p3",
            "prototype": "P3",
            "prototype_name": "COMBINED_PROTOTYPE_V1",
            "classification": "diagnostic_transform_prototype",
            "stage_o_training_classification": "adapted_transform_model_pilot",
            "canonical_replacement": False,
            "statistics_fit": "isolated_train_only_snapshots_11_90",
            "channel_policies": {
                "Bcc1": "canonical",
                "Bcc2": "no_softclip",
                "Bcc3": "no_softclip",
                "rho": "canonical",
                "press": "canonical",
                "vel1": "canonical",
                "vel2": "canonical",
                "vel3": "no_softclip",
            },
        },
        "checksums": {
            "dataset": values["provenance"]["dataset"]["sha256"],
            "manifest": values["provenance"]["manifest"]["sha256"],
            "p3_statistics": values["preprocessing"]["stats_checksum"],
            "p3_config": values["preprocessing"]["prototype_config_checksum"],
            "p3_metadata": values["preprocessing"]["prototype_metadata_checksum"],
            "canonical_preprocessing": values["provenance"][
                "canonical_preprocessing"
            ]["sha256"],
            "shells": values["provenance"]["artifacts"]["shells"]["sha256"],
            "stage_k_architecture_config": frozen["stage_k_config"]["sha256"],
            "stage_k_initial_state_file": frozen["stage_k_initial_state_file"][
                "sha256"
            ],
            "shared_initial_state_tensor": pairing["initial_state_tensor_sha256"],
            "pair_order_file": pairing["pair_order_sha256"],
            "stage_m_gate": frozen["stage_m_gate"]["sha256"],
            "stage_n_config": frozen["stage_n_config"]["sha256"],
            "stage_n_decision": frozen["stage_n_decision"]["sha256"],
            "stage_n_prototype_manifest": frozen["prototype_manifest"]["sha256"],
            "stage_o_config": sha256_file(config_path),
        },
        "initial_state": {
            "path": str(initial_path.relative_to(root)),
            "file_sha256": pairing["initial_state_file_sha256"],
            "tensor_state_sha256": pairing["initial_state_tensor_sha256"],
            "strict_load": pairing["strict_load"],
        },
        "pairing": pairing,
        "p3_roundtrip_reproduction": parity,
        "data_split": {
            "source_snapshots": [11, 111],
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
            "seed": 42,
            "loss": "strict_plain_l2_p3_normalized_space",
            "expected_microbatches": 2370,
            "expected_optimizer_updates": 600,
        },
        "evaluation": {
            "legacy_detector": True,
            "oracle_conditioned_gate": True,
            "oracle_conditioned_gate_version": "stage_m_v1",
        },
        "historical_status_unchanged": {
            "Stage K": "C. TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE",
            "Stage L": "3. MIXED_OVERALL",
            "Stage M": "4. MIXED_TRANSPORT_AND_PREPROCESSING_FAILURE",
            "Stage N": "P3 ready; mixed operator-response failure",
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "run_manifest.md").write_text(
        "\n".join(
            [
                "# Stage O run manifest",
                "",
                "- Classification: `adapted_transform_model_pilot`.",
                "- P3 is a frozen diagnostic prototype, not a canonical replacement.",
                "- P3 statistics are isolated train-only statistics for snapshots 11..90.",
                "- Full train/validation P3 round-trip parity: passed.",
                f"- Initial tensor state: `{pairing['initial_state_tensor_sha256']}`.",
                f"- Pair order: `{pairing['pair_order_sha256']}`.",
                f"- Parameters: `{pairing['parameter_count']}`.",
                "- Authorized next gate: real 64^3 RTX 5070 preflight.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": manifest["status"], "pairing": pairing}, indent=2))


if __name__ == "__main__":
    main()
