#!/usr/bin/env python
"""Freeze the Stage K LocalNO initial state and controlled run manifest."""

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
from grmhd.paper_config import (
    build_paper_model,
    load_paper_experiment_config,
    model_tensor_state_sha256,
    recursive_config_differences,
)
from grmhd.paper_protocol import EXPECTED_UPSTREAM_COMMIT, PaperReduced100Protocol
from grmhd.paper_stage_g import load_epoch_pair_order, tensor_state_sha256
from grmhd.upstream_adapters import GRMHDNextStepDataset


EXPECTED_PAIR_ORDER_SHA256 = (
    "5e1e361ea75e01ffd667822d8bfe870d70c5bc69570d52cdf806e637b2df2b52"
)


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "configs/paper_reduced100/stage_k_localno_differential_plain.yaml"
        ),
    )
    parser.add_argument(
        "--fno-plain-config",
        type=Path,
        default=Path("configs/paper_reduced100/plain_l2_fno.yaml"),
    )
    parser.add_argument(
        "--pair-order",
        type=Path,
        default=Path("outputs/paper_reduced100/stage_g/epoch_pair_order.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper_reduced100/stage_k"),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.project_root.resolve()
    output_dir = (
        args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    )
    config_path = args.config if args.config.is_absolute() else root / args.config
    fno_path = (
        args.fno_plain_config
        if args.fno_plain_config.is_absolute()
        else root / args.fno_plain_config
    )
    pair_path = (
        args.pair_order if args.pair_order.is_absolute() else root / args.pair_order
    )
    preflight_path = output_dir / "localno_preflight.json"
    if not preflight_path.is_file():
        raise FileNotFoundError("Run the Stage K GPU preflight before preparation")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "passed":
        raise RuntimeError("Saved Stage K GPU preflight did not pass")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA became unavailable before Stage K preparation")
    device_name = torch.cuda.get_device_name(0)
    if device_name != "NVIDIA GeForce RTX 5070":
        raise RuntimeError(f"Stage K expected RTX 5070, found {device_name!r}")

    config = load_paper_experiment_config(config_path, project_root=root)
    fno_config = load_paper_experiment_config(fno_path, project_root=root)
    if not config.stage_k or fno_config.stage_k:
        raise ValueError("Stage K/Stage G Plain architecture identities are invalid")
    upstream = root / "external/neuraloperator"
    if git(upstream, "rev-parse", "HEAD") != EXPECTED_UPSTREAM_COMMIT:
        raise ValueError("Pinned upstream commit changed")
    if git(upstream, "status", "--short"):
        raise ValueError("Pinned upstream worktree is dirty")

    pair_payload, pair_orders = load_epoch_pair_order(pair_path)
    pair_sha256 = sha256_file(pair_path)
    if pair_sha256 != EXPECTED_PAIR_ORDER_SHA256:
        raise ValueError("Frozen Stage G pair-order checksum changed")
    if len(pair_orders) != 30 or any(len(order) != 79 for order in pair_orders):
        raise ValueError("Frozen pair order does not contain 30 complete 79-pair epochs")

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    localno = build_paper_model(config)
    localno_state = {
        name: value.detach().cpu().clone()
        for name, value in localno.state_dict().items()
        if torch.is_tensor(value)
    }
    initial_hash = tensor_state_sha256(localno_state)
    if initial_hash != model_tensor_state_sha256(localno):
        raise RuntimeError("LocalNO initial tensor-state hash is inconsistent")
    state_path = output_dir / "shared_localno_initial_state.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(localno_state, state_path)
    strict_probe = build_paper_model(config)
    strict_probe.load_state_dict(localno_state, strict=True)
    if model_tensor_state_sha256(strict_probe) != initial_hash:
        raise RuntimeError("LocalNO initial state strict reload changed tensor identity")

    fno = build_paper_model(fno_config)
    localno_count = sum(parameter.numel() for parameter in localno.parameters())
    fno_count = sum(parameter.numel() for parameter in fno.parameters())
    protocol = PaperReduced100Protocol.from_yaml(
        root / "configs/data/paper_reduced100.yaml", project_root=root
    )
    datasets = protocol.make_datasets()
    train_pairs = GRMHDNextStepDataset(datasets["train"])
    validation_pairs = GRMHDNextStepDataset(datasets["validation"])
    differences = recursive_config_differences(
        fno_config.as_dict(), config.as_dict()
    )
    manifest = {
        "schema_version": "paper-stage-k-run-manifest-v1",
        "status": "ready",
        "classification": "adapted_method_reproduction",
        "not_3d_disco": True,
        "not_exact_paper_reproduction": True,
        "project_git_commit": git(root, "rev-parse", "HEAD"),
        "project_branch": git(root, "branch", "--show-current"),
        "project_dirty_during_preparation": bool(git(root, "status", "--short")),
        "upstream_commit": git(upstream, "rev-parse", "HEAD"),
        "upstream_worktree_clean": True,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "numpy": np.__version__,
            "gpu_name": device_name,
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
        },
        "gpu_preflight": {
            key: preflight[key]
            for key in (
                "status",
                "forward_seconds",
                "loss_seconds",
                "backward_seconds",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
                "state_hash_unchanged",
                "differential_module_count",
                "spectral_module_count",
                "disco_module_count",
            )
        },
        "checksums": {
            "dataset": config.values["provenance"]["dataset"]["sha256"],
            "manifest": config.values["provenance"]["manifest"]["sha256"],
            "preprocessing": config.values["provenance"]["preprocessing"]["sha256"],
            "stage_d_artifacts": {
                name: record["sha256"]
                for name, record in config.values["provenance"]["artifacts"].items()
            },
            "stage_e_loss_contract": config.values["provenance"]
            ["stage_e_loss_contract"]["sha256"],
            "stage_k_config": sha256_file(config_path),
            "stage_g_plain_config": sha256_file(fno_path),
            "pair_order_file": pair_sha256,
        },
        "initial_state": {
            "path": str(state_path.relative_to(root)),
            "seed": 42,
            "tensor_state_sha256": initial_hash,
            "strict_reload_sha256": model_tensor_state_sha256(strict_probe),
            "constructor_config": dict(config.values["model"]),
            "parameters": [
                {
                    "name": name,
                    "shape": list(parameter.shape),
                    "numel": parameter.numel(),
                }
                for name, parameter in localno.named_parameters()
            ],
        },
        "model": {
            **config.values["model"],
            "localno_parameter_count": localno_count,
            "fno_plain_parameter_count": fno_count,
            "localno_to_fno_parameter_ratio": localno_count / fno_count,
        },
        "fairness": {
            "same_seed": True,
            "same_split": True,
            "same_preprocessing": True,
            "same_pair_order": True,
            "same_optimizer": True,
            "same_scheduler": True,
            "same_batch_and_accumulation": True,
            "same_epochs": True,
            "same_checkpoint_selector": True,
            "same_evaluation": True,
            "same_tensor_initial_state": False,
            "same_tensor_initial_state_reason": "different_model_architectures",
            "config_differences": differences,
        },
        "protocol": {
            "name": config.values["protocol"]["name"],
            "train_pairs": len(train_pairs),
            "validation_pairs": len(validation_pairs),
            "train_snapshots": config.values["protocol"]["train_snapshots"],
            "validation_snapshots": config.values["protocol"]["validation_snapshots"],
            "dropped_transition": config.values["protocol"]["dropped_transition"],
            "validation_shuffle": pair_payload["validation_shuffle"],
            "thermal": config.values["thermal"],
            "coordinates": config.values["coordinates"],
            "radial_mode": config.values["representation"]["radial"]["selected_mode"],
        },
        "run": {
            "epochs": 30,
            "batch_size": 1,
            "gradient_accumulation": 4,
            "microbatches_per_epoch": 79,
            "microbatches_total": 2370,
            "optimizer_steps_per_epoch": 20,
            "optimizer_steps_total": 600,
            "final_accumulation_count": 3,
            "seed": 42,
            "optimizer": "Adam",
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-4,
            "scheduler": "warmup_cosine_step_per_epoch",
            "warmup_epochs": 2,
            "cosine_min_lr": 1.0e-6,
            "gradient_clip_norm": 1.0,
            "mixed_precision": False,
            "eval_interval": 1,
            "early_stopping": "disabled",
            "checkpoint_selection_metric": (
                "validation normalized per-channel relative L2 arithmetic average"
            ),
        },
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Stage K controlled run manifest",
        "",
        "- Classification: `adapted_method_reproduction`",
        "- Backbone: `3D differential LocalNO`",
        "- DISCO integral: `disabled`",
        "- Exact paper reproduction: `false`",
        f"- Project commit: `{manifest['project_git_commit']}`",
        f"- Upstream commit: `{manifest['upstream_commit']}`",
        f"- GPU: `{device_name}`",
        f"- Dataset SHA256: `{manifest['checksums']['dataset']}`",
        f"- Preprocessing SHA256: `{manifest['checksums']['preprocessing']}`",
        f"- Pair-order SHA256: `{pair_sha256}`",
        f"- LocalNO initial tensor-state SHA256: `{initial_hash}`",
        f"- LocalNO/FNO Plain parameters: `{localno_count}/{fno_count}` "
        f"(ratio `{localno_count / fno_count:.6f}`)",
        "- Epochs/train pairs/validation pairs: `30/79/19`",
        "- Accumulation: `4` (`20` updates/epoch, final group `3`)",
        "- Total microbatches/updates: `2370/600`",
        "- Early stopping: `disabled`",
        "- Selector: validation normalized per-channel relative-L2 arithmetic average",
        "",
        "The architecture differs from the Stage G FNO proxy, so fairness freezes "
        "the seed and complete data/optimization/evaluation protocol rather than "
        "claiming a shared tensor initialization.",
    ]
    (output_dir / "run_manifest.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
