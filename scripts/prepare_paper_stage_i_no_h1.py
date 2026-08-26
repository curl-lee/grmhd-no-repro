#!/usr/bin/env python
"""Freeze the Stage I no-H1 Run A manifest and config-difference audit."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import torch

from grmhd.dataset import sha256_file
from grmhd.paper_config import (
    build_paper_model,
    load_paper_experiment_config,
    recursive_config_differences,
    stage_i_difference_is_allowed,
)
from grmhd.paper_stage_g import tensor_state_sha256, validate_epoch_pair_order
from grmhd.paper_stage_h import validate_frozen_input_manifest


EXPECTED_UPSTREAM = "86a8bc7812a31b42c4f7895693cf4ac11521c066"
EXPECTED_SHARED = "02d83278d5e74a27c83130e99b488ca18eae5477dad9732a4552b694056b6cb1"
EXPECTED_ORDER = "5e1e361ea75e01ffd667822d8bfe870d70c5bc69570d52cdf806e637b2df2b52"
RUN_REASON = "isolate_H1_contribution_under_spherical_grid_adaptation"


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def relative(root: Path, path: Path) -> str:
    return str(path.resolve().relative_to(root))


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs/paper_reduced100/stage_i"
    output.mkdir(parents=True, exist_ok=True)
    branch = git(root, "branch", "--show-current")
    if branch != "main":
        raise RuntimeError(f"Unexpected branch: {branch}")
    if git(root, "status", "--short"):
        raise RuntimeError("Stage I manifest requires a clean project worktree")
    upstream = git(root, "-C", "external/neuraloperator", "rev-parse", "HEAD")
    if upstream != EXPECTED_UPSTREAM:
        raise RuntimeError("Pinned upstream commit changed")
    if git(root, "-C", "external/neuraloperator", "status", "--short"):
        raise RuntimeError("Pinned upstream worktree is dirty")

    stage_g_path = root / "outputs/paper_reduced100/stage_g/run_manifest.json"
    stage_g = json.loads(stage_g_path.read_text(encoding="utf-8"))
    frozen_path = root / "outputs/paper_reduced100/stage_h/frozen_inputs.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    validate_frozen_input_manifest(root, frozen)

    full_path = root / "configs/paper_reduced100/full_fno_proxy.yaml"
    no_h1_path = (
        root / "configs/paper_reduced100/extensions/no_h1_control.yaml"
    )
    full = load_paper_experiment_config(full_path, project_root=root)
    no_h1 = load_paper_experiment_config(no_h1_path, project_root=root)
    differences = recursive_config_differences(full.as_dict(), no_h1.as_dict())
    unexpected = [
        item
        for item in differences
        if not stage_i_difference_is_allowed(str(item["path"]))
    ]
    non_h1_paths = [
        item["path"]
        for item in differences
        if not str(item["path"]).startswith("loss.h1.")
        and str(item["path"])
        not in {
            "experiment_name",
            "output_dir",
            "checkpoint_prefix",
            "reproduction_metadata.reproduction_level",
            "reproduction_metadata.paper_faithful_full",
            "reproduction_metadata.extension_reason",
        }
    ]
    if unexpected or non_h1_paths:
        raise RuntimeError(
            f"No-H1 config has unpaired differences: {unexpected or non_h1_paths}"
        )
    diff_payload = {
        "schema_version": "paper-stage-i-no-h1-config-diff-v1",
        "status": "passed",
        "reference_config": relative(root, full_path),
        "diagnostic_config": relative(root, no_h1_path),
        "reference_config_sha256": sha256_file(full_path),
        "diagnostic_config_sha256": sha256_file(no_h1_path),
        "allowed_differences": [
            "experiment_name",
            "output_dir",
            "extension metadata (including checkpoint_prefix)",
            "loss.h1.*",
        ],
        "differences": differences,
        "unexpected_differences": [],
        "non_h1_training_objective_differences": [],
        "selected_h1_training_coefficient": 0.0,
        "non_h1_fields_exact": True,
        "no_h1_is_plain_l2": False,
    }
    write_json(output / "no_h1_config_diff.json", diff_payload)
    diff_lines = [
        "# Stage I Run A no-H1 config difference",
        "",
        "- Status: `passed`",
        f"- Full config: `{relative(root, full_path)}`",
        f"- no-H1 config: `{relative(root, no_h1_path)}`",
        "- Selected H1 training contribution: `0`",
        "- All non-H1 training-objective fields: exact",
        "- no-H1 is Plain L2: `false`",
        "- `checkpoint_prefix` is classified as extension output metadata.",
        "",
        "## Differences",
        "",
    ]
    diff_lines.extend(f"- `{item['path']}`" for item in differences)
    (output / "no_h1_config_diff.md").write_text(
        "\n".join(diff_lines) + "\n",
        encoding="utf-8",
    )

    shared_path = (
        root / "outputs/paper_reduced100/stage_g/shared_initial_state.pt"
    )
    order_path = (
        root / "outputs/paper_reduced100/stage_g/epoch_pair_order.json"
    )
    shared_state = torch.load(
        shared_path,
        map_location="cpu",
        weights_only=True,
    )
    shared_hash = tensor_state_sha256(shared_state)
    order_hash = sha256_file(order_path)
    if shared_hash != EXPECTED_SHARED or order_hash != EXPECTED_ORDER:
        raise RuntimeError("Frozen Stage G pairing hashes changed")
    order_payload = json.loads(order_path.read_text(encoding="utf-8"))
    epoch_orders = validate_epoch_pair_order(order_payload)
    if len(epoch_orders) != 30 or any(
        sorted(order) != list(range(79)) for order in epoch_orders
    ):
        raise RuntimeError("Frozen pair order is not 30 complete train permutations")

    model = build_paper_model(no_h1)
    model.load_state_dict(shared_state, strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 331832:
        raise RuntimeError("Stage I model parameter count changed")
    if tensor_state_sha256(model.state_dict()) != EXPECTED_SHARED:
        raise RuntimeError("Stage I model did not strictly load the shared state")

    preflight = json.loads(
        (output / "gpu_preflight.json").read_text(encoding="utf-8")
    )
    if preflight["status"] != "passed":
        raise RuntimeError("The frozen three-way Stage I preflight did not pass")
    stage_h_decision = git(
        root,
        "log",
        "-1",
        "--format=%H",
        "--",
        "docs/PAPER_STAGE_H_DECISION.md",
    )
    provenance = no_h1.values["provenance"]
    manifest = {
        "schema_version": "paper-stage-i-no-h1-run-manifest-v1",
        "status": "ready_for_single_no_h1_preflight",
        "experiment": "stage_i_no_h1",
        "project_commit": git(root, "rev-parse", "HEAD"),
        "project_branch": branch,
        "upstream_commit": upstream,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "numpy": np.__version__,
            "gpu": preflight["environment"]["device_name"],
            "gpu_capability": preflight["environment"]["capability"],
        },
        "classification": {
            "reproduction_level": "diagnostic_extension",
            "paper_faithful_full": False,
            "extension_reason": RUN_REASON,
            "comparison_parent": "stage_g_paper_adapted_full",
            "no_h1_is_plain_l2": False,
        },
        "checksums": {
            "dataset": {
                "path": provenance["dataset"]["path"],
                "sha256": provenance["dataset"]["sha256"],
            },
            "preprocessing": provenance["preprocessing"],
            "stage_d_artifacts": provenance["artifacts"],
            "stage_e_loss_contract": provenance["stage_e_loss_contract"],
            "no_h1_config": sha256_file(no_h1_path),
            "shared_initial_state_tensor": shared_hash,
            "shared_initial_state_file": sha256_file(shared_path),
            "pair_order_file": order_hash,
            "stage_g_run_manifest": sha256_file(stage_g_path),
            "stage_h_frozen_inputs": sha256_file(frozen_path),
        },
        "stage_h_decision_commit": stage_h_decision,
        "model": {
            "parameter_count": parameter_count,
            **no_h1.values["model"],
        },
        "split": {
            "train_pairs": 79,
            "validation_pairs": 19,
            "train_snapshots": [11, 91],
            "validation_snapshots": [91, 111],
            "dropped_transition": [90, 91],
            "validation_shuffle": False,
            "pair_order_epochs": len(epoch_orders),
        },
        "training": {
            "epochs": 30,
            "microbatches_per_epoch": 79,
            "microbatches_total": 2370,
            "batch_size": 1,
            "gradient_accumulation": 4,
            "optimizer_steps_per_epoch": 20,
            "optimizer_steps_total": 600,
            "final_accumulation_count": 3,
            "optimizer": no_h1.values["optimizer"],
            "scheduler": no_h1.values["scheduler"],
            "gradient_clip_norm": 1.0,
            "mixed_precision": False,
            "early_stopping": False,
            "seed": 42,
        },
        "loss": {
            "selected_h1_training_coefficient": 0.0,
            "diagnostic_current_upstream_h1_trained": False,
            "non_h1_contract": {
                name: no_h1.values["loss"][name]
                for name in ("base", "roi", "bounds", "envelope", "dissipation")
            },
        },
        "evaluation_protocol": {
            "checkpoint_selection": (
                "validation normalized arithmetic-average relative L2"
            ),
            "validation_pairs": 19,
            "gt_rollout_steps": 19,
            "no_gt_rollout_steps": 100,
            "selected_steps": [1, 5, 10, 19, 25, 50, 75, 100],
            "frozen_scoring_source": (
                "src/grmhd/paper_stage_g_evaluation.py"
            ),
        },
        "frozen_stage_g_checkpoint_hashes_validated": True,
        "config_diff_audit": {
            "status": "passed",
            "path": (
                "outputs/paper_reduced100/stage_i/no_h1_config_diff.json"
            ),
            "non_h1_fields_exact": True,
        },
    }
    write_json(output / "run_manifest.json", manifest)
    lines = [
        "# Stage I Run A manifest",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Project commit: `{manifest['project_commit']}`",
        f"- Upstream: `{upstream}`",
        f"- GPU: `{manifest['environment']['gpu']}`",
        f"- Config SHA256: `{manifest['checksums']['no_h1_config']}`",
        f"- Shared state tensor SHA256: `{shared_hash}`",
        f"- Pair-order SHA256: `{order_hash}`",
        f"- Stage H decision commit: `{stage_h_decision}`",
        "- Experiment: `stage_i_no_h1`",
        "- Reproduction level: `diagnostic_extension`",
        "- Paper-faithful Full: `false`",
        f"- Extension reason: `{RUN_REASON}`",
        "- Comparison parent: `stage_g_paper_adapted_full`",
        "- Budget: 30 epochs, 2370 microbatches, 600 optimizer steps",
        "- Training started during preparation: `false`",
    ]
    (output / "run_manifest.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
