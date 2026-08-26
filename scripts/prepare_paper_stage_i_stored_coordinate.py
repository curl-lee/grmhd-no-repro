#!/usr/bin/env python
"""Validate frozen controls and authorize the Stage I stored-H1 Run C."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import torch

from grmhd.dataset import sha256_file
from grmhd.paper_checkpoint import (
    load_paper_checkpoint,
    stage_i_selected_h1_definition,
)
from grmhd.paper_config import build_paper_model, load_paper_experiment_config
from grmhd.paper_stage_g import tensor_state_sha256, validate_epoch_pair_order
from grmhd.paper_stage_h import validate_frozen_input_manifest


EXPECTED_BRANCH = "main"
EXPECTED_UPSTREAM = "86a8bc7812a31b42c4f7895693cf4ac11521c066"
EXPECTED_SHARED = "02d83278d5e74a27c83130e99b488ca18eae5477dad9732a4552b694056b6cb1"
EXPECTED_ORDER = "5e1e361ea75e01ffd667822d8bfe870d70c5bc69570d52cdf806e637b2df2b52"
RUN_REASON = "stored_spherical_coordinate_H1_adaptation_diagnosis"


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_recorded_run(
    root: Path,
    manifest: Mapping[str, Any],
    *,
    run_name: str,
    directory_name: str,
    config_path: Path,
    expected_mode: str,
    expected_epochs: tuple[int, int],
) -> dict[str, Any]:
    record = manifest["runs"][run_name]
    recorded_hashes = record["file_sha256"]
    output = root / "outputs/paper_reduced100/stage_i" / directory_name
    observed = {
        name: sha256_file(output / name) for name in recorded_hashes
    }
    if observed != recorded_hashes:
        changed = sorted(name for name in observed if observed[name] != recorded_hashes[name])
        raise RuntimeError(f"Frozen {run_name} files changed: {changed}")

    config = load_paper_experiment_config(config_path, project_root=root)
    reloads: dict[str, Any] = {}
    for checkpoint_name, epoch in zip(
        ("best_validation_l2", "last"), expected_epochs, strict=True
    ):
        model = build_paper_model(config)
        loaded = load_paper_checkpoint(
            output / checkpoint_name,
            config=config,
            model=model,
            expected_config_checksum=sha256_file(config_path),
        )
        stage_i = loaded.metadata.get("stage_i")
        if not isinstance(stage_i, Mapping):
            raise RuntimeError(f"{run_name} {checkpoint_name} Stage I metadata missing")
        if loaded.epoch != epoch or stage_i["diagnostic_h1_mode"] != expected_mode:
            raise RuntimeError(f"{run_name} {checkpoint_name} strict reload mismatch")
        reloads[checkpoint_name] = {
            "epoch": loaded.epoch,
            "model_strict_reload": True,
            "metadata_validated": True,
        }
    return {"file_sha256": observed, "strict_reload": reloads}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs/paper_reduced100/stage_i"
    if git(root, "branch", "--show-current") != EXPECTED_BRANCH:
        raise RuntimeError("Unexpected project branch")
    if git(root, "status", "--short"):
        raise RuntimeError("Run C manifest requires a clean project worktree")
    upstream = git(root, "-C", "external/neuraloperator", "rev-parse", "HEAD")
    if upstream != EXPECTED_UPSTREAM:
        raise RuntimeError("Pinned upstream commit changed")
    if git(root, "-C", "external/neuraloperator", "status", "--short"):
        raise RuntimeError("Pinned upstream worktree is dirty")

    frozen_path = root / "outputs/paper_reduced100/stage_h/frozen_inputs.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    validate_frozen_input_manifest(root, frozen)
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    shared_path = root / "outputs/paper_reduced100/stage_g/shared_initial_state.pt"
    order_path = root / "outputs/paper_reduced100/stage_g/epoch_pair_order.json"
    shared_state = torch.load(shared_path, map_location="cpu", weights_only=True)
    shared_hash = tensor_state_sha256(shared_state)
    order_hash = sha256_file(order_path)
    if shared_hash != EXPECTED_SHARED or order_hash != EXPECTED_ORDER:
        raise RuntimeError("Frozen Stage G pairing hashes changed")
    order_payload = json.loads(order_path.read_text(encoding="utf-8"))
    orders = validate_epoch_pair_order(order_payload)
    if len(orders) != 30 or any(sorted(order) != list(range(79)) for order in orders):
        raise RuntimeError("Frozen pair order is not 30 complete permutations")
    if order_payload["validation_shuffle"] is not False:
        raise RuntimeError("Validation order unexpectedly shuffles")

    run_a = validate_recorded_run(
        root,
        manifest,
        run_name="no_h1",
        directory_name="no_h1",
        config_path=root / "configs/paper_reduced100/extensions/no_h1_control.yaml",
        expected_mode="no_h1",
        expected_epochs=(25, 30),
    )
    run_b = validate_recorded_run(
        root,
        manifest,
        run_name="unit_index",
        directory_name="unit_index_h1",
        config_path=root / "configs/paper_reduced100/extensions/unit_index_h1.yaml",
        expected_mode="unit_index",
        expected_epochs=(27, 30),
    )

    config_diff_path = output / "config_diff.json"
    config_diff = json.loads(config_diff_path.read_text(encoding="utf-8"))
    if config_diff["status"] != "passed":
        raise RuntimeError("Four-way config/pairing audit failed")
    preflight_path = output / "stored_coordinate_preflight.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    records = {record["mode"]: record for record in preflight["extensions"]}
    if preflight["status"] != "passed" or set(records) != {
        "stored_coordinate_volume_proxy"
    }:
        raise RuntimeError("Stored-coordinate-only GPU preflight did not pass")
    mode_checks = records["stored_coordinate_volume_proxy"]["mode_checks"]
    required_checks = (
        "volume_weights_finite",
        "volume_weights_nonnegative",
        "volume_weights_normalized",
        "batch_reduction_independent",
        "radial_spacing_nonuniform",
        "all_coordinate_spacings_positive",
    )
    if not all(mode_checks[name] is True for name in required_checks):
        raise RuntimeError("Stored-coordinate geometry preflight gate failed")

    run_c_dir = output / "stored_coordinate_volume_h1"
    if run_c_dir.exists() and any(run_c_dir.iterdir()):
        raise RuntimeError("Run C output directory is not empty")
    config_path = root / (
        "configs/paper_reduced100/extensions/stored_coordinate_volume_h1.yaml"
    )
    config = load_paper_experiment_config(config_path, project_root=root)
    reproduction_metadata = config.values["reproduction_metadata"]
    expected_classification = {
        "reproduction_level": "diagnostic_extension",
        "paper_faithful_full": False,
        "extension_reason": RUN_REASON,
    }
    if any(
        reproduction_metadata.get(key) != value
        for key, value in expected_classification.items()
    ):
        raise RuntimeError("Run C classification changed")
    model = build_paper_model(config)
    model.load_state_dict(shared_state, strict=True)
    if sum(parameter.numel() for parameter in model.parameters()) != 331832:
        raise RuntimeError("Run C model parameter count changed")
    if tensor_state_sha256(model.state_dict()) != EXPECTED_SHARED:
        raise RuntimeError("Run C shared initial state load changed")
    selected_definition = stage_i_selected_h1_definition(config)
    if selected_definition is None:
        raise RuntimeError("Run C selected H1 definition missing")

    provenance = config.values["provenance"]
    manifest.update(
        {
            "schema_version": "paper-stage-i-run-manifest-v3",
            "status": "ready_for_stored_coordinate_training",
            "project_commit": git(root, "rev-parse", "HEAD"),
            "project_branch": EXPECTED_BRANCH,
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
        }
    )
    manifest.setdefault("checksums", {}).update(
        {
            "dataset": provenance["dataset"],
            "preprocessing": provenance["preprocessing"],
            "stage_d_artifacts": provenance["artifacts"],
            "stage_e_loss_contract": provenance["stage_e_loss_contract"],
            "stage_h_frozen_inputs": sha256_file(frozen_path),
            "stored_coordinate_config": sha256_file(config_path),
            "stored_coordinate_preflight": sha256_file(preflight_path),
            "config_diff": sha256_file(config_diff_path),
            "shared_initial_state_tensor": shared_hash,
            "shared_initial_state_file": sha256_file(shared_path),
            "pair_order_file": order_hash,
        }
    )
    manifest["model"] = {**config.values["model"], "parameter_count": 331832}
    manifest["split"] = {
        "train_pairs": 79,
        "validation_pairs": 19,
        "dropped_transition": [90, 91],
        "validation_shuffle": False,
        "pair_order_epochs": 30,
    }
    manifest["training"] = {
        "epochs": 30,
        "microbatches_per_epoch": 79,
        "microbatches_total": 2370,
        "batch_size": 1,
        "gradient_accumulation": 4,
        "optimizer_steps_per_epoch": 20,
        "optimizer_steps_total": 600,
        "final_accumulation_count": 3,
        "optimizer": config.values["optimizer"],
        "scheduler": config.values["scheduler"],
        "gradient_clip_norm": 1.0,
        "mixed_precision": False,
        "early_stopping": False,
        "seed": 42,
    }
    manifest["runs"]["no_h1"].update(run_a)
    manifest["runs"]["unit_index"].update(run_b)
    manifest["runs"]["stored_coordinate_volume_proxy"] = {
        "status": "authorized_preflight_passed_not_started",
        "experiment": "stage_i_stored_coordinate_volume_h1",
        "classification": {
            "reproduction_level": "diagnostic_extension",
            "paper_faithful_full": False,
            "extension_reason": RUN_REASON,
            "comparison_parent": "stage_g_paper_adapted_full",
            "paired_controls": [
                "stage_g_full",
                "stage_g_plain",
                "stage_i_no_h1",
                "stage_i_unit_index_h1",
            ],
        },
        "config": str(config_path.relative_to(root)),
        "selected_h1_definition": selected_definition,
        "diagnostic_current_upstream_h1_trained": False,
        "diagnostic_unit_index_h1_trained": False,
        "preflight": str(preflight_path.relative_to(root)),
        "output_directory_exists": run_c_dir.exists(),
    }
    write_json(manifest_path, manifest)
    lines = [
        "# Stage I Run C manifest",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Project commit: `{manifest['project_commit']}`",
        f"- Upstream: `{upstream}`",
        f"- GPU: `{manifest['environment']['gpu']}`",
        f"- Config SHA256: `{manifest['checksums']['stored_coordinate_config']}`",
        f"- Coordinate SHA256: `{selected_definition['coordinate_combined_sha256']}`",
        f"- Volume-weight SHA256: `{selected_definition['volume_proxy']['weight_sha256']}`",
        f"- Shared-state tensor SHA256: `{shared_hash}`",
        f"- Pair-order SHA256: `{order_hash}`",
        "- Run A/B file hashes and best/last model metadata: `validated`",
        "- Run C preflight: `passed`",
        "- Run C budget: 30 epochs, 2370 microbatches, 600 optimizer steps",
        "- Training started during manifest preparation: `false`",
    ]
    (output / "run_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(manifest["runs"]["stored_coordinate_volume_proxy"], indent=2))


if __name__ == "__main__":
    main()
