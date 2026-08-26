#!/usr/bin/env python
"""Validate frozen controls and prepare the Stage I Run B manifest."""

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
from grmhd.paper_config import (
    build_paper_model,
    load_paper_experiment_config,
)
from grmhd.paper_stage_g import tensor_state_sha256, validate_epoch_pair_order
from grmhd.paper_stage_h import validate_frozen_input_manifest
from grmhd.paper_trainer import (
    build_paper_optimizer,
    build_warmup_cosine_scheduler,
)


EXPECTED_BRANCH = "main"
EXPECTED_UPSTREAM = "86a8bc7812a31b42c4f7895693cf4ac11521c066"
EXPECTED_SHARED = "02d83278d5e74a27c83130e99b488ca18eae5477dad9732a4552b694056b6cb1"
EXPECTED_ORDER = "5e1e361ea75e01ffd667822d8bfe870d70c5bc69570d52cdf806e637b2df2b52"
RUN_REASON = "isolate_unit_cube_spacing_amplification"
RUN_A_HASHES = {
    "best_validation_l2/manifest.pt": (
        "1201e63323636422c1ff3d98e9784eae00b39773d55f7435a92469af9a465b28"
    ),
    "best_validation_l2/optimizer.pt": (
        "144336d1f84fe059a954f2c436d107d6cabeedaf65ceb270218998ea6dee37c4"
    ),
    "best_validation_l2/paper_grmhd_metadata.json": (
        "687e67a50d4e37625564118a6822e37b6eb8cbdfcb660b4fbe3eb344bb5787c5"
    ),
    "best_validation_l2/paper_metadata.pkl": (
        "7340d5df6df5f2a3e254840e5ef44fa2c735c4b3af45c7fa939521f4722093a8"
    ),
    "best_validation_l2/paper_state_dict.pt": (
        "25c2d3850a358d1cc9d025ab5c5a5c98a1e5cb4a9fb857c8771924b61cb224b7"
    ),
    "best_validation_l2/scheduler.pt": (
        "142d0fe5d84dfe283042744143373b222cd24ac8432ce34c64574230fe7ae422"
    ),
    "last/manifest.pt": (
        "1b2198015cb8ff67ddd3ed28e06635294212316f2c344364e5c7fc362df239c3"
    ),
    "last/optimizer.pt": (
        "476c682513c4a4a0634853395e5325e2bbe164db9834588daa3ae414d6e978e3"
    ),
    "last/paper_grmhd_metadata.json": (
        "ba87e84fed5bc313771e8a14a6c1f44567ab4023e44d1c11609e51d7a418ef53"
    ),
    "last/paper_metadata.pkl": (
        "7340d5df6df5f2a3e254840e5ef44fa2c735c4b3af45c7fa939521f4722093a8"
    ),
    "last/paper_state_dict.pt": (
        "5a2cba35ad4457697d92372ae2bd22f4cc7e58bb38c266676ebd42ac952d4b5e"
    ),
    "last/scheduler.pt": (
        "665575ab94754602b9c3d35819df94058f5343ff98b25c41fe98d7ef287fe78c"
    ),
    "evaluation_summary.json": (
        "afbfb751ceec47be1e2a04d89b0090f49158094f5dff6c9c1bc9c1bbc150accb"
    ),
    "metrics.json": (
        "c1eced89c21d2c8edc8f26f7cec3c21ce024be812c35fd1f7e2328a53ae7924e"
    ),
    "gt_rollout.json": (
        "e5ead9646ad2cc26f54d25b3fe4262af160163492578ffdfcd5a3a0ab4b505ab"
    ),
    "no_gt_rollout.json": (
        "e302840241b3c179d614bba01faf099d598a66387cdd283257a56367db617fd9"
    ),
}


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


def strict_reload_run_a(
    *,
    root: Path,
    config_path: Path,
    shared_hash: str,
    order_hash: str,
) -> dict[str, Any]:
    config = load_paper_experiment_config(config_path, project_root=root)
    output = root / "outputs/paper_reduced100/stage_i/no_h1"
    reloaded: dict[str, Any] = {}
    for name, expected_epoch in (("best_validation_l2", 25), ("last", 30)):
        model = build_paper_model(config)
        optimizer = build_paper_optimizer(
            model,
            learning_rate=config.values["optimizer"]["learning_rate"],
            weight_decay=config.values["optimizer"]["weight_decay"],
        )
        scheduler = build_warmup_cosine_scheduler(
            optimizer,
            total_epochs=30,
            warmup_epochs=2,
            min_learning_rate=config.values["scheduler"]["min_learning_rate"],
        )
        loaded = load_paper_checkpoint(
            output / name,
            config=config,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_config_checksum=sha256_file(config_path),
        )
        stage_g = loaded.metadata.get("stage_g")
        stage_i = loaded.metadata.get("stage_i")
        if not isinstance(stage_g, Mapping) or not isinstance(stage_i, Mapping):
            raise RuntimeError(f"Run A {name} pairing metadata is missing")
        checks = {
            "epoch": loaded.epoch == expected_epoch,
            "model": loaded.model is model,
            "optimizer": loaded.optimizer is optimizer,
            "scheduler": loaded.scheduler is scheduler,
            "shared_state": (
                stage_g["shared_initial_state_sha256"] == shared_hash
            ),
            "pair_order": stage_g["pair_order_sha256"] == order_hash,
            "mode": stage_i["diagnostic_h1_mode"] == "no_h1",
            "selected_h1_zero": (
                stage_i["selected_h1_training_coefficient"] == 0.0
            ),
            "upstream_diagnostic_only": (
                stage_i["diagnostic_current_upstream_h1_trained"] is False
            ),
        }
        if not all(checks.values()):
            raise RuntimeError(f"Run A {name} strict reload failed: {checks}")
        reloaded[name] = {
            "epoch": loaded.epoch,
            "optimizer_step": stage_g["optimizer_step"],
            "checks": checks,
        }
    return reloaded


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs/paper_reduced100/stage_i"
    branch = git(root, "branch", "--show-current")
    if branch != EXPECTED_BRANCH:
        raise RuntimeError(f"Unexpected branch: {branch}")
    if git(root, "status", "--short"):
        raise RuntimeError("Run B manifest requires a clean project worktree")
    upstream = git(root, "-C", "external/neuraloperator", "rev-parse", "HEAD")
    if upstream != EXPECTED_UPSTREAM:
        raise RuntimeError("Pinned upstream commit changed")
    if git(root, "-C", "external/neuraloperator", "status", "--short"):
        raise RuntimeError("Pinned upstream worktree is dirty")

    frozen_path = root / "outputs/paper_reduced100/stage_h/frozen_inputs.json"
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    validate_frozen_input_manifest(root, frozen)

    shared_path = (
        root / "outputs/paper_reduced100/stage_g/shared_initial_state.pt"
    )
    order_path = (
        root / "outputs/paper_reduced100/stage_g/epoch_pair_order.json"
    )
    shared_state = torch.load(shared_path, map_location="cpu", weights_only=True)
    shared_hash = tensor_state_sha256(shared_state)
    order_hash = sha256_file(order_path)
    if shared_hash != EXPECTED_SHARED or order_hash != EXPECTED_ORDER:
        raise RuntimeError("Frozen Stage G pairing hashes changed")
    orders = validate_epoch_pair_order(
        json.loads(order_path.read_text(encoding="utf-8"))
    )
    if len(orders) != 30:
        raise RuntimeError("Frozen pair order does not cover 30 epochs")

    run_a_dir = output / "no_h1"
    observed_run_a_hashes = {
        name: sha256_file(run_a_dir / name) for name in RUN_A_HASHES
    }
    if observed_run_a_hashes != RUN_A_HASHES:
        changed = {
            name: {
                "expected": RUN_A_HASHES[name],
                "actual": observed_run_a_hashes[name],
            }
            for name in RUN_A_HASHES
            if observed_run_a_hashes[name] != RUN_A_HASHES[name]
        }
        raise RuntimeError(f"Run A frozen hashes changed: {changed}")

    no_h1_path = (
        root / "configs/paper_reduced100/extensions/no_h1_control.yaml"
    )
    run_a_reload = strict_reload_run_a(
        root=root,
        config_path=no_h1_path,
        shared_hash=shared_hash,
        order_hash=order_hash,
    )
    run_a_evaluation = json.loads(
        (run_a_dir / "evaluation_summary.json").read_text(encoding="utf-8")
    )
    run_a_metrics = json.loads(
        (run_a_dir / "metrics.json").read_text(encoding="utf-8")
    )
    normalized = run_a_evaluation["best"]["validation"]["model"]["metrics"][
        "E_norm"
    ]
    expected_values = {
        "best_epoch": run_a_metrics["best_epoch"] == 25,
        "average": abs(normalized["arithmetic_average"] - 0.4828240411753495)
        < 1e-15,
        "global": abs(normalized["global_relative_l2"] - 0.3864974195988429)
        < 1e-15,
        "nonfinite": run_a_metrics["runtime"]["nonfinite_count"] == 0,
    }
    if not all(expected_values.values()):
        raise RuntimeError(
            f"Run A evaluation values changed: {expected_values}"
        )
    training_doc = (
        root / "docs/PAPER_STAGE_I_TRAINING.md"
    ).read_text(encoding="utf-8")
    for expected_text in ("0.482824", "0.386497", "0.14355", "0.98877"):
        if expected_text not in training_doc:
            raise RuntimeError(
                f"Run A training document omitted {expected_text}"
            )

    stored_output = output / "stored_coordinate_volume_h1"
    if stored_output.exists():
        raise RuntimeError("Stored-coordinate training output already exists")
    config_diff = json.loads(
        (output / "config_diff.json").read_text(encoding="utf-8")
    )
    if config_diff["status"] != "passed":
        raise RuntimeError("Four-way Stage I config parity did not pass")
    preflight = json.loads(
        (output / "unit_index_preflight.json").read_text(encoding="utf-8")
    )
    if preflight["status"] != "passed":
        raise RuntimeError("Run B unit-index GPU preflight did not pass")
    unit_records = {
        record["mode"]: record for record in preflight["extensions"]
    }
    if set(unit_records) != {"unit_index"}:
        raise RuntimeError("Run B preflight was not unit-index-only")
    unit_record = unit_records["unit_index"]
    if not unit_record["mode_checks"]["stage_h_4096_parity"]:
        raise RuntimeError("Run B unit-index spacing parity failed")

    unit_path = (
        root / "configs/paper_reduced100/extensions/unit_index_h1.yaml"
    )
    unit = load_paper_experiment_config(unit_path, project_root=root)
    model = build_paper_model(unit)
    model.load_state_dict(shared_state, strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 331832:
        raise RuntimeError("Run B model parameter count changed")

    provenance = unit.values["provenance"]
    manifest = {
        "schema_version": "paper-stage-i-run-manifest-v2",
        "status": "ready_for_unit_index_training",
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
        "checksums": {
            "dataset": provenance["dataset"],
            "preprocessing": provenance["preprocessing"],
            "stage_d_artifacts": provenance["artifacts"],
            "stage_e_loss_contract": provenance["stage_e_loss_contract"],
            "stage_h_frozen_inputs": sha256_file(frozen_path),
            "full_config": sha256_file(
                root / "configs/paper_reduced100/full_fno_proxy.yaml"
            ),
            "no_h1_config": sha256_file(no_h1_path),
            "unit_index_config": sha256_file(unit_path),
            "stored_coordinate_config": sha256_file(
                root
                / "configs/paper_reduced100/extensions/"
                "stored_coordinate_volume_h1.yaml"
            ),
            "shared_initial_state_tensor": shared_hash,
            "shared_initial_state_file": sha256_file(shared_path),
            "pair_order_file": order_hash,
            "config_diff": sha256_file(output / "config_diff.json"),
            "unit_index_preflight": sha256_file(
                output / "unit_index_preflight.json"
            ),
        },
        "model": {
            **unit.values["model"],
            "parameter_count": parameter_count,
        },
        "split": {
            "train_pairs": 79,
            "validation_pairs": 19,
            "dropped_transition": [90, 91],
            "validation_shuffle": False,
            "pair_order_epochs": len(orders),
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
            "optimizer": unit.values["optimizer"],
            "scheduler": unit.values["scheduler"],
            "gradient_clip_norm": 1.0,
            "mixed_precision": False,
            "early_stopping": False,
            "seed": 42,
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
        "runs": {
            "no_h1": {
                "status": "completed_and_frozen",
                "experiment": "stage_i_no_h1",
                "best_epoch": 25,
                "best_validation_average": normalized[
                    "arithmetic_average"
                ],
                "best_validation_global": normalized["global_relative_l2"],
                "strict_reload": run_a_reload,
                "file_sha256": observed_run_a_hashes,
            },
            "unit_index": {
                "status": "authorized_preflight_passed_not_started",
                "experiment": "stage_i_unit_index_h1",
                "classification": {
                    "reproduction_level": "diagnostic_extension",
                    "paper_faithful_full": False,
                    "extension_reason": RUN_REASON,
                    "comparison_parent": "stage_g_paper_adapted_full",
                    "paired_controls": [
                        "stage_g_full",
                        "stage_g_plain",
                        "stage_i_no_h1",
                    ],
                },
                "config": relative(root, unit_path),
                "selected_h1_definition": (
                    stage_i_selected_h1_definition(unit)
                ),
                "diagnostic_current_upstream_h1_trained": False,
                "preflight": (
                    "outputs/paper_reduced100/stage_i/"
                    "unit_index_preflight.json"
                ),
            },
            "stored_coordinate_volume_proxy": {
                "status": "not_executed",
                "output_directory_exists": False,
            },
        },
    }
    write_json(output / "run_manifest.json", manifest)
    lines = [
        "# Stage I Run B manifest",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Project commit: `{manifest['project_commit']}`",
        f"- Upstream: `{upstream}`",
        f"- GPU: `{manifest['environment']['gpu']}`",
        f"- Unit config SHA256: `{manifest['checksums']['unit_index_config']}`",
        f"- Shared state tensor SHA256: `{shared_hash}`",
        f"- Pair-order SHA256: `{order_hash}`",
        "- Model parameters: `331832`",
        "- Run A best/last strict reload: `passed/passed`",
        "- Stage G/H frozen hash validation: `passed`",
        "- Unit-index preflight: `passed`",
        "- Current/unit-index H1 raw ratio: `4096` within tolerance",
        "- Run B: 30 epochs, 2370 microbatches, 600 optimizer steps",
        "- Stored-coordinate training: `not executed`",
        "- Training started during manifest preparation: `false`",
    ]
    (output / "run_manifest.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
