#!/usr/bin/env python
"""Create the frozen Stage G shared state, epoch order, and run manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from grmhd.dataset import sha256_file
from grmhd.paper_config import (
    build_paper_model,
    load_paper_experiment_config,
    model_tensor_state_sha256,
    paired_config_audit,
)
from grmhd.paper_data_processor import PaperDataProcessor
from grmhd.paper_protocol import PaperReduced100Protocol
from grmhd.paper_stage_g import (
    STAGE_G_ACCUMULATION,
    STAGE_G_EPOCHS,
    STAGE_G_SEED,
    generate_epoch_pair_order,
    tensor_state_sha256,
)
from grmhd.upstream_adapters import GRMHDNextStepDataset


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper_reduced100/stage_g"),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.project_root.resolve()
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to prepare a CPU Stage G run")
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(device)
    if "RTX 5070" not in device_name:
        raise RuntimeError(f"Stage G expected RTX 5070, found {device_name!r}")

    full_path = root / "configs/paper_reduced100/full_fno_proxy.yaml"
    plain_path = root / "configs/paper_reduced100/plain_l2_fno.yaml"
    full = load_paper_experiment_config(full_path, project_root=root)
    plain = load_paper_experiment_config(plain_path, project_root=root)
    audit = paired_config_audit(full, plain)
    if audit["status"] != "passed" or audit["unexpected_differences"]:
        raise RuntimeError("Stage G paired config audit failed")
    if git(root / "external/neuraloperator", "status", "--short"):
        raise RuntimeError("Pinned upstream worktree is dirty")

    output_dir.mkdir(parents=True, exist_ok=True)
    order_payload = generate_epoch_pair_order()
    order_path = output_dir / "epoch_pair_order.json"
    order_path.write_text(
        json.dumps(order_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pair_order_sha256 = sha256_file(order_path)

    torch.manual_seed(STAGE_G_SEED)
    torch.cuda.manual_seed_all(STAGE_G_SEED)
    shared_model = build_paper_model(full)
    shared_state = {
        name: value.detach().cpu().clone()
        for name, value in shared_model.state_dict().items()
        if torch.is_tensor(value)
    }
    shared_hash = tensor_state_sha256(shared_state)
    if shared_hash != model_tensor_state_sha256(shared_model):
        raise RuntimeError("Shared state tensor hash disagrees with model state hash")
    shared_path = output_dir / "shared_initial_state.pt"
    torch.save(shared_state, shared_path)

    full_model = build_paper_model(full).to(device).eval()
    plain_model = build_paper_model(plain).to(device).eval()
    full_model.load_state_dict(shared_state, strict=True)
    plain_model.load_state_dict(shared_state, strict=True)
    if model_tensor_state_sha256(full_model) != shared_hash:
        raise RuntimeError("Full epoch-0 model differs from shared state")
    if model_tensor_state_sha256(plain_model) != shared_hash:
        raise RuntimeError("Plain epoch-0 model differs from shared state")

    protocol = PaperReduced100Protocol.from_yaml(
        root / "configs/data/paper_reduced100.yaml", project_root=root
    )
    datasets = protocol.make_datasets()
    train_dataset = GRMHDNextStepDataset(datasets["train"])
    validation_dataset = GRMHDNextStepDataset(datasets["validation"])
    batch = next(iter(DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0)))
    full_processor = PaperDataProcessor.from_config(full).to(device)
    plain_processor = PaperDataProcessor.from_config(plain).to(device)
    full_sample = full_processor.preprocess(batch)
    plain_sample = plain_processor.preprocess(batch)
    if not torch.equal(full_sample["x"], plain_sample["x"]):
        raise RuntimeError("Full/Plain epoch-0 model inputs differ")
    with torch.no_grad():
        full_output = full_model(x=full_sample["x"])
        plain_output = plain_model(x=plain_sample["x"])
    output_bitwise_equal = torch.equal(full_output, plain_output)
    output_close = torch.allclose(full_output, plain_output, rtol=0, atol=0)
    if not output_bitwise_equal:
        raise RuntimeError("Full/Plain epoch-0 raw outputs are not bitwise identical")

    preflight_path = output_dir / "gpu_preflight.json"
    if not preflight_path.is_file():
        raise FileNotFoundError("Run the Stage G GPU preflight before preparing the manifest")
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    if preflight.get("status") != "passed":
        raise RuntimeError("Saved Stage G GPU preflight did not pass")

    parameter_count = sum(parameter.numel() for parameter in full_model.parameters())
    manifest = {
        "schema_version": "paper-stage-g-run-manifest-v1",
        "status": "ready",
        "project_git_commit": git(root, "rev-parse", "HEAD"),
        "project_branch": git(root, "branch", "--show-current"),
        "project_dirty_during_preparation": bool(git(root, "status", "--short")),
        "upstream_commit": git(root / "external/neuraloperator", "rev-parse", "HEAD"),
        "upstream_worktree_clean": True,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "numpy": np.__version__,
            "gpu_name": device_name,
            "gpu_capability": list(torch.cuda.get_device_capability(device)),
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
            )
        },
        "checksums": {
            "dataset": full.values["provenance"]["dataset"]["sha256"],
            "manifest": full.values["provenance"]["manifest"]["sha256"],
            "preprocessing": full.values["provenance"]["preprocessing"]["sha256"],
            "stage_d_artifacts": {
                name: record["sha256"]
                for name, record in full.values["provenance"]["artifacts"].items()
            },
            "stage_e_loss_contract": full.values["provenance"][
                "stage_e_loss_contract"
            ]["sha256"],
            "full_config": sha256_file(full_path),
            "plain_config": sha256_file(plain_path),
            "pair_order_file": pair_order_sha256,
        },
        "shared_initial_state": {
            "path": str(shared_path.relative_to(root)),
            "tensor_state_sha256": shared_hash,
            "full_epoch0_state_sha256": model_tensor_state_sha256(full_model),
            "plain_epoch0_state_sha256": model_tensor_state_sha256(plain_model),
            "full_plain_raw_output_bitwise_equal": output_bitwise_equal,
            "full_plain_raw_output_strict_close": output_close,
        },
        "paired_config_audit": {
            "status": audit["status"],
            "allowed_difference_patterns": audit["allowed_difference_patterns"],
            "unexpected_difference_count": len(audit["unexpected_differences"]),
        },
        "protocol": {
            "name": full.values["protocol"]["name"],
            "train_pairs": len(train_dataset),
            "validation_pairs": len(validation_dataset),
            "train_snapshots": full.values["protocol"]["train_snapshots"],
            "validation_snapshots": full.values["protocol"]["validation_snapshots"],
            "dropped_transition": full.values["protocol"]["dropped_transition"],
            "validation_shuffle": False,
            "thermal": full.values["thermal"],
            "coordinates": full.values["coordinates"],
            "radial_mode": full.values["representation"]["radial"]["selected_mode"],
        },
        "model": {
            **full.values["model"],
            "parameter_count": parameter_count,
        },
        "run": {
            "epochs": STAGE_G_EPOCHS,
            "batch_size": 1,
            "gradient_accumulation": STAGE_G_ACCUMULATION,
            "microbatches_per_epoch": 79,
            "optimizer_steps_per_epoch": 20,
            "optimizer_steps_total": 600,
            "final_accumulation_count": 3,
            "seed": STAGE_G_SEED,
            "learning_rate": 1.0e-3,
            "weight_decay": 1.0e-4,
            "warmup_epochs": 2,
            "cosine_min_lr": 1.0e-6,
            "gradient_clip_norm": 1.0,
            "mixed_precision": False,
            "eval_interval": 1,
            "early_stopping": "disabled",
            "early_stopping_reason": "paper patience=100 exceeds the 30-epoch pilot",
            "checkpoint_selection_metric": (
                "validation normalized per-channel relative L2 arithmetic average"
            ),
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Stage G controlled run manifest",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Project commit: `{manifest['project_git_commit']}`",
        f"- Upstream commit: `{manifest['upstream_commit']}`",
        f"- GPU: `{device_name}`",
        f"- Parameter count: `{parameter_count}`",
        f"- Shared initial tensor-state SHA256: `{shared_hash}`",
        f"- Pair-order SHA256: `{pair_order_sha256}`",
        f"- Full/Plain epoch-0 output bitwise equal: `{str(output_bitwise_equal).lower()}`",
        "- Epochs: `30`",
        "- Train/validation pairs: `79/19`",
        "- Gradient accumulation: `4` (`20` steps/epoch; final count `3`)",
        "- Checkpoint metric: validation normalized per-channel relative L2 arithmetic average",
        "- Early stopping: disabled",
        "",
        "This is a reduced100, spherical Kerr-Schild, press-adapted, FNO-proxy, "
        "30-epoch resource-scaled pilot. It is not directly comparable to the paper's "
        "1200-epoch 3D DISCO LocalNO result.",
    ]
    (output_dir / "run_manifest.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
