#!/usr/bin/env python
"""Gate Stage L provenance and reproduce the frozen Stage K detector flags."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import torch
import yaml
from neuralop.layers.spectral_convolution import SpectralConv

from grmhd import CHANNELS
from grmhd.dataset import sha256_file
from grmhd.paper_checkpoint import validate_paper_checkpoint_metadata
from grmhd.paper_config import (
    build_paper_model,
    load_paper_experiment_config,
    model_tensor_state_sha256,
)
from grmhd.paper_protocol import EXPECTED_UPSTREAM_COMMIT, PaperReduced100Protocol
from grmhd.paper_stage_g_evaluation import artifact_diagnostics


EXPECTED_ROOT = Path("/home/curl/projects/grmhd-no-repro")
EXPECTED_BRANCH = "main"
EXPECTED_STAGE_K_DECISION = "C. TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE"
EXPECTED_COLLAPSE_FLAGS = {
    25: ["Bcc3:possible_field_collapse"],
    50: ["Bcc3:possible_field_collapse"],
    75: ["Bcc3:possible_field_collapse", "vel3:possible_field_collapse"],
    100: ["Bcc3:possible_field_collapse", "vel3:possible_field_collapse"],
}


def git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def directory_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(item for item in path.iterdir() if item.is_file())
    if not files:
        raise FileNotFoundError(f"Checkpoint directory is empty: {path}")
    for item in files:
        digest.update(item.name.encode("utf-8"))
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def resolve(root: Path, path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else root / value


def json_load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return value


def discover_experiment(
    stage_root: Path,
    *,
    config_checksum: str,
    stage_name: str,
    expected_run_kind: str | None = None,
) -> Path:
    matches: list[Path] = []
    for sidecar in sorted(
        stage_root.glob("*/best_validation_l2/paper_grmhd_metadata.json")
    ):
        metadata = json_load(sidecar)
        run_kind = (metadata.get("stage_k") or {}).get("run_kind")
        if metadata.get("config_checksum") == config_checksum and (
            expected_run_kind is None or run_kind == expected_run_kind
        ):
            matches.append(sidecar.parent.parent)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {stage_name} experiment from checkpoint manifests, "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def checkpoint_record(
    *,
    root: Path,
    checkpoint_dir: Path,
    config_path: Path,
    expected_config_checksum: str,
) -> dict[str, Any]:
    config = load_paper_experiment_config(config_path, project_root=root)
    model = build_paper_model(config)
    metadata_path = checkpoint_dir / "paper_grmhd_metadata.json"
    state_path = checkpoint_dir / "paper_state_dict.pt"
    manifest_path = checkpoint_dir / "manifest.pt"
    metadata = json_load(metadata_path)
    validate_paper_checkpoint_metadata(
        metadata, config, expected_config_checksum=expected_config_checksum
    )
    with torch.no_grad():
        with torch.serialization.safe_globals([torch._C._nn.gelu, SpectralConv]):
            state = torch.load(state_path, map_location="cpu", weights_only=True)
        incompatibilities = model.load_state_dict(state, strict=True)
        if incompatibilities.missing_keys or incompatibilities.unexpected_keys:
            raise RuntimeError("Strict model reload reported incompatible keys")
        model.eval()
        tensor_hash = model_tensor_state_sha256(model)
    checkpoint_manifest = torch.load(
        manifest_path, map_location="cpu", weights_only=True
    )
    epoch = checkpoint_manifest.get("epoch")
    if epoch is None or int(epoch) != int(metadata["epoch"]):
        raise RuntimeError("Checkpoint manifest epoch differs from validated metadata")
    return {
        "path": str(checkpoint_dir.relative_to(root)),
        "epoch": int(epoch),
        "strict_model_reload": True,
        "eval_mode": not model.training,
        "optimizer_created": False,
        "scheduler_created": False,
        "model_state_file_sha256": sha256_file(state_path),
        "model_tensor_state_sha256": tensor_hash,
        "metadata_sha256": sha256_file(metadata_path),
        "manifest_sha256": sha256_file(manifest_path),
        "checkpoint_directory_sha256": directory_sha256(checkpoint_dir),
    }


def verify_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != "paper-stage-l-collapse-attribution-v1":
        raise ValueError("Stage L attribution config schema changed")
    classification = config.get("classification", {})
    if classification.get("training_allowed") is not False:
        raise ValueError("Stage L must prohibit training")
    if classification.get("stage_k_decision_frozen") != EXPECTED_STAGE_K_DECISION:
        raise ValueError("Stage K decision is not frozen")
    detector = config.get("frozen_detector", {})
    expected = {
        "implementation": "grmhd.paper_stage_g_evaluation.artifact_diagnostics",
        "implementation_source_sha256": (
            "3a9ea391b663df6b58a0c75942283658ee0214669d234e2c5a9594cc5f36e95d"
        ),
        "source_file_sha256": (
            "64125810309c129ca56c326c6a6be7f60f94c706b8362b62810947587d7101e8"
        ),
        "threshold": 0.05,
        "comparator": "strict_less_than",
        "denominator_floor": 1.0e-12,
        "torch_std_correction": 1,
        "shell_aggregation": "none",
        "time_aggregation": "none_per_step",
    }
    for name, value in expected.items():
        if detector.get(name) != value:
            raise ValueError(f"Frozen detector field changed: {name}")
    retention = config.get("retention_contract", {})
    if retention.get("severe_loss_threshold") != 0.5:
        raise ValueError("Stage L severe-loss threshold changed")
    if retention.get("severe_loss_comparator") != "strict_less_than":
        raise ValueError("Stage L severe-loss comparator changed")


def markdown_manifest(manifest: Mapping[str, Any]) -> str:
    p = manifest["provenance"]
    checkpoints = manifest["checkpoints"]
    lines = [
        "# Stage L provenance gate",
        "",
        f"- Status: `{manifest['status']}`",
        "- Classification: `post_hoc_attribution`",
        "- Training/backward/optimizer/scheduler: `not used`",
        f"- Project commit: `{p['project_commit']}`",
        f"- Upstream commit: `{p['upstream_commit']}`",
        f"- HDF5 SHA256: `{p['dataset']['actual_sha256']}`",
        f"- Preprocessing SHA256: `{p['preprocessing']['actual_sha256']}`",
        f"- Pair-order SHA256: `{p['pair_order']['actual_sha256']}`",
        f"- Detector implementation SHA256: `{p['detector']['implementation_sha256']}`",
        f"- Detector config SHA256: `{p['detector']['config_sha256']}`",
        f"- Detector contract SHA256: `{p['detector']['contract_sha256']}`",
        f"- Stage L config SHA256: `{p['stage_l_config_sha256']}`",
        "",
        "## Model-only strict reload",
        "",
    ]
    for model_name, records in checkpoints.items():
        for checkpoint_name, record in records.items():
            lines.append(
                f"- `{model_name}/{checkpoint_name}`: epoch `{record['epoch']}`, "
                f"state file `{record['model_state_file_sha256']}`, strict/eval "
                f"`{record['strict_model_reload']}/{record['eval_mode']}`"
            )
    lines.extend(
        [
            "",
            "All checkpoint directories were discovered by matching checkpoint sidecars "
            "to the config checksums frozen in the Stage G/K run manifests. No optimizer "
            "or scheduler was instantiated or loaded.",
        ]
    )
    return "\n".join(lines) + "\n"


def markdown_contract(contract: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# Frozen Stage K field-collapse detector contract",
            "",
            f"- Function: `{contract['function']}`",
            f"- Function-source SHA256: `{contract['function_source_sha256']}`",
            f"- Full source-file SHA256: `{contract['source_file_sha256']}`",
            f"- Contract SHA256: `{contract['contract_sha256']}`",
            "- Prediction domain: canonical decoded physical prediction after the "
            "rho/press-only evaluation bound clamp and the frozen inverse clamp.",
            "- no-GT reference: raw physical validation snapshot 91.",
            "- Reduction: separately per channel, one standard deviation over all "
            "`(phi, theta, r)` voxels; PyTorch correction is 1.",
            "- Formula: `prediction.std() / max(reference.std(), 1e-12)`.",
            "- Collapse flag: strict ratio `< 0.05`.",
            "- Shell aggregation: none. Time aggregation: none; every step is flagged "
            "independently.",
            "- NaN/Inf: the rollout rejects nonfinite decoded predictions before calling "
            "the detector; the detector has no separate nonfinite branch.",
            "- Saturation: there is no mask exemption. Bcc/velocity predictions can be "
            "limited by the frozen inverse clamp; only rho/press additionally receive "
            "the evaluation bounds clamp.",
        ]
    ) + "\n"


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/paper_reduced100/stage_l_collapse_attribution.yaml"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper_reduced100/stage_l"),
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    root = args.project_root.resolve()
    if root != EXPECTED_ROOT:
        raise RuntimeError(f"Stage L repository path changed: {root}")
    if git(root, "branch", "--show-current") != EXPECTED_BRANCH:
        raise RuntimeError("Stage L branch changed")
    upstream = root / "external/neuraloperator"
    upstream_commit = git(upstream, "rev-parse", "HEAD")
    if upstream_commit != EXPECTED_UPSTREAM_COMMIT:
        raise RuntimeError("Pinned neuraloperator commit changed")
    if git(upstream, "status", "--short"):
        raise RuntimeError("Pinned neuraloperator worktree is dirty")

    config_path = resolve(root, args.config)
    output_dir = resolve(root, args.output_dir)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError("Stage L config must contain a mapping")
    verify_config(config)
    function_source = inspect.getsource(artifact_diagnostics)
    function_hash = hashlib.sha256(function_source.encode("utf-8")).hexdigest()
    source_file_hash = sha256_file(
        resolve(root, config["frozen_detector"]["source_path"])
    )
    if function_hash != config["frozen_detector"]["implementation_source_sha256"]:
        raise RuntimeError("Frozen detector implementation checksum changed")
    if source_file_hash != config["frozen_detector"]["source_file_sha256"]:
        raise RuntimeError("Frozen detector source-file checksum changed")
    paths = config["provenance"]
    stage_g_manifest_path = resolve(root, paths["stage_g_run_manifest"])
    stage_k_manifest_path = resolve(root, paths["stage_k_run_manifest"])
    stage_g_manifest = json_load(stage_g_manifest_path)
    stage_k_manifest = json_load(stage_k_manifest_path)
    stage_k_decision = json_load(resolve(root, paths["stage_k_decision"]))
    if (
        stage_k_decision.get("choice") != "C"
        or stage_k_decision.get("decision")
        != "TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE"
    ):
        raise RuntimeError("Frozen Stage K decision changed")

    stage_k_config_path = root / "configs/paper_reduced100/stage_k_localno_differential_plain.yaml"
    stage_g_full_config_path = root / "configs/paper_reduced100/full_fno_proxy.yaml"
    stage_g_plain_config_path = root / "configs/paper_reduced100/plain_l2_fno.yaml"
    stage_k_config = load_paper_experiment_config(stage_k_config_path, project_root=root)
    expected_hashes = stage_k_config.values["provenance"]
    for name in ("dataset", "manifest", "preprocessing"):
        actual = sha256_file(resolve(root, expected_hashes[name]["path"]))
        if actual != expected_hashes[name]["sha256"]:
            raise RuntimeError(f"Frozen {name} checksum changed")
    for name, record in expected_hashes["artifacts"].items():
        if sha256_file(resolve(root, record["path"])) != record["sha256"]:
            raise RuntimeError(f"Frozen Stage D artifact changed: {name}")
    for name in ("stage_e_loss_contract", "oracle_baseline", "loss_contract_document"):
        record = expected_hashes[name]
        if sha256_file(resolve(root, record["path"])) != record["sha256"]:
            raise RuntimeError(f"Frozen provenance object changed: {name}")

    if stage_g_manifest["upstream_commit"] != upstream_commit:
        raise RuntimeError("Stage G upstream provenance changed")
    if stage_k_manifest["upstream_commit"] != upstream_commit:
        raise RuntimeError("Stage K upstream provenance changed")
    for key in ("dataset", "manifest", "preprocessing"):
        expected = expected_hashes[key]["sha256"]
        if stage_g_manifest["checksums"][key] != expected:
            raise RuntimeError(f"Stage G {key} provenance differs from frozen config")
        if stage_k_manifest["checksums"][key] != expected:
            raise RuntimeError(f"Stage K {key} provenance differs from frozen config")

    pair_path = resolve(root, paths["stage_g_pair_order"])
    pair_hash = sha256_file(pair_path)
    if pair_hash != stage_g_manifest["checksums"]["pair_order_file"]:
        raise RuntimeError("Stage G pair-order hash changed")
    if pair_hash != stage_k_manifest["checksums"]["pair_order_file"]:
        raise RuntimeError("Stage K pair-order hash changed")

    config_records = {
        "fno_full": (
            stage_g_full_config_path,
            stage_g_manifest["checksums"]["full_config"],
            stage_g_manifest_path.parent,
            None,
        ),
        "fno_plain": (
            stage_g_plain_config_path,
            stage_g_manifest["checksums"]["plain_config"],
            stage_g_manifest_path.parent,
            None,
        ),
        "localno_plain": (
            stage_k_config_path,
            stage_k_manifest["checksums"]["stage_k_config"],
            stage_k_manifest_path.parent,
            "pilot30",
        ),
    }
    checkpoint_records: dict[str, Any] = {}
    experiment_dirs: dict[str, Path] = {}
    for model_name, (
        model_config_path,
        config_hash,
        stage_root,
        expected_run_kind,
    ) in config_records.items():
        if sha256_file(model_config_path) != config_hash:
            raise RuntimeError(f"Frozen model config changed: {model_name}")
        experiment_dir = discover_experiment(
            stage_root,
            config_checksum=config_hash,
            stage_name=model_name,
            expected_run_kind=expected_run_kind,
        )
        experiment_dirs[model_name] = experiment_dir
        checkpoint_records[model_name] = {
            checkpoint_name: checkpoint_record(
                root=root,
                checkpoint_dir=experiment_dir / checkpoint_name,
                config_path=model_config_path,
                expected_config_checksum=config_hash,
            )
            for checkpoint_name in ("best_validation_l2", "last")
        }

    selected_path = resolve(root, paths["stage_k_selected_states"])
    if selected_path.parent != experiment_dirs["localno_plain"]:
        raise RuntimeError("Stage K selected-state path does not match checkpoint manifests")
    selected_payload = torch.load(selected_path, map_location="cpu", weights_only=True)
    if selected_payload.get("schema_version") != "paper-stage-k-selected-states-v1":
        raise RuntimeError("Stage K selected-state schema changed")
    if selected_payload.get("checkpoint_epoch") != checkpoint_records[
        "localno_plain"
    ]["best_validation_l2"]["epoch"]:
        raise RuntimeError("Stage K selected states do not match the best checkpoint")
    selected_states = selected_payload["selected_steps"]

    gt_rollout_path = resolve(root, paths["stage_k_gt_rollout"])
    no_gt_rollout_path = resolve(root, paths["stage_k_no_gt_rollout"])
    summary_gt_rollout_path = resolve(root, paths["stage_k_summary_gt_rollout"])
    summary_no_gt_rollout_path = resolve(root, paths["stage_k_summary_no_gt_rollout"])
    no_gt_rollout = json_load(no_gt_rollout_path)
    recorded = {int(record["step"]): record for record in no_gt_rollout["records"]}
    protocol = PaperReduced100Protocol.from_yaml(
        root / "configs/data/paper_reduced100.yaml", project_root=root
    )
    initial = protocol.make_datasets()["validation"].load_snapshot(91).unsqueeze(0)

    reproduction_rows: list[dict[str, Any]] = []
    reproduction_steps: dict[str, Any] = {}
    maximum_metric_delta = 0.0
    for step in config["analysis"]["no_gt_selected_steps"]:
        with torch.no_grad():
            reproduced = artifact_diagnostics(
                selected_states[step],
                initial,
                initial,
                reference_kind="initial_snapshot_91",
            )
        saved = recorded[step]["artifacts"]
        saved_collapse = [
            flag for flag in saved["flags"] if flag.endswith("possible_field_collapse")
        ]
        reproduced_collapse = [
            flag
            for flag in reproduced["flags"]
            if flag.endswith("possible_field_collapse")
        ]
        exact_collapse_match = reproduced_collapse == saved_collapse
        exact_all_flag_match = reproduced["flags"] == saved["flags"]
        if reproduced_collapse != EXPECTED_COLLAPSE_FLAGS[step]:
            raise RuntimeError(f"Stage K collapse flags failed reproduction at step {step}")
        if not exact_collapse_match or not exact_all_flag_match:
            raise RuntimeError(f"Stage K recorded flags differ at step {step}")
        for channel in CHANNELS:
            ratio = reproduced["channels"][channel][
                "std_ratio_prediction_over_reference"
            ]
            saved_ratio = saved["channels"][channel][
                "std_ratio_prediction_over_reference"
            ]
            delta = abs(ratio - saved_ratio)
            maximum_metric_delta = max(maximum_metric_delta, delta)
            reproduction_rows.append(
                {
                    "step": step,
                    "channel": channel,
                    "reproduced_std_ratio": ratio,
                    "recorded_std_ratio": saved_ratio,
                    "absolute_delta_cpu_vs_recorded_cuda": delta,
                    "threshold": config["frozen_detector"]["threshold"],
                    "reproduced_collapse": (
                        f"{channel}:possible_field_collapse" in reproduced_collapse
                    ),
                    "recorded_collapse": (
                        f"{channel}:possible_field_collapse" in saved_collapse
                    ),
                    "exact_flag_match": exact_collapse_match,
                }
            )
        reproduction_steps[str(step)] = {
            "recorded_all_flags": saved["flags"],
            "reproduced_all_flags": reproduced["flags"],
            "recorded_collapse_flags": saved_collapse,
            "reproduced_collapse_flags": reproduced_collapse,
            "exact_all_flag_match": exact_all_flag_match,
            "exact_collapse_flag_match": exact_collapse_match,
        }

    detector_config = dict(config["frozen_detector"])
    detector_config_hash = canonical_sha256(detector_config)
    detector_contract_core = {
        "schema_version": "paper-stage-l-detector-contract-v1",
        "function": detector_config["implementation"],
        "function_source_sha256": function_hash,
        "source_path": detector_config["source_path"],
        "source_file_sha256": source_file_hash,
        "detector_config_sha256": detector_config_hash,
        "input_contract": detector_config,
        "call_path": {
            "caller": "scripts/evaluate_paper_stage_g.py:evaluate_rollout",
            "stage_k_summary": "scripts/summarize_paper_stage_k.py",
            "prediction": "physical_prediction_from_decode_prediction_apply_evaluation_clamp_true",
            "no_gt_reference": "raw_physical_initial_snapshot_91",
            "reference_input": "raw_physical_initial_snapshot_91",
        },
        "formula": {
            "prediction_std": "torch.std(prediction[0, channel])",
            "reference_std": "torch.std(reference[0, channel])",
            "ratio": "prediction_std / max(reference_std, 1e-12)",
            "flag": "ratio < 0.05",
        },
    }
    contract_hash = canonical_sha256(detector_contract_core)
    detector_contract = {**detector_contract_core, "contract_sha256": contract_hash}

    provenance_records = {
        "project_commit": git(root, "rev-parse", "HEAD"),
        "project_branch": git(root, "branch", "--show-current"),
        "upstream_commit": upstream_commit,
        "upstream_worktree_clean": True,
        "project_worktree_clean_at_initial_gate": True,
        "dataset": {
            "path": expected_hashes["dataset"]["path"],
            "expected_sha256": expected_hashes["dataset"]["sha256"],
            "actual_sha256": sha256_file(resolve(root, expected_hashes["dataset"]["path"])),
        },
        "manifest": {
            "path": expected_hashes["manifest"]["path"],
            "expected_sha256": expected_hashes["manifest"]["sha256"],
            "actual_sha256": sha256_file(resolve(root, expected_hashes["manifest"]["path"])),
        },
        "preprocessing": {
            "path": expected_hashes["preprocessing"]["path"],
            "expected_sha256": expected_hashes["preprocessing"]["sha256"],
            "actual_sha256": sha256_file(resolve(root, expected_hashes["preprocessing"]["path"])),
        },
        "stage_d_artifacts": {
            name: {
                "path": record["path"],
                "expected_sha256": record["sha256"],
                "actual_sha256": sha256_file(resolve(root, record["path"])),
            }
            for name, record in expected_hashes["artifacts"].items()
        },
        "stage_e_loss_contract": {
            "path": expected_hashes["stage_e_loss_contract"]["path"],
            "expected_sha256": expected_hashes["stage_e_loss_contract"]["sha256"],
            "actual_sha256": sha256_file(
                resolve(root, expected_hashes["stage_e_loss_contract"]["path"])
            ),
        },
        "oracle_semantics": {
            "path": expected_hashes["oracle_baseline"]["path"],
            "expected_sha256": expected_hashes["oracle_baseline"]["sha256"],
            "actual_sha256": sha256_file(
                resolve(root, expected_hashes["oracle_baseline"]["path"])
            ),
            "version": expected_hashes["oracle_reference_semantics_version"],
        },
        "pair_order": {
            "path": str(pair_path.relative_to(root)),
            "expected_sha256": stage_k_manifest["checksums"]["pair_order_file"],
            "actual_sha256": pair_hash,
        },
        "stage_k_rollouts": {
            "gt": {"path": str(gt_rollout_path.relative_to(root)), "sha256": sha256_file(gt_rollout_path)},
            "no_gt": {"path": str(no_gt_rollout_path.relative_to(root)), "sha256": sha256_file(no_gt_rollout_path)},
            "summary_gt": {"path": str(summary_gt_rollout_path.relative_to(root)), "sha256": sha256_file(summary_gt_rollout_path)},
            "summary_no_gt": {"path": str(summary_no_gt_rollout_path.relative_to(root)), "sha256": sha256_file(summary_no_gt_rollout_path)},
            "selected_states": {"path": str(selected_path.relative_to(root)), "sha256": sha256_file(selected_path)},
        },
        "detector": {
            "implementation_sha256": function_hash,
            "source_file_sha256": source_file_hash,
            "config_sha256": detector_config_hash,
            "contract_sha256": contract_hash,
        },
        "stage_l_config_sha256": sha256_file(config_path),
    }
    manifest = {
        "schema_version": "paper-stage-l-run-manifest-v1",
        "status": "detector_gate_passed",
        "scope_completed": [
            "provenance_gate",
            "detector_source_audit",
            "stage_k_flag_reproduction",
            "frozen_attribution_config",
        ],
        "scope_not_started": [
            "variance_attribution",
            "shell_attribution",
            "spectrum_attribution",
        ],
        "classification": "post_hoc_attribution",
        "no_training": True,
        "no_backward": True,
        "no_optimizer_created": True,
        "no_scheduler_created": True,
        "stage_k_decision_unchanged": EXPECTED_STAGE_K_DECISION,
        "environment": {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "torch": torch.__version__,
        },
        "provenance": provenance_records,
        "checkpoints": checkpoint_records,
        "detector_reproduction": {
            "status": "passed",
            "selected_steps": config["analysis"]["no_gt_selected_steps"],
            "exact_all_flags": True,
            "exact_collapse_flags": True,
            "maximum_std_ratio_delta_cpu_vs_recorded_cuda": maximum_metric_delta,
        },
    }
    reproduction = {
        "schema_version": "paper-stage-l-detector-reproduction-v1",
        "status": "passed",
        "detector_contract_sha256": contract_hash,
        "selected_states_sha256": provenance_records["stage_k_rollouts"]["selected_states"]["sha256"],
        "recorded_no_gt_rollout_sha256": provenance_records["stage_k_rollouts"]["no_gt"]["sha256"],
        "reference_snapshot": 91,
        "reference_domain": "raw_physical",
        "reproduction_device": "cpu",
        "recorded_device": "cuda",
        "maximum_std_ratio_delta_cpu_vs_recorded_cuda": maximum_metric_delta,
        "steps": reproduction_steps,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "run_manifest.md").write_text(
        markdown_manifest(manifest), encoding="utf-8"
    )
    (output_dir / "detector_contract.json").write_text(
        json.dumps(detector_contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "detector_contract.md").write_text(
        markdown_contract(detector_contract), encoding="utf-8"
    )
    (output_dir / "detector_reproduction.json").write_text(
        json.dumps(reproduction, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "detector_reproduction.csv", reproduction_rows)
    print(json.dumps(manifest["detector_reproduction"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
