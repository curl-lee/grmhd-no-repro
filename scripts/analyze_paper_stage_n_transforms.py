#!/usr/bin/env python
"""Fit Stage N train-only prototypes and validate oracle round trips in two phases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Mapping

import h5py
import numpy as np
import torch
import yaml

from grmhd import CHANNELS
from grmhd.dataset import sha256_file
from grmhd.paper_preprocessing import PaperPreprocessor
from grmhd.paper_stage_g_evaluation import radial_shell_indices
from grmhd.paper_stage_n import (
    CANONICAL,
    CORE_METRICS,
    PrototypePreprocessor,
    aggregate_roundtrip,
    assert_train_only_indices,
    channel_prototype_decision,
    classify_readiness,
    combined_prototype_decision,
    prototype_specs,
    roundtrip_metrics,
    select_bcc2_candidate,
)


EXPECTED = {
    "upstream": "86a8bc7812a31b42c4f7895693cf4ac11521c066",
    "data": "cd3966bed1f617f49d775151cff87ae92eb2432a560bb82ae9355605e7aea70a",
    "manifest": "f6f3497717b577db25ff96f58daa5b98debe5300c329e30b543805d11d21b9d6",
    "normalizer": "1645714424d95b02de2283fc16a8ef05f3864d15f11025580af13f3e67964001",
    "shells": "99580fc2c62431f46e6702972660b060aa1294920e5f9530b45b43ce08c190b7",
    "fno_full": "a18e8711aae8071218959d34d217bb52b51a1fcc23f4e11bdcc5713dbff111e2",
    "fno_plain": "923b233de1d0abfda97f99fda9142dc437dc91f5dbbc3f19e0e5d28f5e862f1b",
    "localno_best": "f69008f91da80a4a7a6374adc253bbcdf37f58f0d142ea80f2b3f399e1f02a9f",
    "localno_last": "d8ab630075126b68a5c005ea9cb7d3ac55c17f53e826a47cd74cf04a6f826b81",
    "localno_selected": "441a1d8348b240cb5df16063ee71de1ba375d97b64e79010ec88330191fbea66",
    "stage_l_config": "8f9c170be023da938ce6b6d1414ebc459965467080ea397a85fe98998ff34983",
    "stage_l_decision": "867c8e155f843e9f5bbdd52de5280f515138fc5e6b29a743e8d7c3d6cca83595",
    "stage_l_attribution": "cb173caf676df312d3f3eaf6cd46ea63f408144476adeb7a8131fcb733596ddf",
    "stage_m_config": "91a360fe9087f413fccaeaa6cf56874e980773aa7f584b3bf5b2917e8a677206",
    "stage_m_decision": "7b9c3ba58da2594e8a4159ad79b22b569d02a525c132d0ce692aa274db59808e",
    "stage_m_candidate": "a829a3760a71a006f54ae18b399c569a903898d27f3ef3a72236982fd033e5f7",
}


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing empty Stage N CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def provenance_gate(root: Path, config_path: Path) -> dict[str, Any]:
    paths = {
        "data": "data_proc/grmhd_regrid_inner_r200_64.h5",
        "manifest": "outputs/paper_reduced100/manifest.json",
        "normalizer": "outputs/paper_reduced100/stats/normalizer.npz",
        "shells": "outputs/paper_reduced100/priors/shell_metadata.json",
        "fno_full": "outputs/paper_reduced100/stage_g/pilot30_full_fno/best_validation_l2/paper_state_dict.pt",
        "fno_plain": "outputs/paper_reduced100/stage_g/pilot30_plain_l2/best_validation_l2/paper_state_dict.pt",
        "localno_best": "outputs/paper_reduced100/stage_k/localno_differential_plain/best_validation_l2/paper_state_dict.pt",
        "localno_last": "outputs/paper_reduced100/stage_k/localno_differential_plain/last/paper_state_dict.pt",
        "localno_selected": "outputs/paper_reduced100/stage_k/localno_differential_plain/selected_states.pt",
        "stage_l_config": "configs/paper_reduced100/stage_l_collapse_attribution.yaml",
        "stage_l_decision": "outputs/paper_reduced100/stage_l/stage_l_decision.json",
        "stage_l_attribution": "outputs/paper_reduced100/stage_l/collapse_attribution.json",
        "stage_m_config": "configs/paper_reduced100/stage_m_transform_floor_audit.yaml",
        "stage_m_decision": "outputs/paper_reduced100/stage_m/stage_m_decision.json",
        "stage_m_candidate": "outputs/paper_reduced100/stage_m/candidate_gate_config.json",
    }
    checks = {}
    for name, relative in paths.items():
        actual = sha256_file(root / relative)
        checks[name] = {"path": relative, "expected_sha256": EXPECTED[name], "actual_sha256": actual, "match": actual == EXPECTED[name]}
        if actual != EXPECTED[name]:
            raise RuntimeError(f"Stage N frozen provenance mismatch: {name}")
    upstream = git(root, "-C", "external/neuraloperator", "rev-parse", "HEAD")
    if upstream != EXPECTED["upstream"] or git(root, "-C", "external/neuraloperator", "status", "--short"):
        raise RuntimeError("Stage N pinned upstream mismatch or dirty worktree")
    if git(root, "branch", "--show-current") != "main":
        raise RuntimeError("Stage N transform audit is running on the wrong branch")
    manifest = json.loads((root / paths["manifest"]).read_text())
    h5_path = root / manifest["dataset"]["configured_path"]
    with h5py.File(h5_path, "r") as handle:
        shape = tuple(int(value) for value in handle["snapshots"].shape)
        channels = [value.decode() if isinstance(value, bytes) else str(value) for value in handle["channels"][...]]
    if shape != (111, 8, 64, 64, 64) or channels != list(CHANNELS):
        raise RuntimeError("Stage N dataset shape/channel gate failed")
    return {
        "checks": checks,
        "project_start_commit": "39dde155093436685c20c4f1d97269a02ae1189d",
        "project_current_commit": git(root, "rev-parse", "HEAD"),
        "branch": git(root, "branch", "--show-current"),
        "upstream_commit": upstream,
        "dataset_shape": list(shape),
        "channels": channels,
        "stage_n_config_sha256": sha256_file(config_path),
    }


def summary_rows(
    detailed: list[dict[str, Any]], specs: Mapping[str, Any], split: str
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    rows: list[dict[str, Any]] = []
    nested: dict[str, dict[str, dict[str, Any]]] = {}
    for key, spec in specs.items():
        nested[key] = {}
        for channel, name in enumerate(CHANNELS):
            selected = [row for row in detailed if row["prototype"] == key and row["channel"] == name]
            aggregate = aggregate_roundtrip(selected)
            aggregate.update({"prototype": key, "prototype_name": spec.name, "split": split, "channel": name, "policy": spec.channel_policies[channel]})
            rows.append(aggregate)
            nested[key][name] = aggregate
    return rows, nested


def evaluate_split(
    h5_path: Path,
    indices: range,
    processors: Mapping[str, PrototypePreprocessor],
    shell_index: np.ndarray,
    split: str,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, dict[str, Any]]]]:
    detailed: list[dict[str, Any]] = []
    with h5py.File(h5_path, "r") as handle:
        snapshots = handle["snapshots"]
        for snapshot in indices:
            raw = np.asarray(snapshots[snapshot], dtype=np.float32)
            for key, processor in processors.items():
                encoded, decoded = processor.round_trip(raw, channel_axis=0)
                if not np.isfinite(encoded).all() or not np.isfinite(decoded).all():
                    raise FloatingPointError(f"{key} produced NaN/Inf at {split} snapshot {snapshot}")
                for channel, name in enumerate(CHANNELS):
                    policy = processor.spec.channel_policies[channel]
                    metrics = roundtrip_metrics(
                        raw[channel], decoded[channel], encoded[channel],
                        shell_index=shell_index,
                        softclip_applied=policy != "no_softclip",
                    )
                    detailed.append({"prototype": key, "snapshot": snapshot, "split": split, "channel": name, "policy": policy, **metrics})
    return summary_rows(detailed, {key: value.spec for key, value in processors.items()}, split)


def readiness_for(
    nested: Mapping[str, Mapping[str, Mapping[str, Any]]], prototype: str, channel: str
) -> dict[str, Any]:
    controls = [name for name in CHANNELS if name != channel]
    return classify_readiness(
        target_summary=nested[prototype][channel],
        canonical_target_summary=nested["P0"][channel],
        control_summaries=[(nested[prototype][name], nested["P0"][name]) for name in controls],
    )


def create_manifest(root: Path, config_path: Path, provenance: Mapping[str, Any], status: str) -> dict[str, Any]:
    return {
        "schema_version": "paper-stage-n-run-manifest-v1",
        "status": status,
        "classification": "diagnostic_transform_and_operator_response_audit",
        "no_training": True,
        "no_optimizer_scheduler_backward": True,
        "prototype_used_for_model_input": False,
        "project_branch": git(root, "branch", "--show-current"),
        "project_start_commit": provenance["project_start_commit"],
        "project_current_commit": git(root, "rev-parse", "HEAD"),
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "torch": torch.__version__, "torch_cuda": torch.version.cuda},
        "provenance": provenance,
        "frozen_historical_status": {
            "stage_k": "C. TRAINING_COMPLETE_BUT_ROLLOUT_UNSTABLE",
            "stage_l": "3. MIXED_OVERALL",
            "stage_m": "4. MIXED_TRANSPORT_AND_PREPROCESSING_FAILURE",
        },
        "stage_n_config": str(config_path.relative_to(root)),
    }


def run_train(root: Path, config_path: Path, output: Path) -> None:
    transform_dir = output / "transform_prototypes"
    if (transform_dir / "validation_roundtrip.json").exists():
        raise RuntimeError("Stage N validation already exists; refusing to refit after validation")
    provenance = provenance_gate(root, config_path)
    config = yaml.safe_load(config_path.read_text())
    manifest = json.loads((root / config["provenance"]["manifest"]).read_text())
    h5_path = root / manifest["dataset"]["configured_path"]
    train_indices = assert_train_only_indices(range(*config["split"]["train_snapshots"]))
    with h5py.File(h5_path, "r") as handle:
        _, shell_index = radial_shell_indices(np.asarray(handle["coords/r"][...]), 8)
    names = {
        "P0": "p0_canonical", "P1": "p1_bcc3_vel3_no_softclip",
        "P2A": "p2a_bcc2_minimal_inverse_clamp", "P2B": "p2b_bcc2_no_softclip",
        "P3": "p3_combined_v1",
    }
    specs = prototype_specs()
    processors: dict[str, PrototypePreprocessor] = {}
    artifacts: dict[str, Any] = {}
    for key, spec in specs.items():
        processor = PrototypePreprocessor.fit_hdf5(
            h5_path, spec=spec, training_indices=train_indices,
            expected_source_hdf5_checksum=EXPECTED["data"], protocol_name=manifest["protocol_name"],
        )
        processors[key] = processor
        artifacts[key] = processor.save(output / "prototypes" / names[key])
    canonical = PaperPreprocessor.load(
        root / config["provenance"]["canonical_normalizer"],
        h5_path=h5_path, expected_source_hdf5_checksum=EXPECTED["data"],
        expected_training_indices=train_indices, expected_protocol_name=manifest["protocol_name"],
    )
    p0 = processors["P0"].base
    stats_parity = all(np.array_equal(left, right) for left, right in ((p0.epsilon, canonical.epsilon), (p0.median, canonical.median), (p0.scale, canonical.scale)))
    if not stats_parity:
        raise RuntimeError("Refitted P0 statistics do not reproduce the canonical normalizer")
    rows, nested = evaluate_split(h5_path, range(*config["split"]["train_snapshots"]), processors, shell_index, "train")
    readiness = {
        "P1_Bcc3": readiness_for(nested, "P1", "Bcc3"),
        "P1_vel3": readiness_for(nested, "P1", "vel3"),
        "P2A_Bcc2": readiness_for(nested, "P2A", "Bcc2"),
        "P2B_Bcc2": readiness_for(nested, "P2B", "Bcc2"),
    }
    bcc2 = select_bcc2_candidate(readiness["P2A_Bcc2"], readiness["P2B_Bcc2"])
    if bcc2["policy"] is not None:
        p3_spec = prototype_specs(bcc2["policy"])["P3"]
        p3 = PrototypePreprocessor.fit_hdf5(
            h5_path, spec=p3_spec, training_indices=train_indices,
            expected_source_hdf5_checksum=EXPECTED["data"], protocol_name=manifest["protocol_name"],
        )
        processors["P3"] = p3
        artifacts["P3"] = p3.save(output / "prototypes" / names["P3"])
        p3_rows, p3_nested = evaluate_split(h5_path, range(*config["split"]["train_snapshots"]), {"P3": p3}, shell_index, "train")
        rows.extend(p3_rows)
        nested["P3"] = p3_nested["P3"]
    transform_dir.mkdir(parents=True, exist_ok=True)
    write_csv(transform_dir / "train_roundtrip.csv", rows)
    write_json(transform_dir / "train_roundtrip.json", {"schema_version": "paper-stage-n-train-roundtrip-v1", "rows": rows, "readiness": readiness})
    frozen = {
        "schema_version": "paper-stage-n-prototype-manifest-v1",
        "classification": "diagnostic_transform_prototype",
        "canonical_replacement": False,
        "authorized_for_training": False,
        "authorized_for_checkpoint_evaluation": False,
        "paper_faithful": False,
        "fit_snapshot_indices": list(train_indices),
        "validation_snapshot_indices_read": [],
        "validation_excluded_from_fit": True,
        "hdf5_checksum": EXPECTED["data"],
        "canonical_normalizer_checksum": EXPECTED["normalizer"],
        "p0_statistics_bitwise_canonical": stats_parity,
        "prototype_artifacts": artifacts,
        "train_readiness": readiness,
        "bcc2_candidate_frozen_from_train_only": bcc2,
        "combined_created": "P3" in processors,
        "combined_parameters_frozen_before_validation": True,
    }
    write_json(transform_dir / "prototype_manifest.json", frozen)
    run_manifest = create_manifest(root, config_path, provenance, "combined_train_decision_complete_validation_not_read")
    run_manifest["transform_prototypes"] = frozen
    write_json(output / "run_manifest.json", run_manifest)
    (output / "run_manifest.md").write_text(
        "# Stage N run manifest\n\n- Transform train-only phase: complete.\n- Validation read: false.\n- LocalNO loaded: false.\n- Training/backward/optimizer/scheduler: false.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": run_manifest["status"], "bcc2": bcc2, "readiness": readiness}, indent=2))


def run_validation(root: Path, config_path: Path, output: Path) -> None:
    transform_dir = output / "transform_prototypes"
    validation_path = transform_dir / "validation_roundtrip.json"
    if validation_path.exists():
        raise RuntimeError("Stage N validation is single-read; refusing to overwrite it")
    frozen = json.loads((transform_dir / "prototype_manifest.json").read_text())
    if frozen["validation_snapshot_indices_read"] or not frozen["combined_parameters_frozen_before_validation"]:
        raise RuntimeError("Stage N prototype freeze gate failed")
    provenance = provenance_gate(root, config_path)
    config = yaml.safe_load(config_path.read_text())
    manifest = json.loads((root / config["provenance"]["manifest"]).read_text())
    h5_path = root / manifest["dataset"]["configured_path"]
    train_indices = assert_train_only_indices(frozen["fit_snapshot_indices"])
    bcc2 = frozen["bcc2_candidate_frozen_from_train_only"]
    specs = prototype_specs(bcc2["policy"])
    names = {
        "P0": "p0_canonical", "P1": "p1_bcc3_vel3_no_softclip",
        "P2A": "p2a_bcc2_minimal_inverse_clamp", "P2B": "p2b_bcc2_no_softclip",
        "P3": "p3_combined_v1",
    }
    processors = {
        key: PrototypePreprocessor.load(output / "prototypes" / names[key], spec=spec)
        for key, spec in specs.items()
    }
    # Verify independently reconstructed parameters against the frozen artifacts
    # before opening any validation snapshot.
    for key, processor in processors.items():
        expected_artifact = frozen["prototype_artifacts"][key]["normalizer_artifact_sha256"]
        artifact = output / "prototypes" / names[key] / "normalizer.npz"
        if sha256_file(artifact) != expected_artifact:
            raise RuntimeError(f"Frozen Stage N prototype artifact changed: {key}")
        with np.load(artifact, allow_pickle=False) as values:
            if not all(np.array_equal(values[name], getattr(processor.base, name)) for name in ("epsilon", "median", "scale")):
                raise RuntimeError(f"Reconstructed Stage N statistics differ: {key}")
    with h5py.File(h5_path, "r") as handle:
        _, shell_index = radial_shell_indices(np.asarray(handle["coords/r"][...]), 8)
    rows, nested = evaluate_split(h5_path, range(*config["split"]["validation_snapshots"]), processors, shell_index, "validation")
    train_payload = json.loads((transform_dir / "train_roundtrip.json").read_text())
    readiness = train_payload["readiness"]
    selected_key = bcc2["selected"]
    channel_inputs = {
        "Bcc2": (selected_key, readiness[f"{selected_key}_Bcc2"]["ready"]),
        "Bcc3": ("P1", readiness["P1_Bcc3"]["ready"]),
        "vel3": ("P1", readiness["P1_vel3"]["ready"]),
    }
    decisions = {}
    for channel, (key, ready) in channel_inputs.items():
        if key is None:
            decisions[channel] = "C. PARTIAL_RECOVERY_ONLY"
        else:
            decisions[channel] = channel_prototype_decision(
                train_ready=ready, validation_summary=nested[key][channel],
                canonical_validation_l2=nested["P0"][channel]["median_raw_to_oracle_relative_l2"],
            )
    p3_finite = "P3" in nested and all(nested["P3"][name]["all_finite"] for name in CHANNELS)
    combined = combined_prototype_decision(decisions, p3_finite=p3_finite)
    write_csv(transform_dir / "validation_roundtrip.csv", rows)
    write_json(validation_path, {"schema_version": "paper-stage-n-validation-roundtrip-v1", "single_read": True, "snapshot_indices": list(range(*config["split"]["validation_snapshots"])), "rows": rows})
    write_json(transform_dir / "channel_decisions.json", {"schema_version": "paper-stage-n-channel-decisions-v1", "decisions": decisions, "bcc2_candidate_selected_before_validation": selected_key, "validation_did_not_select_parameters": True})
    write_json(transform_dir / "combined_decision.json", {"schema_version": "paper-stage-n-combined-decision-v1", "decision": combined, "prototype": "P3", "authorized_for_training": False, "validation_did_not_change_candidate": True})
    frozen["validation_snapshot_indices_read"] = list(range(*config["split"]["validation_snapshots"]))
    frozen["channel_decisions"] = decisions
    frozen["combined_decision"] = combined
    write_json(transform_dir / "prototype_manifest.json", frozen)
    run_manifest = create_manifest(root, config_path, provenance, "transform_workflow_complete_operator_not_started")
    run_manifest["transform_prototypes"] = frozen
    write_json(output / "run_manifest.json", run_manifest)
    (output / "run_manifest.md").write_text(
        "# Stage N run manifest\n\n- Train-only prototype fitting: complete.\n- Single-read validation round-trip: complete.\n- LocalNO loaded: false.\n- Training/backward/optimizer/scheduler: false.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": run_manifest["status"], "channel_decisions": decisions, "combined_decision": combined}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("train", "validation"), required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/paper_reduced100/stage_n_transform_operator_audit.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/paper_reduced100/stage_n"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    if args.phase == "train":
        run_train(root, config, output)
    else:
        run_validation(root, config, output)


if __name__ == "__main__":
    main()
