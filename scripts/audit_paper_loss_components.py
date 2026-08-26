#!/usr/bin/env python
"""Audit Stage E component values and gradients without a model or optimizer."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping

import h5py
import numpy as np
import torch

from grmhd.dataset import sha256_file
from grmhd.paper_bounds import PaperPhysicalBounds
from grmhd.paper_dissipation import PaperDissipativeReference
from grmhd.paper_losses import PaperCompositeLoss, PaperLossContext, PlainL2Loss
from grmhd.paper_preprocessing import PaperPreprocessor
from grmhd.paper_priors import PAPER_COORDINATE_SYSTEM, PaperResidualEnvelope, read_prior_json
from grmhd.paper_protocol import PaperReduced100Protocol
from grmhd.paper_radial import PaperRadialBaseline
from grmhd.paper_velocity_roi import (
    PaperVelocityROI,
    stored_component_speed_proxy,
    top_fraction_mask,
)


EXPECTED_HDF5_SHA256 = "cd3966bed1f617f49d775151cff87ae92eb2432a560bb82ae9355605e7aea70a"
COMPONENT_FIELDS = {
    "base": "base_fidelity_weighted",
    "h1": "h1_weighted",
    "roi": "roi_weighted",
    "bounds": "bounds_weighted",
    "envelope": "envelope_weighted",
    "dissipation": "dissipation_weighted",
    "total": "total",
}
ISOLATED_CHANNELS = {
    "isolated_Bcc1": 0,
    "isolated_Bcc3": 2,
    "isolated_rho": 3,
    "isolated_press": 4,
    "isolated_vel1": 5,
    "isolated_vel3": 7,
}


def _git_output(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _flatten(prefix: str, value: Any, output: dict[str, Any]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, output)
    elif isinstance(value, list):
        output[prefix] = json.dumps(value, separators=(",", ":"))
    elif value is None or isinstance(value, (str, int, float, bool)):
        output[prefix] = value


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    flattened: list[dict[str, Any]] = []
    fields: set[str] = set()
    for record in records:
        row: dict[str, Any] = {}
        _flatten("", record, row)
        flattened.append(row)
        fields.update(row)
    ordered = [
        name
        for name in ("split", "batch", "source_index", "target_index", "case", "component")
        if name in fields
    ] + sorted(fields - {"split", "batch", "source_index", "target_index", "case", "component"})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ordered)
        writer.writeheader()
        writer.writerows(flattened)


def _mean(records: list[dict[str, Any]], path: tuple[str, ...]) -> float:
    values = []
    for record in records:
        value: Any = record
        for key in path:
            value = value[key]
        values.append(float(value))
    return float(np.mean(values))


def _component_markdown(records: list[dict[str, Any]]) -> str:
    lines = [
        "# paper_reduced100 Stage E loss-component audit",
        "",
        "No model, Trainer, optimizer, or checkpoint was used. Each row below is the mean",
        "over the fixed batches for that split/case.",
        "",
        "| split | case | total | base | H1 | ROI | bounds | envelope | dissipation | finite |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    keys = sorted({(record["split"], record["case"]) for record in records})
    for split, case in keys:
        selected = [r for r in records if r["split"] == split and r["case"] == case]
        lines.append(
            f"| {split} | {case} | {_mean(selected, ('components', 'total')):.8g} | "
            f"{_mean(selected, ('components', 'base_fidelity_weighted')):.8g} | "
            f"{_mean(selected, ('components', 'h1_weighted')):.8g} | "
            f"{_mean(selected, ('components', 'roi_weighted')):.8g} | "
            f"{_mean(selected, ('components', 'bounds_weighted')):.8g} | "
            f"{_mean(selected, ('components', 'envelope_weighted')):.8g} | "
            f"{_mean(selected, ('components', 'dissipation_weighted')):.8g} | "
            f"{all(r['finite'] for r in selected)} |"
        )
    lines.extend(
        [
            "",
            "For `normalized_target`, base, H1, and ROI are exactly zero. Bounds, envelope,",
            "and dissipation are independent priors and are not required to vanish.",
        ]
    )
    return "\n".join(lines) + "\n"


def _gradient_markdown(records: list[dict[str, Any]]) -> str:
    lines = [
        "# paper_reduced100 Stage E gradient audit",
        "",
        "Every component gradient was taken from the same leaf prediction tensor for its",
        "batch/case. Zero gradients are valid when the corresponding component is zero.",
        "",
        "| split | case | component | mean value | mean gradient norm | finite |",
        "| --- | --- | --- | ---: | ---: | --- |",
    ]
    keys = sorted(
        {(record["split"], record["case"], record["component"]) for record in records}
    )
    for split, case, component in keys:
        selected = [
            r
            for r in records
            if r["split"] == split and r["case"] == case and r["component"] == component
        ]
        lines.append(
            f"| {split} | {case} | {component} | "
            f"{float(np.mean([r['weighted_value'] for r in selected])):.8g} | "
            f"{float(np.mean([r['gradient_norm'] for r in selected])):.8g} | "
            f"{all(r['finite'] for r in selected)} |"
        )
    return "\n".join(lines) + "\n"


def _prediction_cases(
    normalized_input: torch.Tensor,
    normalized_target: torch.Tensor,
    *,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=normalized_target.device).manual_seed(seed)
    cases = {
        "normalized_target": normalized_target.clone(),
        "persistence_normalized_input": normalized_input.clone(),
        "zero_tensor": torch.zeros_like(normalized_target),
        "deterministic_random": torch.randn(
            normalized_target.shape,
            generator=generator,
            dtype=normalized_target.dtype,
            device=normalized_target.device,
        ),
    }
    spatial_pattern = torch.linspace(
        -1.0,
        1.0,
        normalized_target[0, 0].numel(),
        dtype=normalized_target.dtype,
        device=normalized_target.device,
    ).reshape_as(normalized_target[0, 0])
    for name, channel in ISOLATED_CHANNELS.items():
        prediction = normalized_target.clone()
        prediction[0, channel] += spatial_pattern
        cases[name] = prediction
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config", type=Path, default=Path("configs/data/paper_reduced100.yaml")
    )
    parser.add_argument(
        "--normalizer", type=Path, default=Path("outputs/paper_reduced100/stats/normalizer.npz")
    )
    parser.add_argument(
        "--prior-dir", type=Path, default=Path("outputs/paper_reduced100/priors")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/paper_reduced100/losses")
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = args.config if args.config.is_absolute() else root / args.config
    normalizer_path = args.normalizer if args.normalizer.is_absolute() else root / args.normalizer
    prior_dir = args.prior_dir if args.prior_dir.is_absolute() else root / args.prior_dir
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol = PaperReduced100Protocol.from_yaml(config, project_root=root)
    hdf5_checksum = sha256_file(protocol.dataset_path)
    if hdf5_checksum != EXPECTED_HDF5_SHA256:
        raise ValueError("Stage E canonical HDF5 checksum mismatch")
    preprocessing_checksum = sha256_file(normalizer_path)
    preprocessor = PaperPreprocessor.load(
        normalizer_path,
        h5_path=protocol.dataset_path,
        expected_source_hdf5_checksum=hdf5_checksum,
        expected_training_indices=protocol.train_indices,
        expected_protocol_name=protocol.protocol_name,
    )
    expected = {
        "source_hdf5_checksum": hdf5_checksum,
        "preprocessing_stats_checksum": preprocessing_checksum,
        "training_indices": protocol.train_indices,
        "protocol_name": protocol.protocol_name,
        "thermal_channel": protocol.thermal_channel,
    }
    radial = PaperRadialBaseline.load(prior_dir / "radial_selected.json", **expected)
    bounds = PaperPhysicalBounds.load(prior_dir / "physical_bounds.json", **expected)
    envelope = PaperResidualEnvelope.load(prior_dir / "residual_envelope.json", **expected)
    roi = PaperVelocityROI.load(prior_dir / "velocity_roi.json", **expected)
    dissipation = PaperDissipativeReference.load(
        prior_dir / "dissipative_reference.json", **expected
    )
    read_prior_json(
        prior_dir / "shell_metadata.json",
        expected_schema="paper-shell-metadata-v1",
        **expected,
    )
    full_loss = PaperCompositeLoss(
        bounds=bounds,
        envelope=envelope,
        roi=roi,
        dissipation=dissipation,
        radial_metadata=radial.metadata,
    )
    plain_loss = PlainL2Loss()
    batches = (
        ("train", 0, 11, 12, 0),
        ("train", 1, 12, 13, 1),
        ("train", 2, 13, 14, 374),
        ("train", 3, 14, 15, 375),
        ("validation", 0, 91, 92, 376),
        ("validation", 1, 92, 93, 500),
    )
    component_records: list[dict[str, Any]] = []
    gradient_records: list[dict[str, Any]] = []
    with h5py.File(protocol.dataset_path, "r") as handle:
        for split, batch_index, source_index, target_index, epoch in batches:
            print(
                f"auditing {split} batch {batch_index}: {source_index}->{target_index}, epoch={epoch}",
                flush=True,
            )
            raw_input = torch.from_numpy(
                np.asarray(handle["snapshots"][source_index], dtype=np.float32)
            ).unsqueeze(0)
            raw_target = torch.from_numpy(
                np.asarray(handle["snapshots"][target_index], dtype=np.float32)
            ).unsqueeze(0)
            normalized_input = preprocessor.encode(raw_input, channel_axis=1)
            normalized_target = preprocessor.encode(raw_target, channel_axis=1)
            oracle_target = preprocessor.decode(normalized_target, channel_axis=1)
            canonical_mask = top_fraction_mask(
                stored_component_speed_proxy(oracle_target), roi.top_fraction
            )
            raw_mask = top_fraction_mask(
                stored_component_speed_proxy(raw_target), roi.top_fraction
            )
            baseline = radial.state(
                tuple(int(value) for value in normalized_target.shape[2:]),
                normalized=True,
                dtype=normalized_target.dtype,
                device=normalized_target.device,
                batch_size=1,
            )
            context = PaperLossContext(
                normalized_input=normalized_input,
                normalized_target=normalized_target,
                raw_physical_target=raw_target,
                oracle_physical_target=oracle_target,
                canonical_roi_mask=canonical_mask,
                raw_roi_diagnostic_mask=raw_mask,
                radial_baseline_normalized=baseline,
                normalized_bounds=bounds.normalized_bounds,
                epoch=epoch,
                snapshot_indices=(target_index,),
                protocol_metadata={
                    "protocol_name": protocol.protocol_name,
                    "thermal_channel": protocol.thermal_channel,
                    "paper_adaptation": protocol.paper_adaptation,
                    "validation_not_used_for_fit": True,
                    "selected_radial_mode": radial.mode,
                    "coordinate_system": PAPER_COORDINATE_SYSTEM,
                    "canonical_roi_source": "oracle_physical_target",
                },
            )
            cases = _prediction_cases(
                normalized_input,
                normalized_target,
                seed=20260722 + source_index,
            )
            for case_name, initial_prediction in cases.items():
                prediction = initial_prediction.detach().clone().requires_grad_(True)
                result = full_loss.components(prediction, context=context)
                detached = result.detached_log()
                components = detached["components"]
                diagnostics = detached["diagnostics"]
                plain_value = plain_loss(prediction, normalized_target)
                finite = all(
                    np.isfinite(float(value)) for value in components.values()
                ) and bool(torch.isfinite(plain_value))
                component_record = {
                    "split": split,
                    "batch": batch_index,
                    "batch_size": 1,
                    "source_index": source_index,
                    "target_index": target_index,
                    "case": case_name,
                    "epoch": epoch,
                    "roi_ramp": components["roi_ramp"],
                    "plain_l2": float(plain_value.detach()),
                    "components": components,
                    "diagnostics": diagnostics,
                    "finite": finite,
                }
                component_records.append(component_record)
                for component, field in COMPONENT_FIELDS.items():
                    value = getattr(result, field)
                    gradient = torch.autograd.grad(
                        value,
                        prediction,
                        retain_graph=True,
                        allow_unused=True,
                    )[0]
                    if gradient is None:
                        raise RuntimeError(
                            f"Missing {component} gradient for {split}/{case_name}"
                        )
                    gradient_norm = torch.linalg.vector_norm(gradient)
                    gradient_finite = bool(torch.isfinite(gradient).all()) and bool(
                        torch.isfinite(gradient_norm)
                    )
                    gradient_records.append(
                        {
                            "split": split,
                            "batch": batch_index,
                            "batch_size": 1,
                            "source_index": source_index,
                            "target_index": target_index,
                            "case": case_name,
                            "epoch": epoch,
                            "roi_ramp": components["roi_ramp"],
                            "component": component,
                            "weighted_value": float(value.detach()),
                            "gradient_norm": float(gradient_norm.detach()),
                            "gradient_nonzero": bool(gradient_norm > 0),
                            "finite": gradient_finite,
                        }
                    )
                    finite = finite and gradient_finite
                component_record["finite"] = finite
    status = "passed" if all(r["finite"] for r in component_records + gradient_records) else "failed"
    contract = {
        "schema_version": "paper-stage-e-loss-contract-v1",
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project": {
            "branch": _git_output(root, "branch", "--show-current"),
            "commit": _git_output(root, "rev-parse", "HEAD"),
            "dirty_at_audit": bool(_git_output(root, "status", "--short")),
        },
        "protocol": {
            "name": protocol.protocol_name,
            "hdf5_sha256": hdf5_checksum,
            "preprocessing_sha256": preprocessing_checksum,
            "train_indices": list(protocol.train_indices),
            "validation_indices": list(protocol.validation_indices),
            "thermal_channel": protocol.thermal_channel,
            "paper_adaptation": protocol.paper_adaptation,
            "validation_not_used_for_fit": True,
            "selected_radial_mode": radial.mode,
        },
        "stage_d_artifacts": {
            path.name: _file_sha256(path)
            for path in sorted(prior_dir.iterdir())
            if path.is_file()
        },
        "full_loss": full_loss.metadata,
        "plain_l2": plain_loss.metadata,
        "tensor_spaces": {
            "prediction": "canonical_normalized_state",
            "fidelity_target": "canonical_normalized_target",
            "roi_source": "supplied canonical mask from oracle_physical_target",
            "radial_reference": "selected literal canonical_normalized_rho_press",
            "physical_errors_in_training": False,
        },
        "audit_batches": [
            {
                "split": split,
                "batch": batch,
                "source_index": source,
                "target_index": target,
                "epoch": epoch,
            }
            for split, batch, source, target, epoch in batches
        ],
        "prediction_cases": [
            "normalized_target",
            "persistence_normalized_input",
            "zero_tensor",
            "deterministic_random",
            *ISOLATED_CHANNELS,
        ],
        "training_or_model_execution": False,
    }
    payload = {
        "schema_version": "paper-stage-e-loss-component-audit-v1",
        "status": status,
        "contract": contract,
        "records": component_records,
    }
    _write_json(output_dir / "loss_contract.json", contract)
    _write_json(output_dir / "loss_component_audit.json", payload)
    _write_csv(output_dir / "loss_component_audit.csv", component_records)
    (output_dir / "loss_component_audit.md").write_text(
        _component_markdown(component_records), encoding="utf-8"
    )
    _write_csv(output_dir / "gradient_audit.csv", gradient_records)
    (output_dir / "gradient_audit.md").write_text(
        _gradient_markdown(gradient_records), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": status,
                "component_records": len(component_records),
                "gradient_records": len(gradient_records),
                "output_dir": str(output_dir),
                "training_or_model_execution": False,
            },
            indent=2,
        )
    )
    if status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
