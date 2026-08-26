#!/usr/bin/env python
"""Audit the Stage I four-way config and frozen pairing gate without training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from grmhd.dataset import sha256_file
from grmhd.paper_config import (
    STAGE_I_PAIR_ALLOWED_DIFFERENCES,
    build_paper_model,
    build_paper_training_loss,
    load_paper_experiment_config,
    recursive_config_differences,
    stage_i_difference_is_allowed,
)
from grmhd.paper_data_processor import PaperDataProcessor
from grmhd.paper_protocol import PaperReduced100Protocol
from grmhd.paper_stage_g import (
    tensor_state_sha256,
    validate_epoch_pair_order,
)
from grmhd.upstream_adapters import GRMHDNextStepDataset


EXPECTED_SHARED_HASH = (
    "02d83278d5e74a27c83130e99b488ca18eae5477dad9732a4552b694056b6cb1"
)
EXPECTED_PAIR_ORDER_HASH = (
    "5e1e361ea75e01ffd667822d8bfe870d70c5bc69570d52cdf806e637b2df2b52"
)
NON_H1_COMPONENTS = (
    "base_fidelity_raw",
    "base_fidelity_weighted",
    "roi_raw",
    "roi_ramp",
    "roi_weighted",
    "bounds_rho_raw",
    "bounds_press_raw",
    "bounds_weighted",
    "envelope_rho_raw",
    "envelope_press_raw",
    "envelope_weighted",
    "dissipation_raw",
    "dissipation_weighted",
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "outputs/paper_reduced100/stage_i"
    paths = {
        "full": root / "configs/paper_reduced100/full_fno_proxy.yaml",
        "no_h1": root
        / "configs/paper_reduced100/extensions/no_h1_control.yaml",
        "unit_index": root
        / "configs/paper_reduced100/extensions/unit_index_h1.yaml",
        "stored_coordinate_volume_proxy": root
        / "configs/paper_reduced100/extensions/stored_coordinate_volume_h1.yaml",
    }
    configs = {
        name: load_paper_experiment_config(path, project_root=root)
        for name, path in paths.items()
    }
    full = configs["full"]
    differences: dict[str, list[dict[str, Any]]] = {}
    unexpected: dict[str, list[dict[str, Any]]] = {}
    for name in ("no_h1", "unit_index", "stored_coordinate_volume_proxy"):
        records = recursive_config_differences(
            full.as_dict(),
            configs[name].as_dict(),
        )
        differences[name] = records
        unexpected[name] = [
            record
            for record in records
            if not stage_i_difference_is_allowed(str(record["path"]))
        ]

    shared_path = (
        root / "outputs/paper_reduced100/stage_g/shared_initial_state.pt"
    )
    pair_order_path = (
        root / "outputs/paper_reduced100/stage_g/epoch_pair_order.json"
    )
    shared_state = torch.load(
        shared_path,
        map_location="cpu",
        weights_only=True,
    )
    shared_hash = tensor_state_sha256(shared_state)
    pair_order_hash = sha256_file(pair_order_path)
    pair_order_payload = json.loads(pair_order_path.read_text(encoding="utf-8"))
    epoch_orders = validate_epoch_pair_order(pair_order_payload)

    generator = torch.Generator().manual_seed(42)
    probe = torch.randn(1, 16, 16, 16, 16, generator=generator)
    model_records: dict[str, dict[str, Any]] = {}
    reference_output: torch.Tensor | None = None
    for name, config in configs.items():
        model = build_paper_model(config).eval()
        model.load_state_dict(shared_state, strict=True)
        with torch.no_grad():
            model_output = model(x=probe)
        if reference_output is None:
            reference_output = model_output
        model_records[name] = {
            "parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "tensor_state_sha256": tensor_state_sha256(model.state_dict()),
            "strict_shared_state_load": True,
            "raw_epoch_zero_output_bitwise_equal": torch.equal(
                model_output,
                reference_output,
            ),
            "raw_epoch_zero_output_sha256": tensor_state_sha256(
                {"output": model_output}
            ),
        }

    protocol = PaperReduced100Protocol.from_yaml(
        root / "configs/data/paper_reduced100.yaml",
        project_root=root,
    )
    datasets = protocol.make_datasets()
    train_dataset = GRMHDNextStepDataset(datasets["train"])
    validation_dataset = GRMHDNextStepDataset(datasets["validation"])
    batch = next(
        iter(DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0))
    )
    processor = PaperDataProcessor.from_config(full)
    processor.set_epoch(0)
    processor.preprocess(batch)
    fields = processor.last_batch
    if fields is None:
        raise RuntimeError("Stage I pairing processor did not retain the batch")
    prediction = fields["normalized_input"].clone()
    loss_results = {}
    with torch.no_grad():
        for name, config in configs.items():
            loss_results[name] = build_paper_training_loss(config).eval().components(
                prediction,
                context=fields["context"],
            )
    reference_components = loss_results["full"]
    unit_result = loss_results["unit_index"]
    stored_result = loss_results["stored_coordinate_volume_proxy"]
    unit_current_upstream = unit_result.diagnostics[
        "diagnostic_current_upstream_h1_raw"
    ]
    unit_spacing_ratio = float(
        (
            unit_current_upstream
            / unit_result.h1_raw.clamp_min(1.0e-30)
        ).cpu()
    )
    stored_metadata = stored_result.diagnostics["selected_h1_metadata"]
    component_parity: dict[str, dict[str, bool]] = {}
    for name in ("no_h1", "unit_index", "stored_coordinate_volume_proxy"):
        component_parity[name] = {
            component: torch.equal(
                getattr(loss_results[name], component),
                getattr(reference_components, component),
            )
            for component in NON_H1_COMPONENTS
        }

    checks = {
        "no_unexpected_config_differences": not any(unexpected.values()),
        "shared_state_hash_exact": shared_hash == EXPECTED_SHARED_HASH,
        "pair_order_hash_exact": pair_order_hash == EXPECTED_PAIR_ORDER_HASH,
        "parameter_counts_exact": all(
            record["parameter_count"] == 331832
            for record in model_records.values()
        ),
        "initial_states_exact": all(
            record["tensor_state_sha256"] == EXPECTED_SHARED_HASH
            for record in model_records.values()
        ),
        "epoch_zero_outputs_bitwise_equal": all(
            record["raw_epoch_zero_output_bitwise_equal"]
            for record in model_records.values()
        ),
        "non_h1_components_exact": all(
            all(values.values()) for values in component_parity.values()
        ),
        "train_validation_pairs_exact": (
            len(train_dataset) == 79 and len(validation_dataset) == 19
        ),
        "epoch_order_exact": (
            len(epoch_orders) == 30
            and all(
                len(order) == 79 and sorted(order) == list(range(79))
                for order in epoch_orders
            )
            and pair_order_payload["validation_shuffle"] is False
        ),
        "unit_index_selected_implementation": (
            unit_result.diagnostics["selected_h1_metadata"]["mode"]
            == "unit_index"
            and unit_result.diagnostics["selected_h1_metadata"]["spacings"]
            == [1.0, 1.0, 1.0]
        ),
        "unit_index_spacing_ratio_4096": bool(
            np.isclose(unit_spacing_ratio, 4096.0, rtol=2.0e-5, atol=2.0e-3)
        ),
        "current_upstream_h1_is_detached_diagnostic": (
            bool(unit_current_upstream.requires_grad) is False
        ),
        "stored_coordinate_selected_implementation": (
            stored_metadata["mode"] == "stored_coordinate_volume_proxy"
            and stored_metadata["radial_coordinate"] == "physical_r"
            and stored_metadata["phi_boundary"] == "periodic_centered"
            and stored_metadata["theta_boundary"]
            == "open_three_point_lagrange"
            and stored_metadata["r_boundary"]
            == "open_nonuniform_three_point_lagrange"
            and stored_metadata["reduction"]
            == "normalized_r2_sin_theta_coordinate_volume_proxy"
            and stored_metadata["pole_sin_floor"] == 0.0
            and stored_metadata["volume_weight_normalization"]
            == "explicit_sum_to_one"
        ),
        "stored_current_and_unit_diagnostics_detached": (
            stored_result.diagnostics[
                "diagnostic_current_upstream_h1_raw"
            ].requires_grad
            is False
            and stored_result.diagnostics[
                "diagnostic_unit_index_h1_raw"
            ].requires_grad
            is False
        ),
        "stored_coordinate_training_not_executed": not (
            output / "stored_coordinate_volume_h1"
        ).exists(),
        "no_training_executed": True,
        "no_optimizer_created": True,
        "no_optimizer_step": True,
        "no_scheduler_step": True,
    }
    payload = {
        "schema_version": "paper-stage-i-config-pairing-audit-v1",
        "status": "passed" if all(checks.values()) else "failed",
        "allowed_difference_patterns": sorted(
            STAGE_I_PAIR_ALLOWED_DIFFERENCES
        ),
        "config_paths": {
            name: str(path.relative_to(root)) for name, path in paths.items()
        },
        "config_sha256": {
            name: sha256_file(path) for name, path in paths.items()
        },
        "differences_from_full": differences,
        "unexpected_differences": unexpected,
        "frozen_pairing": {
            "shared_initial_state_tensor_sha256": shared_hash,
            "pair_order_file_sha256": pair_order_hash,
            "seed": 42,
            "train_pairs": len(train_dataset),
            "validation_pairs": len(validation_dataset),
            "epochs": 30,
            "batch_size": 1,
            "gradient_accumulation": 4,
            "validation_shuffle": pair_order_payload["validation_shuffle"],
        },
        "models": model_records,
        "non_h1_component_parity": component_parity,
        "unit_index_h1": {
            "selected_implementation": "H1_unit_index",
            "current_upstream_raw_h1": float(unit_current_upstream.cpu()),
            "selected_raw_h1": float(unit_result.h1_raw.cpu()),
            "current_to_selected_raw_ratio": unit_spacing_ratio,
            "expected_ratio": 4096.0,
            "current_upstream_detached_diagnostic_only": True,
        },
        "stored_coordinate_h1": {
            "selected_implementation": "H3_stored_r",
            "selected_raw_h1": float(stored_result.h1_raw.cpu()),
            "current_upstream_raw_h1": float(
                stored_result.diagnostics[
                    "diagnostic_current_upstream_h1_raw"
                ].cpu()
            ),
            "unit_index_raw_h1": float(
                stored_result.diagnostics[
                    "diagnostic_unit_index_h1_raw"
                ].cpu()
            ),
            "metadata": stored_metadata,
            "current_and_unit_diagnostics_detached_only": True,
        },
        "checks": checks,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "config_diff.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Stage I H1 extension config/pairing audit",
        "",
        f"- Status: `{payload['status']}`",
        f"- Shared state SHA256: `{shared_hash}`",
        f"- Pair-order SHA256: `{pair_order_hash}`",
        "- Parameter count: `331832` for all four variants",
        "- Epoch-zero state: exact shared-state match",
        "- Epoch-zero raw output: bitwise identical",
        "- Non-H1 loss components: exact parity",
        f"- Current/unit-index raw H1 ratio: `{unit_spacing_ratio:.8g}`",
        "- Stored-coordinate H1: physical-r coordinate derivative with "
        "normalized spherical-coordinate volume proxy",
        "- Current upstream H1: detached diagnostic only",
        "- Stored-coordinate training: not executed",
        "- Data order: 30 complete permutations of 79 train pairs",
        "- Validation shuffle: `false`",
        "- Training executed: `false`",
        "",
        "## Unexpected differences",
        "",
    ]
    if any(unexpected.values()):
        for name, records in unexpected.items():
            for record in records:
                lines.append(f"- `{name}`: `{record['path']}`")
    else:
        lines.append("None.")
    (output / "config_diff.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if payload["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
