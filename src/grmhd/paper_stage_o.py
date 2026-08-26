"""Frozen P3/Stage-K pairing gates for the authorized Stage O pilot."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import numpy as np
import torch

from . import CHANNELS
from .dataset import sha256_file
from .paper_config import (
    EXPECTED_STAGE_O_P3_CONFIG_SHA256,
    EXPECTED_STAGE_O_P3_METADATA_SHA256,
    EXPECTED_STAGE_O_P3_STATS_SHA256,
    EXPECTED_STAGE_O_PAIR_ORDER_SHA256,
    ResolvedPaperConfig,
    build_paper_model,
    model_tensor_state_sha256,
)
from .paper_preprocessing import PaperPreprocessor
from .paper_stage_g import load_epoch_pair_order, tensor_state_sha256
from .paper_stage_g_evaluation import radial_shell_indices
from .paper_stage_n import (
    CANONICAL,
    NO_SOFTCLIP,
    PrototypePreprocessor,
    aggregate_roundtrip,
    prototype_specs,
    roundtrip_metrics,
)


EXPECTED_INITIAL_TENSOR_STATE_SHA256 = (
    "7e10895cb283be8532e4c036e8a526c536a632cce880c500080ea1d14fd89311"
)
EXPECTED_PARAMETER_COUNT = 358296
P3_TARGET_CHANNELS = ("Bcc2", "Bcc3", "vel3")
P3_CONTROL_CHANNELS = tuple(name for name in CHANNELS if name not in P3_TARGET_CHANNELS)


def load_frozen_p3(config: ResolvedPaperConfig) -> PrototypePreprocessor:
    if not config.uses_p3:
        raise ValueError("Frozen P3 requires an authorized P3 experiment config")
    values = config.values["preprocessing"]
    directory = config.resolve_path(str(values["artifact"]))
    checks = {
        directory / "normalizer.npz": EXPECTED_STAGE_O_P3_STATS_SHA256,
        directory / "resolved_config.json": EXPECTED_STAGE_O_P3_CONFIG_SHA256,
        directory / "normalizer.json": EXPECTED_STAGE_O_P3_METADATA_SHA256,
    }
    for path, expected in checks.items():
        if sha256_file(path) != expected:
            raise ValueError(f"Frozen P3 artifact changed: {path.name}")
    processor = PrototypePreprocessor.load(
        directory, spec=prototype_specs(NO_SOFTCLIP)["P3"]
    )
    processor.validate_compatibility(
        h5_path=config.resolve_path(str(config.values["protocol"]["dataset"])),
        expected_source_hdf5_checksum=config.values["provenance"]["dataset"][
            "sha256"
        ],
        expected_training_indices=range(11, 91),
        expected_protocol_name=(
            f"{config.values['protocol']['name']}__stage_n__p3"
        ),
    )
    return processor


def validate_stage_k_pairing(
    config: ResolvedPaperConfig,
    *,
    initial_state_path: str | Path,
    pair_order_path: str | Path,
    epochs: int = 30,
) -> dict[str, Any]:
    if not config.uses_p3:
        raise ValueError("Stage K/P3 pairing requires an authorized P3 config")
    initial_path = Path(initial_state_path)
    order_path = Path(pair_order_path)
    state = torch.load(initial_path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError("Frozen Stage K initial state is not a state_dict")
    state_hash = tensor_state_sha256(state)
    if state_hash != EXPECTED_INITIAL_TENSOR_STATE_SHA256:
        raise ValueError("Frozen Stage K initial tensor state changed")
    model = build_paper_model(config)
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("Stage O initial state did not load strictly")
    if model_tensor_state_sha256(model) != state_hash:
        raise ValueError("Stage O model identity differs after strict initial load")
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != EXPECTED_PARAMETER_COUNT:
        raise ValueError("Stage O LocalNO parameter count changed")
    _, orders = load_epoch_pair_order(order_path)
    if sha256_file(order_path) != EXPECTED_STAGE_O_PAIR_ORDER_SHA256:
        raise ValueError("Frozen Stage G/K pair order changed")
    if len(orders) < epochs:
        raise ValueError("Frozen pair order does not cover Stage O")
    expected = list(range(79))
    for epoch, order in enumerate(orders[:epochs]):
        if sorted(order) != expected:
            raise ValueError(f"Stage O epoch {epoch} pair order has duplicate/omission")
    parameters = [
        {"name": name, "shape": list(parameter.shape), "numel": parameter.numel()}
        for name, parameter in model.named_parameters()
    ]
    return {
        "initial_state_file_sha256": sha256_file(initial_path),
        "initial_state_tensor_sha256": state_hash,
        "pair_order_sha256": EXPECTED_STAGE_O_PAIR_ORDER_SHA256,
        "epochs_covered": epochs,
        "pairs_per_epoch": 79,
        "parameter_count": count,
        "parameters": parameters,
        "strict_load": True,
    }


def _aggregate_p3_split(
    *,
    h5_path: Path,
    indices: Iterable[int],
    preprocessor: PrototypePreprocessor,
    split: str,
) -> list[dict[str, Any]]:
    detailed: list[dict[str, Any]] = []
    with h5py.File(h5_path, "r") as handle:
        snapshots = handle["snapshots"]
        _, shell_index = radial_shell_indices(
            np.asarray(handle["coords/r"][...], dtype=np.float64), 8
        )
        for snapshot in indices:
            raw = np.asarray(snapshots[int(snapshot)], dtype=np.float32)
            encoded, decoded = preprocessor.round_trip(raw, channel_axis=0)
            if not np.isfinite(encoded).all() or not np.isfinite(decoded).all():
                raise FloatingPointError(
                    f"P3 produced NaN/Inf at {split} snapshot {snapshot}"
                )
            for channel, name in enumerate(CHANNELS):
                policy = preprocessor.spec.channel_policies[channel]
                detailed.append(
                    {
                        "prototype": "P3",
                        "snapshot": int(snapshot),
                        "split": split,
                        "channel": name,
                        "policy": policy,
                        **roundtrip_metrics(
                            raw[channel],
                            decoded[channel],
                            encoded[channel],
                            shell_index=shell_index,
                            softclip_applied=policy != NO_SOFTCLIP,
                        ),
                    }
                )
    rows = []
    for channel, name in enumerate(CHANNELS):
        selected = [row for row in detailed if row["channel"] == name]
        aggregate = aggregate_roundtrip(selected)
        aggregate.update(
            {
                "prototype": "P3",
                "prototype_name": "COMBINED_PROTOTYPE_V1",
                "split": split,
                "channel": name,
                "policy": preprocessor.spec.channel_policies[channel],
            }
        )
        rows.append(aggregate)
    return rows


def _compare_rows(
    observed: list[Mapping[str, Any]],
    frozen: list[Mapping[str, Any]],
    *,
    rtol: float = 2.0e-7,
    atol: float = 1.0e-12,
) -> dict[str, Any]:
    observed_by_channel = {str(row["channel"]): row for row in observed}
    frozen_by_channel = {
        str(row["channel"]): row
        for row in frozen
        if row.get("prototype") == "P3"
    }
    if set(observed_by_channel) != set(CHANNELS) or set(frozen_by_channel) != set(
        CHANNELS
    ):
        raise ValueError("P3 parity rows do not contain exactly eight channels")
    maximum_relative_difference = 0.0
    compared_values = 0
    for channel in CHANNELS:
        actual = observed_by_channel[channel]
        expected = frozen_by_channel[channel]
        for key, expected_value in expected.items():
            if key not in actual:
                raise ValueError(f"P3 parity is missing {channel}.{key}")
            actual_value = actual[key]
            if isinstance(expected_value, bool) or isinstance(expected_value, str):
                if actual_value != expected_value:
                    raise ValueError(f"P3 parity changed at {channel}.{key}")
            elif expected_value is None:
                if actual_value is not None:
                    raise ValueError(f"P3 parity defined previously-null {channel}.{key}")
            elif isinstance(expected_value, (int, float)):
                compared_values += 1
                if not np.isclose(
                    float(actual_value), float(expected_value), rtol=rtol, atol=atol
                ):
                    raise ValueError(
                        f"P3 parity changed at {channel}.{key}: "
                        f"expected={expected_value}, actual={actual_value}"
                    )
                scale = max(abs(float(expected_value)), atol)
                maximum_relative_difference = max(
                    maximum_relative_difference,
                    abs(float(actual_value) - float(expected_value)) / scale,
                )
    return {
        "passed": True,
        "rtol": rtol,
        "atol": atol,
        "compared_values": compared_values,
        "maximum_scaled_difference": maximum_relative_difference,
    }


def reproduce_p3_roundtrip(config: ResolvedPaperConfig) -> dict[str, Any]:
    preprocessor = load_frozen_p3(config)
    h5_path = config.resolve_path(str(config.values["protocol"]["dataset"]))
    stage_n = config.project_root / "outputs/paper_reduced100/stage_n"
    train_frozen = json.loads(
        (stage_n / "transform_prototypes/train_roundtrip.json").read_text(
            encoding="utf-8"
        )
    )["rows"]
    validation_frozen = json.loads(
        (stage_n / "transform_prototypes/validation_roundtrip.json").read_text(
            encoding="utf-8"
        )
    )["rows"]
    train = _aggregate_p3_split(
        h5_path=h5_path,
        indices=range(11, 91),
        preprocessor=preprocessor,
        split="train",
    )
    validation = _aggregate_p3_split(
        h5_path=h5_path,
        indices=range(91, 111),
        preprocessor=preprocessor,
        split="validation",
    )
    canonical = PaperPreprocessor.load(
        config.resolve_path(
            str(config.values["provenance"]["canonical_preprocessing"]["path"])
        ),
        h5_path=h5_path,
        expected_source_hdf5_checksum=config.values["provenance"]["dataset"][
            "sha256"
        ],
        expected_training_indices=range(11, 91),
        expected_protocol_name=config.values["protocol"]["name"],
    )
    stats_bitwise_canonical = all(
        np.array_equal(getattr(preprocessor.base, name), getattr(canonical, name))
        for name in ("epsilon", "median", "scale")
    )
    if not stats_bitwise_canonical:
        raise ValueError("Frozen P3 train-only statistics changed from their fitted values")
    decision = json.loads(
        (stage_n / "stage_n_decision.json").read_text(encoding="utf-8")
    )
    if any(
        decision["transform_channel_decisions"].get(name)
        != "A. PROTOTYPE_READY"
        for name in P3_TARGET_CHANNELS
    ):
        raise ValueError("Frozen Stage N P3 target decision is no longer READY")
    return {
        "passed": True,
        "train_indices": list(range(11, 91)),
        "validation_indices": list(range(91, 111)),
        "validation_indices_used_for_fit": [],
        "validation_leakage": False,
        "statistics_bitwise_canonical": stats_bitwise_canonical,
        "target_channel_decisions": decision["transform_channel_decisions"],
        "combined_decision": decision["combined_prototype_decision"],
        "train_parity": _compare_rows(train, train_frozen),
        "validation_parity": _compare_rows(validation, validation_frozen),
        "train_rows": train,
        "validation_rows": validation,
        "control_channels": list(P3_CONTROL_CHANNELS),
        "target_channels": list(P3_TARGET_CHANNELS),
    }
