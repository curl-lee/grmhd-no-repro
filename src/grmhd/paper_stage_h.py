"""Shared read-only infrastructure for Stage H post-hoc diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

from .dataset import sha256_file
from .paper_checkpoint import load_paper_checkpoint
from .paper_config import (
    ResolvedPaperConfig,
    build_paper_model,
    load_paper_experiment_config,
)
from .paper_data_processor import PaperDataProcessor
from .paper_protocol import PaperReduced100Protocol
from .paper_stage_g import tensor_state_sha256
from .upstream_adapters import GRMHDNextStepDataset


STAGE_H_SCHEMA_VERSION = "paper-stage-h-diagnostics-v1"
FULL_CONFIG_PATH = Path("configs/paper_reduced100/full_fno_proxy.yaml")
PLAIN_CONFIG_PATH = Path("configs/paper_reduced100/plain_l2_fno.yaml")
STAGE_G_ROOT = Path("outputs/paper_reduced100/stage_g")
SHARED_STATE_PATH = STAGE_G_ROOT / "shared_initial_state.pt"
PAIR_ORDER_PATH = STAGE_G_ROOT / "epoch_pair_order.json"
RUN_MANIFEST_PATH = STAGE_G_ROOT / "run_manifest.json"
OUTPUT_ROOT = Path("outputs/paper_reduced100/stage_h")

FIXED_TRAIN_PAIRS = ((11, 12), (20, 21), (40, 41), (60, 61), (89, 90))
FIXED_VALIDATION_PAIRS = (
    (91, 92),
    (95, 96),
    (100, 101),
    (105, 106),
    (109, 110),
)


@dataclass(frozen=True)
class ModelStateSpec:
    name: str
    mode: str
    config_path: Path
    checkpoint_dir: Path | None
    shared_initial: bool = False


MODEL_STATES = (
    ModelStateSpec(
        "shared_initial",
        "shared",
        FULL_CONFIG_PATH,
        None,
        shared_initial=True,
    ),
    ModelStateSpec(
        "full_best",
        "full",
        FULL_CONFIG_PATH,
        STAGE_G_ROOT / "pilot30_full_fno/best_validation_l2",
    ),
    ModelStateSpec(
        "full_last",
        "full",
        FULL_CONFIG_PATH,
        STAGE_G_ROOT / "pilot30_full_fno/last",
    ),
    ModelStateSpec(
        "plain_best",
        "plain",
        PLAIN_CONFIG_PATH,
        STAGE_G_ROOT / "pilot30_plain_l2/best_validation_l2",
    ),
    ModelStateSpec(
        "plain_last",
        "plain",
        PLAIN_CONFIG_PATH,
        STAGE_G_ROOT / "pilot30_plain_l2/last",
    ),
)


def project_root_from_file() -> Path:
    return Path(__file__).resolve().parents[2]


def json_value(value: Any) -> Any:
    if torch.is_tensor(value):
        detached = value.detach().cpu()
        return float(detached) if detached.numel() == 1 else detached.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_value(value), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("cannot write empty Stage H CSV")
    keys = sorted({str(key) for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(json_value(row[key]), sort_keys=True)
                    if isinstance(row.get(key), (dict, list, tuple))
                    else json_value(row.get(key))
                    for key in keys
                }
            )


def require_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("Stage H real diagnostics require the validated CUDA device")
    device = torch.device("cuda:0")
    name = torch.cuda.get_device_name(device)
    if "RTX 5070" not in name:
        raise RuntimeError(f"Stage H expected RTX 5070, found {name!r}")
    return device


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def checkpoint_and_evaluation_paths(root: Path) -> list[Path]:
    paths = [
        root / SHARED_STATE_PATH,
        root / PAIR_ORDER_PATH,
        root / RUN_MANIFEST_PATH,
        root / STAGE_G_ROOT / "pilot30_full_fno/evaluation_summary.json",
        root / STAGE_G_ROOT / "pilot30_plain_l2/evaluation_summary.json",
    ]
    for spec in MODEL_STATES:
        if spec.checkpoint_dir is not None:
            paths.extend(
                sorted(
                    path
                    for path in (root / spec.checkpoint_dir).iterdir()
                    if path.is_file()
                )
            )
    return paths


def frozen_input_manifest(root: Path) -> dict[str, Any]:
    full = load_paper_experiment_config(root / FULL_CONFIG_PATH, project_root=root)
    stage_g = json.loads((root / RUN_MANIFEST_PATH).read_text(encoding="utf-8"))
    tracked = checkpoint_and_evaluation_paths(root)
    if not all(path.is_file() for path in tracked):
        missing = [str(path) for path in tracked if not path.is_file()]
        raise FileNotFoundError(f"Stage H frozen inputs missing: {missing}")
    upstream_status = _git(
        root, "-C", "external/neuraloperator", "status", "--short"
    )
    if upstream_status:
        raise ValueError("Pinned upstream worktree is dirty")
    upstream_commit = _git(
        root, "-C", "external/neuraloperator", "rev-parse", "HEAD"
    )
    if upstream_commit != full.values["provenance"]["upstream"]["commit"]:
        raise ValueError("Pinned upstream commit changed")
    return {
        "schema_version": STAGE_H_SCHEMA_VERSION,
        "project_branch": _git(root, "branch", "--show-current"),
        "project_commit": _git(root, "rev-parse", "HEAD"),
        "upstream_commit": upstream_commit,
        "provenance": full.values["provenance"],
        "stage_g_shared_tensor_state_sha256": stage_g["shared_initial_state"][
            "tensor_state_sha256"
        ],
        "stage_g_pair_order_sha256": stage_g["checksums"]["pair_order_file"],
        "fixed_pairs": {
            "train": [list(pair) for pair in FIXED_TRAIN_PAIRS],
            "validation": [list(pair) for pair in FIXED_VALIDATION_PAIRS],
        },
        "files": {
            str(path.relative_to(root)): {
                "sha256": sha256_file(path),
                "size": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in tracked
        },
    }


def validate_frozen_input_manifest(
    root: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate frozen inputs while preserving the Stage H starting commit.

    Stage H records the project commit at which diagnostics began. Diagnostic
    code and documentation may subsequently be committed, so current HEAD is
    allowed to advance linearly from that starting point, including on a later
    diagnostic branch. Every actual frozen input and the pinned upstream state
    must remain identical; the recorded historical branch is provenance, not a
    content checksum.
    """

    current = frozen_input_manifest(root)
    expected_without_head = dict(expected)
    current_without_head = dict(current)
    expected_commit = str(expected_without_head.pop("project_commit"))
    current_commit = str(current_without_head.pop("project_commit"))
    expected_without_head.pop("project_branch")
    current_branch = str(current_without_head.pop("project_branch"))
    if not current_branch:
        raise ValueError("Stage H validation requires a named project branch")
    if expected_without_head != current_without_head:
        raise ValueError("Stage H frozen inputs changed after the first audit")
    ancestry = subprocess.run(
        ["git", "merge-base", "--is-ancestor", expected_commit, current_commit],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    publication_manifest = root / "docs/publish_manifest.md"
    clean_publication_snapshot = (
        current_branch == "main"
        and publication_manifest.is_file()
        and expected_commit in publication_manifest.read_text(encoding="utf-8")
    )
    if ancestry.returncode != 0 and not clean_publication_snapshot:
        raise ValueError(
            "Stage H project history no longer descends from the frozen "
            f"starting commit {expected_commit}"
        )
    return dict(expected)


def freeze_or_validate_inputs(root: Path, output_root: Path) -> dict[str, Any]:
    current = frozen_input_manifest(root)
    path = output_root / "frozen_inputs.json"
    if path.exists():
        expected = json.loads(path.read_text(encoding="utf-8"))
        return validate_frozen_input_manifest(root, expected)
    else:
        write_json(path, current)
    return current


def load_coordinates(config: ResolvedPaperConfig) -> dict[str, torch.Tensor]:
    with h5py.File(
        config.resolve_path(config.values["protocol"]["dataset"]), "r"
    ) as handle:
        return {
            name: torch.as_tensor(
                np.asarray(handle[f"coords/{name}"][...], dtype=np.float64)
            )
            for name in ("phi", "theta", "r")
        }


def make_fixed_datasets(root: Path) -> dict[str, GRMHDNextStepDataset]:
    protocol = PaperReduced100Protocol.from_yaml(
        root / "configs/data/paper_reduced100.yaml", project_root=root
    )
    datasets = protocol.make_datasets()
    return {
        split: GRMHDNextStepDataset(datasets[split])
        for split in ("train", "validation")
    }


def fixed_samples() -> Iterable[tuple[str, int, int]]:
    for split, pairs in (
        ("train", FIXED_TRAIN_PAIRS),
        ("validation", FIXED_VALIDATION_PAIRS),
    ):
        for source, target in pairs:
            yield split, source, target


def collate_fixed_pair(
    datasets: Mapping[str, GRMHDNextStepDataset],
    *,
    split: str,
    source: int,
    target: int,
) -> dict[str, Any]:
    first = 11 if split == "train" else 91
    item = source - first
    if target != source + 1:
        raise ValueError("Stage H samples must be one-step pairs")
    sample = datasets[split][item]
    if int(sample["source_index"]) != source or int(sample["target_index"]) != target:
        raise ValueError("Stage H dataset relative index resolved to wrong snapshots")
    return default_collate([sample])


def load_model_state(
    root: Path,
    spec: ModelStateSpec,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, int, Mapping[str, Any], str]:
    config = load_paper_experiment_config(
        root / spec.config_path, project_root=root
    )
    model = build_paper_model(config)
    if spec.shared_initial:
        state = torch.load(
            root / SHARED_STATE_PATH, map_location="cpu", weights_only=True
        )
        model.load_state_dict(state, strict=True)
        epoch = 0
        metadata: Mapping[str, Any] = {
            "mode": "shared",
            "epoch": 0,
            "source": str(SHARED_STATE_PATH),
        }
    else:
        if spec.checkpoint_dir is None:
            raise ValueError("checkpoint model state lacks a directory")
        loaded = load_paper_checkpoint(
            root / spec.checkpoint_dir,
            config=config,
            model=model,
            optimizer=None,
            scheduler=None,
            expected_config_checksum=sha256_file(root / spec.config_path),
        )
        model = loaded.model
        epoch = loaded.epoch
        metadata = loaded.metadata
    model.to(device).eval()
    allowed_dtypes = {torch.float32, torch.complex64}
    if any(parameter.dtype not in allowed_dtypes for parameter in model.parameters()):
        raise TypeError(
            "Stage H requires float32/complex64 model state with mixed precision disabled"
        )
    state_hash = tensor_state_sha256(model.state_dict())
    return model, epoch, metadata, state_hash


def make_processor(
    root: Path,
    *,
    device: torch.device,
    epoch: int,
) -> PaperDataProcessor:
    config = load_paper_experiment_config(
        root / FULL_CONFIG_PATH, project_root=root
    )
    processor = PaperDataProcessor.from_config(config).to(device)
    processor.eval()
    processor.set_epoch(epoch)
    return processor


def forward_fixed_pair(
    *,
    model: torch.nn.Module,
    processor: PaperDataProcessor,
    batch: Mapping[str, Any],
) -> tuple[torch.Tensor, Mapping[str, Any]]:
    processed = processor.preprocess(batch)
    prediction = model(x=processed["x"])
    prediction, fields = processor.postprocess(prediction, processed)
    if prediction.shape != fields["normalized_target"].shape:
        raise RuntimeError("Stage H prediction/target shape mismatch")
    if not torch.isfinite(prediction).all():
        raise FloatingPointError("Stage H prediction contains NaN/Inf")
    return prediction, fields


def module_group(parameter_name: str) -> list[str]:
    groups = []
    if parameter_name.startswith("lifting."):
        groups.append("lifting")
    elif parameter_name.startswith("projection."):
        groups.append("projection")
    elif ".convs." in parameter_name:
        groups.append("spectral_convolution")
    elif ".fno_skips." in parameter_name or ".channel_mlp_skips." in parameter_name:
        groups.append("skip_path")
    elif ".channel_mlp." in parameter_name:
        groups.append("channel_mlp")
    else:
        groups.append("other")
    parts = parameter_name.split(".")
    for marker in ("convs", "fno_skips", "channel_mlp", "channel_mlp_skips"):
        if marker in parts:
            position = parts.index(marker)
            if position + 1 < len(parts) and parts[position + 1].isdigit():
                groups.append(f"fno_block_{parts[position + 1]}")
                break
    return groups


__all__ = [
    "FIXED_TRAIN_PAIRS",
    "FIXED_VALIDATION_PAIRS",
    "MODEL_STATES",
    "OUTPUT_ROOT",
    "STAGE_H_SCHEMA_VERSION",
    "collate_fixed_pair",
    "fixed_samples",
    "forward_fixed_pair",
    "freeze_or_validate_inputs",
    "json_value",
    "load_coordinates",
    "load_model_state",
    "make_fixed_datasets",
    "make_processor",
    "module_group",
    "project_root_from_file",
    "require_cuda_device",
    "write_csv",
    "write_json",
]
