#!/usr/bin/env python
"""Evaluate untrained paper baselines against raw and canonical-oracle targets."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from typing import Any

import torch

from grmhd import CHANNELS
from grmhd.models import build_model, trainable_parameter_count
from grmhd.paper_metrics import ClampMaskAccumulator, PaperMetricAccumulator
from grmhd.paper_preprocessing import PaperPreprocessor
from grmhd.paper_protocol import PaperReduced100Protocol
from grmhd.paper_references import (
    PaperReferenceStates,
    paper_reference_metadata,
)
from grmhd.shells import radial_shells_tensor


BASELINE_ORDER = (
    "oracle",
    "persistence",
    "zero_normalized",
    "random_upstream_fno",
)
METRIC_ORDER = (
    "E_norm",
    "E_model_oracle",
    "E_model_raw",
    "E_oracle_raw",
    "excess_absolute",
)


def _run_git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(requested)


def _metric_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for baseline_name in BASELINE_ORDER:
        metric_result = results[baseline_name]["oracle_metrics"]
        for metric_name in METRIC_ORDER:
            record = metric_result["metrics"][metric_name]
            for channel in CHANNELS:
                rows.append(
                    {
                        "baseline": baseline_name,
                        "section": "relative_l2",
                        "metric": metric_name,
                        "channel": channel,
                        "value": record["per_channel"][channel],
                        "numerator_squared_sum": record.get(
                            "numerator_squared_sum", {}
                        ).get(channel),
                        "denominator_squared_sum": record.get(
                            "denominator_squared_sum", {}
                        ).get(channel),
                    }
                )
            rows.append(
                {
                    "baseline": baseline_name,
                    "section": "relative_l2",
                    "metric": metric_name,
                    "channel": "arithmetic_average",
                    "value": record["arithmetic_average"],
                    "numerator_squared_sum": None,
                    "denominator_squared_sum": None,
                }
            )
        decomposition = metric_result["squared_error_numerator_decomposition"]
        for channel in CHANNELS:
            for component, value in decomposition["per_channel"][channel].items():
                rows.append(
                    {
                        "baseline": baseline_name,
                        "section": "squared_error_numerator_decomposition",
                        "metric": component,
                        "channel": channel,
                        "value": value,
                        "numerator_squared_sum": None,
                        "denominator_squared_sum": None,
                    }
                )
        clamp_result = results[baseline_name]["clamp_masks"]
        for channel in CHANNELS:
            for clamp_name, value in clamp_result["channels"][channel].items():
                rows.append(
                    {
                        "baseline": baseline_name,
                        "section": "clamp_masks",
                        "metric": clamp_name,
                        "channel": channel,
                        "value": value,
                        "numerator_squared_sum": None,
                        "denominator_squared_sum": None,
                    }
                )
    return rows


def _format_number(value: float | int | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.9g}" if isinstance(value, float) else str(value)


def _markdown_report(payload: dict[str, Any]) -> str:
    evaluation = payload["evaluation"]
    reference_metadata = payload["reference_semantics"]
    lines = [
        "# paper_reduced100 oracle-aware metric baseline",
        "",
        f"- Status: `{payload['status']}`",
        f"- Split / pairs: `{evaluation['split']}` / `{evaluation['pair_count']}`",
        f"- Snapshot pairs: `{evaluation['first_pair']}` through `{evaluation['last_pair']}`",
        f"- Device: `{payload['runtime']['device']}`",
        f"- Reference semantics: `{reference_metadata['reference_semantics_version']}`",
        f"- Gamma / inverse clamp: `{reference_metadata['canonical_gamma']}` / "
        f"`{reference_metadata['inverse_clamp_fraction']}`",
        f"- Thermal channel: `{reference_metadata['thermal_channel']}` "
        "(`press` paper adaptation; unverified EOS conversion disabled)",
        "- Components: native spherical Kerr-Schild coordinate components, not Cartesian vectors",
        "- Reduction: sum squared values over every evaluated pair/voxel within each channel, "
        "then take the relative L2 square root",
        "- Channel summary: unweighted arithmetic mean of the eight channel-relative values",
        "",
        "## Reference semantics",
        "",
        "- `raw_physical_target`: HDF5 target with no canonical round trip.",
        "- `oracle_physical_target`: `decode(encode(raw_physical_target))` under the frozen "
        "0.99-gamma inverse clamp.",
        "- `model_physical_prediction`: `decode(normalized_prediction)` under that same inverse.",
        "",
        "## Eight-channel arithmetic averages",
        "",
        "| baseline | E_norm | E_model_oracle | E_model_raw | E_oracle_raw floor | excess_absolute |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for baseline_name in BASELINE_ORDER:
        metrics = payload["baselines"][baseline_name]["oracle_metrics"]["metrics"]
        lines.append(
            f"| {baseline_name} | "
            + " | ".join(
                _format_number(metrics[metric_name]["arithmetic_average"])
                for metric_name in METRIC_ORDER
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "The floor is target-side canonical encode/decode loss. `excess_absolute` is a "
            "difference of relative norms, not an additive error allocation.",
        ]
    )

    for baseline_name in BASELINE_ORDER:
        baseline = payload["baselines"][baseline_name]
        metrics = baseline["oracle_metrics"]["metrics"]
        clamp_channels = baseline["clamp_masks"]["channels"]
        lines.extend(
            [
                "",
                f"## {baseline_name}",
                "",
                "| channel | E_norm | E_model_oracle | E_model_raw | E_oracle_raw | excess_absolute |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for channel in CHANNELS:
            lines.append(
                f"| {channel} | "
                + " | ".join(
                    _format_number(metrics[metric_name]["per_channel"][channel])
                    for metric_name in METRIC_ORDER
                )
                + " |"
            )
        lines.extend(
            [
                "",
                "| channel | target clamp | model clamp | intersection | model-only | target-only | "
                "Jaccard | precision | recall | sign agreement |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for channel in CHANNELS:
            clamp = clamp_channels[channel]
            keys = (
                "target_clamp_fraction",
                "model_clamp_fraction",
                "intersection_fraction",
                "model_only_fraction",
                "target_only_fraction",
                "jaccard",
                "precision",
                "recall",
                "sign_agreement_within_intersection",
            )
            lines.append(
                f"| {channel} | "
                + " | ".join(_format_number(clamp[key]) for key in keys)
                + " |"
            )

    oracle_checks = payload["oracle_invariant_checks"]
    lines.extend(
        [
            "",
            "## Oracle invariant checks",
            "",
            f"- Normalized oracle error is zero: `{oracle_checks['E_norm_zero']}`.",
            f"- Model-to-oracle physical error is zero: "
            f"`{oracle_checks['E_model_oracle_zero']}`.",
            f"- Oracle model-to-raw error equals the canonical floor: "
            f"`{oracle_checks['E_model_raw_equals_E_oracle_raw']}`.",
            "",
            "## Squared-error diagnostic",
            "",
            "The JSON/CSV record the exact numerator identity "
            "`||model-raw||^2 = ||model-oracle||^2 + ||oracle-raw||^2 + "
            "2<model-oracle, oracle-raw>` per channel. Relative norms do not obey this simple "
            "additivity, so the report makes no such claim.",
            "",
            "The `random_upstream_fno` row is an untrained, seeded random initialization of the "
            "fixed upstream FNO. It is a metric-pipeline smoke baseline, not a scientific model.",
            "",
        ]
    )
    return "\n".join(lines)


def _oracle_checks(oracle_metrics: dict[str, Any]) -> dict[str, Any]:
    metrics = oracle_metrics["metrics"]
    normalized_zero = all(metrics["E_norm"]["per_channel"][name] == 0 for name in CHANNELS)
    model_oracle_zero = all(
        metrics["E_model_oracle"]["per_channel"][name] == 0 for name in CHANNELS
    )
    model_raw_equals_floor = all(
        metrics["E_model_raw"]["per_channel"][name]
        == metrics["E_oracle_raw"]["per_channel"][name]
        for name in CHANNELS
    )
    checks = {
        "E_norm_zero": normalized_zero,
        "E_model_oracle_zero": model_oracle_zero,
        "E_model_raw_equals_E_oracle_raw": model_raw_equals_floor,
        "exact_comparison": True,
    }
    if not all((normalized_zero, model_oracle_zero, model_raw_equals_floor)):
        raise AssertionError(f"Oracle metric invariant failed: {checks}")
    return checks


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[1]
    config_path = (project_root / args.config).resolve()
    normalizer_path = (project_root / args.normalizer).resolve()
    output_root = (project_root / args.output_root).resolve()
    protocol = PaperReduced100Protocol.from_yaml(config_path, project_root=project_root)
    validation = protocol.validate_hdf5()
    datasets = protocol.make_datasets()
    dataset = datasets[args.split]
    pair_count = len(dataset) if args.max_pairs is None else min(len(dataset), args.max_pairs)
    if pair_count <= 0:
        raise ValueError("Evaluation requires at least one pair")
    preprocessor = PaperPreprocessor.load(
        normalizer_path,
        h5_path=protocol.dataset_path,
        expected_training_indices=protocol.train_indices,
        expected_protocol_name=protocol.protocol_name,
    )
    if preprocessor.gamma != 6.0 or preprocessor.inverse_clamp_fraction != 0.99:
        raise ValueError("Oracle metric evaluation requires frozen gamma=6 / inverse clamp=0.99")

    upstream_root = project_root / "external" / "neuraloperator"
    upstream_commit = _run_git(upstream_root, "rev-parse", "HEAD")
    if upstream_commit != protocol.expected_upstream_commit:
        raise ValueError(
            f"Upstream commit {upstream_commit} != fixed {protocol.expected_upstream_commit}"
        )
    if _run_git(upstream_root, "status", "--porcelain"):
        raise RuntimeError("Fixed upstream neuraloperator worktree must remain clean")

    device = _resolve_device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
    spatial_shape = tuple(int(value) for value in dataset.snapshot_shape[1:])
    shells, shell_metadata = radial_shells_tensor(
        dataset.coords["r"],
        spatial_shape[0],
        spatial_shape[1],
        n_shells=args.n_shells,
        device=device,
    )
    random_fno = build_model(
        "fno",
        in_channels=len(CHANNELS) + args.n_shells,
        out_channels=len(CHANNELS),
        default_in_shape=spatial_shape,
        n_modes=(args.n_modes,) * 3,
        hidden_channels=args.hidden_channels,
        n_layers=args.n_layers,
        positional_embedding=None,
    ).to(device)
    random_fno.eval()

    metric_accumulators = {name: PaperMetricAccumulator() for name in BASELINE_ORDER}
    clamp_accumulators = {
        name: ClampMaskAccumulator(
            gamma=preprocessor.gamma,
            inverse_clamp_fraction=preprocessor.inverse_clamp_fraction,
        )
        for name in BASELINE_ORDER
    }
    reference_metadata = paper_reference_metadata(preprocessor)
    evaluated_pairs: list[list[int]] = []
    start_time = time.perf_counter()
    with torch.no_grad():
        for item in range(pair_count):
            sample = dataset[item]
            raw_physical_input = sample["x"]
            raw_physical_target = sample["y"]
            normalized_input = preprocessor.encode(raw_physical_input, channel_axis=0)
            normalized_target = preprocessor.encode(raw_physical_target, channel_axis=0)
            oracle_physical_target = preprocessor.decode(normalized_target, channel_axis=0)
            model_input = torch.cat(
                [normalized_input.to(device), shells], dim=0
            ).unsqueeze(0)
            random_normalized_prediction = random_fno(model_input).squeeze(0).cpu()
            normalized_predictions = {
                "oracle": normalized_target,
                "persistence": normalized_input,
                "zero_normalized": torch.zeros_like(normalized_target),
                "random_upstream_fno": random_normalized_prediction,
            }
            for baseline_name in BASELINE_ORDER:
                normalized_prediction = normalized_predictions[baseline_name]
                model_physical_prediction = preprocessor.decode(
                    normalized_prediction, channel_axis=0
                )
                references = PaperReferenceStates(
                    raw_physical_target=raw_physical_target,
                    oracle_physical_target=oracle_physical_target,
                    normalized_target=normalized_target,
                    normalized_prediction=normalized_prediction,
                    model_physical_prediction=model_physical_prediction,
                    metadata=reference_metadata,
                )
                metric_accumulators[baseline_name].update(references, channel_axis=0)
                clamp_accumulators[baseline_name].update(
                    normalized_target=normalized_target,
                    normalized_prediction=normalized_prediction,
                    channel_axis=0,
                )
            evaluated_pairs.append([int(sample["index"]), int(sample["target_index"])])
            print(
                f"evaluated {item + 1}/{pair_count}: "
                f"{sample['index']}->{sample['target_index']}",
                flush=True,
            )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - start_time
    baselines = {
        baseline_name: {
            "prediction_semantics": {
                "oracle": "normalized_prediction = normalized_target",
                "persistence": "normalized_prediction = canonical encode(raw_physical_input)",
                "zero_normalized": "normalized_prediction = zeros in canonical normalized space",
                "random_upstream_fno": (
                    "untrained seeded fixed-upstream FNO direct normalized prediction"
                ),
            }[baseline_name],
            "trained": False,
            "oracle_metrics": metric_accumulators[baseline_name].finalize(),
            "clamp_masks": clamp_accumulators[baseline_name].finalize(),
        }
        for baseline_name in BASELINE_ORDER
    }
    oracle_checks = _oracle_checks(baselines["oracle"]["oracle_metrics"])

    project_commit = _run_git(project_root, "rev-parse", "HEAD")
    project_branch = _run_git(project_root, "branch", "--show-current")
    project_dirty = bool(_run_git(project_root, "status", "--porcelain"))
    payload = {
        "schema_version": "paper-oracle-baseline-v1",
        "status": "passed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "name": protocol.protocol_name,
            "config": str(config_path.relative_to(project_root)),
            "dataset": str(protocol.dataset_path),
            "dataset_sha256": preprocessor.source_hdf5_checksum,
            "dataset_shape": validation["snapshot_shape"],
            "thermal_channel": protocol.thermal_channel,
            "paper_adaptation": protocol.paper_adaptation,
            "eos_conversion": protocol.eos_conversion,
            "coordinate_system": protocol.coordinate_system,
        },
        "evaluation": {
            "split": args.split,
            "pair_count": pair_count,
            "available_pair_count": len(dataset),
            "partial_due_to_max_pairs": pair_count != len(dataset),
            "pairs": evaluated_pairs,
            "first_pair": evaluated_pairs[0],
            "last_pair": evaluated_pairs[-1],
            "channel_order": list(CHANNELS),
            "spatial_shape": list(spatial_shape),
        },
        "reference_semantics": reference_metadata,
        "metric_definitions": {
            "E_norm": "||normalized_prediction-normalized_target|| / ||normalized_target||",
            "E_model_oracle": (
                "||model_physical_prediction-oracle_physical_target|| / "
                "||oracle_physical_target||"
            ),
            "E_model_raw": (
                "||model_physical_prediction-raw_physical_target|| / ||raw_physical_target||"
            ),
            "E_oracle_raw": (
                "||oracle_physical_target-raw_physical_target|| / ||raw_physical_target||"
            ),
            "excess_absolute": "E_model_raw - E_oracle_raw",
            "reduction": baselines["oracle"]["oracle_metrics"]["reduction"],
            "channel_average": baselines["oracle"]["oracle_metrics"]["channel_average"],
        },
        "random_upstream_fno": {
            "trained": False,
            "purpose": "metric-path smoke baseline only",
            "seed": args.seed,
            "upstream_commit": upstream_commit,
            "class": f"{type(random_fno).__module__}.{type(random_fno).__qualname__}",
            "in_channels": len(CHANNELS) + args.n_shells,
            "out_channels": len(CHANNELS),
            "n_modes": [args.n_modes] * 3,
            "hidden_channels": args.hidden_channels,
            "n_layers": args.n_layers,
            "positional_embedding": None,
            "trainable_parameter_count": trainable_parameter_count(random_fno),
            "shell_metadata": shell_metadata.as_dict(),
        },
        "runtime": {
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "elapsed_seconds": elapsed_seconds,
            "peak_cuda_memory_mib": (
                torch.cuda.max_memory_allocated(device) / 2**20
                if device.type == "cuda"
                else None
            ),
        },
        "git": {
            "project_branch": project_branch,
            "project_commit": project_commit,
            "project_dirty_at_evaluation": project_dirty,
            "upstream_commit": upstream_commit,
            "upstream_dirty": False,
        },
        "oracle_invariant_checks": oracle_checks,
        "baselines": baselines,
    }

    output_root.mkdir(parents=True, exist_ok=True)
    json_path = output_root / "oracle_metric_baseline.json"
    csv_path = output_root / "oracle_metric_baseline.csv"
    markdown_path = output_root / "oracle_metric_baseline.md"
    json_path.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    rows = _metric_rows(baselines)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "baseline",
                "section",
                "metric",
                "channel",
                "value",
                "numerator_squared_sum",
                "denominator_squared_sum",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    markdown_path.write_text(_markdown_report(payload), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {markdown_path}")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/data/paper_reduced100.yaml"))
    parser.add_argument(
        "--normalizer",
        type=Path,
        default=Path("outputs/paper_reduced100/stats/normalizer.npz"),
    )
    parser.add_argument(
        "--output-root", type=Path, default=Path("outputs/paper_reduced100")
    )
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-shells", type=int, default=8)
    parser.add_argument("--n-modes", type=int, default=8)
    parser.add_argument("--hidden-channels", type=int, default=16)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--max-pairs", type=int)
    arguments = parser.parse_args()
    if arguments.max_pairs is not None and arguments.max_pairs <= 0:
        parser.error("--max-pairs must be positive")
    return arguments


if __name__ == "__main__":
    evaluate(parse_args())
