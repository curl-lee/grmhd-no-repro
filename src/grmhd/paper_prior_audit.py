"""Network-free Stage D diagnostics for frozen paper prior artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import h5py
import numpy as np
import torch

from .paper_bounds import PaperPhysicalBounds
from .paper_dissipation import PaperDissipativeReference
from .paper_preprocessing import PaperPreprocessor
from .paper_radial import PaperRadialBaseline, evaluate_radial_baseline
from .paper_velocity_roi import (
    PaperVelocityROI,
    canonical_oracle_velocity_roi_mask,
    raw_physical_velocity_roi_diagnostic_mask,
    roi_mask_diagnostics,
)


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else float(numerator / denominator)


def audit_bounds_split(
    h5_path: str | Path,
    snapshot_indices: Iterable[int],
    *,
    preprocessor: PaperPreprocessor,
    bounds: PaperPhysicalBounds,
) -> dict[str, Any]:
    indices = tuple(int(index) for index in snapshot_indices)
    records: dict[str, Any] = {}
    with h5py.File(h5_path, "r") as handle:
        for channel, name in ((3, "rho"), (4, "press")):
            total = raw_low = raw_high = normalized_low = normalized_high = inverse_hits = 0
            physical_lower, physical_upper = bounds.physical_bounds[name]
            normalized_lower, normalized_upper = bounds.normalized_bounds[name]
            for index in indices:
                raw = np.asarray(handle["snapshots"][index], dtype=np.float32)
                normalized = preprocessor.encode(raw)
                raw_values = raw[channel]
                normalized_values = normalized[channel]
                total += raw_values.size
                raw_low += int(np.count_nonzero(raw_values < physical_lower))
                raw_high += int(np.count_nonzero(raw_values > physical_upper))
                normalized_low += int(np.count_nonzero(normalized_values < normalized_lower))
                normalized_high += int(np.count_nonzero(normalized_values > normalized_upper))
                inverse_hits += int(
                    np.count_nonzero(
                        np.abs(normalized_values)
                        > preprocessor.gamma * preprocessor.inverse_clamp_fraction
                    )
                )
            records[name] = {
                "value_count": total,
                "raw_physical_low_violation_fraction": raw_low / total,
                "raw_physical_high_violation_fraction": raw_high / total,
                "raw_physical_any_violation_fraction": (raw_low + raw_high) / total,
                "normalized_low_violation_fraction": normalized_low / total,
                "normalized_high_violation_fraction": normalized_high / total,
                "normalized_any_violation_fraction": (normalized_low + normalized_high) / total,
                "evaluation_clamp_hit_fraction_oracle": (
                    normalized_low + normalized_high
                )
                / total,
                "paper_inverse_clamp_target_fraction": inverse_hits / total,
            }
    return {"snapshot_indices": list(indices), "channels": records}


def audit_roi_split(
    h5_path: str | Path,
    snapshot_indices: Iterable[int],
    *,
    preprocessor: PaperPreprocessor,
    roi: PaperVelocityROI,
    shell_index: np.ndarray,
) -> dict[str, Any]:
    indices = tuple(int(index) for index in snapshot_indices)
    snapshot_records: list[dict[str, Any]] = []
    total = canonical_count = raw_count = intersection_count = union_count = disagreement = 0
    canonical_clamp = raw_clamp = 0
    n_shells = int(np.max(shell_index)) + 1
    shell_totals = np.zeros(n_shells, dtype=np.int64)
    shell_canonical = np.zeros(n_shells, dtype=np.int64)
    shell_raw = np.zeros(n_shells, dtype=np.int64)
    shell_intersection = np.zeros(n_shells, dtype=np.int64)
    shell_union = np.zeros(n_shells, dtype=np.int64)
    shell_canonical_clamp = np.zeros(n_shells, dtype=np.int64)
    shell_raw_clamp = np.zeros(n_shells, dtype=np.int64)
    with h5py.File(h5_path, "r") as handle:
        for index in indices:
            raw = torch.from_numpy(
                np.asarray(handle["snapshots"][index], dtype=np.float32)
            )
            canonical_mask, _ = canonical_oracle_velocity_roi_mask(
                raw, preprocessor, top_fraction=roi.top_fraction
            )
            raw_mask = raw_physical_velocity_roi_diagnostic_mask(
                raw, top_fraction=roi.top_fraction
            )
            normalized = preprocessor.encode(raw)
            clamp_mask = torch.any(
                torch.abs(normalized[5:8])
                > preprocessor.gamma * preprocessor.inverse_clamp_fraction,
                dim=0,
            )
            per_snapshot = roi_mask_diagnostics(
                canonical_mask,
                raw_mask,
                clamp_mask,
                shell_index=shell_index,
                snapshot_indices=[index],
            )["snapshot_records"][0]
            snapshot_records.append(per_snapshot)
            inter = canonical_mask & raw_mask
            union = canonical_mask | raw_mask
            total += canonical_mask.numel()
            canonical_count += int(canonical_mask.sum())
            raw_count += int(raw_mask.sum())
            intersection_count += int(inter.sum())
            union_count += int(union.sum())
            disagreement += int((canonical_mask ^ raw_mask).sum())
            canonical_clamp += int((canonical_mask & clamp_mask).sum())
            raw_clamp += int((raw_mask & clamp_mask).sum())
            for shell in range(n_shells):
                radial = torch.as_tensor(shell_index == shell, dtype=torch.bool).view(1, 1, -1)
                radial = radial.expand_as(canonical_mask)
                c = canonical_mask & radial
                r = raw_mask & radial
                shell_totals[shell] += int(radial.sum())
                shell_canonical[shell] += int(c.sum())
                shell_raw[shell] += int(r.sum())
                shell_intersection[shell] += int((c & r).sum())
                shell_union[shell] += int((c | r).sum())
                shell_canonical_clamp[shell] += int((c & clamp_mask).sum())
                shell_raw_clamp[shell] += int((r & clamp_mask).sum())
    shells = [
        {
            "shell": shell,
            "canonical_fraction": float(shell_canonical[shell] / shell_totals[shell]),
            "raw_fraction": float(shell_raw[shell] / shell_totals[shell]),
            "jaccard": _safe_ratio(shell_intersection[shell], shell_union[shell]),
            "canonical_clamp_overlap": _safe_ratio(
                shell_canonical_clamp[shell], shell_canonical[shell]
            ),
            "raw_clamp_overlap": _safe_ratio(shell_raw_clamp[shell], shell_raw[shell]),
        }
        for shell in range(n_shells)
    ]
    inner_total = int(np.sum(shell_totals[:2]))
    inner_canonical = int(np.sum(shell_canonical[:2]))
    inner_raw = int(np.sum(shell_raw[:2]))
    return {
        "snapshot_indices": list(indices),
        "canonical_fraction": canonical_count / total,
        "raw_fraction": raw_count / total,
        "jaccard": _safe_ratio(intersection_count, union_count),
        "precision_canonical_against_raw": _safe_ratio(intersection_count, canonical_count),
        "recall_canonical_against_raw": _safe_ratio(intersection_count, raw_count),
        "disagreement_fraction": disagreement / total,
        "canonical_clamp_overlap": _safe_ratio(canonical_clamp, canonical_count),
        "raw_clamp_overlap": _safe_ratio(raw_clamp, raw_count),
        "shells": shells,
        "inner_two_shells": {
            "canonical_fraction": inner_canonical / inner_total,
            "raw_fraction": inner_raw / inner_total,
            "canonical_clamp_overlap": _safe_ratio(
                int(np.sum(shell_canonical_clamp[:2])), inner_canonical
            ),
            "raw_clamp_overlap": _safe_ratio(
                int(np.sum(shell_raw_clamp[:2])), inner_raw
            ),
        },
        "snapshot_records": snapshot_records,
        "main_mask": "canonical_oracle_physical stored-component speed proxy",
        "raw_mask_role": "diagnostic_extension_only",
    }


def audit_dissipation_split(
    h5_path: str | Path,
    snapshot_indices: Iterable[int],
    *,
    preprocessor: PaperPreprocessor,
    reference: PaperDissipativeReference,
) -> dict[str, Any]:
    indices = tuple(int(index) for index in snapshot_indices)
    records: list[dict[str, float | int]] = []
    with h5py.File(h5_path, "r") as handle:
        for index in indices:
            raw = np.asarray(handle["snapshots"][index], dtype=np.float32)
            normalized = preprocessor.encode(raw).astype(np.float64)
            norm = float(np.sqrt(np.sum(np.square(normalized))))
            gate = float(1.0 / (1.0 + np.exp(np.clip(-reference.beta * (reference.rin - norm), -700, 700))))
            records.append({"snapshot_index": index, "global_l2_norm": norm, "gate": gate})
    norms = np.asarray([record["global_l2_norm"] for record in records], dtype=np.float64)
    gates = np.asarray([record["gate"] for record in records], dtype=np.float64)
    return {
        "snapshot_indices": list(indices),
        "norm_min": float(np.min(norms)),
        "norm_median": float(np.median(norms)),
        "norm_max": float(np.max(norms)),
        "norm_mean": float(np.mean(norms)),
        "norm_std": float(np.std(norms)),
        "fraction_above_Rin": float(np.mean(norms > reference.rin)),
        "fraction_above_Rout": float(np.mean(norms > reference.rout)),
        "gate_min": float(np.min(gates)),
        "gate_median": float(np.median(gates)),
        "gate_max": float(np.max(gates)),
        "snapshot_records": records,
    }


def audit_preprocessing_target_clamp_overlap(
    h5_path: str | Path,
    snapshot_indices: Iterable[int],
    *,
    preprocessor: PaperPreprocessor,
) -> dict[str, Any]:
    indices = tuple(int(index) for index in snapshot_indices)
    total = b_count = velocity_count = intersection_count = union_count = 0
    with h5py.File(h5_path, "r") as handle:
        for index in indices:
            raw = np.asarray(handle["snapshots"][index], dtype=np.float32)
            normalized = preprocessor.encode(raw)
            limit = preprocessor.gamma * preprocessor.inverse_clamp_fraction
            b_mask = np.abs(normalized[2]) > limit
            velocity_mask = np.abs(normalized[7]) > limit
            total += b_mask.size
            b_count += int(np.count_nonzero(b_mask))
            velocity_count += int(np.count_nonzero(velocity_mask))
            intersection_count += int(np.count_nonzero(b_mask & velocity_mask))
            union_count += int(np.count_nonzero(b_mask | velocity_mask))
    return {
        "Bcc3_target_clamp_fraction": b_count / total,
        "vel3_target_clamp_fraction": velocity_count / total,
        "intersection_fraction": intersection_count / total,
        "jaccard": _safe_ratio(intersection_count, union_count),
        "source": "canonical preprocessor and existing preprocessing clamp-audit semantics",
        "statistics_refit": False,
    }


def audit_stage_d_split(
    h5_path: str | Path,
    *,
    split: str,
    snapshot_indices: Iterable[int],
    preprocessor: PaperPreprocessor,
    radial: PaperRadialBaseline,
    bounds: PaperPhysicalBounds,
    roi: PaperVelocityROI,
    dissipation: PaperDissipativeReference,
    shell_edges: Iterable[float],
) -> dict[str, Any]:
    edges = np.asarray(tuple(shell_edges), dtype=np.float64)
    shell_index = np.digitize(np.asarray(radial.r), edges[1:-1], right=False)
    indices = tuple(int(index) for index in snapshot_indices)
    return {
        "split": split,
        "snapshot_indices": list(indices),
        "radial": evaluate_radial_baseline(
            radial,
            str(h5_path),
            snapshot_indices=indices,
            preprocessor=preprocessor,
            shell_edges=edges,
        ),
        "bounds": audit_bounds_split(
            h5_path, indices, preprocessor=preprocessor, bounds=bounds
        ),
        "roi": audit_roi_split(
            h5_path,
            indices,
            preprocessor=preprocessor,
            roi=roi,
            shell_index=shell_index,
        ),
        "dissipation": audit_dissipation_split(
            h5_path, indices, preprocessor=preprocessor, reference=dissipation
        ),
        "preprocessing_target_clamp_overlap": audit_preprocessing_target_clamp_overlap(
            h5_path, indices, preprocessor=preprocessor
        ),
    }


def _flatten_scalars(prefix: str, value: Any, rows: list[dict[str, Any]]) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _flatten_scalars(f"{prefix}.{key}" if prefix else str(key), item, rows)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _flatten_scalars(f"{prefix}[{index}]", item, rows)
    elif value is None or isinstance(value, (str, int, float, bool)):
        rows.append({"metric": prefix, "value": value})


def write_stage_d_audit(
    payload: Mapping[str, Any],
    *,
    json_path: str | Path,
    csv_path: str | Path,
    markdown_path: str | Path,
) -> None:
    json_path = Path(json_path)
    csv_path = Path(csv_path)
    markdown_path = Path(markdown_path)
    for path in (json_path, csv_path, markdown_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(dict(payload), indent=2) + "\n", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    _flatten_scalars("", payload, rows)
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("metric", "value"))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# paper_reduced100 Stage D prior audit",
        "",
        f"- Status: `{payload['status']}`",
        f"- Selected radial mode: `{payload['selected_radial_mode']}`",
        f"- Validation used for fit: `{not payload['validation_not_used_for_fit']}`",
        "- Validation diagnostics are report-only and do not update any artifact.",
        "",
    ]
    for split in ("train", "validation"):
        record = payload["splits"][split]
        lines.extend(
            [
                f"## {split}",
                "",
                "| component | diagnostic | value |",
                "| --- | --- | ---: |",
                f"| radial rho | envelope violation | {record['radial']['channels']['rho']['envelope_violation_fraction']:.9g} |",
                f"| radial press | envelope violation | {record['radial']['channels']['press']['envelope_violation_fraction']:.9g} |",
                f"| bounds rho | raw violation | {record['bounds']['channels']['rho']['raw_physical_any_violation_fraction']:.9g} |",
                f"| bounds press | raw violation | {record['bounds']['channels']['press']['raw_physical_any_violation_fraction']:.9g} |",
                f"| ROI | canonical/raw Jaccard | {record['roi']['jaccard']:.9g} |",
                f"| ROI | canonical clamp overlap | {record['roi']['canonical_clamp_overlap']:.9g} |",
                f"| dissipation | max state norm | {record['dissipation']['norm_max']:.9g} |",
                f"| dissipation | fraction above Rin | {record['dissipation']['fraction_above_Rin']:.9g} |",
                "",
            ]
        )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
