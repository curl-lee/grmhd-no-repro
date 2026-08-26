#!/usr/bin/env python
"""Read-only Stage J inventory of GRMHD data and provenance candidates.

The audit intentionally reads only HDF5 metadata and small provenance datasets;
it never loads field arrays, rewrites data, infers an EOS, or converts variables.
Content hashes cover the complete files so that later audits can reject silent
changes.  Physics metadata is allow-listed to avoid copying unrelated config
contents (including possible credentials) into the report.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import tomllib
from typing import Any

import h5py
import numpy as np
import yaml


DATA_SUFFIXES = {".athdf", ".h5", ".hdf5"}
CONFIG_SUFFIXES = {".ini", ".yaml", ".toml", ".json"}
SNAPSHOT_PATTERN = re.compile(r"\.(\d+)\.athdf$")
SENSITIVE_FRAGMENTS = (
    "secret",
    "password",
    "passwd",
    "token",
    "api_key",
    "private_key",
    "credential",
)
PHYSICS_KEYS = {
    "a",
    "adiabatic_index",
    "axis_order",
    "channels",
    "coordinate_system",
    "coordinates",
    "cycle",
    "eint",
    "energy",
    "eos",
    "equation_of_state",
    "gamma",
    "gamma_adi",
    "integrator",
    "internal_energy",
    "max_level",
    "method",
    "nx1",
    "nx2",
    "nx3",
    "problem",
    "problem_id",
    "r_max",
    "r_min",
    "snapshot_count",
    "source_file_count",
    "source_raw_dir",
    "spin",
    "thermal_channel",
    "time",
    "variable",
    "variable_names",
    "x1max",
    "x1min",
    "x1rat",
    "x2max",
    "x2min",
    "x3max",
    "x3min",
}


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decoded(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return decoded(value.item())
    if isinstance(value, np.ndarray):
        return [decoded(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [decoded(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def timestamp(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def base_record(path: Path, *, source: str) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "path": str(path.absolute()),
        "resolved_path": str(resolved),
        "source": source,
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
        "modified_time_utc": timestamp(resolved),
    }


def hdf5_attributes(handle: h5py.File) -> dict[str, Any]:
    result: dict[str, Any] = {
        f"/{key}": decoded(value) for key, value in handle.attrs.items()
    }

    def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
        for key, value in obj.attrs.items():
            result[f"/{name}@{key}"] = decoded(value)

    handle.visititems(visitor)
    return result


def eos_metadata(attributes: Mapping[str, Any]) -> dict[str, Any]:
    terms = (
        "gamma",
        "gamma_adi",
        "adiabatic_index",
        "eos",
        "equation_of_state",
        "eint",
        "internal_energy",
        "total_energy",
    )
    return {
        key: value
        for key, value in attributes.items()
        if any(term in key.lower() for term in terms)
    }


def small_dataset(handle: h5py.File, name: str) -> list[Any] | None:
    if name not in handle or not isinstance(handle[name], h5py.Dataset):
        return None
    dataset = handle[name]
    if dataset.size > 10_000:
        return None
    return decoded(dataset[...])


def inspect_hdf5(path: Path, *, source: str) -> dict[str, Any]:
    record = base_record(path, source=source)
    with h5py.File(path, "r") as handle:
        attributes = hdf5_attributes(handle)
        objects: dict[str, Any] = {}

        def visitor(name: str, obj: h5py.Group | h5py.Dataset) -> None:
            if isinstance(obj, h5py.Dataset):
                objects[name] = {
                    "shape": list(obj.shape),
                    "dtype": str(obj.dtype),
                }

        handle.visititems(visitor)
        variable_names = decoded(handle.attrs.get("VariableNames", []))
        channels = small_dataset(handle, "channels")
        source_files = small_dataset(handle, "source_files")
        times = small_dataset(handle, "source_times")
        if times is None:
            times = small_dataset(handle, "times")
        snapshot_match = SNAPSHOT_PATTERN.search(path.name)
        record.update(
            {
                "kind": "raw_athdf" if path.suffix == ".athdf" else "processed_hdf5",
                "variable_names": variable_names or channels or [],
                "coordinate_system": decoded(
                    handle.attrs.get(
                        "Coordinates",
                        handle.get("metadata", {}).attrs.get("coordinates")
                        if "metadata" in handle
                        else None,
                    )
                ),
                "snapshot_number": (
                    int(snapshot_match.group(1)) if snapshot_match else None
                ),
                "time": (
                    float(handle.attrs["Time"])
                    if "Time" in handle.attrs
                    else None
                ),
                "cycle": (
                    int(handle.attrs["NumCycles"])
                    if "NumCycles" in handle.attrs
                    else None
                ),
                "mesh_metadata": {
                    key: decoded(handle.attrs[key])
                    for key in (
                        "MaxLevel",
                        "MeshBlockSize",
                        "NumMeshBlocks",
                        "RootGridSize",
                        "RootGridX1",
                        "RootGridX2",
                        "RootGridX3",
                    )
                    if key in handle.attrs
                },
                "objects": objects,
                "eos_gamma_metadata": eos_metadata(attributes),
                "source_files": source_files or [],
                "source_times": times or [],
                "provenance_relation": (
                    "raw_snapshot_sequence"
                    if path.suffix == ".athdf"
                    else "processed_or_derived_hdf5"
                ),
            }
        )
    return record


def is_sensitive(path: str) -> bool:
    lowered = path.lower()
    return any(fragment in lowered for fragment in SENSITIVE_FRAGMENTS)


def physics_key(path: str) -> bool:
    terminal = re.split(r"[.\[\]/]", path.lower())[-1]
    return terminal in PHYSICS_KEYS or any(
        term in terminal
        for term in ("gamma", "eos", "eint", "internal_energy", "spin")
    )


def flatten_physics(value: Any, prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if not is_sensitive(path):
                result.update(flatten_physics(child, path))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if prefix and physics_key(prefix) and len(value) <= 64:
            result[prefix] = decoded(value)
    elif prefix and physics_key(prefix):
        result[prefix] = decoded(value)
    return result


def parse_athinput(text: str) -> dict[str, Any]:
    section = "root"
    result: dict[str, Any] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("<") and line.endswith(">"):
            section = line[1:-1].strip()
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        path = f"{section}.{key}"
        if is_sensitive(path) or not physics_key(path):
            continue
        scalar = value.split("#", 1)[0].strip()
        if scalar.lower() in {"true", "false"}:
            parsed: Any = scalar.lower() == "true"
        else:
            try:
                parsed = int(scalar)
            except ValueError:
                try:
                    parsed = float(scalar)
                except ValueError:
                    parsed = scalar[:256]
        result[path] = parsed
    return result


def inspect_config(path: Path, *, source: str) -> dict[str, Any]:
    record = base_record(path, source=source)
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.name.startswith("athinput"):
        metadata = parse_athinput(text)
        kind = "athena_input"
    else:
        try:
            if path.suffix == ".json":
                parsed = json.loads(text)
            elif path.suffix == ".toml":
                parsed = tomllib.loads(text)
            elif path.suffix == ".yaml":
                parsed = yaml.safe_load(text)
            else:
                parsed = {}
            metadata = flatten_physics(parsed)
        except (json.JSONDecodeError, tomllib.TOMLDecodeError, yaml.YAMLError):
            metadata = {}
        kind = "metadata_or_config"
    record.update(
        {
            "kind": kind,
            "physics_metadata": metadata,
            "provenance_relation": (
                "unrelated_athena_sample"
                if source == "athena_public_samples"
                else "project_or_dataset_metadata_candidate"
            ),
        }
    )
    return record


def discover(root: Path, *, data: bool, project: bool) -> list[Path]:
    if not root.exists():
        return []
    result: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if ".git" in path.parts or "__pycache__" in path.parts:
            continue
        if "stage_j" in path.parts and "paper_reduced100" in path.parts:
            continue
        if data and (path.suffix in DATA_SUFFIXES or path.name.startswith("athinput")):
            result.append(path)
        elif data and path.suffix in {".ini", ".yaml", ".toml"}:
            result.append(path)
        elif project and (
            path.name.startswith("athinput") or path.suffix in CONFIG_SUFFIXES
        ):
            result.append(path)
    return sorted(set(result), key=lambda item: str(item))


def exact_config_match(
    record: Mapping[str, Any], raw_reference: Mapping[str, Any]
) -> bool:
    metadata = record.get("physics_metadata", {})
    if not isinstance(metadata, Mapping):
        return False
    expected_grid = raw_reference["mesh_metadata"]["RootGridSize"]
    expected_x1 = raw_reference["mesh_metadata"]["RootGridX1"]
    values = {key.lower(): value for key, value in metadata.items()}
    coordinate_values = [
        str(value).lower() for key, value in values.items() if "coord" in key
    ]
    return (
        any("kerr-schild" in value for value in coordinate_values)
        and values.get("mesh.nx1") == expected_grid[0]
        and values.get("mesh.nx2") == expected_grid[1]
        and values.get("mesh.nx3") == expected_grid[2]
        and math.isclose(float(values.get("mesh.x1min", math.inf)), expected_x1[0])
        and math.isclose(float(values.get("mesh.x1max", math.inf)), expected_x1[1])
    )


def summarize(
    records: list[dict[str, Any]], *, raw_link: Path, canonical_hdf5: Path
) -> dict[str, Any]:
    raw = sorted(
        (record for record in records if record["kind"] == "raw_athdf"),
        key=lambda item: item["snapshot_number"],
    )
    indices = [int(record["snapshot_number"]) for record in raw]
    times = [float(record["time"]) for record in raw]
    cycles = [int(record["cycle"]) for record in raw]
    missing = (
        sorted(set(range(indices[0], indices[-1] + 1)) - set(indices))
        if indices
        else []
    )
    dts = np.diff(np.asarray(times, dtype=np.float64)).tolist()
    raw_names = {Path(record["resolved_path"]).name: record for record in raw}
    processed_checks: dict[str, Any] = {}
    for record in records:
        if record["kind"] != "processed_hdf5" or not record["source_files"]:
            continue
        source_files = [str(value) for value in record["source_files"]]
        source_times = [float(value) for value in record["source_times"]]
        present = all(name in raw_names for name in source_files)
        exact_times = present and len(source_files) == len(source_times) and all(
            math.isclose(
                source_times[index],
                float(raw_names[name]["time"]),
                rel_tol=0.0,
                abs_tol=0.0,
            )
            for index, name in enumerate(source_files)
        )
        processed_checks[record["resolved_path"]] = {
            "source_file_count": len(source_files),
            "all_source_files_present": present,
            "source_times_exact": exact_times,
            "source_range": [source_files[0], source_files[-1]],
        }
        record["provenance_relation"] = (
            "traceable_to_local_raw_snapshots"
            if present and exact_times
            else "processed_provenance_unverified"
        )

    configs = [record for record in records if record["kind"] == "athena_input"]
    matching_configs = (
        [record["resolved_path"] for record in configs if exact_config_match(record, raw[0])]
        if raw
        else []
    )
    return {
        "raw_link": str(raw_link.absolute()),
        "raw_link_resolved": str(raw_link.resolve()),
        "raw_snapshot_count": len(raw),
        "raw_snapshot_number_range": [indices[0], indices[-1]] if indices else None,
        "missing_snapshot_numbers": missing,
        "exactly_300_snapshots": len(raw) == 300 and not missing,
        "last_250_available": len(raw) >= 250,
        "time_range": [times[0], times[-1]] if times else None,
        "cycle_range": [cycles[0], cycles[-1]] if cycles else None,
        "time_strictly_increasing": all(b > a for a, b in zip(times, times[1:])),
        "cadence": {
            "count": len(dts),
            "min": min(dts) if dts else None,
            "max": max(dts) if dts else None,
            "mean": statistics.fmean(dts) if dts else None,
            "median": statistics.median(dts) if dts else None,
            "approximately_constant_rtol_1e-3": (
                bool(
                    np.allclose(
                        np.asarray(dts),
                        statistics.median(dts),
                        rtol=1.0e-3,
                        atol=0.0,
                    )
                )
                if dts
                else None
            ),
        },
        "coordinate_systems": sorted(
            {str(record["coordinate_system"]) for record in raw}
        ),
        "variable_sets": sorted(
            {tuple(record["variable_names"]) for record in raw}
        ),
        "raw_eos_gamma_metadata_present": any(
            bool(record["eos_gamma_metadata"]) for record in raw
        ),
        "matching_run_config_status": (
            "exact_matching_candidate_found" if matching_configs else "not_found"
        ),
        "matching_run_configs": matching_configs,
        "athena_input_candidate_count": len(configs),
        "processed_provenance": processed_checks,
        "canonical_processed_hdf5": str(canonical_hdf5.resolve()),
        "canonical_processed_present": canonical_hdf5.is_file(),
        "snapshot_protocol_status": "BLOCKED_300_SNAPSHOT_PROTOCOL",
        "thermal_channel": "press",
        "eint_status": "BLOCKED_UNVERIFIED_EOS",
        "conversion_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--datasets-root", type=Path, default=Path("/home/curl/datasets"))
    parser.add_argument("--raw-link", type=Path)
    parser.add_argument(
        "--athena-sample-root",
        type=Path,
        default=Path("/home/curl/athena-public-version/inputs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/paper_reduced100/stage_j/data_provenance.json"),
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    raw_link = args.raw_link or root / "data_raw"
    output = args.output if args.output.is_absolute() else root / args.output
    canonical = root / "data_proc/grmhd_regrid_inner_r200_64.h5"

    candidates: list[tuple[Path, str]] = []
    candidates.extend(
        (path, "datasets_root")
        for path in discover(args.datasets_root, data=True, project=False)
    )
    if raw_link.exists():
        candidates.extend(
            (path, "raw_link_target")
            for path in discover(raw_link.resolve(), data=True, project=False)
        )
    candidates.extend(
        (path, "project") for path in discover(root, data=False, project=True)
    )
    candidates.extend(
        (path, "athena_public_samples")
        for path in discover(args.athena_sample_root, data=True, project=False)
        if path.name.startswith("athinput")
    )
    unique: dict[str, tuple[Path, str]] = {}
    for path, source in candidates:
        unique.setdefault(str(path.resolve()), (path, source))

    records: list[dict[str, Any]] = []
    for index, (path, source) in enumerate(unique.values(), start=1):
        print(f"[{index}/{len(unique)}] {path}", flush=True)
        if path.suffix in DATA_SUFFIXES:
            record = inspect_hdf5(path, source=source)
        else:
            record = inspect_config(path, source=source)
        records.append(record)
    records.sort(key=lambda item: item["resolved_path"])
    payload = {
        "schema_version": "paper-stage-j-data-provenance-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "audit_scope": {
            "datasets_root": str(args.datasets_root.resolve()),
            "project_root": str(root),
            "raw_link": str(raw_link.absolute()),
            "raw_link_target": str(raw_link.resolve()),
            "athena_sample_root": str(args.athena_sample_root.resolve()),
            "field_arrays_read": False,
            "data_files_modified": False,
            "secrets_copied": False,
        },
        "summary": summarize(records, raw_link=raw_link, canonical_hdf5=canonical),
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
