#!/usr/bin/env python3
"""No-training frozen-operator radial-position identifiability diagnostic."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
import yaml

from grmhd.shells import radial_shells_tensor
from grmhd.stage_x_audit import classify_radial_identifiability
from train_stage_s import build_frozen_model


def tensor_hash(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def symmetric_distance(left: torch.Tensor, right: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm((left - right).double())
    denominator = torch.sqrt(
        0.5 * (torch.sum(left.double().square()) + torch.sum(right.double().square()))
    )
    return float((numerator / torch.clamp(denominator, min=1.0e-30)).cpu())


def cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    a = left.double().reshape(-1)
    b = right.double().reshape(-1)
    return float((torch.dot(a, b) / torch.clamp(torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b), min=1e-30)).cpu())


def make_pattern(shape: tuple[int, int, int], center_r: int, half_width: int) -> torch.Tensor:
    """A deterministic identical 3D stencil translated only along tensor r."""

    nphi, ntheta, nr = shape
    value = torch.zeros((1, 8, nphi, ntheta, nr), dtype=torch.float32)
    phi0, theta0 = nphi // 2, ntheta // 2
    coordinates = torch.arange(-half_width, half_width + 1, dtype=torch.float32)
    pp, tt, rr = torch.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    envelope = torch.exp(-0.5 * (pp.square() + tt.square() + rr.square()))
    modulation = 1.0 + 0.15 * pp - 0.10 * tt + 0.05 * rr
    stencil = envelope * modulation
    if not half_width <= center_r < nr - half_width:
        raise ValueError("pattern would cross the radial boundary")
    for channel in range(8):
        scale = ((-1.0) ** channel) * (channel + 1) / 8.0
        value[
            0,
            channel,
            phi0 - half_width : phi0 + half_width + 1,
            theta0 - half_width : theta0 + half_width + 1,
            center_r - half_width : center_r + half_width + 1,
        ] = scale * stencil
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-config", type=Path, default=Path("configs/stage_x/method_gap_audit.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/stage_s/expanded_localno_p3_residual.yaml"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/stage_x/coordinates/radial_position_identifiability.csv"))
    args = parser.parse_args()

    audit = yaml.safe_load(args.audit_config.read_text(encoding="utf-8"))
    config = yaml.safe_load(args.model_config.read_text(encoding="utf-8"))
    checkpoint_path = Path(audit["coordinates"]["frozen_checkpoint"])
    indices = [int(value) for value in audit["coordinates"]["radial_indices"]]
    half_width = int(audit["coordinates"]["local_pattern_half_width"])
    thresholds = audit["coordinates"]["identifiability_thresholds"]

    if not torch.cuda.is_available():
        raise RuntimeError("Stage X frozen 64^3 diagnostic requires CUDA; CPU fallback is disabled")
    device = torch.device("cuda")
    model = build_frozen_model(config)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        for parameter in model.parameters():
            parameter.requires_grad_(False)

    dataset = Path(config["data"]["dataset"])
    with h5py.File(dataset, "r") as handle:
        r = np.asarray(handle["coords/r"], dtype=np.float64)
        shape = tuple(int(value) for value in handle["snapshots"].shape[2:])
    shells, metadata = radial_shells_tensor(r, shape[0], shape[1], n_shells=8)
    shells = shells.unsqueeze(0).to(device=device, dtype=torch.float32)
    zeros_physical = torch.zeros((1, 8, *shape), device=device, dtype=torch.float32)
    zeros_shells = torch.zeros_like(shells)

    responses: dict[tuple[str, int], torch.Tensor] = {}
    hashes: dict[tuple[str, int], str] = {}
    anchor = indices[len(indices) // 2]
    with torch.inference_mode():
        baselines = {
            "shell_off": model(x=torch.cat((zeros_physical, zeros_shells), dim=1)),
            "shell_on": model(x=torch.cat((zeros_physical, shells), dim=1)),
        }
        for mode, shell_input in (("shell_off", zeros_shells), ("shell_on", shells)):
            for index in indices:
                pattern = make_pattern(shape, index, half_width).to(device)
                response = model(x=torch.cat((pattern, shell_input), dim=1)) - baselines[mode]
                aligned = torch.roll(response, shifts=anchor - index, dims=-1).detach()
                if not torch.isfinite(aligned).all():
                    raise FloatingPointError(f"nonfinite response for {mode}/r-index={index}")
                responses[(mode, index)] = aligned
                hashes[(mode, index)] = tensor_hash(aligned)

    rows: list[dict[str, object]] = []
    off_distances: list[float] = []
    on_distances: list[float] = []
    names = dict(zip(indices, ("inner", "middle", "outer"), strict=True))
    for left_index, right_index in ((indices[0], indices[1]), (indices[0], indices[2]), (indices[1], indices[2])):
        values: dict[str, float] = {}
        for mode in ("shell_off", "shell_on"):
            left = responses[(mode, left_index)]
            right = responses[(mode, right_index)]
            distance = symmetric_distance(left, right)
            values[mode] = distance
            rows.append({
                "record_type": "pair_response",
                "shell_mode": mode,
                "left_region": names[left_index],
                "right_region": names[right_index],
                "left_r_index": left_index,
                "right_r_index": right_index,
                "left_r": float(r[left_index]),
                "right_r": float(r[right_index]),
                "symmetric_relative_distance": distance,
                "cosine": cosine(left, right),
                "left_response_sha256": hashes[(mode, left_index)],
                "right_response_sha256": hashes[(mode, right_index)],
            })
        off_distances.append(values["shell_off"])
        on_distances.append(values["shell_on"])

    classification = classify_radial_identifiability(
        off_distances,
        on_distances,
        off_max=float(thresholds["shell_off_relative_distance_max"]),
        on_min=float(thresholds["shell_on_relative_distance_min"]),
        has_continuous_coordinates=False,
    )
    rows.append({
        "record_type": "classification",
        "shell_mode": "paired",
        "radial_position_identifiability": classification,
        "shell_off_max_distance": max(off_distances),
        "shell_on_min_distance": min(on_distances),
        "shell_on_to_off_min_ratio": min(
            on / max(off, 1.0e-30) for on, off in zip(on_distances, off_distances, strict=True)
        ),
        "continuous_coordinate_channels": False,
        "shell_count": metadata.n_shells,
        "shell_edges_json": json.dumps(metadata.edges),
        "checkpoint": str(checkpoint_path),
        "completed_epoch": int(payload["training_state"]["completed_epoch"]),
        "device": torch.cuda.get_device_name(0),
        "training_performed": False,
    })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({
        "output": str(args.out),
        "classification": classification,
        "shell_off_distances": off_distances,
        "shell_on_distances": on_distances,
        "device": torch.cuda.get_device_name(0),
    }, indent=2))


if __name__ == "__main__":
    main()
