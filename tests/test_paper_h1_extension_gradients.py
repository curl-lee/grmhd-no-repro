from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from grmhd.paper_h1_extensions import DiagnosticH1Extension
from grmhd.shells import radial_shells_tensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def small_geometry(size: int = 8):
    phi = (torch.arange(size, dtype=torch.float64) + 0.5) * (
        2.0 * torch.pi / size
    )
    theta = (torch.arange(size, dtype=torch.float64) + 0.5) * (
        torch.pi / size
    )
    radius = torch.exp(
        torch.linspace(
            torch.log(torch.tensor(1.2, dtype=torch.float64)),
            torch.log(torch.tensor(20.0, dtype=torch.float64)),
            size,
        )
    )
    shells, _ = radial_shells_tensor(
        radius.numpy(),
        size,
        size,
        n_shells=8,
    )
    return {"phi": phi, "theta": theta, "r": radius}, shells.bool()


def test_stored_coordinate_h1_has_independent_batch_reduction():
    coordinates, shells = small_geometry()
    loss = DiagnosticH1Extension(
        mode="stored_coordinate_volume_proxy",
        coordinates=coordinates,
        shell_masks=shells,
    )
    generator = torch.Generator().manual_seed(71)
    prediction = torch.randn(2, 8, 8, 8, 8, generator=generator)
    target = torch.randn(2, 8, 8, 8, 8, generator=generator)
    combined = loss.components(prediction, target)
    first = loss.components(prediction[:1], target[:1])
    second = loss.components(prediction[1:], target[1:])
    torch.testing.assert_close(
        combined.raw_h1,
        (first.raw_h1 + second.raw_h1) / 2,
    )
    torch.testing.assert_close(
        combined.per_channel,
        (first.per_channel + second.per_channel) / 2,
    )
    torch.testing.assert_close(
        combined.per_shell,
        (first.per_shell + second.per_shell) / 2,
    )


def test_saved_stage_i_gpu_preflight_is_strictly_read_only_and_passed():
    path = (
        PROJECT_ROOT
        / "outputs/paper_reduced100/stage_i/gpu_preflight.json"
    )
    if not path.exists():
        pytest.skip("local Stage I GPU preflight is Git ignored")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
    assert payload["environment"]["device_name"] == "NVIDIA GeForce RTX 5070"
    assert payload["environment"]["capability"] == [12, 0]
    assert payload["frozen_pairing"][
        "shared_state_tensor_sha256"
    ] == (
        "02d83278d5e74a27c83130e99b488ca18eae5477dad9732a4552b694056b6cb1"
    )
    assert payload["safety"] == {
        "all_three_passed_before_training": True,
        "checkpoint_write": False,
        "cpu_fallback": False,
        "optimizer_created": False,
        "optimizer_step": False,
        "scheduler_step": False,
        "training_started": False,
    }
    records = {record["mode"]: record for record in payload["extensions"]}
    assert set(records) == {
        "no_h1",
        "unit_index",
        "stored_coordinate_volume_proxy",
    }
    assert all(record["status"] == "passed" for record in records.values())
    assert records["no_h1"]["mode_checks"][
        "selected_h1_parameter_gradient_exact_zero"
    ] is True
    assert records["no_h1"]["mode_checks"][
        "selected_h1_prediction_gradient_exact_zero"
    ] is True
    assert records["unit_index"]["mode_checks"]["stage_h_4096_parity"] is True
    stored = records["stored_coordinate_volume_proxy"]["mode_checks"]
    assert stored["volume_weights_normalized"] is True
    assert stored["radial_spacing_nonuniform"] is True
    assert stored["diagnostic_proxy_only"] is True
    assert stored["covariant_GRMHD_H1"] is False


def test_stage_i_preflight_source_cannot_create_training_state():
    source = (
        PROJECT_ROOT / "scripts/preflight_paper_stage_i.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "torch.optim",
        "build_paper_optimizer",
        "build_warmup_cosine_scheduler",
        "optimizer.step(",
        "scheduler.step(",
        "torch.save(",
        "save_paper_checkpoint",
        "Trainer(",
    ):
        assert forbidden not in source
    assert "result.total.backward()" in source
