from __future__ import annotations

import torch

from grmhd.paper_h1_diagnostics import current_upstream_h1
from grmhd.paper_h1_extensions import (
    DiagnosticH1Extension,
    DiagnosticPaperCompositeLoss,
)
from grmhd.paper_losses import PaperLossContext
from grmhd.shells import radial_shells_tensor


def geometry(size: int = 64):
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


def test_unit_index_has_exact_4096_spacing_ratio_on_64_cubed():
    coordinates, shells = geometry()
    generator = torch.Generator().manual_seed(19)
    target = torch.zeros(1, 8, 64, 64, 64)
    prediction = torch.randn(target.shape, generator=generator)
    unit = DiagnosticH1Extension(
        mode="unit_index",
        coordinates=coordinates,
        shell_masks=shells,
    ).components(prediction, target)
    current = current_upstream_h1(prediction, target)
    torch.testing.assert_close(
        current / unit.raw_h1,
        current.new_tensor(4096.0),
        rtol=2.0e-5,
        atol=2.0e-3,
    )
    torch.testing.assert_close(unit.per_shell.sum(), unit.raw_h1)
    torch.testing.assert_close(
        torch.stack(tuple(unit.per_direction.values())).sum(),
        unit.raw_h1,
    )
    torch.testing.assert_close(unit.per_channel.sum(), unit.raw_h1)


def test_no_h1_selected_value_and_prediction_gradient_are_exactly_zero():
    coordinates, shells = geometry(size=8)
    prediction = torch.randn(
        1,
        8,
        8,
        8,
        8,
        requires_grad=True,
    )
    target = torch.zeros_like(prediction)
    result = DiagnosticH1Extension(
        mode="no_h1",
        coordinates=coordinates,
        shell_masks=shells,
    ).components(prediction, target)
    gradient = torch.autograd.grad(result.weighted_h1, prediction)[0]
    assert result.raw_h1.item() == 0.0
    assert result.weighted_h1.item() == 0.0
    assert torch.count_nonzero(gradient) == 0
    assert torch.count_nonzero(result.per_channel) == 0
    assert torch.count_nonzero(result.per_shell) == 0


def test_stored_coordinate_volume_proxy_is_finite_additive_and_explicitly_proxy():
    coordinates, shells = geometry(size=8)
    prediction = torch.randn(2, 8, 8, 8, 8, requires_grad=True)
    target = torch.randn_like(prediction)
    result = DiagnosticH1Extension(
        mode="stored_coordinate_volume_proxy",
        coordinates=coordinates,
        shell_masks=shells,
    ).components(prediction, target)
    assert torch.isfinite(result.raw_h1)
    assert torch.isfinite(result.per_channel).all()
    assert torch.isfinite(result.per_shell).all()
    torch.testing.assert_close(result.per_channel.sum(), result.raw_h1)
    torch.testing.assert_close(result.per_shell.sum(), result.raw_h1)
    assert result.metadata["covariant_GRMHD_H1"] is False
    assert result.metadata["proper_Kerr_Schild_volume"] == "unverified"
    assert result.metadata["stored_components_covariant_derivative"] is False
    assert result.metadata["stored_vector_covariant_derivative"] is False
    assert result.metadata["diagnostic_proxy_only"] is True
    gradient = torch.autograd.grad(result.weighted_h1, prediction)[0]
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0


def test_extension_rejects_shell_channels_as_prediction_targets():
    coordinates, shells = geometry(size=8)
    loss = DiagnosticH1Extension(
        mode="unit_index",
        coordinates=coordinates,
        shell_masks=shells,
    )
    prediction = torch.zeros(1, 16, 8, 8, 8)
    target = torch.zeros_like(prediction)
    try:
        loss(prediction, target)
    except ValueError as error:
        assert "extra shell channels" in str(error)
    else:
        raise AssertionError("Stage I H1 accepted 16 output channels")


def test_composite_logs_unit_index_h1_detached_from_selected_training_term(
    paper_loss_factory,
):
    case = paper_loss_factory("stage-i-stored-composite")
    shape = (4, 4, 4)
    radius = torch.logspace(0.0, 1.0, shape[-1], dtype=torch.float64)
    shells, _ = radial_shells_tensor(
        radius.numpy(),
        shape[0],
        shape[1],
        n_shells=8,
    )
    coordinates = {
        "phi": (
            torch.arange(shape[0], dtype=torch.float64) + 0.5
        ) * (2.0 * torch.pi / shape[0]),
        "theta": torch.linspace(0.2, 2.9, shape[1], dtype=torch.float64),
        "r": radius,
    }
    loss = DiagnosticPaperCompositeLoss(
        diagnostic_h1_mode="stored_coordinate_volume_proxy",
        coordinates=coordinates,
        shell_masks=shells.bool(),
        bounds=case["bounds"],
        envelope=case["envelope"],
        roi=case["roi"],
        dissipation=case["dissipation"],
        radial_metadata=case["radial"].metadata,
    )
    generator = torch.Generator().manual_seed(72)
    normalized_target = torch.randn(
        1, 8, *shape, generator=generator
    ) * 0.01
    raw_target = torch.randn(1, 8, *shape, generator=generator) * 0.01
    raw_target[:, 3:5] = raw_target[:, 3:5].abs() + 0.1
    context = PaperLossContext(
        normalized_input=torch.zeros_like(normalized_target),
        normalized_target=normalized_target,
        raw_physical_target=raw_target,
        oracle_physical_target=raw_target.clone(),
        canonical_roi_mask=torch.zeros(1, *shape, dtype=torch.bool),
        raw_roi_diagnostic_mask=torch.zeros(1, *shape, dtype=torch.bool),
        radial_baseline_normalized=torch.zeros_like(normalized_target),
        normalized_bounds=case["bounds"].normalized_bounds,
        epoch=0,
        snapshot_indices=(1,),
        protocol_metadata=case["context"].protocol_metadata,
    )
    prediction = (
        normalized_target.detach().clone()
        + 0.01 * torch.randn(
            normalized_target.shape,
            generator=generator,
        )
    ).requires_grad_(True)
    result = loss.components(prediction, context=context)
    selected = result.h1_raw
    unit = result.diagnostics["diagnostic_unit_index_h1_raw"]
    current = result.diagnostics["diagnostic_current_upstream_h1_raw"]
    assert selected.requires_grad is True
    assert unit.requires_grad is False
    assert current.requires_grad is False
    assert torch.isfinite(selected)
    assert torch.isfinite(unit)
    assert torch.isfinite(current)
    selected_gradient = torch.autograd.grad(selected, prediction)[0]
    assert torch.count_nonzero(selected_gradient) > 0
