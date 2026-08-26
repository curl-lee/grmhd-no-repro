import json

import pytest
import torch

from grmhd.coupling_contract import (
    BoundaryExchange,
    BoundaryState,
    CoarseState,
    CouplingSchedule,
    FineState,
    GridMetadata,
    ProlongationOperator,
    RestrictionOperator,
    TimeStepRatio,
    interpolate_boundary,
)


def grid(shape=(8, 8, 8), channels=("rho", "press")) -> GridMetadata:
    return GridMetadata(
        shape=shape,
        channel_names=channels,
        coordinates="toy_cartesian_uniform",
        spacing=(1.0, 1.0, 1.0),
        component_basis="Cartesian coordinate basis",
        magnetic_staggering="none",
    )


def test_state_shapes_channels_times_and_ownership_are_explicit():
    metadata = grid()
    tensor = torch.zeros(1, 2, 8, 8, 8)
    coarse = CoarseState(tensor, metadata, time_index=3, physical_time=1.5)
    fine = FineState(tensor, metadata, time_index=6, physical_time=1.5)

    assert coarse.tensor.shape == (1, 2, 8, 8, 8)
    assert coarse.autoregressive_owner == "coarse_solver"
    assert fine.autoregressive_owner == "fine_solver"
    with pytest.raises(ValueError, match="spatial shape"):
        CoarseState(torch.zeros(1, 2, 7, 8, 8), metadata, 0, 0.0)
    with pytest.raises(ValueError, match="channel count"):
        FineState(torch.zeros(1, 1, 8, 8, 8), metadata, 0, 0.0)


def test_restriction_and_prolongation_shapes_and_constant_preservation():
    fine = torch.full((2, 3, 8, 10, 12), 2.75)
    restricted = RestrictionOperator((2, 2, 3))(fine)
    prolonged = ProlongationOperator((8, 10, 12))(restricted)

    assert restricted.shape == (2, 3, 4, 5, 4)
    assert prolonged.shape == fine.shape
    assert torch.equal(restricted, torch.full_like(restricted, 2.75))
    assert torch.equal(prolonged, torch.full_like(prolonged, 2.75))


def test_trilinear_prolongation_preserves_linear_field():
    z, y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 3),
        torch.linspace(-2.0, 2.0, 4),
        torch.linspace(0.0, 3.0, 5),
        indexing="ij",
    )
    coarse = (1.5 * z - 0.25 * y + 2.0 * x + 0.75)[None, None]
    result = ProlongationOperator((7, 9, 11))(coarse)
    zf, yf, xf = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 7),
        torch.linspace(-2.0, 2.0, 9),
        torch.linspace(0.0, 3.0, 11),
        indexing="ij",
    )
    expected = (1.5 * zf - 0.25 * yf + 2.0 * xf + 0.75)[None, None]

    torch.testing.assert_close(result, expected, atol=2e-6, rtol=2e-6)


def test_temporal_synchronization_and_boundary_interpolation():
    schedule = CouplingSchedule(
        TimeStepRatio(4),
        coarse_start_index=10,
        fine_start_index=40,
        physical_start_time=2.0,
        coarse_dt=0.8,
    )
    assert schedule.fine_dt == pytest.approx(0.2)
    assert schedule.coarse_bracket(40) == (10, 10, 0.0)
    assert schedule.coarse_bracket(43) == (10, 11, 0.75)
    assert schedule.coarse_bracket(44) == (11, 11, 0.0)
    assert schedule.is_synchronization_step(44)
    assert not schedule.is_synchronization_step(43)
    assert schedule.physical_time(43) == pytest.approx(2.6)

    metadata = grid()
    left = BoundaryState(torch.zeros(1, 2, 8, 8, 8), metadata, 10, 2.0, 1)
    right = BoundaryState(torch.full((1, 2, 8, 8, 8), 4.0), metadata, 11, 2.8, 1)
    middle = interpolate_boundary(left, right, 0.75, time_index=43, physical_time=2.6)
    assert torch.equal(middle.tensor, torch.full_like(middle.tensor, 3.0))


def test_boundary_exchange_updates_only_selected_ghost_zones_and_channels():
    interior = torch.zeros(1, 3, 6, 7, 8)
    boundary = torch.full_like(interior, 9.0)
    exchange = BoundaryExchange(
        1,
        faces=("z_low", "x_high"),
        channel_indices=(0, 2),
        magnetic_channel_indices=(),
    )
    result = exchange(interior, boundary)

    assert torch.equal(result[:, 0, 0], torch.full_like(result[:, 0, 0], 9.0))
    assert torch.equal(result[:, 2, :, :, -1], torch.full_like(result[:, 2, :, :, -1], 9.0))
    assert torch.count_nonzero(result[:, 1]) == 0
    assert torch.count_nonzero(result[:, :, 1:-1, 1:-1, 1:-1]) == 0
    assert torch.count_nonzero(interior) == 0  # input was not overwritten in place


def test_unverified_cell_centered_magnetic_exchange_is_rejected():
    with pytest.raises(ValueError, match="CT/EMF"):
        BoundaryExchange(
            1,
            channel_indices=(0, 1, 2, 3),
            magnetic_channel_indices=(0, 1, 2),
        )


def test_reference_operators_have_finite_backward():
    fine = torch.randn(1, 2, 8, 8, 8, requires_grad=True)
    boundary = torch.randn(1, 2, 8, 8, 8, requires_grad=True)
    restricted = RestrictionOperator()(fine)
    prolonged = ProlongationOperator((8, 8, 8))(restricted)
    exchanged = BoundaryExchange(1)(prolonged, boundary)
    exchanged.square().mean().backward()

    assert fine.grad is not None and torch.isfinite(fine.grad).all()
    assert boundary.grad is not None and torch.isfinite(boundary.grad).all()


def test_reference_operators_are_deterministic_and_parameter_free():
    torch.manual_seed(7)
    value = torch.randn(1, 2, 8, 8, 8)
    restriction = RestrictionOperator()
    prolongation = ProlongationOperator((8, 8, 8))

    first = prolongation(restriction(value))
    second = prolongation(restriction(value))
    assert torch.equal(first, second)
    assert restriction.state_dict() == {}
    assert prolongation.state_dict() == {}


def test_metadata_schedule_and_state_descriptors_are_json_serializable():
    metadata = grid()
    schedule = CouplingSchedule(TimeStepRatio(3), coarse_dt=0.6)
    state = CoarseState(torch.zeros(1, 2, 8, 8, 8), metadata, 0, 0.0)

    encoded = json.dumps(
        {
            "grid": metadata.to_dict(),
            "schedule": schedule.to_dict(),
            "state": state.descriptor(),
        },
        sort_keys=True,
    )
    decoded = json.loads(encoded)

    assert GridMetadata.from_dict(decoded["grid"]) == metadata
    assert CouplingSchedule.from_dict(decoded["schedule"]) == schedule
    assert decoded["state"]["tensor_shape"] == [1, 2, 8, 8, 8]
