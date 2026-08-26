from __future__ import annotations

import h5py
import numpy as np

from build_regrid_from_athdf import (
    TargetGrid,
    build_target_bin_average_mapping,
    build_trilinear_mapping,
    interpolation_brackets,
    periodic_interpolation_brackets,
    regrid_channel_trilinear,
    regrid_channel_target_bin_average,
)
from grmhd import CHANNELS


def make_single_block(path) -> None:
    r = np.asarray([1.0, 2.0])
    theta = np.asarray([0.25, 0.75])
    phi = np.asarray([0.5, 1.5])
    base = (
        phi[:, None, None]
        + 2.0 * theta[None, :, None]
        + 3.0 * r[None, None, :]
    )
    values = np.stack([base + channel for channel in range(8)], axis=0)[:, None]
    with h5py.File(path, "w") as handle:
        handle.attrs["DatasetNames"] = np.asarray(["prim"], dtype="S")
        handle.attrs["NumVariables"] = np.asarray([8], dtype=np.int32)
        handle.attrs["VariableNames"] = np.asarray(CHANNELS, dtype="S")
        handle.create_dataset("prim", data=values.astype(np.float32))
        handle.create_dataset("Levels", data=np.asarray([0], dtype=np.int32))
        handle.create_dataset(
            "LogicalLocations", data=np.asarray([[0, 0, 0]], dtype=np.int64)
        )
        for name, data in {
            "x1f": [[0.5, 1.5, 2.5]],
            "x1v": [r],
            "x2f": [[0.0, 0.5, 1.0]],
            "x2v": [theta],
            "x3f": [[0.0, 1.0, 2.0]],
            "x3v": [phi],
        }.items():
            handle.create_dataset(name, data=np.asarray(data, dtype=np.float64))


def test_interpolation_brackets_clamp_edges_and_broadcast() -> None:
    values = np.asarray([0.0, 1.0, 1.5, 2.0, 3.0])
    lower, upper, weight = interpolation_brackets(values, np.asarray([1.0, 2.0]))
    np.testing.assert_array_equal(lower, [0, 0, 0, 1, 1])
    np.testing.assert_array_equal(upper, [0, 0, 1, 1, 1])
    np.testing.assert_allclose(weight, [0.0, 0.0, 0.5, 0.0, 0.0])


def test_periodic_brackets_normalize_query_to_selected_block() -> None:
    lower, upper, weight = periodic_interpolation_brackets(
        np.asarray([-1.75, 2.75]),
        np.asarray([0.25, 0.75]),
        0.0,
        1.0,
        2.0,
    )
    np.testing.assert_array_equal(lower, [0, 1])
    np.testing.assert_array_equal(upper, [0, 1])
    np.testing.assert_array_equal(weight, [0.0, 0.0])


def test_leaf_trilinear_exact_for_linear_field_and_source_centres(tmp_path) -> None:
    path = tmp_path / "single.athdf"
    make_single_block(path)
    grid = TargetGrid(
        r=np.asarray([1.0, 1.5]),
        theta=np.asarray([0.25, 0.5]),
        phi=np.asarray([0.5, 1.25]),
        r_edges=np.asarray([0.5, 1.5, 2.5]),
        theta_edges=np.asarray([0.0, 0.5, 1.0]),
        phi_edges=np.asarray([0.0, 1.0, 2.0]),
    )
    with h5py.File(path, "r") as handle:
        mapping = build_trilinear_mapping(handle, grid)
        output = regrid_channel_trilinear(handle, "Bcc1", mapping)
    expected = (
        grid.phi[:, None, None]
        + 2.0 * grid.theta[None, :, None]
        + 3.0 * grid.r[None, None, :]
    )
    np.testing.assert_allclose(output, expected, rtol=0, atol=1e-7)
    assert output[0, 0, 0] == 4.0  # exact source-cell interpolation consistency


def test_target_bin_coordinate_weighted_average_and_empty_fallback(tmp_path) -> None:
    path = tmp_path / "single.athdf"
    make_single_block(path)
    grid = TargetGrid(
        r=np.asarray([1.0, 2.0]),
        theta=np.asarray([0.125, 0.375, 0.625, 0.875]),
        phi=np.asarray([0.5, 1.5]),
        r_edges=np.asarray([0.5, 1.5, 2.5]),
        theta_edges=np.asarray([0.0, 0.25, 0.5, 0.75, 1.0]),
        phi_edges=np.asarray([0.0, 1.0, 2.0]),
    )
    with h5py.File(path, "r") as handle:
        mapping = build_target_bin_average_mapping(handle, grid)
        output = regrid_channel_target_bin_average(handle, "rho", mapping)
    # Half of the theta bins contain no source centre and use deterministic nearest fallback.
    assert np.count_nonzero(mapping.target_weight_sums == 0) == 8
    assert output.shape == (2, 4, 2)
    assert np.isfinite(output).all()
