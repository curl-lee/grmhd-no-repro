import numpy as np
import torch

from grmhd.shells import radial_shells, radial_shells_tensor


def test_shell_shape_one_hot_and_log_edges():
    r = np.geomspace(1.0, 200.0, 64)
    shells, metadata = radial_shells(r, nphi=5, ntheta=7)
    assert shells.shape == (8, 5, 7, 64)
    assert shells.dtype == np.float32
    np.testing.assert_array_equal(shells.sum(axis=0), np.ones((5, 7, 64)))
    assert np.all(np.diff(metadata.edges) > 0)
    ratios = np.asarray(metadata.edges[1:]) / np.asarray(metadata.edges[:-1])
    np.testing.assert_allclose(ratios, ratios[0])
    # Every phi/theta location receives the same radial coordinate-derived shell.
    np.testing.assert_array_equal(shells[:, 0, 0], shells[:, -1, -1])


def test_tensor_shells_do_not_require_normalization():
    values, metadata = radial_shells_tensor(np.geomspace(2.0, 20.0, 16), 3, 4)
    assert values.shape == (8, 3, 4, 16)
    assert values.dtype == torch.float32
    assert torch.all(values.sum(dim=0) == 1)
    assert metadata.coordinate == "spherical Kerr-Schild r"
