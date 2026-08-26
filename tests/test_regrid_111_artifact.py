from pathlib import Path

import h5py
import numpy as np
import pytest

from grmhd import CHANNELS


def test_workspace_regrid_has_111_snapshots_when_available():
    path = Path("data_proc/grmhd_regrid_inner_r200_64.h5")
    if not path.exists():
        pytest.skip("external regrid artifact is not present")
    with h5py.File(path, "r") as handle:
        assert handle["snapshots"].shape == (111, 8, 64, 64, 64)
        assert handle["times"].shape == (111,)
        assert np.all(np.diff(handle["times"][...]) > 0)
        assert len(handle["source_files"]) == 111
        assert list(handle["channels"].asstr()[...]) == list(CHANNELS)
