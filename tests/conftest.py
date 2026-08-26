from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from grmhd import CHANNELS
from grmhd.dataset import sha256_file
from grmhd.paper_preprocessing import PaperPreprocessor
from grmhd.paper_bounds import fit_paper_physical_bounds
from grmhd.paper_dissipation import fit_dissipative_reference
from grmhd.paper_losses import PaperCompositeLoss, PaperLossContext
from grmhd.paper_priors import (
    PAPER_COORDINATE_SYSTEM,
    PAPER_PROTOCOL_NAME,
    PaperResidualEnvelope,
    PriorProvenance,
)
from grmhd.paper_radial import fit_appendix_literal_press_proxy
from grmhd.paper_velocity_roi import (
    PaperVelocityROI,
    canonical_oracle_velocity_roi_mask,
    raw_physical_velocity_roi_diagnostic_mask,
)


@pytest.fixture
def paper_prior_factory(tmp_path):
    def make(name: str, *, validation_multiplier: float = 1.0):
        path = tmp_path / f"{name}.h5"
        r = np.geomspace(1.0, 20.0, 4)
        shape = (3, 3, len(r))
        values = np.empty((8, 8, *shape), dtype=np.float32)
        radial = r.reshape(1, 1, -1)
        phi = np.arange(shape[0], dtype=np.float32).reshape(-1, 1, 1)
        theta = np.arange(shape[1], dtype=np.float32).reshape(1, -1, 1)
        for index in range(len(values)):
            phase = 1.0 + 0.02 * index + 0.005 * phi + 0.003 * theta
            values[index, 0] = 0.01 * np.sin(radial + index)
            values[index, 1] = 0.02 * np.cos(radial + index)
            values[index, 2] = -0.015 * np.sin(0.5 * radial + index)
            values[index, 3] = phase * np.power(radial, -0.7)
            values[index, 4] = 0.1 * phase * np.power(radial, -0.9)
            values[index, 5] = 0.01 * phase * radial
            values[index, 6] = -0.02 * phase * radial
            values[index, 7] = 0.03 * phase * radial
        values[5:, 3:5] *= validation_multiplier
        values[5:, 5:8] *= validation_multiplier
        with h5py.File(path, "w") as handle:
            handle.create_dataset("snapshots", data=values)
            handle.create_dataset("times", data=np.arange(len(values), dtype=np.float64))
            handle.create_dataset(
                "channels", data=np.asarray(CHANNELS, dtype=h5py.string_dtype())
            )
            coords = handle.create_group("coords")
            coords.create_dataset("r", data=r)
            coords.create_dataset("theta", data=np.linspace(0.2, 2.9, shape[1]))
            coords.create_dataset("phi", data=np.linspace(0.1, 6.0, shape[0]))
        train_indices = tuple(range(5))
        validation_indices = tuple(range(5, 8))
        checksum = sha256_file(path)
        preprocessor = PaperPreprocessor.fit_hdf5(
            path,
            training_indices=train_indices,
            protocol_name=PAPER_PROTOCOL_NAME,
            expected_source_hdf5_checksum=checksum,
        )
        provenance = PriorProvenance(
            source_hdf5_checksum=checksum,
            preprocessing_stats_checksum=f"stats-{name}",
            training_indices=train_indices,
        )
        return {
            "path": path,
            "r": r,
            "values": values,
            "train_indices": train_indices,
            "validation_indices": validation_indices,
            "preprocessor": preprocessor,
            "provenance": provenance,
        }

    return make


@pytest.fixture
def paper_loss_factory(paper_prior_factory):
    def make(name: str, *, epoch: int | float = 375):
        case = paper_prior_factory(name)
        radial = fit_appendix_literal_press_proxy(
            str(case["path"]),
            training_indices=case["train_indices"],
            r=case["r"],
            preprocessor=case["preprocessor"],
            provenance=case["provenance"],
        )
        bounds = fit_paper_physical_bounds(
            case["path"],
            training_indices=case["train_indices"],
            preprocessor=case["preprocessor"],
            provenance=case["provenance"],
        )
        envelope = PaperResidualEnvelope(case["provenance"], radial.mode)
        roi = PaperVelocityROI(case["provenance"])
        dissipation = fit_dissipative_reference(
            case["path"],
            training_indices=case["train_indices"],
            preprocessor=case["preprocessor"],
            provenance=case["provenance"],
        )
        raw_input = torch.from_numpy(case["values"][0]).unsqueeze(0)
        raw_target = torch.from_numpy(case["values"][1]).unsqueeze(0)
        normalized_input = case["preprocessor"].encode(raw_input, channel_axis=1)
        normalized_target = case["preprocessor"].encode(raw_target, channel_axis=1)
        oracle_target = case["preprocessor"].decode(normalized_target, channel_axis=1)
        canonical_mask, _ = canonical_oracle_velocity_roi_mask(
            raw_target, case["preprocessor"]
        )
        raw_mask = raw_physical_velocity_roi_diagnostic_mask(raw_target)
        baseline = radial.state((3, 3, 4), normalized=True, batch_size=1)
        context = PaperLossContext(
            normalized_input=normalized_input,
            normalized_target=normalized_target,
            raw_physical_target=raw_target,
            oracle_physical_target=oracle_target,
            canonical_roi_mask=canonical_mask,
            raw_roi_diagnostic_mask=raw_mask,
            radial_baseline_normalized=baseline,
            normalized_bounds=bounds.normalized_bounds,
            epoch=epoch,
            snapshot_indices=(1,),
            protocol_metadata={
                "protocol_name": PAPER_PROTOCOL_NAME,
                "thermal_channel": "press",
                "paper_adaptation": True,
                "validation_not_used_for_fit": True,
                "selected_radial_mode": radial.mode,
                "coordinate_system": PAPER_COORDINATE_SYSTEM,
                "canonical_roi_source": "oracle_physical_target",
            },
        )
        loss = PaperCompositeLoss(
            bounds=bounds,
            envelope=envelope,
            roi=roi,
            dissipation=dissipation,
            radial_metadata=radial.metadata,
        )
        return {
            **case,
            "radial": radial,
            "bounds": bounds,
            "envelope": envelope,
            "roi": roi,
            "dissipation": dissipation,
            "context": context,
            "loss": loss,
        }

    return make
