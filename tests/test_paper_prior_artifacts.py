from __future__ import annotations

import pytest

from grmhd.paper_priors import (
    PAPER_PROTOCOL_NAME,
    PaperResidualEnvelope,
    PriorProvenance,
    read_prior_json,
    write_prior_json,
)


def test_provenance_requires_fit_identity_and_rejects_all_mismatches(tmp_path):
    provenance = PriorProvenance(
        source_hdf5_checksum="hdf5",
        preprocessing_stats_checksum="stats",
        training_indices=(1, 2, 3),
    )
    envelope = PaperResidualEnvelope(provenance, "appendix_literal_press_proxy")
    path = tmp_path / "envelope.json"
    write_prior_json(path, envelope.as_dict())
    payload = read_prior_json(
        path,
        expected_schema=PaperResidualEnvelope.schema_version,
        source_hdf5_checksum="hdf5",
        preprocessing_stats_checksum="stats",
        training_indices=(1, 2, 3),
        protocol_name=PAPER_PROTOCOL_NAME,
        thermal_channel="press",
    )
    restored = PaperResidualEnvelope.from_dict(payload)
    assert restored == envelope
    expected = {
        "source_hdf5_checksum": "hdf5",
        "preprocessing_stats_checksum": "stats",
        "training_indices": (1, 2, 3),
        "protocol_name": PAPER_PROTOCOL_NAME,
        "thermal_channel": "press",
    }
    for key, value, match in (
        ("source_hdf5_checksum", "wrong", "HDF5 checksum mismatch"),
        ("preprocessing_stats_checksum", "wrong", "preprocessing checksum mismatch"),
        ("training_indices", (1, 2), "training indices mismatch"),
        ("protocol_name", "wrong", "protocol mismatch"),
        ("thermal_channel", "eint", "thermal channel mismatch"),
    ):
        with pytest.raises(ValueError, match=match):
            read_prior_json(
                path,
                expected_schema=envelope.schema_version,
                **{**expected, key: value},
            )


def test_validation_used_for_fit_and_noncanonical_metadata_are_rejected():
    with pytest.raises(ValueError, match="Validation data cannot"):
        PriorProvenance(
            source_hdf5_checksum="hdf5",
            preprocessing_stats_checksum="stats",
            training_indices=(1,),
            validation_not_used_for_fit=False,
        )
    with pytest.raises(ValueError, match="thermal adaptation"):
        PriorProvenance(
            source_hdf5_checksum="hdf5",
            preprocessing_stats_checksum="stats",
            training_indices=(1,),
            thermal_channel="eint",
        )
