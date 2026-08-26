from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch
import yaml

from grmhd.paper_bounds import PaperPhysicalBounds
from grmhd.paper_config import (
    PAIR_ALLOWED_DIFFERENCES,
    build_paper_model,
    build_paper_training_loss,
    difference_is_allowed,
    load_paper_experiment_config,
    paired_config_audit,
    recursive_config_differences,
    write_paired_config_audit,
)
from grmhd.paper_dissipation import PaperDissipativeReference
from grmhd.paper_losses import PaperCompositeLoss, PlainL2Loss
from grmhd.paper_priors import PaperResidualEnvelope
from grmhd.paper_radial import PaperRadialBaseline
from grmhd.paper_velocity_roi import PaperVelocityROI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_PATH = PROJECT_ROOT / "configs/paper_reduced100/full_fno_proxy.yaml"
PLAIN_PATH = PROJECT_ROOT / "configs/paper_reduced100/plain_l2_fno.yaml"


@pytest.fixture(scope="module")
def paired_configs():
    return (
        load_paper_experiment_config(FULL_PATH, project_root=PROJECT_ROOT),
        load_paper_experiment_config(PLAIN_PATH, project_root=PROJECT_ROOT),
    )


def test_resolved_pair_diff_is_strictly_whitelisted(paired_configs):
    full, plain = paired_configs
    differences = recursive_config_differences(full.as_dict(), plain.as_dict())
    assert PAIR_ALLOWED_DIFFERENCES == {
        "experiment_name",
        "loss.*",
        "output_dir",
        "checkpoint_prefix",
    }
    assert differences
    assert all(difference_is_allowed(item["path"]) for item in differences)
    assert {item["path"] for item in differences}.issuperset(
        {"experiment_name", "output_dir", "checkpoint_prefix", "loss.name"}
    )

    changed = plain.as_dict()
    changed["runtime"]["seed"] += 1
    unexpected = recursive_config_differences(full.as_dict(), changed)
    assert any(
        item["path"] == "runtime.seed" and not difference_is_allowed(item["path"])
        for item in unexpected
    )


def test_models_have_equal_parameter_count_and_identical_seeded_state(paired_configs):
    full, plain = paired_configs
    seed = full.values["runtime"]["seed"]
    torch.manual_seed(seed)
    full_model = build_paper_model(full)
    torch.manual_seed(seed)
    plain_model = build_paper_model(plain)
    assert sum(parameter.numel() for parameter in full_model.parameters()) == sum(
        parameter.numel() for parameter in plain_model.parameters()
    )
    full_state = full_model.state_dict()
    plain_state = plain_model.state_dict()
    assert full_state.keys() == plain_state.keys()
    for key in full_state:
        if torch.is_tensor(full_state[key]):
            assert torch.equal(full_state[key], plain_state[key]), key
        else:
            assert full_state[key] == plain_state[key], key


def test_plain_builder_never_instantiates_full_prior(paired_configs, monkeypatch):
    _, plain = paired_configs

    def forbidden(*args, **kwargs):
        raise AssertionError("Plain config attempted to instantiate a Full prior")

    for prior in (
        PaperRadialBaseline,
        PaperPhysicalBounds,
        PaperResidualEnvelope,
        PaperVelocityROI,
        PaperDissipativeReference,
    ):
        monkeypatch.setattr(prior, "load", forbidden)
    loss = build_paper_training_loss(plain)
    assert isinstance(loss, PlainL2Loss)
    assert loss.metadata["enabled_priors"] == []


def test_full_builder_loads_frozen_stage_d_contract(paired_configs):
    full, _ = paired_configs
    loss = build_paper_training_loss(full)
    assert isinstance(loss, PaperCompositeLoss)
    assert loss.metadata["schema_version"] == "paper-composite-loss-v1"
    assert loss.metadata["h1_weight"] == 0.05
    assert loss.metadata["roi_kappa"] == 8
    assert loss.metadata["radial_mode"] == "appendix_literal_press_proxy"


@pytest.mark.parametrize("source", [FULL_PATH, PLAIN_PATH])
def test_both_modes_reject_stale_artifact_checksum(source, tmp_path):
    values = yaml.safe_load(source.read_text(encoding="utf-8"))
    values["provenance"]["artifacts"]["bounds"]["sha256"] = "0" * 64
    stale = tmp_path / source.name
    stale.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="Stale Stage D bounds checksum"):
        load_paper_experiment_config(stale, project_root=PROJECT_ROOT)


def test_plain_evaluation_clamp_is_not_part_of_training_loss(paired_configs):
    _, plain = paired_configs
    assert plain.values["evaluation"]["rho_press_eval_clamp"] is True
    loss = build_paper_training_loss(plain)
    prediction = torch.zeros(1, 8, 2, 2, 2)
    target = torch.zeros_like(prediction)
    prediction[:, 3] = 100.0
    prediction[:, 4] = -100.0
    expected = prediction.square().mean(dim=(0, 2, 3, 4)).sum()
    actual = loss(prediction, target)
    assert torch.equal(actual, expected)
    assert actual.item() == 20000.0


def test_resolved_pair_preserves_complete_checksum_metadata(paired_configs):
    full, plain = paired_configs
    for config in (full, plain):
        provenance = config.values["provenance"]
        assert set(provenance["artifacts"]) == {
            "radial",
            "bounds",
            "roi",
            "envelope",
            "dissipation",
            "shells",
        }
        checksum_records = [
            provenance["dataset"],
            provenance["manifest"],
            provenance["preprocessing"],
            *provenance["artifacts"].values(),
            provenance["stage_e_loss_contract"],
            provenance["oracle_baseline"],
            provenance["loss_contract_document"],
        ]
        assert all(len(record["sha256"]) == 64 for record in checksum_records)
        assert (
            provenance["oracle_reference_semantics_version"]
            == "paper-reference-semantics-v1"
        )
        assert provenance["validation_not_used_for_fit"] is True


def test_audit_output_records_no_unexpected_difference(paired_configs, tmp_path):
    full, plain = paired_configs
    audit = paired_config_audit(full, plain)
    assert audit["status"] == "passed"
    assert audit["unexpected_differences"] == []
    assert audit["model_initialization"]["counts_equal"] is True
    assert audit["model_initialization"]["state_identical"] is True
    write_paired_config_audit(audit, tmp_path)
    assert (tmp_path / "paired_config_diff.json").is_file()
    markdown = (tmp_path / "paired_config_diff.md").read_text(encoding="utf-8")
    assert "Unexpected differences: `0`" in markdown
