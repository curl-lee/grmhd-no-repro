from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from grmhd.dataset import sha256_file
from grmhd.paper_checkpoint import (
    build_paper_checkpoint_metadata,
    load_paper_checkpoint,
    save_paper_checkpoint,
)
from grmhd.paper_config import build_paper_model, load_paper_experiment_config
from grmhd.paper_trainer import build_paper_optimizer, build_warmup_cosine_scheduler
from grmhd.paper_velocity_roi import paper_roi_ramp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FULL_PATH = PROJECT_ROOT / "configs/paper_reduced100/full_fno_proxy.yaml"
PLAIN_PATH = PROJECT_ROOT / "configs/paper_reduced100/plain_l2_fno.yaml"


def make_state(config):
    torch.manual_seed(91)
    model = build_paper_model(config)
    optimizer = build_paper_optimizer(
        model,
        learning_rate=config.values["optimizer"]["learning_rate"],
        weight_decay=config.values["optimizer"]["weight_decay"],
    )
    scheduler = build_warmup_cosine_scheduler(
        optimizer,
        total_epochs=2,
        warmup_epochs=2,
        min_learning_rate=config.values["scheduler"]["min_learning_rate"],
    )
    return model, optimizer, scheduler


def test_checkpoint_strict_model_optimizer_scheduler_and_prediction_reload(tmp_path):
    config = load_paper_experiment_config(FULL_PATH, project_root=PROJECT_ROOT)
    model, optimizer, scheduler = make_state(config)
    # Populate gradients without optimizer.step; checkpoint integration is not a training test.
    probe = torch.randn(1, 16, 8, 8, 8)
    prediction = model(x=probe)
    prediction.square().mean().backward()
    scheduler.last_epoch = 1
    metadata = build_paper_checkpoint_metadata(
        config,
        project_commit="test-project-commit",
        config_checksum=sha256_file(FULL_PATH),
        epoch=7,
        experiment_name="checkpoint-test",
        gradient_accumulation=4,
    )
    checkpoint = tmp_path / "best.pt"
    save_paper_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
    )

    reloaded_model, reloaded_optimizer, reloaded_scheduler = make_state(config)
    loaded = load_paper_checkpoint(
        checkpoint,
        config=config,
        model=reloaded_model,
        optimizer=reloaded_optimizer,
        scheduler=reloaded_scheduler,
        expected_config_checksum=sha256_file(FULL_PATH),
    )
    assert loaded.epoch == 7
    assert loaded.metadata["roi_ramp"] == paper_roi_ramp(7, 375)
    assert loaded.optimizer.state_dict() == optimizer.state_dict()
    assert loaded.scheduler.state_dict() == scheduler.state_dict()
    with torch.no_grad():
        reloaded_prediction = loaded.model(x=probe)
    torch.testing.assert_close(reloaded_prediction, prediction.detach(), rtol=0, atol=0)
    assert (checkpoint / "paper_state_dict.pt").is_file()
    assert (checkpoint / "optimizer.pt").is_file()
    assert (checkpoint / "scheduler.pt").is_file()
    assert (checkpoint / "manifest.pt").is_file()


def test_checkpoint_rejects_stale_checksum_before_state_load(tmp_path):
    config = load_paper_experiment_config(FULL_PATH, project_root=PROJECT_ROOT)
    model, optimizer, scheduler = make_state(config)
    metadata = build_paper_checkpoint_metadata(
        config,
        project_commit="test",
        config_checksum=sha256_file(FULL_PATH),
        epoch=2,
        experiment_name="stale-test",
        gradient_accumulation=1,
    )
    checkpoint = tmp_path / "last.pt"
    save_paper_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
    )
    sidecar = checkpoint / "paper_grmhd_metadata.json"
    stale = json.loads(sidecar.read_text(encoding="utf-8"))
    stale["preprocessing_checksum"] = "0" * 64
    sidecar.write_text(json.dumps(stale), encoding="utf-8")
    target, target_optimizer, target_scheduler = make_state(config)
    before = {name: value.clone() for name, value in target.state_dict().items() if torch.is_tensor(value)}
    with pytest.raises(ValueError, match="preprocessing_checksum"):
        load_paper_checkpoint(
            checkpoint,
            config=config,
            model=target,
            optimizer=target_optimizer,
            scheduler=target_scheduler,
        )
    for name, value in target.state_dict().items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)


@pytest.mark.parametrize(
    ("saved_path", "load_path", "saved_mode", "load_mode"),
    [
        (FULL_PATH, PLAIN_PATH, "full", "plain"),
        (PLAIN_PATH, FULL_PATH, "plain", "full"),
    ],
)
def test_full_plain_checkpoint_cross_load_is_rejected(
    saved_path, load_path, saved_mode, load_mode, tmp_path
):
    saved_config = load_paper_experiment_config(saved_path, project_root=PROJECT_ROOT)
    load_config = load_paper_experiment_config(load_path, project_root=PROJECT_ROOT)
    model, optimizer, scheduler = make_state(saved_config)
    metadata = build_paper_checkpoint_metadata(
        saved_config,
        project_commit="test",
        config_checksum=sha256_file(saved_path),
        epoch=1,
        experiment_name=f"{saved_mode}-test",
        gradient_accumulation=1,
    )
    checkpoint = tmp_path / f"{saved_mode}.pt"
    save_paper_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
    )
    target, target_optimizer, target_scheduler = make_state(load_config)
    with pytest.raises(ValueError, match="mode"):
        load_paper_checkpoint(
            checkpoint,
            config=load_config,
            model=target,
            optimizer=target_optimizer,
            scheduler=target_scheduler,
        )


def test_checkpoint_rejects_epoch_ramp_mismatch(tmp_path):
    config = load_paper_experiment_config(PLAIN_PATH, project_root=PROJECT_ROOT)
    model, optimizer, scheduler = make_state(config)
    metadata = build_paper_checkpoint_metadata(
        config,
        project_commit="test",
        config_checksum=sha256_file(PLAIN_PATH),
        epoch=375,
        experiment_name="ramp-test",
        gradient_accumulation=1,
    )
    metadata["roi_ramp"] = 0.0
    checkpoint = tmp_path / "ramp.pt"
    save_paper_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
    )
    target, target_optimizer, target_scheduler = make_state(config)
    with pytest.raises(ValueError, match="epoch/ROI ramp"):
        load_paper_checkpoint(
            checkpoint,
            config=config,
            model=target,
            optimizer=target_optimizer,
            scheduler=target_scheduler,
        )


@pytest.mark.parametrize(
    ("config_path", "h1_weight"),
    [(FULL_PATH, 0.05), (PLAIN_PATH, None)],
)
def test_stage_g_checkpoint_metadata_roundtrip(config_path, h1_weight, tmp_path):
    config = load_paper_experiment_config(config_path, project_root=PROJECT_ROOT)
    model, optimizer, scheduler = make_state(config)
    stage_g = {
        "shared_initial_state_sha256": "1" * 64,
        "pair_order_sha256": "2" * 64,
        "optimizer_step": 20,
        "h1_weight": h1_weight,
        "gradient_clip_norm": 1.0,
        "resource_scaled_training": True,
    }
    metadata = build_paper_checkpoint_metadata(
        config,
        project_commit="test",
        config_checksum=sha256_file(config_path),
        epoch=1,
        experiment_name="stage-g-test",
        gradient_accumulation=4,
        stage_g=stage_g,
    )
    checkpoint = tmp_path / "stage-g"
    save_paper_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
    )
    target, target_optimizer, target_scheduler = make_state(config)
    loaded = load_paper_checkpoint(
        checkpoint,
        config=config,
        model=target,
        optimizer=target_optimizer,
        scheduler=target_scheduler,
        expected_config_checksum=sha256_file(config_path),
    )
    assert loaded.metadata["stage_g"] == stage_g
