from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from neuralop.layers.differential_conv import FiniteDifferenceConvolution
from neuralop.layers.discrete_continuous_convolution import (
    EquidistantDiscreteContinuousConv2d,
)
from neuralop.layers.spectral_convolution import SpectralConv

from grmhd.models import PersistenceBaseline, build_model
from grmhd.paper_checkpoint import (
    build_paper_checkpoint_metadata,
    load_paper_checkpoint,
    save_paper_checkpoint,
)
from grmhd.paper_config import (
    PLAIN_LOSS_CONTRACT,
    build_paper_model,
    build_paper_training_loss,
    load_paper_experiment_config,
    model_tensor_state_sha256,
)
from grmhd.paper_data_processor import PaperDataProcessor
from grmhd.paper_losses import PlainL2Loss
from grmhd.paper_protocol import PaperReduced100Protocol
from grmhd.paper_stage_g import tensor_state_sha256
from grmhd.paper_trainer import build_paper_optimizer, build_warmup_cosine_scheduler
from grmhd.upstream_adapters import GRMHDNextStepDataset
from scripts.train_paper_reduced import evaluate_rollout3


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_K_PATH = (
    PROJECT_ROOT
    / "configs/paper_reduced100/stage_k_localno_differential_plain.yaml"
)
FNO_PLAIN_PATH = PROJECT_ROOT / "configs/paper_reduced100/plain_l2_fno.yaml"


@pytest.fixture(scope="module")
def stage_k_config():
    return load_paper_experiment_config(STAGE_K_PATH, project_root=PROJECT_ROOT)


def tiny_localno(seed: int = 42):
    torch.manual_seed(seed)
    return build_model(
        "localno_differential_3d",
        in_channels=16,
        out_channels=8,
        default_in_shape=(8, 8, 8),
        n_modes=(2, 2, 2),
        hidden_channels=4,
        n_layers=2,
        positional_embedding=None,
        fin_diff_kernel_size=3,
        mix_derivatives=True,
        conv_padding_mode="periodic",
        use_channel_mlp=False,
        local_no_skip="linear",
        norm=None,
    )


def test_stage_k_config_builds_frozen_3d_differential_localno(stage_k_config):
    model = build_paper_model(stage_k_config)
    assert stage_k_config.stage_k is True
    assert stage_k_config.architecture == "localno_differential_3d"
    assert model.n_dim == 3
    assert sum(parameter.numel() for parameter in model.parameters()) == 358_296
    assert len(model.local_no_blocks.differential) == 4
    assert len(model.local_no_blocks.convs) == 4
    assert len(model.local_no_blocks.local_convs) == 0
    assert sum(isinstance(module, FiniteDifferenceConvolution) for module in model.modules()) == 4
    assert sum(isinstance(module, SpectralConv) for module in model.modules()) == 4
    assert not any(
        isinstance(module, EquidistantDiscreteContinuousConv2d)
        for module in model.modules()
    )
    assert not any(isinstance(module, nn.Conv2d) for module in model.modules())
    assert sum(isinstance(module, nn.Conv3d) for module in model.modules()) == 4


def test_stage_k_shape_forward_and_backward_are_finite():
    model = tiny_localno()
    value = torch.randn(1, 16, 8, 8, 8, requires_grad=True)
    prediction = model(x=value)
    assert prediction.shape == (1, 8, 8, 8, 8)
    assert torch.isfinite(prediction).all()
    prediction.square().mean().backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_volumetric_disco_factory_and_config_are_rejected(tmp_path):
    with pytest.raises(
        ValueError, match="volumetric 3D DISCO is unavailable in pinned upstream"
    ):
        build_model(
            "localno_disco",
            in_channels=16,
            out_channels=8,
            default_in_shape=(8, 8, 8),
            n_modes=(2, 2, 2),
            hidden_channels=4,
            n_layers=1,
            positional_embedding=None,
        )

    values = yaml.safe_load(STAGE_K_PATH.read_text(encoding="utf-8"))
    values["model"]["use_disco"] = True
    invalid = tmp_path / "stage_k_invalid_disco.yaml"
    invalid.write_text(yaml.safe_dump(values, sort_keys=False), encoding="utf-8")
    with pytest.raises(
        ValueError, match="volumetric 3D DISCO is unavailable in pinned upstream"
    ):
        load_paper_experiment_config(invalid, project_root=PROJECT_ROOT)


def test_localno_state_dict_strict_reload_and_seed_are_deterministic():
    first = tiny_localno(seed=42)
    second = tiny_localno(seed=42)
    assert model_tensor_state_sha256(first) == model_tensor_state_sha256(second)
    state = first.state_dict()
    second.load_state_dict(state, strict=True)
    assert tensor_state_sha256(state) == tensor_state_sha256(second.state_dict())


def test_shells_are_input_only_and_prediction_is_direct(stage_k_config):
    protocol = PaperReduced100Protocol.from_yaml(
        PROJECT_ROOT / "configs/data/paper_reduced100.yaml",
        project_root=PROJECT_ROOT,
    )
    train = GRMHDNextStepDataset(protocol.make_datasets()["train"])
    processor = PaperDataProcessor.from_config(stage_k_config)
    sample = processor.preprocess(next(iter(DataLoader(train, batch_size=1))))
    assert sample["x"].shape == (1, 16, 64, 64, 64)
    assert sample["y"].shape == (1, 8, 64, 64, 64)
    torch.testing.assert_close(
        sample["x"][:, 8:].sum(dim=1), torch.ones(1, 64, 64, 64)
    )
    prediction = PersistenceBaseline(out_channels=8)(x=sample["x"])
    assert prediction.shape == sample["y"].shape
    torch.testing.assert_close(prediction, sample["x"][:, :8])
    assert stage_k_config.values["model"]["prediction_mode"] == "direct"


def test_fno_plain_path_and_parameter_count_are_unchanged():
    config = load_paper_experiment_config(FNO_PLAIN_PATH, project_root=PROJECT_ROOT)
    torch.manual_seed(42)
    first = build_paper_model(config)
    torch.manual_seed(42)
    second = build_paper_model(config)
    assert config.stage_k is False and config.architecture == "fno"
    assert first.__class__.__name__ == "FNO"
    assert sum(parameter.numel() for parameter in first.parameters()) == 331_832
    assert model_tensor_state_sha256(first) == model_tensor_state_sha256(second)


def test_stage_k_plain_loss_has_exact_stage_g_plain_parity(stage_k_config):
    assert stage_k_config.values["loss"] == PLAIN_LOSS_CONTRACT
    loss = build_paper_training_loss(stage_k_config)
    assert isinstance(loss, PlainL2Loss)
    prediction = torch.randn(2, 8, 4, 3, 2)
    target = torch.randn_like(prediction)
    expected = (prediction - target).square().mean(dim=(0, 2, 3, 4)).sum()
    torch.testing.assert_close(loss(prediction, target), expected, rtol=0, atol=0)


def test_stage_k_checkpoint_strict_reload(stage_k_config, tmp_path):
    torch.manual_seed(42)
    model = build_paper_model(stage_k_config)
    optimizer = build_paper_optimizer(model, learning_rate=1e-3, weight_decay=1e-4)
    scheduler = build_warmup_cosine_scheduler(
        optimizer, total_epochs=2, warmup_epochs=2, min_learning_rate=1e-6
    )
    stage_k = {
        "classification": "adapted_method_reproduction",
        "architecture": "localno_differential_3d",
        "n_dim": 3,
        "differential_enabled": True,
        "disco_enabled": False,
        "parameter_count": 358_296,
        "input_channels": 16,
        "output_channels": 8,
        "prediction_mode": "direct",
        "plain_loss_contract": dict(stage_k_config.values["loss"]),
        "initial_state_sha256": "1" * 64,
        "pair_order_sha256": "2" * 64,
        "optimizer_step": 40,
        "gradient_clip_norm": 1.0,
        "resource_scaled_training": True,
        "run_kind": "engineering_smoke",
    }
    metadata = build_paper_checkpoint_metadata(
        stage_k_config,
        project_commit="test",
        config_checksum="3" * 64,
        epoch=2,
        experiment_name="stage-k-test",
        gradient_accumulation=4,
        stage_k=stage_k,
    )
    checkpoint = tmp_path / "stage-k"
    save_paper_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        metadata=metadata,
    )
    target = build_paper_model(stage_k_config)
    target_optimizer = build_paper_optimizer(
        target, learning_rate=1e-3, weight_decay=1e-4
    )
    target_scheduler = build_warmup_cosine_scheduler(
        target_optimizer, total_epochs=2, warmup_epochs=2, min_learning_rate=1e-6
    )
    loaded = load_paper_checkpoint(
        checkpoint,
        config=stage_k_config,
        model=target,
        optimizer=target_optimizer,
        scheduler=target_scheduler,
        expected_config_checksum="3" * 64,
    )
    assert loaded.metadata["stage_k"] == stage_k
    assert model_tensor_state_sha256(loaded.model) == model_tensor_state_sha256(model)


def test_three_step_rollout_transform_counter_is_exact(stage_k_config):
    protocol = PaperReduced100Protocol.from_yaml(
        PROJECT_ROOT / "configs/data/paper_reduced100.yaml",
        project_root=PROJECT_ROOT,
    )
    validation = protocol.make_datasets()["validation"]
    processor = PaperDataProcessor.from_config(stage_k_config)
    report = evaluate_rollout3(
        model=PersistenceBaseline(out_channels=8),
        processor=processor,
        validation_dataset=validation,
        epoch=2,
    )
    assert report["finite"] is True
    assert report["rho_press_positive"] is True
    assert report["no_double_transform"] is True
    assert report["transform_count_delta"] == {
        "input_encode": 3,
        "target_encode": 3,
        "oracle_decode": 3,
        "prediction_decode": 3,
    }
