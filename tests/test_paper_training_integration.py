from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from grmhd.paper_config import (
    build_paper_training_loss,
    load_paper_experiment_config,
)
from grmhd.paper_data_processor import PaperDataProcessor
from grmhd.paper_protocol import PaperReduced100Protocol
from grmhd.paper_trainer import PaperTrainerLossAdapter
from grmhd.paper_velocity_roi import stored_component_speed_proxy, top_fraction_mask
from grmhd.upstream_adapters import (
    GRMHDNextStepDataset,
    UpstreamFNOConfig,
    build_upstream_fno,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "full": PROJECT_ROOT / "configs/paper_reduced100/full_fno_proxy.yaml",
    "plain": PROJECT_ROOT / "configs/paper_reduced100/plain_l2_fno.yaml",
}


@pytest.fixture(scope="module")
def real_train_batch():
    protocol = PaperReduced100Protocol.from_yaml(
        PROJECT_ROOT / "configs/data/paper_reduced100.yaml", project_root=PROJECT_ROOT
    )
    dataset = GRMHDNextStepDataset(protocol.make_datasets()["train"])
    return next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))


def make_processor(mode: str) -> tuple[object, PaperDataProcessor]:
    config = load_paper_experiment_config(CONFIGS[mode], project_root=PROJECT_ROOT)
    return config, PaperDataProcessor.from_config(config)


def test_processor_to_moves_registered_representation_buffers():
    _, processor = make_processor("full")
    returned = processor.to("meta")
    assert returned is processor
    assert processor.device.type == "meta"
    assert processor.shells.device.type == "meta"
    assert processor.radial_baseline_normalized.device.type == "meta"


def tiny_fno(seed: int = 17):
    torch.manual_seed(seed)
    return build_upstream_fno(
        UpstreamFNOConfig(
            in_channels=16,
            out_channels=8,
            n_modes=(2, 2, 2),
            hidden_channels=4,
            n_layers=1,
        )
    )


def test_processor_produces_explicit_same_space_real_batch(real_train_batch):
    _, full_processor = make_processor("full")
    _, plain_processor = make_processor("plain")
    full_processor.set_epoch(1)
    plain_processor.set_epoch(1)
    full_model_sample = full_processor.preprocess(real_train_batch)
    plain_model_sample = plain_processor.preprocess(real_train_batch)
    full = full_processor.last_batch
    plain = plain_processor.last_batch
    assert full is not None and plain is not None

    assert full_model_sample["x"].shape == (1, 16, 64, 64, 64)
    assert full_model_sample["y"].shape == (1, 8, 64, 64, 64)
    assert full_model_sample["x"].dtype == torch.float32
    assert full_model_sample["x"].device.type == "cpu"
    torch.testing.assert_close(full_model_sample["x"][:, :8], full["normalized_input"])
    torch.testing.assert_close(full_model_sample["x"][:, 8:].sum(dim=1), torch.ones(1, 64, 64, 64))
    torch.testing.assert_close(full_model_sample["y"], full["normalized_target"])
    assert full["source_snapshot_index"] == (11,)
    assert full["target_snapshot_index"] == (12,)
    assert full["input_time"].dtype == torch.float64
    assert full["target_time"].item() > full["input_time"].item()

    expected_input = full_processor.preprocessor.encode(
        real_train_batch["physical_input"], channel_axis=1
    )
    expected_target = full_processor.preprocessor.encode(
        real_train_batch["physical_target"], channel_axis=1
    )
    expected_oracle = full_processor.preprocessor.decode(expected_target, channel_axis=1)
    expected_roi = top_fraction_mask(stored_component_speed_proxy(expected_oracle), 0.20)
    torch.testing.assert_close(full["normalized_input"], expected_input)
    torch.testing.assert_close(full["normalized_target"], expected_target)
    torch.testing.assert_close(full["oracle_physical_target"], expected_oracle)
    assert torch.equal(full["canonical_roi_mask"], expected_roi)
    assert full["context"].raw_roi_diagnostic_mask is None
    assert full["raw_roi_diagnostic_mask"] is not None
    assert set(full["normalized_bounds"]) == {"rho", "press"}
    baseline = full["radial_baseline_normalized"]
    assert torch.count_nonzero(baseline[:, :3]) == 0
    assert torch.count_nonzero(baseline[:, 5:]) == 0
    assert torch.count_nonzero(baseline[:, 3:5]) > 0
    assert (
        full["protocol_metadata"]["representation_semantics"]
        == "direct_canonical_state_radial_envelope_reference_only"
    )

    for key in (
        "model_input",
        "normalized_input",
        "normalized_target",
        "raw_physical_target",
        "oracle_physical_target",
        "canonical_roi_mask",
        "radial_baseline_normalized",
    ):
        if torch.is_tensor(full[key]):
            torch.testing.assert_close(full[key], plain[key], rtol=0, atol=0)
    assert full_processor.transform_counts == {
        "input_encode": 1,
        "target_encode": 1,
        "oracle_decode": 1,
        "prediction_decode": 0,
    }
    assert plain_processor.transform_counts == full_processor.transform_counts


def test_full_and_plain_have_identical_initial_model_output(real_train_batch):
    _, full_processor = make_processor("full")
    _, plain_processor = make_processor("plain")
    full_sample = full_processor.preprocess(real_train_batch)
    plain_sample = plain_processor.preprocess(real_train_batch)
    full_model = tiny_fno(seed=23)
    plain_model = tiny_fno(seed=23)
    full_output = full_model(x=full_sample["x"])
    plain_output = plain_model(x=plain_sample["x"])
    torch.testing.assert_close(full_output, plain_output, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["full", "plain"])
def test_real_batch_upstream_fno_loss_backward_without_optimizer_step(
    mode, real_train_batch
):
    config, processor = make_processor(mode)
    processor.set_epoch(1)
    loss = PaperTrainerLossAdapter(build_paper_training_loss(config))
    loss.set_epoch(1)
    model = tiny_fno()
    model_sample = processor.preprocess(real_train_batch)
    normalized_prediction = model(x=model_sample["x"])
    normalized_prediction, loss_sample = processor.postprocess(
        normalized_prediction, model_sample
    )
    value = loss(normalized_prediction, **loss_sample)
    assert value.ndim == 0 and torch.isfinite(value)
    assert normalized_prediction.shape == loss_sample["normalized_target"].shape
    assert normalized_prediction.dtype == loss_sample["normalized_target"].dtype
    assert normalized_prediction.device == loss_sample["normalized_target"].device
    value.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    assert any(gradient is not None for gradient in gradients)
    assert all(
        torch.isfinite(gradient).all() for gradient in gradients if gradient is not None
    )
    assert processor.transform_counts["prediction_decode"] == 0
    assert loss_sample["context"].epoch == 1
    if mode == "full":
        assert loss.last_result is not None
        assert loss.last_result.roi_ramp.item() == pytest.approx(1 / 375)
    else:
        assert loss.last_result is None
        assert loss.last_log["disabled_components"] == [
            "h1",
            "roi",
            "bounds_training_penalty",
            "radial_envelope",
            "dissipation",
        ]
