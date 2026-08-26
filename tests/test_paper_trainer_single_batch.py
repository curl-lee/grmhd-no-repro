from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from grmhd.paper_config import build_paper_training_loss, load_paper_experiment_config
from grmhd.paper_data_processor import PaperDataProcessor
from grmhd.paper_protocol import PaperReduced100Protocol
from grmhd.paper_trainer import (
    PaperTrainerAdapter,
    PaperTrainerLossAdapter,
    parameter_gradient_norm,
)
from grmhd.paper_velocity_roi import paper_roi_ramp
from grmhd.upstream_adapters import (
    GRMHDNextStepDataset,
    UpstreamFNOConfig,
    build_upstream_fno,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def batch():
    protocol = PaperReduced100Protocol.from_yaml(
        PROJECT_ROOT / "configs/data/paper_reduced100.yaml", project_root=PROJECT_ROOT
    )
    dataset = GRMHDNextStepDataset(protocol.make_datasets()["train"])
    return next(iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)))


def make_model(seed: int):
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


@pytest.mark.parametrize(
    ("epoch", "expected"),
    [(0, 0.0), (1, 1 / 375), (374, 374 / 375), (375, 1.0), (900, 1.0)],
)
def test_roi_epoch_ramp_contract(epoch, expected):
    assert paper_roi_ramp(epoch, 375) == pytest.approx(expected)


@pytest.mark.parametrize("mode", ["full", "plain"])
def test_upstream_trainer_batch_matches_manual_path(mode, batch):
    config_path = PROJECT_ROOT / f"configs/paper_reduced100/{'full_fno_proxy' if mode == 'full' else 'plain_l2_fno'}.yaml"
    config = load_paper_experiment_config(config_path, project_root=PROJECT_ROOT)
    manual_model = make_model(seed=31)
    trainer_model = make_model(seed=31)
    manual_processor = PaperDataProcessor.from_config(config)
    trainer_processor = PaperDataProcessor.from_config(config)
    manual_loss = PaperTrainerLossAdapter(build_paper_training_loss(config))
    trainer_loss = PaperTrainerLossAdapter(build_paper_training_loss(config))
    epoch = 374
    manual_processor.set_epoch(epoch)
    manual_loss.set_epoch(epoch)

    manual_sample = manual_processor.preprocess(batch)
    manual_output = manual_model(**manual_sample)
    manual_output, manual_fields = manual_processor.postprocess(
        manual_output, manual_sample
    )
    manual_value = manual_loss(manual_output, **manual_fields)
    manual_value.backward()
    manual_gradient_norm = parameter_gradient_norm(manual_model.parameters())

    optimizer = torch.optim.SGD(trainer_model.parameters(), lr=0.0)
    adapter = PaperTrainerAdapter(
        model=trainer_model,
        data_processor=trainer_processor,
        loss=trainer_loss,
        optimizer=optimizer,
        n_epochs=1,
        device="cpu",
    )
    adapter.set_epoch(epoch)
    trainer_result = adapter.compute_batch(batch, clear_gradients=True)
    trainer_result.loss.backward()
    trainer_gradient_norm = parameter_gradient_norm(trainer_model.parameters())

    torch.testing.assert_close(
        trainer_result.normalized_prediction, manual_output, rtol=0, atol=0
    )
    torch.testing.assert_close(trainer_result.loss, manual_value, rtol=1e-6, atol=1e-6)
    assert trainer_gradient_norm == pytest.approx(manual_gradient_norm, rel=1e-6)
    assert trainer_result.context.protocol_metadata == manual_fields["protocol_metadata"]
    assert trainer_result.context.epoch == epoch
    if mode == "full":
        assert manual_loss.last_result is not None and trainer_loss.last_result is not None
        for key, manual_component in manual_loss.last_result.component_tensors().items():
            torch.testing.assert_close(
                trainer_loss.last_result.component_tensors()[key],
                manual_component,
                rtol=1e-6,
                atol=1e-6,
            )
        assert trainer_loss.last_result.roi_ramp.item() == pytest.approx(374 / 375)
    else:
        assert trainer_loss.last_log == manual_loss.last_log

    # This integration path intentionally never calls optimizer.step().
    for manual_parameter, trainer_parameter in zip(
        manual_model.parameters(), trainer_model.parameters()
    ):
        torch.testing.assert_close(manual_parameter, trainer_parameter, rtol=0, atol=0)
