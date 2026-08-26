from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from grmhd.dataset import sha256_file
from grmhd.paper_config import (
    build_paper_model,
    build_paper_training_loss,
    load_paper_experiment_config,
)
from grmhd.paper_data_processor import PaperDataProcessor
from grmhd.paper_protocol import PaperReduced100Protocol
from grmhd.paper_stage_g import (
    tensor_state_sha256,
    validate_epoch_pair_order,
)
from grmhd.upstream_adapters import GRMHDNextStepDataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATHS = {
    "full": PROJECT_ROOT / "configs/paper_reduced100/full_fno_proxy.yaml",
    "no_h1": PROJECT_ROOT
    / "configs/paper_reduced100/extensions/no_h1_control.yaml",
    "unit_index": PROJECT_ROOT
    / "configs/paper_reduced100/extensions/unit_index_h1.yaml",
    "stored_coordinate_volume_proxy": PROJECT_ROOT
    / "configs/paper_reduced100/extensions/stored_coordinate_volume_h1.yaml",
}
SHARED_STATE_PATH = (
    PROJECT_ROOT / "outputs/paper_reduced100/stage_g/shared_initial_state.pt"
)
PAIR_ORDER_PATH = (
    PROJECT_ROOT / "outputs/paper_reduced100/stage_g/epoch_pair_order.json"
)
EXPECTED_SHARED_HASH = (
    "02d83278d5e74a27c83130e99b488ca18eae5477dad9732a4552b694056b6cb1"
)
EXPECTED_PAIR_ORDER_HASH = (
    "5e1e361ea75e01ffd667822d8bfe870d70c5bc69570d52cdf806e637b2df2b52"
)
NON_H1_COMPONENTS = (
    "base_fidelity_raw",
    "base_fidelity_weighted",
    "roi_raw",
    "roi_ramp",
    "roi_weighted",
    "bounds_rho_raw",
    "bounds_press_raw",
    "bounds_weighted",
    "envelope_rho_raw",
    "envelope_press_raw",
    "envelope_weighted",
    "dissipation_raw",
    "dissipation_weighted",
)


@pytest.fixture(scope="module")
def configs():
    return {
        name: load_paper_experiment_config(path, project_root=PROJECT_ROOT)
        for name, path in CONFIG_PATHS.items()
    }


def load_shared_state():
    return torch.load(SHARED_STATE_PATH, map_location="cpu", weights_only=True)


def test_stage_i_reuses_exact_stage_g_state_and_pair_order(configs):
    state = load_shared_state()
    assert tensor_state_sha256(state) == EXPECTED_SHARED_HASH
    assert sha256_file(PAIR_ORDER_PATH) == EXPECTED_PAIR_ORDER_HASH
    order = json.loads(PAIR_ORDER_PATH.read_text(encoding="utf-8"))
    orders = validate_epoch_pair_order(order)
    assert len(orders) == 30
    assert all(len(epoch) == 79 for epoch in orders)
    assert order["validation_shuffle"] is False

    full = configs["full"].values
    for name, config in configs.items():
        values = config.values
        for key in (
            "protocol",
            "thermal",
            "coordinates",
            "preprocessing",
            "representation",
            "evaluation",
            "model",
            "optimizer",
            "scheduler",
            "runtime",
            "provenance",
        ):
            assert values[key] == full[key], (name, key)


def test_all_variants_strictly_load_same_state_and_match_epoch_zero_output(configs):
    state = load_shared_state()
    generator = torch.Generator().manual_seed(42)
    probe = torch.randn(1, 16, 16, 16, 16, generator=generator)
    reference_output = None
    for name, config in configs.items():
        model = build_paper_model(config).eval()
        model.load_state_dict(state, strict=True)
        assert sum(parameter.numel() for parameter in model.parameters()) == 331832
        assert tensor_state_sha256(model.state_dict()) == EXPECTED_SHARED_HASH
        with torch.no_grad():
            output = model(x=probe)
        if reference_output is None:
            reference_output = output
        else:
            assert torch.equal(output, reference_output), name


@pytest.fixture(scope="module")
def real_loss_case(configs):
    protocol = PaperReduced100Protocol.from_yaml(
        PROJECT_ROOT / "configs/data/paper_reduced100.yaml",
        project_root=PROJECT_ROOT,
    )
    dataset = GRMHDNextStepDataset(protocol.make_datasets()["train"])
    batch = next(
        iter(DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0))
    )
    processor = PaperDataProcessor.from_config(configs["full"])
    processor.set_epoch(0)
    processor.preprocess(batch)
    fields = processor.last_batch
    assert fields is not None
    prediction = fields["normalized_input"].clone()
    return prediction, fields["context"]


def test_all_non_h1_components_have_exact_parity(configs, real_loss_case):
    prediction, context = real_loss_case
    results = {}
    with torch.no_grad():
        for name, config in configs.items():
            loss = build_paper_training_loss(config).eval()
            results[name] = loss.components(prediction, context=context)
    reference = results["full"]
    for name in ("no_h1", "unit_index", "stored_coordinate_volume_proxy"):
        result = results[name]
        for component in NON_H1_COMPONENTS:
            assert torch.equal(
                getattr(result, component),
                getattr(reference, component),
            ), (name, component)
        assert result.metadata["paper_faithful_full"] is False
    assert results["no_h1"].h1_raw.item() == 0.0
    assert results["no_h1"].h1_weighted.item() == 0.0
