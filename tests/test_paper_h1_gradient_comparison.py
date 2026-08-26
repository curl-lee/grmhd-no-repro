import json
import math
from pathlib import Path

import pytest
import torch

from grmhd.paper_h1_diagnostics import (
    flatten_optional_gradients,
    gradient_pair_metrics,
)
from grmhd.paper_stage_g import tensor_state_sha256
from grmhd.paper_stage_h import validate_frozen_input_manifest


def test_autograd_gradient_comparison_does_not_update_model():
    torch.manual_seed(41)
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 5), torch.nn.Tanh(), torch.nn.Linear(5, 3)
    )
    before = tensor_state_sha256(model.state_dict())
    inputs = torch.randn(2, 4)
    prediction = model(inputs)
    first_loss = prediction.square().mean()
    second_loss = prediction.abs().mean()
    parameters = tuple(model.parameters())
    first = torch.autograd.grad(first_loss, parameters, retain_graph=True)
    second = torch.autograd.grad(second_loss, parameters)
    first_vector = flatten_optional_gradients(first, parameters)
    second_vector = flatten_optional_gradients(second, parameters)
    metrics = gradient_pair_metrics(first_vector, second_vector)
    assert torch.isfinite(torch.stack(list(metrics.values()))).all()
    assert tensor_state_sha256(model.state_dict()) == before
    assert all(parameter.grad is None for parameter in parameters)


def test_real_stage_h_outputs_preserve_models_and_forbid_updates_when_available():
    root = Path(__file__).resolve().parents[1]
    output = (
        root
        / "outputs/paper_reduced100/stage_h/h1_gradient_audit.json"
    )
    if not output.exists():
        pytest.skip("local Stage H generated diagnostics are Git ignored")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["safety"] == {
        "autograd_api": "torch.autograd.grad",
        "batch_size": 1,
        "checkpoint_write": False,
        "eval_mode": True,
        "mixed_precision": False,
        "optimizer_created": False,
        "optimizer_step": False,
        "parameter_grad_tensors_remained_none": True,
        "scheduler_step": False,
    }
    for state in payload["fixed_model_states"].values():
        assert state["before"] == state["after"]
        assert state["unchanged"] is True
    assert math.isfinite(payload["overall"]["h1_to_base_norm_ratio_mean"])


def test_real_stage_h_frozen_files_and_upstream_are_unchanged_when_available():
    root = Path(__file__).resolve().parents[1]
    path = root / "outputs/paper_reduced100/stage_h/frozen_inputs.json"
    if not path.exists():
        pytest.skip("local Stage H frozen-input manifest is Git ignored")
    expected = json.loads(path.read_text(encoding="utf-8"))
    assert validate_frozen_input_manifest(root, expected) == expected
