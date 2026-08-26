from __future__ import annotations

import torch

from neuralop.models import FNO

from grmhd.models import parameters_without_grad, trainable_parameter_count
from grmhd.upstream_adapters import UpstreamFNOConfig, build_upstream_fno


def _exercise(in_channels: int, expected_parameters: int, tmp_path):
    config = UpstreamFNOConfig(in_channels=in_channels)
    model = build_upstream_fno(config)
    assert type(model) is FNO
    assert type(model).__module__ == "neuralop.models.fno"
    assert model.in_channels == in_channels
    assert model.out_channels == 8
    assert tuple(model.n_modes) == (8, 8, 8)
    assert model.hidden_channels == 16
    assert model.n_layers == 4
    assert model.positional_embedding is None
    assert trainable_parameter_count(model) == expected_parameters

    x = torch.randn(1, in_channels, 16, 16, 16)
    output = model(x)
    assert output.shape == (1, 8, 16, 16, 16)
    loss = output.square().mean()
    loss.backward()
    assert torch.isfinite(output).all()
    assert not parameters_without_grad(model)
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )

    checkpoint = tmp_path / f"fno_{in_channels}.pt"
    torch.save(model.state_dict(), checkpoint)
    reloaded = build_upstream_fno(config)
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=False), strict=True)
    reloaded.eval()
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(reloaded(x), model(x), rtol=0, atol=0)


def test_upstream_fno_eight_state_channels(tmp_path):
    _exercise(8, 331_576, tmp_path)


def test_upstream_fno_state_plus_shell_channels(tmp_path):
    _exercise(16, 331_832, tmp_path)
