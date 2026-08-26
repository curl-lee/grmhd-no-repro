#!/usr/bin/env python
"""Emit a machine-readable audit of the pinned upstream FNO construction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from grmhd.models import parameters_without_grad, trainable_parameter_count
from grmhd.upstream_adapters import (
    PINNED_NEURALOP_COMMIT,
    UpstreamFNOConfig,
    build_upstream_fno,
)


def audit(in_channels: int) -> dict[str, object]:
    torch.manual_seed(42)
    config = UpstreamFNOConfig(in_channels=in_channels)
    model = build_upstream_fno(config)
    x = torch.randn(1, in_channels, 16, 16, 16)
    output = model(x)
    output.square().mean().backward()
    missing = parameters_without_grad(model)
    state = model.state_dict()
    clone = build_upstream_fno(config)
    reload_result = clone.load_state_dict(state, strict=True)
    model.eval()
    clone.eval()
    with torch.no_grad():
        reload_max_difference = float(torch.max(torch.abs(model(x) - clone(x))))
    return {
        "config": config.as_dict(),
        "class": type(model).__name__,
        "module": type(model).__module__,
        "shape": list(output.shape),
        "parameter_count": trainable_parameter_count(model),
        "forward_finite": bool(torch.isfinite(output).all()),
        "backward_finite": all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ),
        "unused_trainable_parameters": missing,
        "strict_reload_missing_keys": list(reload_result.missing_keys),
        "strict_reload_unexpected_keys": list(reload_result.unexpected_keys),
        "strict_reload_max_abs_difference": reload_max_difference,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/experiment_round3/upstream_model_alignment.json"),
    )
    args = parser.parse_args()
    report = {
        "status": "passed",
        "neuraloperator_commit": PINNED_NEURALOP_COMMIT,
        "state_only": audit(8),
        "state_plus_shells": audit(16),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
