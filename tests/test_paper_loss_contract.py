from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from grmhd.paper_losses import PaperLossContext


def test_context_fixes_tensor_spaces_and_protocol(paper_loss_factory):
    case = paper_loss_factory("loss_contract")
    context = case["context"]
    prediction = context.normalized_target.clone()
    context.validate(prediction)
    assert context.raw_physical_target.data_ptr() != context.oracle_physical_target.data_ptr()
    wrong = replace(
        context,
        protocol_metadata={**context.protocol_metadata, "canonical_roi_source": "raw"},
    )
    with pytest.raises(ValueError, match="canonical_roi_source"):
        wrong.validate(prediction)
    with pytest.raises(ValueError, match="batch size"):
        replace(context, snapshot_indices=(1, 2)).validate(prediction)


def test_structured_result_preserves_graph_and_detaches_only_logs(paper_loss_factory):
    case = paper_loss_factory("loss_result")
    prediction = case["context"].normalized_target.clone().requires_grad_()
    result = case["loss"].components(prediction, context=case["context"])
    assert result.total.requires_grad
    assert all(value.ndim == 0 for value in result.component_tensors().values())
    assert all(value.dtype == prediction.dtype for value in result.component_tensors().values())
    result.total.backward()
    assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
    logged = result.detached_log()
    assert isinstance(logged["components"]["total"], float)
    assert logged["metadata"]["radial_mode"] == "appendix_literal_press_proxy"


def test_context_rejects_bounds_or_baseline_space_mismatch(paper_loss_factory):
    case = paper_loss_factory("loss_context_guard")
    context = case["context"]
    prediction = context.normalized_target
    with pytest.raises(ValueError, match="bounds differ"):
        case["loss"].components(
            prediction,
            context=replace(
                context,
                normalized_bounds={**context.normalized_bounds, "rho": (-9.0, 9.0)},
            ),
        )
    with pytest.raises(ValueError, match="broadcast-compatible"):
        replace(
            context,
            radial_baseline_normalized=torch.zeros(8, 2, 2, 2),
        ).validate(prediction)
