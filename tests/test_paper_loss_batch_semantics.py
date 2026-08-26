from __future__ import annotations

from dataclasses import replace

import torch


def repeat_context(context, count: int):
    def repeat_state(value):
        if value.ndim == 5 and value.shape[0] == 1:
            return value.repeat(count, 1, 1, 1, 1)
        return value

    return replace(
        context,
        normalized_input=repeat_state(context.normalized_input),
        normalized_target=repeat_state(context.normalized_target),
        raw_physical_target=repeat_state(context.raw_physical_target),
        oracle_physical_target=repeat_state(context.oracle_physical_target),
        canonical_roi_mask=context.canonical_roi_mask.repeat(count, 1, 1, 1),
        raw_roi_diagnostic_mask=(
            None
            if context.raw_roi_diagnostic_mask is None
            else context.raw_roi_diagnostic_mask.repeat(count, 1, 1, 1)
        ),
        radial_baseline_normalized=repeat_state(context.radial_baseline_normalized),
        snapshot_indices=tuple(range(count)),
    )


def test_duplicate_batch_and_incomplete_last_batch_use_mean_reduction(paper_loss_factory):
    case = paper_loss_factory("batch_duplicate")
    context1 = case["context"]
    prediction1 = torch.zeros_like(context1.normalized_target)
    value1 = case["loss"](prediction1, context=context1)
    context2 = repeat_context(context1, 2)
    prediction2 = prediction1.repeat(2, 1, 1, 1, 1)
    value2 = case["loss"](prediction2, context=context2)
    torch.testing.assert_close(value2, value1)
    assert torch.isfinite(case["loss"](prediction2[:1], context=repeat_context(context1, 1)))


def test_different_samples_average_and_batch_order_do_not_matter(paper_loss_factory):
    case = paper_loss_factory("batch_order")
    context1 = case["context"]
    first = context1.normalized_target + 0.25
    second = context1.normalized_target - 1.5
    first_value = case["loss"](first, context=context1)
    second_value = case["loss"](second, context=context1)
    context2 = repeat_context(context1, 2)
    batch = torch.cat((first, second), dim=0)
    batch_value = case["loss"](batch, context=context2)
    reversed_value = case["loss"](batch.flip(0), context=context2)
    torch.testing.assert_close(batch_value, (first_value + second_value) / 2)
    torch.testing.assert_close(reversed_value, batch_value)


def test_dissipative_norm_and_roi_mask_remain_per_sample(paper_loss_factory):
    case = paper_loss_factory("batch_per_sample")
    context = repeat_context(case["context"], 2)
    context = replace(
        context,
        normalized_input=torch.cat(
            (context.normalized_input[:1], 2 * context.normalized_input[1:]), dim=0
        ),
    )
    prediction = torch.zeros_like(context.normalized_target)
    result = case["loss"].components(prediction, context=context)
    expected_norms = torch.stack(
        [
            torch.linalg.vector_norm(context.normalized_input[index])
            for index in range(2)
        ]
    )
    assert result.diagnostics["input_norm_mean"] == expected_norms.mean()
    counts = context.canonical_roi_mask.flatten(start_dim=1).sum(dim=1)
    assert counts[0] == counts[1]
