import torch

from grmhd.paper_h1_diagnostics import h1_diagnostic


def test_direction_and_channel_contributions_sum_to_total():
    torch.manual_seed(23)
    error = torch.randn(3, 4, 6, 7, 8, dtype=torch.float64)
    result = h1_diagnostic(error, variant="H0_current_upstream")
    torch.testing.assert_close(
        sum(result.per_direction.values()), result.total, rtol=1.0e-12, atol=1.0e-12
    )
    torch.testing.assert_close(
        result.per_channel.sum(), result.total, rtol=1.0e-12, atol=1.0e-12
    )


def test_batches_do_not_cross_mix():
    torch.manual_seed(29)
    first = torch.randn(1, 2, 6, 7, 8, dtype=torch.float64)
    second = torch.randn(1, 2, 6, 7, 8, dtype=torch.float64)
    combined = h1_diagnostic(
        torch.cat((first, second)), variant="H0_current_upstream"
    )
    separate_first = h1_diagnostic(first, variant="H0_current_upstream")
    separate_second = h1_diagnostic(second, variant="H0_current_upstream")
    torch.testing.assert_close(
        combined.total,
        0.5 * (separate_first.total + separate_second.total),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
