import torch

from grmhd.paper_h1_diagnostics import additive_region_rows, h1_diagnostic


def test_disjoint_shell_contributions_sum_to_full_h1():
    torch.manual_seed(31)
    error = torch.randn(1, 2, 6, 7, 8, dtype=torch.float64)
    result = h1_diagnostic(error, variant="H0_current_upstream")
    shell_index = torch.arange(8).view(1, 1, 8).expand(6, 7, 8)
    masks = {f"shell_{index}": shell_index == index for index in range(8)}
    rows = additive_region_rows(
        result, masks, channel_names=("a", "b")
    )
    aggregate = [
        row
        for row in rows
        if row["channel"] == "all" and row["direction"] == "all"
    ]
    total = sum(row["contribution_fraction"] for row in aggregate)
    torch.testing.assert_close(
        total, torch.tensor(1.0, dtype=error.dtype), rtol=1.0e-12, atol=1.0e-12
    )


def test_region_overlap_and_gradient_fractions_are_finite():
    torch.manual_seed(37)
    error = torch.randn(1, 2, 6, 7, 8, dtype=torch.float64, requires_grad=True)
    result = h1_diagnostic(error, variant="H0_current_upstream")
    gradient = torch.autograd.grad(result.total, error)[0]
    mask = torch.zeros(6, 7, 8, dtype=torch.bool)
    mask[..., :4] = True
    overlap = torch.zeros_like(mask)
    overlap[:, :3] = True
    rows = additive_region_rows(
        result,
        {"inner": mask},
        channel_names=("a", "b"),
        gradient_density=gradient.square(),
        overlap_masks={"probe": overlap},
    )
    aggregate = rows[0]
    for key in (
        "contribution_fraction",
        "enrichment",
        "gradient_norm_fraction",
        "probe_voxel_overlap",
        "probe_h1_overlap",
    ):
        assert torch.isfinite(aggregate[key])
