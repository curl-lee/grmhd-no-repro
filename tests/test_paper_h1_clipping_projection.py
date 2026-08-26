import torch

from grmhd.paper_h1_diagnostics import simulate_global_norm_clip


def test_clipping_projection_scales_components_without_changing_direction():
    base = torch.tensor([3.0, 0.0])
    h1 = torch.tensor([0.0, 4.0])
    total = base + h1
    result = simulate_global_norm_clip(
        {"base": base, "h1": h1, "total": total}, max_norm=1.0
    )
    torch.testing.assert_close(result["scale_factor"], torch.tensor(0.2))
    torch.testing.assert_close(result["total_norm_before"], torch.tensor(5.0))
    torch.testing.assert_close(result["total_norm_after"], torch.tensor(1.0))
    torch.testing.assert_close(result["direction_cosine"], torch.tensor(1.0))
    for component in ("base", "h1", "total"):
        projection = result["component_projections"][component]
        torch.testing.assert_close(
            projection["after"], 0.2 * projection["before"]
        )
