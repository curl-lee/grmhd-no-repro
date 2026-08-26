import math

import pytest

from grmhd.checkpoint_metrics import composite_stability_metric, relative_l2


def test_relative_l2_uses_aggregate_sums() -> None:
    assert relative_l2(9.0, 4.0) == pytest.approx(1.5)
    assert relative_l2(0.0, 0.0) == 0.0
    assert math.isinf(relative_l2(1.0, 0.0))


def test_composite_stability_formula() -> None:
    value = composite_stability_metric(0.2, 0.4, 0.5)
    assert value == pytest.approx(0.2 + 0.5 * 0.4 + 0.1 * 0.5)


@pytest.mark.parametrize("values", [(-1.0, 0.0), (0.0, -1.0)])
def test_relative_l2_rejects_negative_sums(values: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        relative_l2(*values)
