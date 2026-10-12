import pytest

from ige.utils.stats import wilson_ci

pytestmark = pytest.mark.unit


def test_bounds_and_symmetry():
    lo, hi = wilson_ci(38, 77)
    assert 0.0 < lo < 38 / 77 < hi < 1.0
    # k and n-k mirror around 0.5
    lo2, hi2 = wilson_ci(77 - 38, 77)
    assert lo == pytest.approx(1 - hi2, abs=1e-12)
    assert hi == pytest.approx(1 - lo2, abs=1e-12)


def test_extremes_stay_inside_unit_interval():
    lo0, hi0 = wilson_ci(0, 77)
    assert lo0 == 0.0 and 0.0 < hi0 < 0.06   # rule-of-three neighborhood
    lo1, hi1 = wilson_ci(77, 77)
    assert hi1 == pytest.approx(1.0, abs=1e-12) and 0.94 < lo1 < 1.0


def test_narrows_with_n():
    n_small = wilson_ci(8, 10)
    n_large = wilson_ci(80, 100)
    assert (n_large[1] - n_large[0]) < (n_small[1] - n_small[0])


def test_degenerate_n_zero():
    assert wilson_ci(0, 0) == [0.0, 1.0]
