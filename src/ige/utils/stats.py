"""Small statistics helpers shared by experiment runners."""

from __future__ import annotations

import math


def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> list[float]:
    """Wilson score interval for a binomial proportion k/n (default 95%).

    Used for frozen-site flip rates: the frozen evaluation is a
    plain binomial — one pre-committed site and dose, n independent held-out
    episodes — so Wilson applies there (unlike the IRR burden ratio, see
    scripts/analyze_irr_ci.py).
    """
    if n <= 0:
        return [0.0, 1.0]
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, center - half), min(1.0, center + half)]
