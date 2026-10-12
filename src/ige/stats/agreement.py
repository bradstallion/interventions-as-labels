"""Inter-rater agreement for the overseer validity study (overseer validity protocol)."""

from __future__ import annotations

from collections import Counter
from typing import Hashable, Sequence


def cohen_kappa(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """Cohen's kappa between two raters over the same items.

    Returns 1.0 for perfect agreement, 0.0 for chance-level, negative for
    below-chance. Raises on empty or mismatched inputs.
    """
    if len(a) != len(b) or not a:
        raise ValueError("raters must have the same, non-zero number of items")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    if expected == 1.0:  # both raters constant and identical
        return 1.0
    return (observed - expected) / (1.0 - expected)
