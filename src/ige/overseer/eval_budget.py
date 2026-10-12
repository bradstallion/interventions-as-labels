"""Evaluation overseer with attention budget and dynamic trust (paper §3 and Appendix B).

Watches only a fraction of episodes, preferring low-trust families; trust
grows on clean observations and resets on interventions. Oversight burden =
(watched + c * interventions) / n. Deterministic given seed. Distinct
overseer_id from the collection labeler (anti-circularity)."""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class BurdenReport:
    n: int
    watched: int
    interventions: int
    missed: int          # needed intervention but was not watched
    burden: float

    def to_dict(self) -> dict:
        return self.__dict__


class EvalBudgetOverseer:
    overseer_id = "eval-budget-v1"

    def __init__(self, budget: float = 0.5, cost: float = 2.0, t0: float = 0.5,
                 seed: int = 0):
        self.budget, self.cost, self.t0, self.seed = budget, cost, t0, seed

    def burden(self, episodes: list[tuple[str, bool]]) -> BurdenReport:
        """episodes: (family, needs_intervention) in stream order."""
        rng = random.Random(self.seed)
        trust: dict[str, float] = {}
        watched = interventions = missed = 0
        for family, needs in episodes:
            t = trust.setdefault(family, self.t0)
            p_watch = min(1.0, 2.0 * self.budget * (1.0 - t))
            if rng.random() < p_watch:
                watched += 1
                if needs:
                    interventions += 1
                    trust[family] = 0.0
                else:
                    trust[family] = min(0.95, t + 0.1)
            elif needs:
                missed += 1
        n = len(episodes)
        return BurdenReport(
            n=n, watched=watched, interventions=interventions, missed=missed,
            burden=(watched + self.cost * interventions) / max(1, n),
        )
