"""Overseer interface.

Every Verdict carries overseer_id: the anti-circularity rule (anti-circularity rule)
is enforced downstream by refusing evaluation runs whose overseer_id matches
the one that labeled the signature's training data.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Protocol

from ige.agent.runner import Trace

INTERVENTION_TYPES = (
    "ask-for-help",
    "wrong-object",
    "correction-propagation",
    "unnecessary-clarification",  # overask — primary τ since policy v2
    "other",
)


@dataclass(frozen=True)
class Verdict:
    intervene: bool
    type: str            # one of INTERVENTION_TYPES, or "" when intervene=False
    correction: str      # canonical reply the agent should have given ("" if none)
    overseer_id: str
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Overseer(Protocol):
    overseer_id: str

    def judge(self, trace: Trace) -> Verdict: ...
