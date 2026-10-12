"""LLMOverseer: rubric-based judge (policy v1), sees only what a human would.

The rubric text lives in configs/overseer/policy_v1.yaml (versioned; any
change bumps the version and re-triggers the kappa validity study). The judge
NEVER receives ground-truth fields (ambiguous / matching_ids / target_id) —
only task, item list, and the agent's replies.
"""

from __future__ import annotations

import re

from ige.agent.backend import LLMBackend
from ige.agent.runner import Trace
from ige.overseer.base import INTERVENTION_TYPES, Verdict

_INTERVENE = re.compile(r"^\s*INTERVENE:\s*(yes|no)\s*$", re.IGNORECASE | re.MULTILINE)
_TYPE = re.compile(r"^\s*TYPE:\s*([a-z\-]+)\s*$", re.IGNORECASE | re.MULTILINE)
_CORRECTION = re.compile(r"^\s*CORRECTION:\s*(.+?)\s*$", re.MULTILINE)


def parse_verdict_text(text: str, overseer_id: str) -> Verdict:
    m = _INTERVENE.search(text)
    if not m:
        # unparseable judge output is a judge failure, surfaced as 'other'
        return Verdict(True, "other", "", overseer_id, rationale=f"unparseable judge output: {text[:80]!r}")
    if m.group(1).lower() == "no":
        return Verdict(False, "", "", overseer_id)
    t = _TYPE.search(text)
    c = _CORRECTION.search(text)
    vtype = (t.group(1).lower() if t else "other")
    if vtype not in INTERVENTION_TYPES:
        vtype = "other"
    return Verdict(True, vtype, c.group(1) if c else "", overseer_id)


class LLMOverseer:
    def __init__(self, backend: LLMBackend, rubric: str, version: str = "policy_v1"):
        self.backend = backend
        self.rubric = rubric
        self.overseer_id = f"llm-{version}-{backend.name}"

    def judge(self, trace: Trace) -> Verdict:
        # reconstruct the human-visible episode from the verbatim trace
        first_user = trace.messages_per_step[0][-1]["content"]
        replies = "\n".join(f"Agent reply {i+1}: {r}" for i, r in enumerate(trace.raw_replies))
        messages = [
            {"role": "system", "content": self.rubric},
            {"role": "user", "content": f"{first_user}\n\n{replies}\n\nYour verdict:"},
        ]
        return parse_verdict_text(self.backend.generate(messages, max_new_tokens=80), self.overseer_id)
