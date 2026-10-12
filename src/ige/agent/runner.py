"""Episode runner with verbatim, replayable traces.

The trace stores the EXACT message list sent to the model at every step
(prompt-dump: the verbatim message list is stored) plus the raw replies. replay() rebuilds
the message lists from the scenario alone and verifies they are identical to
what was stored — the boundary-invariant check that guards the

    vLLM rollout  ->  hooked (TransformerLens) re-execution

hand-off in M2/M3: if replay drifts from the original, activations would be
captured on prompts the agent never actually saw.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ige.agent.backend import LLMBackend
from ige.agent.parser import parse_reply
from ige.agent.prompts import build_messages
from ige.tasks.clarify import Scenario, score_episode


@dataclass
class Trace:
    scenario: dict[str, Any]              # Scenario.to_dict()
    backend: str
    messages_per_step: list[list[dict[str, str]]] = field(default_factory=list)
    raw_replies: list[str] = field(default_factory=list)
    parsed_steps: list[dict[str, Any]] = field(default_factory=list)
    outcome: str = ""
    conversational: bool = False          # step 2 sees the step-1 exchange (ablation zfg)
    # per-step rollout provenance: rendered text, prompt token
    # ids, first reply token, tokenizer/model revision — what the replay
    # boundary must reproduce. None entries for backends that don't report it.
    rollout_info: list[dict[str, Any] | None] = field(default_factory=list)

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.__dict__, indent=1))

    @classmethod
    def load(cls, path: Path | str) -> "Trace":
        return cls(**json.loads(Path(path).read_text()))


def run_episode(
    backend: LLMBackend,
    scenario: Scenario,
    *,
    max_new_tokens: int = 40,
    conversational: bool = False,
) -> Trace:
    trace = Trace(scenario=scenario.to_dict(), backend=backend.name,
                  conversational=conversational)

    def step(clarification: str | None, own_question: str | None) -> dict[str, Any]:
        messages = build_messages(scenario, clarification, own_question)
        reply = backend.generate(messages, max_new_tokens=max_new_tokens)
        parsed = parse_reply(reply)
        trace.messages_per_step.append(messages)
        trace.raw_replies.append(reply)
        trace.parsed_steps.append(parsed)
        trace.rollout_info.append(getattr(backend, "last_info", None))
        return parsed

    first = step(None, None)
    if first["type"] == "ask":
        own_q = trace.raw_replies[0].strip() if conversational else None
        step(scenario.clarification_answer(), own_q)

    trace.outcome = score_episode(scenario, trace.parsed_steps)
    return trace


def rebuild_messages(trace: Trace) -> list[list[dict[str, str]]]:
    """Reconstruct, from the scenario + recorded replies, the message lists
    the episode *should* have produced given the recorded step types."""
    scenario = _scenario_from_dict(trace.scenario)
    rebuilt = [build_messages(scenario, None)]
    if trace.parsed_steps and trace.parsed_steps[0]["type"] == "ask":
        own_q = trace.raw_replies[0].strip() if trace.conversational else None
        rebuilt.append(build_messages(scenario, scenario.clarification_answer(), own_q))
    return rebuilt


def check_replay_invariant(trace: Trace) -> None:
    """Boundary invariant: rebuilt prompts must be bit-identical to stored ones."""
    rebuilt = rebuild_messages(trace)
    if rebuilt != trace.messages_per_step:
        raise AssertionError(
            "replay drift: rebuilt message list differs from the recorded one — "
            "the agent would be re-executed on prompts it never saw"
        )


def _scenario_from_dict(d: dict[str, Any]) -> Scenario:
    from ige.tasks.clarify import Item

    d = dict(d)
    d["items"] = tuple(Item(**i) for i in d["items"])
    d["matching_ids"] = tuple(d["matching_ids"])
    return Scenario(**d)
