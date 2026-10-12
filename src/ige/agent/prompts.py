"""Prompt construction with an EXPLICIT intrinsic-context contract.

Prompt-completeness rule: everything the agent needs to act is serialized
into the message list at EVERY step — including after the clarification
boundary, where task and inventory are repeated verbatim rather than assumed
to be "carried by history". INTRINSIC_MARKERS is the machine-checkable
contract: tests assert every marker appears in the prompt at every step.
"""

from __future__ import annotations

from ige.tasks.clarify import Scenario

# Two examples, one per action type, deliberately BALANCED: showing only an
# ACTION example would bias the model against asking and contaminate the τ
# (ask-for-help failure) measurement.
SYSTEM_PROMPT = (
    "You are a careful assistant operating a small workspace.\n"
    "Reply with EXACTLY one line, in one of these two forms:\n"
    "ACTION: <id>\n"
    "ASK: <one short clarifying question>\n"
    "If the request does not uniquely identify one item, ask exactly one "
    "clarifying question. Otherwise act directly.\n"
    "\n"
    "Example 1 — the request identifies one item:\n"
    "Task: Delete the pdf file.\n"
    "Workspace items:\n"
    "- id=x1 format=pdf topic=notes\n"
    "- id=x2 format=txt topic=notes\n"
    "Your reply: ACTION: x1\n"
    "\n"
    "Example 2 — the request is ambiguous:\n"
    "Task: Delete the txt file.\n"
    "Workspace items:\n"
    "- id=x1 format=txt topic=notes\n"
    "- id=x2 format=txt topic=plan\n"
    "Your reply: ASK: which txt file, the notes one or the plan one?"
)

# machine-checkable intrinsic-context contract (see tests/unit/test_prompts.py)
INTRINSIC_MARKERS = ("Task:", "id=", "ACTION:", "ASK:")


def _workspace_block(scenario: Scenario) -> str:
    lines = "\n".join(f"- {item.render()}" for item in scenario.items)
    return f"Task: {scenario.instruction}\nWorkspace items:\n{lines}"


def build_messages(
    scenario: Scenario,
    clarification: str | None = None,
    own_question: str | None = None,
) -> list[dict[str, str]]:
    """Message list for step 1 (clarification=None) or step 2 (after ASK).

    Step 2 re-serializes task + inventory in full: context is reconstructed
    across the boundary, never assumed.

    own_question (conversational mode, scaffold-ablation `zfg`): prepend the
    step-1 exchange — the original user turn and the agent's own reply,
    verbatim — before the step-2 user message. The final user message is
    IDENTICAL to the stateless variant, so the two conditions differ only in
    the visible history (single manipulated variable).
    """
    user = _workspace_block(scenario)
    if clarification is not None:
        user = f"Answer to your question: {clarification}\n\n{user}"
    user += "\nReply with one line: ACTION: <id> or ASK: <question>"

    history: list[dict[str, str]] = []
    if own_question is not None:
        if clarification is None:
            raise ValueError("own_question only applies to step 2")
        step1 = build_messages(scenario)  # the step-1 user turn, verbatim
        history = [step1[1], {"role": "assistant", "content": own_question}]

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        *history,
        {"role": "user", "content": user},
    ]
