"""Differential prompt diff — the prompt-completeness discipline of the paper (§4.1) as blocking code.

Within a matched pair (ambiguous / unambiguous twin) the ONLY thing allowed
to differ between the step-1 prompts is the manipulated variable: the
workspace item lines. Any other difference is a confound and the harness
must refuse to proceed. This is the check that, in the ALFWorld project,
would have caught the missing task line before the sweep.
"""

from __future__ import annotations

import difflib

Messages = list[dict[str, str]]

# the manipulated variable: inventory item lines ("- id=...")
_ITEM_LINE_PREFIX = "- id="


class PromptDiffError(AssertionError):
    pass


def prompt_diff_lines(msgs_a: Messages, msgs_b: Messages) -> list[str]:
    """Changed lines (with +/- markers) between two message lists, role-aware."""
    if [m["role"] for m in msgs_a] != [m["role"] for m in msgs_b]:
        return ["<role structure differs>"]
    out: list[str] = []
    for ma, mb in zip(msgs_a, msgs_b):
        diff = difflib.unified_diff(
            ma["content"].splitlines(), mb["content"].splitlines(), lineterm="", n=0
        )
        out.extend(l for l in diff if l[:1] in "+-" and l[:3] not in ("+++", "---"))
    return out


def assert_pair_manipulation_only(msgs_amb: Messages, msgs_una: Messages) -> None:
    """Raise PromptDiffError unless every differing line is an item line."""
    changed = prompt_diff_lines(msgs_amb, msgs_una)
    offending = [l for l in changed if not l[1:].lstrip().startswith(_ITEM_LINE_PREFIX)]
    if offending:
        raise PromptDiffError(
            "confound: prompts differ outside the manipulated variable "
            f"(workspace items): {offending[:5]}"
        )
    if not changed:
        raise PromptDiffError("pair prompts are identical — no manipulated variable present")
