"""Parse the model's reply into a structured step."""

from __future__ import annotations

import re
from typing import Any

_ACT = re.compile(r"^\s*ACTION:\s*([A-Za-z0-9_\-]+)\s*\.?\s*$", re.IGNORECASE)
_ASK = re.compile(r"^\s*ASK:\s*(.+?)\s*$", re.IGNORECASE)


def parse_reply(text: str) -> dict[str, Any]:
    """First line matching the contract wins; anything else is 'invalid'.

    Returns {'type': 'act'|'ask'|'invalid', 'arg': str, 'raw': str}.
    """
    for line in text.splitlines():
        m = _ACT.match(line)
        if m:
            # item ids are lowercase by construction; models often shout them
            return {"type": "act", "arg": m.group(1).lower(), "raw": text}
        m = _ASK.match(line)
        if m:
            return {"type": "ask", "arg": m.group(1), "raw": text}
    return {"type": "invalid", "arg": "", "raw": text}
