import pytest

from ige.agent.parser import parse_reply

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "text,expected_type,expected_arg",
    [
        ("ACTION: f1", "act", "f1"),
        ("ACTION: B1", "act", "b1"),
        ("action: b2.", "act", "b2"),
        ("  ACTION:  d3  ", "act", "d3"),
        ("ASK: which file do you mean?", "ask", "which file do you mean?"),
        ("Let me think.\nASK: the red or the blue one?", "ask", "the red or the blue one?"),
        ("Sure!\nACTION: f2\nDone.", "act", "f2"),
        ("I will delete the file f1.", "invalid", ""),
        ("ACTION f1", "invalid", ""),
        ("", "invalid", ""),
    ],
)
def test_parse(text, expected_type, expected_arg):
    parsed = parse_reply(text)
    assert parsed["type"] == expected_type
    assert parsed["arg"] == expected_arg


def test_merge_system_into_user():
    from ige.agent.backend import merge_system_into_user

    msgs = [{"role": "system", "content": "RULES"}, {"role": "user", "content": "TASK"}]
    merged = merge_system_into_user(msgs)
    assert [m["role"] for m in merged] == ["user"]
    assert "RULES" in merged[0]["content"] and "TASK" in merged[0]["content"]
    # no system -> untouched
    assert merge_system_into_user(msgs[1:]) == msgs[1:]
