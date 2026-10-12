"""Prompt-dump + boundary-invariant tests (the boundary-invariant discipline of the paper, §4.1, as executable checks)."""

import pytest

from ige.agent import (
    FakeBackend,
    INTRINSIC_MARKERS,
    Trace,
    check_replay_invariant,
    run_episode,
)
from ige.tasks.clarify import generate

pytestmark = pytest.mark.unit


def oracle_backend(scenario):
    """Asks iff ambiguous, then acts on the target (uses ground truth)."""

    def policy(messages):
        user = messages[-1]["content"]
        if scenario.ambiguous and "Answer to your question" not in user:
            return "ASK: which one do you mean?"
        return f"ACTION: {scenario.target_id}"

    return FakeBackend(policy, name="oracle")


def lazy_backend():
    """Never asks: acts on the first id it sees (τ generator)."""

    def policy(messages):
        user = messages[-1]["content"]
        first_id = user.split("id=")[1].split()[0]
        return f"ACTION: {first_id}"

    return FakeBackend(policy, name="lazy")


@pytest.fixture(scope="module")
def scenarios():
    return generate(12, seed=7)


def test_intrinsic_context_present_at_every_step(scenarios):
    """Every intrinsic marker (task line, inventory, action space) must be in
    the prompt at EVERY step — including after the clarification boundary."""
    for scenario in scenarios:
        trace = run_episode(oracle_backend(scenario), scenario)
        assert len(trace.messages_per_step) == (2 if scenario.ambiguous else 1)
        for step_messages in trace.messages_per_step:
            blob = "\n".join(m["content"] for m in step_messages)
            for marker in INTRINSIC_MARKERS:
                assert marker in blob, f"intrinsic marker {marker!r} missing"


def test_oracle_outcomes(scenarios):
    for scenario in scenarios:
        trace = run_episode(oracle_backend(scenario), scenario)
        assert trace.outcome == ("correct_clarify" if scenario.ambiguous else "correct_act")


def test_lazy_backend_produces_target_failure(scenarios):
    outcomes = {s.scenario_id: run_episode(lazy_backend(), s).outcome for s in scenarios}
    amb_outcomes = {v for k, v in outcomes.items() if k.endswith("amb")}
    assert amb_outcomes <= {"proceed_when_ambiguous", "wrong_object"}
    assert "proceed_when_ambiguous" in amb_outcomes


def test_replay_invariant_holds_and_detects_tampering(scenarios, tmp_path):
    scenario = scenarios[0]
    trace = run_episode(oracle_backend(scenario), scenario)
    check_replay_invariant(trace)

    # save/load roundtrip preserves the invariant
    path = tmp_path / "trace.json"
    trace.save(path)
    loaded = Trace.load(path)
    check_replay_invariant(loaded)
    assert loaded.outcome == trace.outcome

    # a tampered prompt (context lost at the boundary) must be caught
    loaded.messages_per_step[0][-1]["content"] = "Task: (lost)"
    with pytest.raises(AssertionError, match="replay drift"):
        check_replay_invariant(loaded)


def test_conversational_mode_structure_and_invariant(scenarios):
    """Ablation zfg: conversational step-2 prepends the step-1 exchange; the
    final user message must be IDENTICAL to the stateless variant (single
    manipulated variable), and the replay invariant must still hold."""
    scenario = next(s for s in scenarios if s.ambiguous)
    conv = run_episode(oracle_backend(scenario), scenario, conversational=True)
    flat = run_episode(oracle_backend(scenario), scenario, conversational=False)

    roles_conv = [m["role"] for m in conv.messages_per_step[1]]
    assert roles_conv == ["system", "user", "assistant", "user"]
    # the agent's own question is in context, verbatim
    assert conv.messages_per_step[1][2]["content"] == conv.raw_replies[0].strip()
    # final user message identical across conditions
    assert conv.messages_per_step[1][-1] == flat.messages_per_step[1][-1]

    check_replay_invariant(conv)
    conv.messages_per_step[1][2]["content"] = "ASK: something else?"
    import pytest as _pytest
    with _pytest.raises(AssertionError, match="replay drift"):
        check_replay_invariant(conv)
