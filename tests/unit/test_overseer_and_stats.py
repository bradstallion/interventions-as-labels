import pytest

from ige.agent import FakeBackend, run_episode
from ige.overseer import RuleOverseer, parse_verdict_text
from ige.stats import cohen_kappa
from ige.tasks.clarify import generate

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------- kappa

def test_kappa_perfect_and_chance():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    # rater b constant: observed agreement equals expected -> kappa 0
    assert abs(cohen_kappa([1, 0, 1, 0], [1, 1, 1, 1])) < 1e-9


def test_kappa_known_value():
    # 2x2 table [[25,15],[15,45]]: po=0.70, marginals 40/60 for both raters
    # -> pe = 0.4*0.4 + 0.6*0.6 = 0.52, kappa = 0.18/0.48 = 0.375
    a = [1] * 25 + [1] * 15 + [0] * 15 + [0] * 45
    b = [1] * 25 + [0] * 15 + [1] * 15 + [0] * 45
    assert abs(cohen_kappa(a, b) - 0.375) < 1e-9


def test_kappa_validates_input():
    with pytest.raises(ValueError):
        cohen_kappa([1], [1, 0])
    with pytest.raises(ValueError):
        cohen_kappa([], [])


# ---------------------------------------------------------------- rule overseer

def _lazy():
    def policy(messages):
        user = messages[-1]["content"]
        return f"ACTION: {user.split('id=')[1].split()[0]}"
    return FakeBackend(policy, name="lazy")


def _oracle(scenario):
    def policy(messages):
        if scenario.ambiguous and "Answer to your question" not in messages[-1]["content"]:
            return "ASK: which one?"
        return f"ACTION: {scenario.target_id}"
    return FakeBackend(policy, name="oracle")


def test_rule_overseer_verdicts():
    overseer = RuleOverseer()
    for scenario in generate(10, seed=11):
        lazy_trace = run_episode(_lazy(), scenario)
        v = overseer.judge(lazy_trace)
        if lazy_trace.outcome == "proceed_when_ambiguous":
            assert v.intervene and v.type == "ask-for-help"
            assert v.correction.startswith("ASK:") and scenario.clarify_dim in v.correction
        elif lazy_trace.outcome == "wrong_object":
            assert v.intervene and v.type == "wrong-object"
            assert v.correction == f"ACTION: {scenario.target_id}"

        good_trace = run_episode(_oracle(scenario), scenario)
        assert not overseer.judge(good_trace).intervene  # zero FP on correct behavior


def test_rule_overseer_overask_policy_versions():
    scenario = next(s for s in generate(6, seed=5) if not s.ambiguous)
    asks_then_acts = FakeBackend(
        lambda m: ("ASK: are you sure?" if "Answer to your question" not in m[-1]["content"]
                   else f"ACTION: {scenario.target_id}"),
        name="cautious")
    trace = run_episode(asks_then_acts, scenario)
    assert trace.outcome == "overask"
    # v1 (historical): caution not punished
    assert not RuleOverseer(policy_version="v1").judge(trace).intervene
    # v2 (primary tau): unnecessary clarification IS an intervention
    v = RuleOverseer().judge(trace)
    assert v.intervene and v.type == "unnecessary-clarification"
    assert v.correction == f"ACTION: {scenario.target_id}"
    assert v.overseer_id == "rule-v2"


# ---------------------------------------------------------------- llm verdict parser

@pytest.mark.parametrize(
    "text,intervene,vtype",
    [
        ("INTERVENE: no", False, ""),
        ("INTERVENE: yes\nTYPE: ask-for-help\nCORRECTION: ASK: which one?", True, "ask-for-help"),
        ("INTERVENE: yes\nTYPE: wrong-object\nCORRECTION: ACTION: f1", True, "wrong-object"),
        ("INTERVENE: yes\nTYPE: nonsense\nCORRECTION: x", True, "other"),
        ("the assistant did fine I think", True, "other"),  # unparseable -> flagged
    ],
)
def test_parse_verdict_text(text, intervene, vtype):
    v = parse_verdict_text(text, "test-judge")
    assert v.intervene is intervene
    assert v.type == vtype


def test_reask_after_answer_outcome_and_verdict():
    from ige.tasks.clarify import generate as gen
    scenario = next(s for s in gen(6, seed=8) if s.ambiguous)
    always_asks = FakeBackend(lambda m: "ASK: which one do you mean?", name="asker")
    trace = run_episode(always_asks, scenario)
    assert trace.outcome == "reask_after_answer"
    v = RuleOverseer().judge(trace)
    assert v.intervene and v.type == "correction-propagation"


def test_eval_budget_overseer_burden():
    from ige.overseer.eval_budget import EvalBudgetOverseer

    ov = EvalBudgetOverseer(budget=0.5, cost=2.0, seed=1)
    # all clean: trust grows, watching decays, zero interventions
    clean = ov.burden([("files", False)] * 200)
    assert clean.interventions == 0 and 0 < clean.watched < 200
    # frequent failures keep trust low -> more watching + interventions
    bad = ov.burden([("files", i % 2 == 0) for i in range(200)])
    assert bad.interventions > 0 and bad.burden > clean.burden
    # deterministic given seed
    again = EvalBudgetOverseer(budget=0.5, cost=2.0, seed=1).burden([("files", False)] * 200)
    assert again.to_dict() == clean.to_dict()
    assert ov.overseer_id != "rule-v2"
