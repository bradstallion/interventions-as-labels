import json

import pytest

from ige.agent import FakeBackend, build_messages
from ige.data.collect import collect_dataset
from ige.eval.promptdiff import PromptDiffError, assert_pair_manipulation_only
from ige.overseer import RuleOverseer, Verdict
from ige.tasks.clarify import generate

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------- promptdiff

@pytest.fixture()
def twin_messages():
    scenarios = generate(2, seed=9)
    by_pair = {}
    for s in scenarios:
        by_pair.setdefault(s.pair_id, {})[s.ambiguous] = s
    amb, una = next((p[True], p[False]) for p in by_pair.values() if len(p) == 2)
    return build_messages(amb), build_messages(una)


def test_matched_pair_passes(twin_messages):
    assert_pair_manipulation_only(*twin_messages)  # must not raise


def test_injected_confound_is_caught(twin_messages):
    msgs_amb, msgs_una = twin_messages
    tampered = [dict(m) for m in msgs_amb]
    # the ALFWorld bug, reproduced: task line silently dropped in one condition
    tampered[-1]["content"] = tampered[-1]["content"].replace("Task: ", "")
    with pytest.raises(PromptDiffError, match="confound"):
        assert_pair_manipulation_only(tampered, msgs_una)


def test_identical_prompts_rejected(twin_messages):
    msgs_amb, _ = twin_messages
    with pytest.raises(PromptDiffError, match="identical"):
        assert_pair_manipulation_only(msgs_amb, msgs_amb)


# ---------------------------------------------------------------- collection

def _lazy():
    return FakeBackend(
        lambda m: f"ACTION: {m[-1]['content'].split('id=')[1].split()[0]}", name="lazy"
    )


def test_collect_dataset_end_to_end(tmp_path):
    scenarios = generate(8, seed=21)
    result = collect_dataset(_lazy(), RuleOverseer(), scenarios, tmp_path / "d")

    assert result.n_records == 16
    assert result.fp_rate_unambiguous_correct <= 0.05
    assert result.intervention_rate_ambiguous > 0  # lazy agent gets intervened

    lines = [json.loads(l) for l in result.records_path.read_text().splitlines()]
    assert len(lines) == 16
    rec = lines[0]
    for key in ("scenario", "pair_id", "split", "messages_per_step", "verdict", "outcome"):
        assert key in rec
    assert (tmp_path / "d" / "summary.json").exists()


def test_collect_fp_canary_blocks_bad_overseer(tmp_path):
    class TriggerHappy:
        overseer_id = "bad"
        def judge(self, trace):
            return Verdict(True, "other", "", self.overseer_id)

    with pytest.raises(AssertionError, match="FP canary"):
        collect_dataset(_lazy(), TriggerHappy(), generate(6, seed=2), tmp_path / "d")
