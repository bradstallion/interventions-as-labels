"""Rollout->replay identity (added after the external adversarial audit).

The old invariant only compared two replay-time tokenizations of the same
re-rendered text; these tests pin the new, stronger contract: records carry
the rollout's own token ids, and replay must match them or block.
"""

import pytest
import torch

from ige.agent.backend import FakeBackend
from ige.agent.runner import Trace, run_episode
from ige.attribution.render import TokenReplayError, replay_tokens_for_record
from ige.tasks.clarify import generate

pytestmark = pytest.mark.unit


class StubTokenizer:
    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=False):
        return "RENDERED"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1, 2, 3]}


class StubTLModel:
    def to_tokens(self, text, prepend_bos=False):
        return torch.tensor([[1, 2, 3]])


def _record(rollout_ids):
    return {
        "messages_per_step": [[{"role": "user", "content": "x"}]],
        "rollout_info": [
            None if rollout_ids is None else
            {"prompt_token_ids": rollout_ids, "tokenizer": "stub", "model_revision": "r0"}
        ],
    }


def test_matching_rollout_ids_pass():
    ids = replay_tokens_for_record(StubTokenizer(), StubTLModel(), _record([1, 2, 3]))
    assert ids == [1, 2, 3]


def test_mismatching_rollout_ids_block():
    with pytest.raises(TokenReplayError, match="rollout->replay drift"):
        replay_tokens_for_record(StubTokenizer(), StubTLModel(), _record([1, 2, 9]))


def test_legacy_records_fall_back_to_weak_invariant():
    # no rollout_info at all (records without rollout provenance): only the HF/TL check runs
    rec = {"messages_per_step": [[{"role": "user", "content": "x"}]]}
    assert replay_tokens_for_record(StubTokenizer(), StubTLModel(), rec) == [1, 2, 3]
    # rollout_info present but entry None (backend without provenance): same
    assert replay_tokens_for_record(StubTokenizer(), StubTLModel(), _record(None)) == [1, 2, 3]


def test_fake_backend_traces_carry_none_provenance():
    sc = generate(1, seed=0)[0]
    trace = run_episode(FakeBackend(lambda m: "ACTION: take d1"), sc)
    assert trace.rollout_info == [None]


def test_collect_record_serializes_rollout_info():
    # the verification pass caught rollout_info reaching Trace but not disk
    # (collect._record omitted it): pin the serialization end-to-end
    from ige.data.collect import _record

    class VerdictStub:
        def to_dict(self):
            return {"intervene": False}

    sc = generate(1, seed=0)[0]
    trace = run_episode(FakeBackend(lambda m: "ACTION: take d1"), sc)
    rec = _record(trace, VerdictStub())
    assert "rollout_info" in rec
    assert rec["rollout_info"] == [None]  # FakeBackend: entry present, empty


def test_old_trace_json_loads_without_rollout_info(tmp_path):
    sc = generate(1, seed=0)[0]
    trace = run_episode(FakeBackend(lambda m: "ACTION: take d1"), sc)
    d = dict(trace.__dict__)
    d.pop("rollout_info")           # simulate a trace without rollout provenance on disk
    p = tmp_path / "t.json"
    p.write_text(__import__("json").dumps(d))
    old = Trace.load(p)
    assert old.rollout_info == []   # default, no crash
