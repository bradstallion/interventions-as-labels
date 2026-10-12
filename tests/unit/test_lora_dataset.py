"""M5 dataset builder: corrections come from the overseer verdict, retention
from the model's own replies; only TRAIN pairs; prompt tokens are masked."""

from ige.eval.battery import CAPABILITY_BATTERY
from ige.train.lora_bc import build_dataset, encode_example


def make_records(n_pairs=10):
    recs = []
    for i in range(n_pairs):
        pid = f"fam-{i:04d}"
        recs.append({  # unambiguous twin: overasked -> intervention + correction
            "pair_id": pid,
            "scenario": {"scenario_id": f"{pid}-una", "family": "fam", "ambiguous": False},
            "messages_per_step": [[{"role": "system", "content": "s"},
                                   {"role": "user", "content": f"task {i}"}]],
            "raw_replies": ["ASK: which one?\n"],
            "outcome": "overask",
            "verdict": {"intervene": True, "correction": f"ACTION: x{i}"},
        })
        recs.append({  # ambiguous twin: correct clarify -> retention
            "pair_id": pid,
            "scenario": {"scenario_id": f"{pid}-amb", "family": "fam", "ambiguous": True},
            "messages_per_step": [[{"role": "system", "content": "s"},
                                   {"role": "user", "content": f"task {i} amb"}]],
            "raw_replies": ["ASK: legit question?\n"],
            "outcome": "correct_clarify",
            "verdict": {"intervene": False, "correction": ""},
        })
    return recs


def test_build_dataset_mix_and_split():
    recs = make_records(10)
    ex = build_dataset(recs, split_seed=0)
    # only train pairs (70% of 10 pairs = 7 pairs = 14 records)
    assert len(ex) == 14
    kinds = {e["kind"] for e in ex}
    assert kinds == {"correction", "retention"}
    for e in ex:
        if e["kind"] == "correction":
            assert e["target"].startswith("ACTION:")
        else:
            assert e["target"].startswith("ASK:")


def test_build_dataset_skips_empty_corrections():
    recs = make_records(10)
    for r in recs:
        if r["verdict"]["intervene"]:
            r["verdict"]["correction"] = ""
    ex = build_dataset(recs, split_seed=0)
    assert all(e["kind"] == "retention" for e in ex)


class FakeTok:
    """Char-level tokenizer stub with a chat template."""
    def apply_chat_template(self, msgs, add_generation_prompt=True, tokenize=False):
        return "|".join(m["content"] for m in msgs) + "->"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def test_encode_example_masks_prompt():
    ex = {"messages": [{"role": "user", "content": "hi"}], "target": "ACTION: x1"}
    input_ids, labels = encode_example(FakeTok(), ex)
    prompt_len = len("hi->")
    assert labels[:prompt_len] == [-100] * prompt_len
    assert labels[prompt_len:] == input_ids[prompt_len:]
    assert input_ids[-1] == ord("\n")  # stop signal included in the target


def test_battery_is_sound():
    assert len(CAPABILITY_BATTERY) == 50
    assert len({q for q, _ in CAPABILITY_BATTERY}) == 50
    for q, a in CAPABILITY_BATTERY:
        assert a == a.strip().lower()
        # substring scoring: 1-char answers only for digits and the legacy item
        assert len(a) >= 2 or a.isdigit() or q.startswith("First letter")
