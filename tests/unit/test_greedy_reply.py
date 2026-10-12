"""greedy_reply must stop at special tokens (end-of-turn), like HF generate:
otherwise the next turn's role word gets glued to the reply ('ACTION: d1' ->
'ACTION: d1user'; OLMo-3 audit 2026-07-06, deflated edit_eval base row)."""

import torch

from ige.experiments.steer_eval import greedy_reply

VOCAB = {1: "ACTION:", 2: " d", 3: "1", 7: "user", 8: "\n", 99: "<eot>"}
SPECIAL = [99]


class FakeTokenizer:
    all_special_ids = SPECIAL

    def decode(self, ids, skip_special_tokens=False):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if isinstance(ids, int):
            ids = [ids]
        return "".join(VOCAB[i] for i in ids
                       if not (skip_special_tokens and i in SPECIAL))


class ScriptedModel:
    """Emits a fixed continuation regardless of input."""

    def __init__(self, script):
        self.script = list(script)
        self.tokenizer = FakeTokenizer()
        self._i = 0

    def __call__(self, seq):
        nxt = self.script[min(self._i, len(self.script) - 1)]
        self._i += 1
        logits = torch.zeros(1, seq.shape[1], 100)
        logits[0, -1, nxt] = 10.0
        return logits


def test_stops_at_special_token_before_role_word():
    # after 'ACTION: d1' the model emits <eot> then 'user' — must cut at <eot>
    model = ScriptedModel(script=[2, 3, 99, 7, 7, 7])
    reply = greedy_reply(model, torch.tensor([[42]]), first=1)
    assert reply == "ACTION: d1"


def test_still_stops_at_newline():
    model = ScriptedModel(script=[2, 3, 8, 7, 7])
    reply = greedy_reply(model, torch.tensor([[42]]), first=1)
    assert reply == "ACTION: d1\n"
