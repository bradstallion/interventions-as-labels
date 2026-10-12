"""clarify_toy: ask-for-clarification under ambiguity, with ground truth.

Design (anti-confound requirements):

- Scenarios come in MATCHED PAIRS: the SAME rendered instruction appears once
  with an inventory where the descriptor matches exactly one item
  (unambiguous) and once where it matches two (ambiguous). The class label is
  therefore not predictable from the instruction text at all — zero lexical
  leak by construction. The discriminating information lives only in the
  inventory listing, upstream of the decision, which is exactly what the
  probing story needs.

- Ground truth is parametric: the generator knows the matching set, the
  intended referent, and the attribute that disambiguates it.

- 3 template families x paraphrases; splits are assigned by paraphrase index
  so the generalization split contains unseen phrasings.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict
from typing import Any

# ---------------------------------------------------------------- families

FAMILIES: dict[str, dict[str, Any]] = {
    "files": {
        "noun": "file",
        "verb": "delete",
        "dims": {"format": ["pdf", "docx", "txt", "csv"], "topic": ["budget", "report", "notes", "plan"]},
        "paraphrases": [
            "Delete {d}.",
            "Please remove {d}.",
            "Get rid of {d}.",
            "I want {d} deleted.",
        ],
    },
    "fetch": {
        "noun": "book",
        "verb": "fetch",
        "dims": {"color": ["red", "blue", "green", "yellow"], "size": ["small", "large", "medium", "tiny"]},
        "paraphrases": [
            "Bring me {d}.",
            "Could you fetch {d}?",
            "Hand me {d}.",
            "Go get {d}.",
        ],
    },
    "email": {
        "noun": "draft",
        "verb": "send",
        "dims": {"about": ["invoice", "meeting", "offer", "renewal"], "to": ["john", "mary", "alex", "sam"]},
        "paraphrases": [
            "Send {d}.",
            "Please send out {d}.",
            "Ship {d} now.",
            "Can you send {d}?",
        ],
    },
}

# split by paraphrase index: unseen phrasings land in the generalization split
SPLIT_BY_PARAPHRASE = {0: "train", 1: "train", 2: "test", 3: "generalization"}


@dataclass(frozen=True)
class Item:
    id: str
    attrs: dict[str, str]

    def render(self) -> str:
        attrs = " ".join(f"{k}={v}" for k, v in sorted(self.attrs.items()))
        return f"id={self.id} {attrs}"


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    pair_id: str            # shared by the (ambiguous, unambiguous) twin
    family: str
    ambiguous: bool
    instruction: str        # identical within a pair
    items: tuple[Item, ...]
    matching_ids: tuple[str, ...]
    target_id: str          # the referent the "user" actually means
    clarify_dim: str        # attribute dimension that disambiguates
    clarify_value: str      # target's value on that dimension
    paraphrase_idx: int
    split: str

    def clarification_answer(self) -> str:
        return f"I mean the one with {self.clarify_dim}={self.clarify_value}."

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["items"] = [asdict(i) for i in self.items]
        return d


def _make_pair(rng: random.Random, family: str, pair_idx: int) -> tuple[Scenario, Scenario]:
    spec = FAMILIES[family]
    dims = list(spec["dims"].keys())
    desc_dim = rng.choice(dims)                      # dimension used in the instruction
    clar_dim = next(d for d in dims if d != desc_dim)  # dimension that disambiguates
    desc_val = rng.choice(spec["dims"][desc_dim])

    clar_vals = rng.sample(spec["dims"][clar_dim], 3)
    other_desc_vals = [v for v in spec["dims"][desc_dim] if v != desc_val]

    # ids are assigned from a random permutation so that NO id position or
    # suffix correlates with the target role (latent leak found by reading
    # traces: with fixed numbering the target was always "<x>1", an exploit
    # a probe or a model could learn instead of the semantics)
    def make_items(rows: list[tuple[str, str]]) -> list[Item]:
        suffixes = rng.sample(range(1, 4), 3)
        items = [
            Item(id=f"{spec['noun'][0]}{n}", attrs={desc_dim: dv, clar_dim: cv})
            for n, (dv, cv) in zip(suffixes, rows)
        ]
        rng.shuffle(items)
        return items

    # row 0 is the intended referent in both twins
    amb_items = make_items([
        (desc_val, clar_vals[0]),                     # target
        (desc_val, clar_vals[1]),                     # distractor twin (also matches)
        (rng.choice(other_desc_vals), clar_vals[2]),
    ])
    una_items = make_items([
        (desc_val, clar_vals[0]),                     # target (unique match)
        (rng.choice(other_desc_vals), clar_vals[1]),
        (rng.choice(other_desc_vals), clar_vals[2]),
    ])

    p_idx = rng.randrange(len(spec["paraphrases"]))
    d = f"the {desc_val} {spec['noun']}"
    instruction = spec["paraphrases"][p_idx].format(d=d)
    pair_id = f"{family}-{pair_idx:04d}"

    def build(ambiguous: bool, items: list[Item]) -> Scenario:
        matching = tuple(i.id for i in items if i.attrs[desc_dim] == desc_val)
        assert len(matching) == (2 if ambiguous else 1), "generator invariant broken"
        # the intended referent: matches the description AND carries clar_vals[0]
        target = next(
            i for i in items
            if i.attrs[desc_dim] == desc_val and i.attrs[clar_dim] == clar_vals[0]
        )
        return Scenario(
            scenario_id=f"{pair_id}-{'amb' if ambiguous else 'una'}",
            pair_id=pair_id,
            family=family,
            ambiguous=ambiguous,
            instruction=instruction,
            items=tuple(items),
            matching_ids=matching,
            target_id=target.id,
            clarify_dim=clar_dim,
            clarify_value=target.attrs[clar_dim],
            paraphrase_idx=p_idx,
            split=SPLIT_BY_PARAPHRASE[p_idx],
        )

    return build(True, amb_items), build(False, una_items)


def generate(n_pairs: int, seed: int) -> list[Scenario]:
    """Generate n_pairs matched pairs (2*n_pairs scenarios), deterministic in seed."""
    rng = random.Random(seed)
    scenarios: list[Scenario] = []
    families = sorted(FAMILIES)
    for k in range(n_pairs):
        family = families[k % len(families)]
        amb, una = _make_pair(rng, family, k)
        scenarios.extend([amb, una])
    return scenarios


# ---------------------------------------------------------------- scoring

# outcome taxonomy; "proceed_when_ambiguous" is THE target failure mode (τ1)
OUTCOMES = (
    "correct_act",            # unambiguous, acted on the right item
    "correct_clarify",        # ambiguous, asked then acted on the right item
    "proceed_when_ambiguous", # ambiguous, acted without asking  <-- τ1
    "overask",                # unambiguous, asked anyway
    "reask_after_answer",     # asked, got the answer, asked AGAIN  <-- correction not propagated (τ5-like)
    "wrong_object",           # acted on a non-target item
    "clarify_then_wrong",     # asked, got the answer, still acted wrong
    "invalid",                # unparseable / no action
)


def score_episode(scenario: Scenario, steps: list[dict[str, Any]]) -> str:
    """Classify an episode from parsed steps: [{'type': 'act'|'ask'|'invalid', 'arg': str}]."""
    if not steps or steps[0]["type"] == "invalid":
        return "invalid"
    first = steps[0]
    if scenario.ambiguous:
        if first["type"] == "act":
            return "proceed_when_ambiguous"
        if len(steps) < 2:
            return "invalid"
        if steps[1]["type"] == "ask":
            return "reask_after_answer"
        if steps[1]["type"] != "act":
            return "invalid"
        return "correct_clarify" if steps[1]["arg"] == scenario.target_id else "clarify_then_wrong"
    if first["type"] == "ask":
        return "overask"
    return "correct_act" if first["arg"] == scenario.target_id else "wrong_object"
