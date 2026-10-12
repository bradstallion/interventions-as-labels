from collections import Counter

import pytest

from ige.tasks.clarify import SPLIT_BY_PARAPHRASE, generate

pytestmark = pytest.mark.unit

N_PAIRS = 60


@pytest.fixture(scope="module")
def scenarios():
    return generate(N_PAIRS, seed=42)


def test_deterministic_in_seed(scenarios):
    again = generate(N_PAIRS, seed=42)
    assert [s.to_dict() for s in scenarios] == [d.to_dict() for d in again]
    different = generate(N_PAIRS, seed=43)
    assert [s.to_dict() for s in scenarios] != [d.to_dict() for d in different]


def test_balanced_classes(scenarios):
    amb = [s for s in scenarios if s.ambiguous]
    una = [s for s in scenarios if not s.ambiguous]
    assert len(amb) == len(una) == N_PAIRS


def test_zero_lexical_leak_by_construction(scenarios):
    """The instruction multiset must be IDENTICAL across classes: the label
    is not predictable from the instruction text (anti-confound, plan §6)."""
    amb = Counter(s.instruction for s in scenarios if s.ambiguous)
    una = Counter(s.instruction for s in scenarios if not s.ambiguous)
    assert amb == una


def test_pairs_share_instruction(scenarios):
    by_pair: dict[str, list] = {}
    for s in scenarios:
        by_pair.setdefault(s.pair_id, []).append(s)
    for pair in by_pair.values():
        assert len(pair) == 2
        assert pair[0].instruction == pair[1].instruction
        assert {pair[0].ambiguous, pair[1].ambiguous} == {True, False}


def test_ground_truth_invariants(scenarios):
    for s in scenarios:
        assert len(s.matching_ids) == (2 if s.ambiguous else 1)
        assert s.target_id in s.matching_ids
        # the clarification uniquely identifies the target among the matches
        matching_items = [i for i in s.items if i.id in s.matching_ids]
        holders = [i for i in matching_items if i.attrs[s.clarify_dim] == s.clarify_value]
        assert [i.id for i in holders] == [s.target_id]


def test_target_id_not_predictable_from_suffix(scenarios):
    """Anti-leak: the target must not always carry the same id suffix (the
    original generator always made it '<x>1' — found by reading traces)."""
    suffixes = Counter(s.target_id[-1] for s in scenarios)
    assert len(suffixes) == 3
    assert max(suffixes.values()) / len(scenarios) < 0.5  # roughly uniform


def test_unique_ids_and_split_assignment(scenarios):
    ids = [s.scenario_id for s in scenarios]
    assert len(ids) == len(set(ids))
    for s in scenarios:
        assert s.split == SPLIT_BY_PARAPHRASE[s.paraphrase_idx]
    assert {s.split for s in scenarios} == {"train", "test", "generalization"}
