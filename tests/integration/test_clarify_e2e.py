"""Integration: 20 clarify_toy episodes end-to-end with a tiny real instruct
model (M1). Verifies pipeline mechanics — parseability, replay invariant,
outcome taxonomy — NOT behavioral quality (a 0.5B model may fail the task).

Downloads Qwen2.5-0.5B-Instruct (~1GB) on first run.
"""

import pytest

pytestmark = [pytest.mark.integration]

transformers = pytest.importorskip("transformers")

from ige.agent import HFBackend, check_replay_invariant, run_episode  # noqa: E402
from ige.tasks.clarify import OUTCOMES, generate  # noqa: E402


@pytest.fixture(scope="module")
def backend():
    return HFBackend("Qwen/Qwen2.5-0.5B-Instruct")


def test_20_episodes_end_to_end(backend):
    scenarios = generate(10, seed=3)  # 10 matched pairs = 20 episodes
    outcomes = []
    for scenario in scenarios:
        trace = run_episode(backend, scenario)
        check_replay_invariant(trace)          # boundary invariant on every episode
        assert trace.outcome in OUTCOMES
        assert len(trace.raw_replies) >= 1
        outcomes.append(trace.outcome)
    # mechanics, not behavior: most replies must at least parse
    invalid = outcomes.count("invalid")
    assert invalid <= len(outcomes) // 2, f"model output mostly unparseable: {outcomes}"
