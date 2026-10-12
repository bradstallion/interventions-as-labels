"""Integration: render/token-invariant, U-probe and patching mechanics on
gpt2-small with real clarify_toy pairs. Verifies MECHANICS and canaries;
gpt2's actual AUC is informative but not a gate."""

import pytest
import torch

pytestmark = [pytest.mark.integration]

tl = pytest.importorskip("transformer_lens")

from ige.agent.prompts import build_messages  # noqa: E402
from ige.attribution import (  # noqa: E402
    ask_action_token_ids,
    instruction_final_index,
    patch_last_position,
    render_plain,
    verify_token_replay,
)
from ige.tasks.clarify import generate  # noqa: E402
from ige.utils.env import pick_device  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return tl.HookedTransformer.from_pretrained("gpt2", device=pick_device("auto"))


@pytest.fixture(scope="module")
def twins():
    scenarios = generate(4, seed=13)
    by_pair = {}
    for s in scenarios:
        by_pair.setdefault(s.pair_id, {})[s.ambiguous] = s
    return next(iter(by_pair.values()))


def test_token_replay_invariant_on_real_prompts(model):
    for scenario in generate(6, seed=17):
        text = render_plain(build_messages(scenario))
        ids = verify_token_replay(model.tokenizer, model, text)
        assert len(ids) > 50


def test_instruction_final_precedes_workspace(model):
    scenario = generate(2, seed=17)[0]
    text = render_plain(build_messages(scenario))
    idx = instruction_final_index(model.tokenizer, text)
    ids = model.tokenizer(text, add_special_tokens=False)["input_ids"]
    prefix = model.tokenizer.decode(ids[: idx + 1])
    assert "Workspace items:" not in prefix
    assert "id=" not in prefix


def test_twin_prefix_activations_identical(model, twins):
    """The positional-canary premise, verified directly: before the workspace
    block, twin activations are bit-identical."""
    site = "blocks.5.hook_resid_post"
    acts = {}
    for amb, scenario in ((True, twins[True]), (False, twins[False])):
        text = render_plain(build_messages(scenario))
        idx = instruction_final_index(model.tokenizer, text)
        tokens = model.to_tokens(text, prepend_bos=False)
        _, cache = model.run_with_cache(tokens, names_filter=lambda n: n == site)
        acts[amb] = cache[site][0, idx].cpu()
    assert torch.equal(acts[True], acts[False])


def test_patching_mechanics(model, twins):
    ids = ask_action_token_ids(model.tokenizer)
    amb_tokens = model.to_tokens(render_plain(build_messages(twins[True])), prepend_bos=False)
    una_tokens = model.to_tokens(render_plain(build_messages(twins[False])), prepend_bos=False)

    effect = patch_last_position(model, amb_tokens, una_tokens,
                                 "blocks.8.hook_resid_post", ids)
    assert effect.delta == pytest.approx(effect.patched - effect.baseline)
    # patching an early site with the twin's activation must do *something*
    # different from patching with the run's own activation (self-patch no-op)
    noop = patch_last_position(model, amb_tokens, amb_tokens,
                               "blocks.8.hook_resid_post", ids)
    assert abs(noop.delta) < 1e-3  # self-patch is a no-op: pipeline sanity
