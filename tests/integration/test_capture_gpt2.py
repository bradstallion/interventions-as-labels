"""Integration: capture on a tiny real model, checked against TransformerLens
reference values (M0 gate).

Downloads gpt2-small (~500MB) on first run; uses GPU if available.
"""

import pytest
import torch

pytestmark = [pytest.mark.integration]

tl = pytest.importorskip("transformer_lens")

from ige.hooks import ActivationStore, capture  # noqa: E402
from ige.utils.env import pick_device  # noqa: E402

PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Rome was not built in a day.",
    "2 + 2 =",
]
SITES = ["blocks.3.hook_resid_post", "blocks.7.hook_resid_post"]


@pytest.fixture(scope="module")
def model():
    device = pick_device("auto")
    return tl.HookedTransformer.from_pretrained("gpt2", device=device)


def test_capture_matches_run_with_cache(model, tmp_path):
    store = capture(model, PROMPTS, SITES, tmp_path / "run1")
    for idx, prompt in enumerate(PROMPTS):
        _, cache = model.run_with_cache(model.to_tokens(prompt))
        for site in SITES:
            got = store.get(idx, site).float()
            ref = cache[site][0].to("cpu").float()
            assert got.shape == ref.shape
            # fp16 storage: tolerance driven by cast, not by the computation
            assert torch.allclose(got, ref, atol=2e-3, rtol=1e-3)


def test_capture_is_reproducible(model, tmp_path):
    s1 = capture(model, PROMPTS[:2], SITES[:1], tmp_path / "a")
    s2 = capture(model, PROMPTS[:2], SITES[:1], tmp_path / "b")
    for idx in (0, 1):
        assert torch.equal(s1.get(idx, SITES[0]), s2.get(idx, SITES[0]))


def test_capture_rejects_unknown_site(model, tmp_path):
    with pytest.raises(ValueError, match="unknown hook sites"):
        capture(model, PROMPTS[:1], ["blocks.999.hook_nope"], tmp_path / "x")


def test_store_reopens_after_capture(model, tmp_path):
    capture(model, PROMPTS[:1], SITES[:1], tmp_path / "r")
    store = ActivationStore(tmp_path / "r")
    assert store.meta["model"] == "gpt2"
    assert store.sites == SITES[:1]
