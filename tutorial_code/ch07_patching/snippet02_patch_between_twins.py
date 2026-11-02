"""Chapter 7, snippet 2: patch one activation between twins and watch the
ASK-ACTION preference move.

The tutorial shows only the call; this file builds what the call needs, on
gpt2-small (CPU is fine), the way tests/integration/test_attribution_gpt2.py
does: one twin pair from the task generator, plain rendering (gpt2 has no chat
template), the donor's last-position state grafted into the recipient's run.
The site printed in the tutorial, blocks.28.hook_resid_post, is Qwen's; gpt2 has
12 blocks, so SITE defaults to blocks.8 here.
"""
import argparse

import transformer_lens as tl

from ige.agent.prompts import build_messages
from ige.attribution import render_plain
from ige.attribution.patching import ask_action_token_ids, patch_last_position
from ige.tasks.clarify import generate
from ige.utils.env import pick_device

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gpt2")
ap.add_argument("--site", default="blocks.8.hook_resid_post")
ap.add_argument("--seed", type=int, default=13)
args = ap.parse_args()
SITE = args.site

model = tl.HookedTransformer.from_pretrained(args.model, device=pick_device("auto"))

# one twin pair: same instruction, workspace differing in the single detail
by_pair = {}
for s in generate(4, seed=args.seed):
    by_pair.setdefault(s.pair_id, {})[s.ambiguous] = s
twins = next(iter(by_pair.values()))
src_tokens = model.to_tokens(render_plain(build_messages(twins[True])), prepend_bos=False)   # donor: ambiguous
dst_tokens = model.to_tokens(render_plain(build_messages(twins[False])), prepend_bos=False)  # recipient: unambiguous

effect = patch_last_position(model, dst_tokens, src_tokens,
                             SITE,
                             ask_action_token_ids(model.tokenizer))
print(f"{SITE}: baseline ask-action logit diff {effect.baseline:+.3f}, "
      f"patched {effect.patched:+.3f}, delta {effect.delta:+.3f}")
noop = patch_last_position(model, dst_tokens, dst_tokens, SITE,
                           ask_action_token_ids(model.tokenizer))
print(f"self-patch (pipeline sanity, must be ~0): delta {noop.delta:+.6f}")
