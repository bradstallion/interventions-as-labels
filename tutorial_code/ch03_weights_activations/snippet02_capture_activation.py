"""Chapter 3, snippet 2: load a model with hooks, replay a stored episode
token-for-token, and read the residual-stream vector at one site.

The block in the tutorial is the body of this file with the three constants
filled in as printed: Qwen2.5-3B, cuda, blocks.28.hook_resid_post. The flags
let you run the same code path on a laptop, e.g.
    --model gpt2 --device cpu --site blocks.6.hook_resid_post
(gpt2 has no chat template, so the plain rendering of the tests is used).
"""
import argparse
import json

import torch

from ige.attribution.render import render_chat, render_plain, verify_token_replay
from ige.hooks.load import load_hooked

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
ap.add_argument("--device", default="cuda")
ap.add_argument("--site", default="blocks.28.hook_resid_post")
args = ap.parse_args()
MODEL, DEVICE, SITE = args.model, args.device, args.site
DTYPE = torch.bfloat16 if DEVICE.startswith("cuda") else torch.float32

model = load_hooked(MODEL, DEVICE, DTYPE)
rec = json.loads(open(
    "store/derived/m2_collect_qwen3B_bf16_conv/records.jsonl").readline())
if getattr(model.tokenizer, "chat_template", None):
    text = render_chat(model.tokenizer, rec["messages_per_step"][0])
else:  # gpt2: no chat template, use the plain rendering of the tests
    text = render_plain(rec["messages_per_step"][0])
ids = verify_token_replay(model.tokenizer, model, text)  # blocking check
tokens = torch.tensor([ids], device=DEVICE)
_, cache = model.run_with_cache(
    tokens, names_filter=SITE)
h = cache[SITE][0, -1]            # shape: (d_model,)
print(f"{SITE}: shape {tuple(h.shape)}, norm {h.float().norm():.2f}, "
      f"{len(ids)} tokens replayed")
