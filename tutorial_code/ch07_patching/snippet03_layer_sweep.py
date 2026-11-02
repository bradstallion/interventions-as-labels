"""Chapter 7, snippet 3: sweep the patching site across layers and read the
causal profile of the whole-state twin swap.

The block in the tutorial is the loop below. This file adds the setup: a model
with hooks, a few twin pairs from the task generator, the same rendering and
replay check as Chapter 3, and a table of the mean delta per layer (plus a PNG
if matplotlib is installed). Defaults run on gpt2-small on a CPU in about a
minute; on a machine with a GPU, --model Qwen/Qwen2.5-3B-Instruct --device cuda
runs the same sweep on the case study's first model.
"""
import argparse

import torch

from ige.agent.prompts import build_messages
from ige.attribution.patching import ask_action_token_ids, patch_last_position
from ige.attribution.render import render_chat, render_plain, verify_token_replay
from ige.hooks.load import load_hooked
from ige.tasks.clarify import generate

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="gpt2")
ap.add_argument("--device", default="cpu")
ap.add_argument("--pairs", type=int, default=8)
ap.add_argument("--seed", type=int, default=13)
ap.add_argument("--plot", default="", help="optional PNG path for the profile")
args = ap.parse_args()

dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
model = load_hooked(args.model, args.device, dtype)
ids = ask_action_token_ids(model.tokenizer)


def tokens_for(scenario):
    messages = build_messages(scenario)
    if getattr(model.tokenizer, "chat_template", None):
        text = render_chat(model.tokenizer, messages)
    else:  # gpt2: no chat template, use the plain rendering of the tests
        text = render_plain(messages)
    return torch.tensor([verify_token_replay(model.tokenizer, model, text)],
                        device=args.device)


by_pair = {}
for s in generate(args.pairs, seed=args.seed):
    by_pair.setdefault(s.pair_id, {})[s.ambiguous] = s
pairs = [(tokens_for(tw[False]), tokens_for(tw[True])) for tw in by_pair.values()]

profile = []
for layer in range(model.cfg.n_layers):
    site = f"blocks.{layer}.hook_resid_post"
    deltas = [patch_last_position(model, dst_tokens, src_tokens, site, ids).delta
              for dst_tokens, src_tokens in pairs]
    profile.append(sum(deltas) / len(deltas))
    print(f"L{layer:2d}  mean delta {profile[-1]:+.3f}   "
          f"(min {min(deltas):+.3f}, max {max(deltas):+.3f})")

if args.plot:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5, 2.5))
        plt.plot(range(len(profile)), profile, marker="o")
        plt.axhline(0, color="gray", lw=0.8)
        plt.xlabel("layer"); plt.ylabel("mean delta (ask - action)")
        plt.title(f"whole-state twin swap, {args.model}, {len(pairs)} pairs")
        plt.tight_layout(); plt.savefig(args.plot, dpi=150)
        print("profile ->", args.plot)
    except ImportError:
        print("matplotlib not installed; the table above is the profile")
