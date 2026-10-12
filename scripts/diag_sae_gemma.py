"""Diagnostic for the M6-SAE T0 failure (fvu ~8.6 at the last position).

Separates three hypotheses, per the m6_sae predictions file ("cross-check with
an independent implementation before declaring the transfer failed"):
  A. broken SAE implementation      -> FVU high even on the PT model, plain text
  B. broken PT->IT transfer         -> FVU high on IT everywhere
  C. position-specific (formatting/last token has out-of-distribution
     magnitudes on gemma-2)         -> FVU fine on content tokens, bad at -1

Run:  uv run python scripts/diag_sae_gemma.py
"""

from __future__ import annotations

import json

import torch

from ige.attribution.sae import JumpReluSae
from ige.hooks.load import load_hooked

LAYER = 21
SITE = f"blocks.{LAYER}.hook_resid_post"
TEXTS = [
    "The quick brown fox jumps over the lazy dog near the river bank.",
    "In 1969, astronauts first landed on the Moon during the Apollo program.",
    "Photosynthesis converts sunlight, water and carbon dioxide into glucose.",
    "The committee approved the budget after a long and contentious debate.",
    "Delete the pdf file from the workspace and confirm the operation.",
]
CHAT_USER = ("Task: Delete the pdf file.\nWorkspace items:\n"
             "- id=x1 format=pdf topic=notes\n- id=x2 format=txt topic=plan\n"
             "Reply with one line: ACTION: <id> or ASK: <question>")


def fvu(sae: JumpReluSae, x: torch.Tensor) -> float:
    return sae.fvu(x) if len(x) > 1 else float("nan")


def collect(model, texts: list[str], chat: bool) -> dict[str, torch.Tensor]:
    """Return {'content': acts at inner positions, 'last': acts at -1}."""
    content, last = [], []
    tok = model.tokenizer
    with torch.no_grad():
        for t in texts:
            if chat:
                msgs = [{"role": "user", "content": t}]
                text = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                               tokenize=False)
            else:
                text = t
            toks = model.to_tokens(text, prepend_bos=not chat)
            _, cache = model.run_with_cache(toks, names_filter=lambda n: n == SITE)
            acts = cache[SITE][0].float().cpu()
            mid = acts[len(acts) // 3: 2 * len(acts) // 3]  # inner content band
            content.append(mid)
            last.append(acts[-1:])
    return {"content": torch.cat(content), "last": torch.cat(last)}


def report(tag: str, sae: JumpReluSae, groups: dict[str, torch.Tensor]) -> None:
    for k, x in groups.items():
        f = sae.encode(x)
        print(json.dumps({
            "which": f"{tag}/{k}", "n": len(x),
            "fvu": round(fvu(sae, x), 3),
            "median_l0": float((f > 0).sum(-1).float().median()),
            "mean_norm": round(float(x.norm(dim=-1).mean()), 1),
            "max_norm": round(float(x.norm(dim=-1).max()), 1),
        }))


def main() -> None:
    sae = JumpReluSae.load_gemma_scope("google/gemma-scope-2b-pt-res", LAYER, "16k")
    print(f"SAE loaded: {sae.n_features} features")

    pt = load_hooked("google/gemma-2-2b", "cuda:0", torch.bfloat16, no_processing=True)
    report("PT/plain", sae, collect(pt, TEXTS, chat=False))
    del pt
    torch.cuda.empty_cache()

    it = load_hooked("google/gemma-2-2b-it", "cuda:0", torch.bfloat16, no_processing=True)
    report("IT/plain", sae, collect(it, TEXTS, chat=False))
    report("IT/chat", sae, collect(it, [CHAT_USER] * 3 + TEXTS[:2], chat=True))


if __name__ == "__main__":
    main()
