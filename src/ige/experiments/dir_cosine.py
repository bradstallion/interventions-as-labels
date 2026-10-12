"""dir_cosine: per-site cosine between the behavior, U and style
directions. Motivation: on gemma the style direction flips 100% and beats the
behavior direction in gated steering (89% vs 63% IRR), on qwen it is inert
(7% flip) — hypothesis: on gemma the two directions are nearly aligned.

Directions are refit exactly as in steer_eval (logistic probes on TRAIN pairs
for behavior/U; ASK/ACT-suffix mean difference for style). Nothing is decided
here — it is a descriptive analysis; report goes to store/derived."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ige.attribution.probes import split_by_pair
from ige.attribution.render import render_chat, replay_tokens_for_record
from ige.experiments.m4prep_overask import ACT_SUFFIX, ASK_SUFFIX, style_variant
from ige.experiments.probe_overask import behavior_label
from ige.utils.env import pick_device, run_metadata
from ige.utils.seed import set_seed


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a @ b) / (a.norm() * b.norm()))


def run(cfg) -> dict:
    from ige.hooks.load import load_hooked
    from sklearn.linear_model import LogisticRegression

    set_seed(int(cfg.get("seed", 0)))
    device = pick_device(cfg.model.get("device", "auto"))
    dtype = getattr(torch, str(cfg.model.get("dtype", "float32")))
    model = load_hooked(cfg.model.name, device, dtype,
                        bridge=bool(cfg.model.get("bridge", False)))
    tokenizer = model.tokenizer
    sites = [f"blocks.{s}.hook_resid_post" for s in cfg.sites]
    site_set = set(sites)

    records = [json.loads(l) for l in Path(cfg.records).read_text().splitlines()]
    pair_ids = [r["pair_id"] for r in records]
    train_idx, _ = split_by_pair(pair_ids, seed=0)

    feats: dict[str, list[torch.Tensor]] = {s: [] for s in sites}
    with torch.no_grad():
        for r in records:
            tok = torch.tensor([replay_tokens_for_record(tokenizer, model, r)],
                               device=device)
            _, cache = model.run_with_cache(tok, names_filter=lambda n: n in site_set)
            for s in sites:
                feats[s].append(cache[s][0, -1].float().cpu())
    stacked = {s: torch.stack(v) for s, v in feats.items()}

    u_labels = [int(r["scenario"]["ambiguous"]) for r in records]
    b_labels = [behavior_label(r) for r in records]
    tr_b = [i for i in train_idx if b_labels[i] is not None]

    def probe_dir(s, idx, labels):
        clf = LogisticRegression(max_iter=2000).fit(
            stacked[s][idx].numpy(), [labels[i] for i in idx])
        w = torch.tensor(clf.coef_[0], dtype=torch.float32)
        return w / w.norm()

    # style direction: same construction as steer_eval (suffix variants)
    style_pool = tr_b[: int(cfg.get("style_n", 40))]
    acc: dict[str, dict[str, list]] = {s: {"ask": [], "act": []} for s in sites}
    with torch.no_grad():
        for i in style_pool:
            for key, sfx in (("ask", ASK_SUFFIX), ("act", ACT_SUFFIX)):
                text = render_chat(tokenizer, style_variant(
                    records[i]["messages_per_step"][0], sfx))
                tok = torch.tensor([tokenizer(text, add_special_tokens=False)["input_ids"]],
                                   device=device)
                _, cache = model.run_with_cache(tok, names_filter=lambda n: n in site_set)
                for s in sites:
                    acc[s][key].append(cache[s][0, -1].float().cpu())

    report: dict = {"model": cfg.model.name, "sites": {}}
    for s in sites:
        d_beh = probe_dir(s, tr_b, b_labels)
        d_u = probe_dir(s, list(train_idx), u_labels)
        d_style = torch.stack(acc[s]["ask"]).mean(0) - torch.stack(acc[s]["act"]).mean(0)
        d_style = d_style / d_style.norm()
        report["sites"][s] = {
            "cos_behavior_style": _cos(d_beh, d_style),
            "cos_behavior_u": _cos(d_beh, d_u),
            "cos_u_style": _cos(d_u, d_style),
        }
        print(s, json.dumps(report["sites"][s]))

    report["meta"] = run_metadata(dict(sites=list(cfg.sites)))
    out = Path(cfg.get("out", "store/derived/dir_cosine"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    return report
