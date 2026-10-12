"""m4prep_overask: the three controls that make the causal result paper-grade
(control arms of the paper's §4.3).

A. STYLE CONTROL — a "about to ask" direction built WITHOUT ambiguity
   semantics: same unambiguous scenarios, system prompt explicitly instructing
   ALWAYS-ASK vs NEVER-ASK; direction = mean activation difference. If
   patching it flips overask cases like the probe direction does, our
   direction is generic question-style, not task caution.

B. U-DIRECTION PATCHING, both faces —
   face 1: on overask cases, push U further toward "unambiguous"
           (H-caution predicts ~no flips: the model already knows);
   face 2: on acted cases, push U toward "ambiguous"
           (if this induces asking, U is causally read by the decision).

C. FULL-GENERATION VALIDATION — after the strongest behavior-direction patch,
   force the patched first token, then generate freely: the reply must parse
   as ACTION with the correct target id, not degenerate text.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ige.agent.parser import parse_reply
from ige.attribution.direction_patch import patch_direction_last_pos, project_to
from ige.attribution.patching import ask_action_token_ids
from ige.attribution.probes import fit_probe, split_by_pair
from ige.attribution.render import render_chat, replay_tokens_for_record
from ige.experiments.probe_overask import behavior_label
from ige.utils.env import pick_device, run_metadata
from ige.utils.seed import set_seed

ASK_SUFFIX = ("\nImportant: in this workspace, policy requires you to ALWAYS "
              "ask one clarifying question before acting.")
ACT_SUFFIX = ("\nImportant: in this workspace, policy requires you to NEVER "
              "ask questions; always act directly on your best guess.")


def style_variant(messages: list[dict], suffix: str) -> list[dict]:
    out = [dict(m) for m in messages]
    out[0] = {"role": "system", "content": out[0]["content"] + suffix}
    return out


def run(cfg) -> dict:
    from ige.hooks.load import load_hooked

    set_seed(int(cfg.get("seed", 0)))
    device = pick_device(cfg.model.get("device", "auto"))
    dtype = getattr(torch, str(cfg.model.get("dtype", "float32")))
    model = load_hooked(cfg.model.name, device, dtype,
                        bridge=bool(cfg.model.get("bridge", False)))
    tokenizer = model.tokenizer
    ids = ask_action_token_ids(tokenizer)
    sites = [f"blocks.{l}.hook_resid_post" for l in cfg.sites]
    site_set = set(sites)

    records = [json.loads(l) for l in Path(cfg.records).read_text().splitlines()]

    # ---- capture last-position activations + tokens for ALL records
    feats: dict[str, list] = {s: [] for s in sites}
    tokens_all: list[torch.Tensor] = []
    with torch.no_grad():
        for r in records:
            tok = torch.tensor([replay_tokens_for_record(tokenizer, model, r)], device=device)
            tokens_all.append(tok)
            _, cache = model.run_with_cache(tok, names_filter=lambda n: n in site_set)
            for s in sites:
                feats[s].append(cache[s][0, -1].float().cpu())
    stacked = {s: torch.stack(v) for s, v in feats.items()}

    pair_ids = [r["pair_id"] for r in records]
    u_labels = [int(r["scenario"]["ambiguous"]) for r in records]
    b_labels = [behavior_label(r) for r in records]
    train_idx, test_idx = split_by_pair(pair_ids, seed=0)
    train_set = set(train_idx)
    test_overask = [i for i in test_idx if b_labels[i] == 1]
    test_acted = [i for i in test_idx if b_labels[i] == 0]
    toks_overask = [tokens_all[i] for i in test_overask]
    toks_acted = [tokens_all[i] for i in test_acted]

    # ---- style direction: instructed-ask vs instructed-act on TRAIN unambiguous
    style_pool = [i for i in train_idx if b_labels[i] is not None][: int(cfg.get("style_n", 60))]
    style_acc: dict[str, dict[str, list]] = {s: {"ask": [], "act": []} for s in sites}
    with torch.no_grad():
        for i in style_pool:
            for key, suffix in (("ask", ASK_SUFFIX), ("act", ACT_SUFFIX)):
                text = render_chat(tokenizer, style_variant(
                    records[i]["messages_per_step"][0], suffix))
                tok = torch.tensor([tokenizer(text, add_special_tokens=False)["input_ids"]],
                                   device=device)
                _, cache = model.run_with_cache(tok, names_filter=lambda n: n in site_set)
                for s in sites:
                    style_acc[s][key].append(cache[s][0, -1].float().cpu())

    results = []
    rng = np.random.default_rng(0)
    for s in sites:
        # behavior-probe direction (reference, same run for comparability)
        bp_idx = [i for i in range(len(records)) if b_labels[i] is not None]
        probe = fit_probe(stacked[s][bp_idx], [b_labels[i] for i in bp_idx],
                          [pair_ids[i] for i in bp_idx], seed=0)
        d_beh = torch.tensor(probe.direction, dtype=torch.float32)
        proj = stacked[s] @ d_beh
        mu_act = float(np.mean([proj[i] for i in train_idx if b_labels[i] == 0]))
        mu_ask = float(np.mean([proj[i] for i in train_idx if b_labels[i] == 1]))
        for lab, tgt in (("class0_mean", mu_act), ("extrapolated", mu_act - (mu_ask - mu_act))):
            results.append({**patch_direction_last_pos(model, toks_overask, s, d_beh,
                            tgt, ids, lab).to_dict(), "kind": "behavior_direction"})

        # U direction, both faces
        u_probe = fit_probe(stacked[s], u_labels, pair_ids, seed=0)
        d_u = torch.tensor(u_probe.direction, dtype=torch.float32)
        proj_u = stacked[s] @ d_u
        mu_una = float(np.mean([proj_u[i] for i in train_idx if u_labels[i] == 0]))
        mu_amb = float(np.mean([proj_u[i] for i in train_idx if u_labels[i] == 1]))
        for lab, tgt in (("to_unambiguous", mu_una),
                         ("to_unambiguous_extrap", mu_una - (mu_amb - mu_una))):
            results.append({**patch_direction_last_pos(model, toks_overask, s, d_u,
                            tgt, ids, lab).to_dict(), "kind": "u_direction_face1"})
        for lab, tgt in (("to_ambiguous", mu_amb),
                         ("to_ambiguous_extrap", mu_amb + (mu_amb - mu_una))):
            results.append({**patch_direction_last_pos(model, toks_acted, s, d_u,
                            tgt, ids, lab).to_dict(), "kind": "u_direction_face2"})

        # style direction on overask cases
        ask_m = torch.stack(style_acc[s]["ask"]).mean(0)
        act_m = torch.stack(style_acc[s]["act"]).mean(0)
        d_style = (ask_m - act_m)
        d_style = d_style / d_style.norm()
        style_tgt = float(act_m @ d_style)
        results.append({**patch_direction_last_pos(model, toks_overask, s, d_style,
                        style_tgt, ids, "style_to_act").to_dict(), "kind": "style_direction"})

        # matched random controls
        for k in range(2):
            rd = torch.tensor(rng.standard_normal(stacked[s].shape[1]), dtype=torch.float32)
            rd /= rd.norm()
            rmu = float(np.mean([(stacked[s][i] @ rd) for i in train_idx if b_labels[i] == 0]))
            results.append({**patch_direction_last_pos(model, toks_overask, s, rd,
                            rmu, ids, f"random_{k}").to_dict(), "kind": "random_control"})

    # ---- C. full-generation validation at the strongest behavior patch
    gen_site = f"blocks.{cfg.gen_site}.hook_resid_post"
    bp_idx = [i for i in range(len(records)) if b_labels[i] is not None]
    probe = fit_probe(stacked[gen_site][bp_idx], [b_labels[i] for i in bp_idx],
                      [pair_ids[i] for i in bp_idx], seed=0)
    d = torch.tensor(probe.direction, dtype=torch.float32).to(device)
    proj = stacked[gen_site] @ d.cpu()
    mu_act = float(np.mean([proj[i] for i in train_idx if b_labels[i] == 0]))
    mu_ask = float(np.mean([proj[i] for i in train_idx if b_labels[i] == 1]))
    target = mu_act - (mu_ask - mu_act)

    gen_stats = {"n": 0, "flipped": 0, "parsed_action": 0, "correct_target": 0}
    with torch.no_grad():
        for i in test_overask:
            tok = tokens_all[i]

            def hook(value, hook):
                value[0, -1] = project_to(value[0, -1].float(), d.float(),
                                          target).to(value.dtype)
                return value

            with model.hooks(fwd_hooks=[(gen_site, hook)]):
                first = int(model(tok)[0, -1].argmax())
            gen_stats["n"] += 1
            if first != ids[1]:  # did not flip to ACTION's first token
                continue
            gen_stats["flipped"] += 1
            seq = torch.cat([tok, torch.tensor([[first]], device=device)], dim=1)
            for _ in range(30):  # greedy continuation, no hook (one-shot patch)
                nxt = int(model(seq)[0, -1].argmax())
                seq = torch.cat([seq, torch.tensor([[nxt]], device=device)], dim=1)
                if "\n" in tokenizer.decode([nxt]):
                    break
            reply = tokenizer.decode(seq[0, tok.shape[1]:], skip_special_tokens=True)
            parsed = parse_reply(reply)
            if parsed["type"] == "act":
                gen_stats["parsed_action"] += 1
                if parsed["arg"] == records[i]["scenario"]["target_id"]:
                    gen_stats["correct_target"] += 1

    report = {
        "model": cfg.model.name,
        "n_test_overask": len(test_overask),
        "n_test_acted": len(test_acted),
        "results": results,
        "generation_validation": {**gen_stats, "site": gen_site, "target": "extrapolated"},
        "meta": run_metadata(dict(records=str(cfg.records), sites=list(cfg.sites))),
    }
    out = Path(cfg.get("out", "store/derived/m4prep_overask"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"generation_validation": report["generation_validation"]}, indent=1))
    for r in results:
        print(f"{r['site']:26s} {r['kind']:20s} {r['alpha_label']:22s} "
              f"flip={r['flip_rate']:.2f} induce={r['induce_rate']:.2f} "
              f"dlogit={r['mean_delta_logitdiff']:+.2f}")
    return report
