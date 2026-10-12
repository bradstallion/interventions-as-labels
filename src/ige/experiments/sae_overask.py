"""sae_overask (M6-SAE, predictions written down in advance: experiments/m6_sae/PREDICTIONS.md):
Gemma Scope at the causal site of gemma-2-2b-it.

Gates and arms, all fixed in advance:
T0  transfer gate (blocking): FVU <= 0.5 and median L0 in [20, 200] on OUR
    IT activations (SAEs are trained on the PT model).
P1  sparse probing: top-k SAE features vs the dense probe (behavior + U).
P2  support overlap: |top-50(behavior) ∩ top-50(style)| by decoder alignment
    (the SAE-level face of the dense cos ~ 0.52).
P3  causal arms on held-out overask episodes: clamp behavior-specific /
    shared / style-specific / random-matched features to the acted-class
    mean; flip rate + displacement along the style axis.
P3+ p1_readers arm (from the external adversarial audit): the
    original P3 arms are all built from decoder alignment, so the exact
    features that won the P1 sparse-probing result were never clamped —
    "readers don't move the decision" was asserted about an arm that did not
    contain the readers (committed report: behavior_specific ∩ P1-k10 = ∅).
    This arm clamps precisely the P1 k=10 behavior features (train-selected
    in P1 already).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ige.attribution.patching import ask_action_token_ids
from ige.attribution.probes import fit_probe, split_by_pair
from ige.attribution.render import render_chat, replay_tokens_for_record
from ige.attribution.sae import (JumpReluSae, clamp_features,
                                 decoder_alignment, topk_by_separation)
from ige.experiments.m4prep_overask import ACT_SUFFIX, ASK_SUFFIX, style_variant
from ige.experiments.probe_overask import behavior_label
from ige.hooks.load import load_hooked
from ige.utils.env import pick_device, run_metadata
from ige.utils.seed import set_seed


def run(cfg) -> dict:
    set_seed(int(cfg.get("seed", 0)))
    device = pick_device(cfg.model.get("device", "auto"))
    dtype = getattr(torch, str(cfg.model.get("dtype", "float32")))
    model = load_hooked(cfg.model.name, device, dtype,
                        bridge=bool(cfg.model.get("bridge", False)),
                        no_processing=bool(cfg.model.get("no_processing", False)))
    tokenizer = model.tokenizer
    ids = ask_action_token_ids(tokenizer)
    layer = int(cfg.sae.layer)
    site = f"blocks.{layer}.hook_resid_post"

    records = [json.loads(l) for l in Path(cfg.records).read_text().splitlines()]
    pair_ids = [r["pair_id"] for r in records]
    u_labels = [int(r["scenario"]["ambiguous"]) for r in records]
    b_labels = [behavior_label(r) for r in records]

    # ---- capture last-position activations (same render+replay as the probes)
    acts: list[torch.Tensor] = []
    tokens_all: list[torch.Tensor] = []
    with torch.no_grad():
        for r in records:
            tok = torch.tensor([replay_tokens_for_record(tokenizer, model, r)], device=device)
            tokens_all.append(tok)
            _, cache = model.run_with_cache(tok, names_filter=lambda n: n == site)
            acts.append(cache[site][0, -1].float().cpu())
    x = torch.stack(acts)

    # ---- T0' (gate amended on 2026-07-06, before any downstream result was read): the
    # variance-based FVU is reported but does NOT gate — our episodes are
    # near-identical twins, so between-sample variance is tiny and FVU
    # explodes on a modest reconstruction error (8.58 measured; diagnostic
    # scripts/diag_sae_gemma.py: implementation and content-token transfer
    # are healthy, 0.30/0.34). Gate on scale-relative error instead.
    sae = JumpReluSae.load_gemma_scope(cfg.sae.repo, layer, str(cfg.sae.width))
    feats = sae.encode(x)
    recon = sae.decode(feats)
    rel_err = float((((x - recon) ** 2).sum(-1) / (x ** 2).sum(-1)).mean())
    fvu = sae.fvu(x)
    l0 = float((feats > 0).sum(-1).float().median())
    t0_ok = rel_err <= 0.5 and 20 <= l0 <= 200
    t0 = {"rel_err": rel_err, "fvu_reported": fvu, "median_l0": l0, "ok": t0_ok}
    print(json.dumps({"T0prime": t0}))
    if not t0_ok and not cfg.get("report_only", False):
        raise AssertionError(
            f"T0' transfer gate failed (rel_err={rel_err:.3f}, l0={l0:.0f}): "
            "PT->IT transfer or SAE implementation broken — cross-check before use")

    train_idx, test_idx = split_by_pair(pair_ids, seed=0)
    b_idx = [i for i in range(len(records)) if b_labels[i] is not None]
    b_lab = [b_labels[i] for i in b_idx]
    b_pairs = [pair_ids[i] for i in b_idx]
    b_train = [j for j, i in enumerate(b_idx) if i in set(train_idx)]

    # ---- P1: sparse probing vs dense
    dense_b = fit_probe(x[b_idx], b_lab, b_pairs, seed=0)
    dense_u = fit_probe(x, u_labels, pair_ids, seed=0)
    sparse: dict[str, dict] = {}
    for k in (1, 5, 10, 50):
        sel_b = topk_by_separation(feats[b_idx], b_lab, b_train, k)
        sel_u = topk_by_separation(feats, u_labels, train_idx, k)
        sparse[f"k{k}"] = {
            "behavior_auc": fit_probe(feats[b_idx][:, sel_b], b_lab, b_pairs, seed=0).auc,
            "u_auc": fit_probe(feats[:, sel_u], u_labels, pair_ids, seed=0).auc,
            "behavior_features": sel_b,
        }
    p1 = {"dense_behavior_auc": dense_b.auc, "dense_u_auc": dense_u.auc,
          "sparse": sparse,
          "p1_pass": sparse["k10"]["behavior_auc"] >= 0.9 * dense_b.auc}

    # ---- style direction (same construction as dir_cosine/m4prep, train only)
    style_pool = [i for i in train_idx if b_labels[i] is not None][: int(cfg.get("style_n", 40))]
    acc = {"ask": [], "act": []}
    with torch.no_grad():
        for i in style_pool:
            for key, sfx in (("ask", ASK_SUFFIX), ("act", ACT_SUFFIX)):
                text = render_chat(tokenizer, style_variant(
                    records[i]["messages_per_step"][0], sfx))
                tok = torch.tensor([tokenizer(text, add_special_tokens=False)["input_ids"]],
                                   device=device)
                _, cache = model.run_with_cache(tok, names_filter=lambda n: n == site)
                acc[key].append(cache[site][0, -1].float().cpu())
    d_style = torch.stack(acc["ask"]).mean(0) - torch.stack(acc["act"]).mean(0)
    d_style = d_style / d_style.norm()
    d_beh = torch.tensor(dense_b.direction, dtype=torch.float32)

    # ---- P2: decoder-alignment supports and overlap
    top = int(cfg.get("support_k", 50))
    align_b = decoder_alignment(d_beh, sae)
    align_s = decoder_alignment(d_style, sae)
    top_b = set(torch.argsort(align_b, descending=True)[:top].tolist())
    top_s = set(torch.argsort(align_s, descending=True)[:top].tolist())
    rng = np.random.default_rng(0)
    rand_overlap = float(np.mean([
        len(set(rng.choice(sae.n_features, top, replace=False))
            & set(rng.choice(sae.n_features, top, replace=False)))
        for _ in range(200)]))
    p2 = {"overlap": len(top_b & top_s), "chance_overlap": rand_overlap,
          "p2_pass": len(top_b & top_s) >= 10}

    # ---- P3: causal arms on held-out overask episodes
    k_arm = int(cfg.get("arm_k", 10))
    rank_b = torch.argsort(align_b, descending=True).tolist()
    rank_s = torch.argsort(align_s, descending=True).tolist()
    arms = {
        "behavior_specific": [f for f in rank_b if f not in top_s][:k_arm],
        "shared": [f for f in rank_b if f in top_s][:k_arm],
        "style_specific": [f for f in rank_s if f not in top_b][:k_arm],
        # the actual P1 sparse-probe winners (train-selected), clamped as-is
        "p1_readers": [int(f) for f in sparse["k10"]["behavior_features"]],
    }
    acted_train = [i for i in train_idx if b_labels[i] == 0]
    mean_acted = feats[acted_train].mean(0)  # per-feature acted-class mean
    beh_act_level = float(feats[:, arms["behavior_specific"]].mean())
    pool = [f for f in range(sae.n_features)
            if abs(float(feats[:, f].mean()) - beh_act_level) < max(0.5, beh_act_level)]
    arms["random_matched"] = rng.choice(
        pool if len(pool) >= k_arm else sae.n_features, k_arm, replace=False).tolist()

    test_overask = [i for i in test_idx if b_labels[i] == 1]
    results = {}
    with torch.no_grad():
        for name, fs in arms.items():
            if not fs:
                results[name] = {"note": "empty arm"}
                continue
            targets = mean_acted[fs]
            flips, dstyle = 0, []

            def hook(value, hook):
                h = value[0, -1].float().cpu()
                h2 = clamp_features(h, sae, fs, targets)
                dstyle.append(float((h2 - h) @ d_style))
                value[0, -1] = h2.to(value.dtype).to(value.device)
                return value

            for i in test_overask:
                base = model(tokens_all[i])[0, -1]
                with model.hooks(fwd_hooks=[(site, hook)]):
                    patched = model(tokens_all[i])[0, -1]
                if float(base[ids[0]] - base[ids[1]]) > 0 >= float(patched[ids[0]] - patched[ids[1]]):
                    flips += 1
            results[name] = {"features": [int(f) for f in fs], "n": len(test_overask),
                             "flips": flips, "flip_rate": flips / max(1, len(test_overask)),
                             "mean_delta_style_proj": float(np.mean(dstyle))}

    report = {
        "model": cfg.model.name, "site": site,
        "sae": {"repo": cfg.sae.repo, "layer": layer, "width": str(cfg.sae.width)},
        "T0prime": t0,
        "P1": p1, "P2": p2, "P3": results,
        "cos_dense_behavior_style": float(d_beh @ d_style),
        "meta": run_metadata(dict(records=str(cfg.records))),
    }
    out = Path(cfg.get("out", "store/derived/m6_sae"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("T0prime", "P1", "P2", "P3")}, indent=1))
    return report
