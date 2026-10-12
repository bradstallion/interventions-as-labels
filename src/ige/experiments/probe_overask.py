"""probe_overask: attribution for τ_overask on collected records (design
the paper's §4).

Inputs: records.jsonl from collect_pairs (conversational, rule-v2).
Pipeline per record: render step-1 prompt with the model's CHAT template
(same code path as collection) -> verify_token_replay (blocking, chat-mode
boundary) -> capture resid_post at {instruction-final, last} positions.

Outputs:
1. U-probe sweep (ambiguous ground truth, all records, matched pairs) +
   positional canary (instruction-final must be at chance).
2. Behavioral probe sweep (unambiguous only: overask vs act) + content
   baseline (scenario metadata) it must beat.
3. H-misrep vs H-caution readout: U-score (best U layer, train pairs only)
   compared between overasked and acted unambiguous test records.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ige.attribution.probes import fit_probe, probe_sweep, split_by_pair
from ige.attribution.render import (instruction_final_index, render_chat,
                                    replay_tokens_for_record)
from ige.utils.env import pick_device, run_metadata
from ige.utils.seed import set_seed


def _target_pos(s: dict) -> int:
    return next(i for i, it in enumerate(s["items"]) if it["id"] == s["target_id"])


def _desc_val(s: dict) -> str:
    """Descriptor value = the target's attribute on the instruction dimension
    (the dimension that is not clarify_dim)."""
    target = next(it for it in s["items"] if it["id"] == s["target_id"])
    desc_dim = next(d for d in target["attrs"] if d != s["clarify_dim"])
    return f"{desc_dim}={target['attrs'][desc_dim]}"


def content_features(records: list[dict]) -> torch.Tensor:
    """Shallow-content baseline features (audit xxw): family, clarify dim,
    paraphrase, descriptor value, TARGET POSITION in the list (strong driver:
    pos0 -> 99/5 overask), distractor attr values, instruction length.
    NO activations. Note: under greedy decoding behavior is deterministic in
    the full prompt, so this baseline is a *shallow-content floor*, not an
    upper bound — the layer-0 probe is the cleanest floor and is reported
    alongside."""
    fams = sorted({r["scenario"]["family"] for r in records})
    dims = sorted({r["scenario"]["clarify_dim"] for r in records})
    descs = sorted({_desc_val(r["scenario"]) for r in records})
    rows = []
    for r in records:
        s = r["scenario"]
        distractor_vals = sorted(
            v for it in s["items"] if it["id"] != s["target_id"]
            for v in it["attrs"].values()
        )
        rows.append(
            [float(s["family"] == f) for f in fams]
            + [float(s["clarify_dim"] == d) for d in dims]
            + [float(_desc_val(s) == d) for d in descs]
            + [float(s["paraphrase_idx"] == k) for k in range(4)]
            + [float(_target_pos(s) == k) for k in range(3)]
            + [len(distractor_vals) / 10.0, len(s["instruction"]) / 100.0]
        )
    return torch.tensor(rows, dtype=torch.float32)


def behavior_label(record: dict) -> int | None:
    """1 = overask, 0 = acted directly; None = not usable (unambiguous only)."""
    if record["scenario"]["ambiguous"]:
        return None
    if record["outcome"] == "overask":
        return 1
    if record["outcome"] in ("correct_act", "wrong_object"):
        return 0
    return None


def run(cfg) -> dict:
    from ige.hooks.load import load_hooked

    set_seed(int(cfg.get("seed", 0)))
    device = pick_device(cfg.model.get("device", "auto"))
    dtype = getattr(torch, str(cfg.model.get("dtype", "float32")))
    model = load_hooked(cfg.model.name, device, dtype,
                        bridge=bool(cfg.model.get("bridge", False)))
    tokenizer = model.tokenizer

    records = [json.loads(l) for l in Path(cfg.records).read_text().splitlines()]
    sites = [f"blocks.{l}.hook_resid_post" for l in range(model.cfg.n_layers)]

    feats_last: dict[str, list] = {s: [] for s in sites}
    feats_instr: dict[str, list] = {s: [] for s in sites}
    with torch.no_grad():
        for r in records:
            text = render_chat(tokenizer, r["messages_per_step"][0])
            ids = replay_tokens_for_record(tokenizer, model, r)  # blocking
            instr_pos = instruction_final_index(tokenizer, text)
            tokens = torch.tensor([ids], device=device)
            _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in set(sites))
            for s in sites:
                feats_last[s].append(cache[s][0, -1].float().cpu())
                feats_instr[s].append(cache[s][0, instr_pos].float().cpu())

    pair_ids = [r["pair_id"] for r in records]
    u_labels = [int(r["scenario"]["ambiguous"]) for r in records]
    stacked_last = {s: torch.stack(v) for s, v in feats_last.items()}

    # ---- 0. context-sensitivity canary (blocking). Different prompts MUST
    # yield different last-position activations; identical features mean the
    # hooked model is not reading the context (e.g. broken TL load: Olmo-3
    # loaded with a missing W_K and a mis-set attention window produced
    # bit-identical features under the chat render, 2026-07-06).
    for s, t in stacked_last.items():
        if float((t - t[0]).abs().max()) == 0.0:
            raise AssertionError(
                f"context-sensitivity canary failed at {s}: last-position "
                "activations are identical across records — the hooked model "
                "is not conditioning on the varying context (broken load?)")

    # ---- 1. U-probe (all records) + positional canary
    u_report = probe_sweep(stacked_last, u_labels, pair_ids)
    u_canary = probe_sweep({s: torch.stack(v) for s, v in feats_instr.items()},
                           u_labels, pair_ids)
    bad = [s for s, r in u_canary.items() if r["auc"] is not None and r["auc"] > 0.65]
    if bad:
        raise AssertionError(f"positional canary failed at {bad}")

    # ---- 2. behavioral probe (unambiguous only) + content baseline
    b_idx = [i for i, r in enumerate(records) if behavior_label(r) is not None]
    b_labels = [behavior_label(records[i]) for i in b_idx]
    b_pairs = [pair_ids[i] for i in b_idx]
    b_report = probe_sweep({s: t[b_idx] for s, t in stacked_last.items()}, b_labels, b_pairs)
    content = fit_probe(content_features([records[i] for i in b_idx]), b_labels, b_pairs)

    # ---- 3. H-misrep vs H-caution: U-score conditioned on behavior
    # site selection on TRAIN pairs only (audit xxw: no test-side selection)
    train_idx, test_idx = split_by_pair(pair_ids, seed=0)
    train_pairs_only = [pair_ids[i] for i in train_idx]
    best_u_site = max(
        (fit_probe(stacked_last[s][train_idx],
                   [u_labels[i] for i in train_idx], train_pairs_only, seed=1).auc, s)
        for s in sites
    )[1]
    u_probe = fit_probe(stacked_last[best_u_site], u_labels, pair_ids, seed=0)
    direction = torch.tensor(u_probe.direction, dtype=torch.float32)
    test_b = [i for i in test_idx if behavior_label(records[i]) is not None]
    u_scores = {lab: [float(stacked_last[best_u_site][i] @ direction)
                      for i in test_b if behavior_label(records[i]) == lab]
                for lab in (0, 1)}
    h_readout = {
        "best_u_site": best_u_site,
        "mean_u_score_acted": float(np.mean(u_scores[0])) if u_scores[0] else None,
        "mean_u_score_overask": float(np.mean(u_scores[1])) if u_scores[1] else None,
        "n_acted": len(u_scores[0]), "n_overask": len(u_scores[1]),
        "note": "overask >> acted on U-score supports H-misrep; ~equal supports H-caution",
    }

    report = {
        "model": cfg.model.name,
        "n_records": len(records),
        "u_probe": u_report,
        "u_positional_canary": "ok",
        "behavior_probe": b_report,
        "content_baseline_auc": content.auc,
        "content_floor_layer0_auc": b_report[sites[0]]["auc"],
        "h_readout": h_readout,
        "meta": run_metadata(dict(records=str(cfg.records))),
    }
    out = Path(cfg.get("out", "store/derived/probe_overask"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))

    best_b = max(((r["auc"], s) for s, r in b_report.items() if r["auc"] is not None),
                 default=(None, None))
    print(json.dumps({
        "best_u": {"site": best_u_site, "auc": u_report[best_u_site]["auc"]},
        "best_behavior": {"site": best_b[1], "auc": best_b[0]},
        "content_baseline_auc": content.auc,
        "h_readout": h_readout,
    }, indent=1))
    return report
