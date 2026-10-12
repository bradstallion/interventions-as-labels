"""patch_overask: causal test of the behavioral (ask/act) probe direction.

For each configured site: fit the behavior probe on TRAIN pairs, take its
direction, compute the acted-class (label 0) mean projection on TRAIN, then
on held-out TEST overask records set the projection to (a) the acted mean and
(b) an extrapolated value, and measure decision flips. The identical
procedure with random directions (matched per site) is the control that must
stay near zero.

Selection protocol (from the external adversarial audit): the
original report exposed the full per-site sweep on the test cohort, so the
"peak" site was implicitly selected on test (multiplicity bias on the peak
value and on depth comparisons). The primary result is now `frozen_eval`:
site and dose are chosen as the argmax of the same sweep run on TRAIN
overasks only, then that single frozen (site, dose) is evaluated once on the
untouched TEST overasks, with a Wilson CI (a plain binomial at that point).
The per-site test sweep is still reported under `results`, explicitly labeled
exploratory.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ige.attribution.direction_patch import patch_direction_last_pos
from ige.attribution.patching import ask_action_token_ids
from ige.attribution.probes import fit_probe, split_by_pair
from ige.attribution.render import replay_tokens_for_record
from ige.experiments.probe_overask import behavior_label
from ige.utils.env import pick_device, run_metadata
from ige.utils.seed import set_seed


def run(cfg) -> dict:
    from ige.hooks.load import load_hooked

    set_seed(int(cfg.get("seed", 0)))
    device = pick_device(cfg.model.get("device", "auto"))
    dtype = getattr(torch, str(cfg.model.get("dtype", "float32")))
    model = load_hooked(cfg.model.name, device, dtype,
                        bridge=bool(cfg.model.get("bridge", False)))
    tokenizer = model.tokenizer
    ids = ask_action_token_ids(tokenizer)

    records = [json.loads(l) for l in Path(cfg.records).read_text().splitlines()]
    usable = [(i, r) for i, r in enumerate(records) if behavior_label(r) is not None]
    sites = [f"blocks.{l}.hook_resid_post" for l in cfg.sites]

    # capture last-position activations + token tensors for usable records
    feats: dict[str, list] = {s: [] for s in sites}
    tokens_all: list[torch.Tensor] = []
    with torch.no_grad():
        for _, r in usable:
            tok = torch.tensor([replay_tokens_for_record(tokenizer, model, r)], device=device)
            tokens_all.append(tok)
            _, cache = model.run_with_cache(tok, names_filter=lambda n: n in set(sites))
            for s in sites:
                feats[s].append(cache[s][0, -1].float().cpu())

    labels = [behavior_label(r) for _, r in usable]
    pair_ids = [r["pair_id"] for _, r in usable]
    train_idx, test_idx = split_by_pair(pair_ids, seed=0)
    train_overask = [j for j in train_idx if labels[j] == 1]
    test_overask = [j for j in test_idx if labels[j] == 1]

    rng = np.random.default_rng(0)
    results = []
    train_table = []
    per_site: dict[str, tuple] = {}
    for s in sites:
        stacked = torch.stack(feats[s])
        probe = fit_probe(stacked, labels, pair_ids, seed=0)
        d = torch.tensor(probe.direction, dtype=torch.float32)

        proj_train = stacked[train_idx] @ d
        mu0 = float(np.mean([proj_train[k] for k, j in enumerate(train_idx) if labels[j] == 0]))
        mu1 = float(np.mean([proj_train[k] for k, j in enumerate(train_idx) if labels[j] == 1]))
        per_site[s] = (d, mu0, mu1, probe.auc)

        # selection sweep: same doses, measured on TRAIN overasks only
        train_tokens = [tokens_all[j] for j in train_overask]
        for label, target in (("class0_mean", mu0), ("extrapolated", mu0 - (mu1 - mu0))):
            train_table.append({**patch_direction_last_pos(
                model, train_tokens, s, d, target, ids, label).to_dict(),
                "kind": "probe_direction", "cohort": "train", "train_auc": probe.auc})

        # exploratory per-site sweep on test (pre-frozen-selection protocol, kept for
        # continuity; NOT the primary claim — see frozen_eval)
        test_tokens = [tokens_all[j] for j in test_overask]
        for label, target in (("class0_mean", mu0), ("extrapolated", mu0 - (mu1 - mu0))):
            results.append({**patch_direction_last_pos(
                model, test_tokens, s, d, target, ids, label).to_dict(),
                "kind": "probe_direction", "train_auc": probe.auc})

        # matched random-direction controls (same procedure end-to-end)
        for k in range(int(cfg.get("n_random_controls", 3))):
            rd = torch.tensor(rng.standard_normal(stacked.shape[1]), dtype=torch.float32)
            rd /= rd.norm()
            rmu0 = float(np.mean([(stacked[j] @ rd) for j in train_idx if labels[j] == 0]))
            results.append({**patch_direction_last_pos(
                model, test_tokens, s, rd, rmu0, ids, f"random_{k}").to_dict(),
                "kind": "random_control"})

    # frozen-site evaluation: (site, dose) = argmax flip on TRAIN; one look
    # at test. max() keeps the first of ties -> deterministic in site order.
    from ige.utils.stats import wilson_ci
    best = max(train_table, key=lambda r: r["flip_rate"])
    site_star, label_star = best["site"], best["alpha_label"]
    d, mu0, mu1, auc_star = per_site[site_star]
    target_star = mu0 if label_star == "class0_mean" else mu0 - (mu1 - mu0)
    frozen = patch_direction_last_pos(
        model, [tokens_all[j] for j in test_overask],
        site_star, d, target_star, ids, label_star).to_dict()
    frozen["wilson_ci95_flip_rate"] = wilson_ci(
        frozen["ask_to_action"], frozen["n_base_ask"])
    frozen_randoms = [r for r in results
                      if r["site"] == site_star and r["kind"] == "random_control"]
    for r in frozen_randoms:
        r["wilson_ci95_flip_rate"] = wilson_ci(r["ask_to_action"], r["n_base_ask"])

    report = {
        "model": cfg.model.name,
        "n_train_overask": len(train_overask),
        "n_test_overask": len(test_overask),
        "selection": {"cohort": "train", "site": site_star,
                      "alpha_label": label_star, "train_auc": auc_star,
                      "table": train_table},
        "frozen_eval": {**frozen,
                        "random_controls_at_site": frozen_randoms,
                        "note": "site+dose frozen on train overasks; single "
                                "evaluation on untouched test overasks"},
        "results": results,
        "meta": run_metadata(dict(records=str(cfg.records), sites=list(cfg.sites))),
    }
    out = Path(cfg.get("out", "store/derived/patch_overask"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps([{k: r[k] for k in ("site", "kind", "alpha_label", "flip_rate",
                                          "mean_delta_logitdiff")} for r in results], indent=1))
    return report
