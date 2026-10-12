"""probe_u: layer-wise U-probe (ambiguity representation) on matched pairs.

Pipeline: regenerate scenarios (deterministic) -> render prompts -> verify
token replay (blocking) -> capture resid_post at all layers, positions
{instruction-final, last} -> probe sweep with canaries at both positions.

Expectations (anti-confound design): AUC rises somewhere at the LAST position (after
the workspace block); AUC at the INSTRUCTION-FINAL position sits at chance —
twin activations there are bit-identical, so any signal is a pipeline bug.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ige.agent.prompts import build_messages
from ige.attribution.probes import probe_sweep
from ige.attribution.render import instruction_final_index, render_plain, verify_token_replay
from ige.tasks.clarify import generate
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

    scenarios = generate(int(cfg.n_pairs), seed=int(cfg.get("seed", 0)))
    sites = [f"blocks.{l}.hook_resid_post" for l in range(model.cfg.n_layers)]

    feats_last: dict[str, list[torch.Tensor]] = {s: [] for s in sites}
    feats_instr: dict[str, list[torch.Tensor]] = {s: [] for s in sites}
    labels: list[int] = []
    pair_ids: list[str] = []

    with torch.no_grad():
        for scenario in scenarios:
            text = render_plain(build_messages(scenario))
            ids = verify_token_replay(tokenizer, model, text)  # blocking boundary invariant
            instr_pos = instruction_final_index(tokenizer, text)
            tokens = torch.tensor([ids], device=device)
            _, cache = model.run_with_cache(tokens, names_filter=lambda n: n in set(sites))
            for s in sites:
                feats_last[s].append(cache[s][0, -1].cpu())
                feats_instr[s].append(cache[s][0, instr_pos].cpu())
            labels.append(int(scenario.ambiguous))
            pair_ids.append(scenario.pair_id)

    report = {
        "model": cfg.model.name,
        "n_scenarios": len(scenarios),
        "last_position": probe_sweep(
            {s: torch.stack(v) for s, v in feats_last.items()}, labels, pair_ids),
        "instruction_final_canary": probe_sweep(
            {s: torch.stack(v) for s, v in feats_instr.items()}, labels, pair_ids),
        "meta": run_metadata({"n_pairs": cfg.n_pairs, "seed": cfg.get("seed", 0)}),
    }

    # positional canary is blocking: signal before the manipulation = bug
    bad = [s for s, r in report["instruction_final_canary"].items()
           if r["auc"] is not None and r["auc"] > 0.65]
    if bad:
        raise AssertionError(
            f"positional canary failed at {bad}: probe separates classes at a "
            "position where twin activations must be identical — pipeline bug"
        )

    out = Path(cfg.get("out", "store/derived/probe_u"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    best = max((r["auc"], s) for s, r in report["last_position"].items()
               if r["auc"] is not None)
    print(json.dumps({"best_last_position": {"site": best[1], "auc": best[0]},
                      "positional_canary": "ok", "out": str(out)}, indent=1))
    return report
