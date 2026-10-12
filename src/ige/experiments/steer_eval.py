"""steer_eval: M4 first pass — gated steering vs baselines, IRR with the
budget/trust eval overseer (criteria written down in advance: experiments/m4_steer/PREDICTIONS.md).

Runtime gate (NO ground truth): intervene on the first decision token iff
U-probe(L28) predicts unambiguous AND behavior-probe(L28) predicts ask.
Steering = project_to on the behavior direction (extrapolated target), first
token only; the continuation runs unpatched. Fresh scenarios (eval seed),
directions and thresholds fitted on the seed-1 TRAIN pairs only."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ige.agent.parser import parse_reply
from ige.agent.prompts import build_messages
from ige.attribution.direction_patch import project_to
from ige.attribution.probes import fit_probe, split_by_pair
from ige.attribution.render import (render_chat, replay_tokens_for_record,
                                    verify_token_replay)  # noqa: F401
from ige.experiments.m4prep_overask import ACT_SUFFIX, ASK_SUFFIX, style_variant
from ige.experiments.probe_overask import behavior_label
from ige.overseer.eval_budget import EvalBudgetOverseer
from ige.overseer.rule import RuleOverseer
from ige.tasks.clarify import Scenario, generate, score_episode
from ige.utils.env import pick_device, run_metadata
from ige.utils.seed import set_seed

PROMPT_ONLY_SUFFIX = ("\nDo not ask unnecessary questions: if the request "
                      "uniquely identifies one item, act directly.")

# extended 50-item battery (the original 12 are its prefix; M4 first-pass
# reports used the 12-item version)
from ige.eval.battery import CAPABILITY_BATTERY  # noqa: E402


class Steerer:
    """Gated one-shot steering wired into a TL model."""

    def __init__(self, model, site, gate_u, gate_b, direction, target):
        self.model, self.site = model, site
        self.gate_u, self.gate_b = gate_u, gate_b  # (dir, bias) tuples or None
        self.direction, self.target = direction, target
        self.fired = 0
        self.calls = 0

    def _gate(self, resid) -> bool:
        if self.gate_u is None:
            return True
        du, bu = self.gate_u
        db, bb = self.gate_b
        u_says_unambiguous = float(resid @ du + bu) < 0
        b_says_ask = float(resid @ db + bb) > 0
        return u_says_unambiguous and b_says_ask

    @torch.no_grad()
    def first_token(self, tokens) -> int:
        """Decide the first reply token, steering iff the gate fires."""
        self.calls += 1
        _, cache = self.model.run_with_cache(
            tokens, names_filter=lambda n: n == self.site)
        resid = cache[self.site][0, -1].float().cpu()
        if not self._gate(resid):
            return int(self.model(tokens)[0, -1].argmax())
        self.fired += 1

        def hook(value, hook):
            value[0, -1] = project_to(value[0, -1].float(),
                                      self.direction.to(value.device),
                                      self.target).to(value.dtype)
            return value

        with self.model.hooks(fwd_hooks=[(self.site, hook)]):
            return int(self.model(tokens)[0, -1].argmax())


def greedy_reply(model, tokens, first: int, max_new: int = 40) -> str:
    # Stop at ANY special token (end-of-turn included), like HF generate does.
    # Without this, models whose end-of-turn marker carries no newline (OLMo-3)
    # run past the turn boundary and the role word gets glued to the reply:
    # 'ACTION: d1' -> 'ACTION: d1user' -> scored wrong_object (audit 2026-07-06,
    # scripts/audit_olmo3_step2.py — deflated the base row of edit_eval).
    stop_ids = set(getattr(model.tokenizer, "all_special_ids", None) or [])
    seq = torch.cat([tokens, torch.tensor([[first]], device=tokens.device)], dim=1)
    for _ in range(max_new):
        nxt = int(model(seq)[0, -1].argmax())
        if nxt in stop_ids:
            break
        seq = torch.cat([seq, torch.tensor([[nxt]], device=tokens.device)], dim=1)
        if "\n" in model.tokenizer.decode([nxt]):
            break
    return model.tokenizer.decode(seq[0, tokens.shape[1]:], skip_special_tokens=True)


def run_episode_steered(model, scenario: Scenario, steerer, system_suffix=None) -> dict:
    steps, replies, messages_per_step = [], [], []
    clar, own_q = None, None
    for _ in range(2):
        messages = build_messages(scenario, clar, own_q)
        if system_suffix:
            messages = style_variant(messages, system_suffix)
        messages_per_step.append(messages)
        text = render_chat(model.tokenizer, messages)
        tokens = torch.tensor(
            [model.tokenizer(text, add_special_tokens=False)["input_ids"]],
            device=next(model.parameters()).device)
        # steering only at the step-1 decision; step 2 runs plain
        first = steerer.first_token(tokens) if clar is None else int(model(tokens)[0, -1].argmax())
        reply = greedy_reply(model, tokens, first)
        parsed = parse_reply(reply)
        steps.append(parsed)
        replies.append(reply)
        if parsed["type"] != "ask" or clar is not None:
            break
        clar, own_q = scenario.clarification_answer(), reply.strip()
    return {"outcome": score_episode(scenario, steps), "steps": steps,
            "replies": replies, "messages_per_step": messages_per_step}


def run(cfg) -> dict:
    from ige.hooks.load import load_hooked

    set_seed(int(cfg.get("seed", 0)))
    device = pick_device(cfg.model.get("device", "auto"))
    dtype = getattr(torch, str(cfg.model.get("dtype", "float32")))
    model = load_hooked(cfg.model.name, device, dtype,
                        bridge=bool(cfg.model.get("bridge", False)))
    tokenizer = model.tokenizer
    site = f"blocks.{cfg.site}.hook_resid_post"

    # ---- fit directions/gates on seed-1 TRAIN pairs only
    records = [json.loads(l) for l in Path(cfg.records).read_text().splitlines()]
    pair_ids = [r["pair_id"] for r in records]
    train_idx, _ = split_by_pair(pair_ids, seed=0)
    feats = []
    with torch.no_grad():
        for r in records:
            tok = torch.tensor([replay_tokens_for_record(tokenizer, model, r)], device=device)
            _, cache = model.run_with_cache(tok, names_filter=lambda n: n == site)
            feats.append(cache[site][0, -1].float().cpu())
    stacked = torch.stack(feats)

    u_labels = [int(r["scenario"]["ambiguous"]) for r in records]
    b_labels = [behavior_label(r) for r in records]
    b_idx = [i for i in range(len(records)) if b_labels[i] is not None]

    from sklearn.linear_model import LogisticRegression
    def fit_full(idx, labels):
        clf = LogisticRegression(max_iter=2000).fit(
            stacked[idx].numpy(), [labels[i] for i in idx])
        w = torch.tensor(clf.coef_[0], dtype=torch.float32)
        return w, float(clf.intercept_[0])

    tr = [i for i in train_idx]
    w_u, b_u = fit_full(tr, u_labels)
    tr_b = [i for i in tr if b_labels[i] is not None]
    w_b, b_b = fit_full(tr_b, b_labels)
    d_beh = w_b / w_b.norm()
    proj = stacked @ d_beh
    mu_act = float(np.mean([proj[i] for i in tr_b if b_labels[i] == 0]))
    mu_ask = float(np.mean([proj[i] for i in tr_b if b_labels[i] == 1]))
    target = mu_act - (mu_ask - mu_act)

    # style + random directions (same gate, same target-style procedure)
    style_pool = tr_b[:40]
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
    d_style /= d_style.norm()
    t_style = float(torch.stack(acc["act"]).mean(0) @ d_style)
    rng = np.random.default_rng(0)
    d_rand = torch.tensor(rng.standard_normal(stacked.shape[1]), dtype=torch.float32)
    d_rand /= d_rand.norm()
    t_rand = float(np.mean([(stacked[i] @ d_rand) for i in tr_b if b_labels[i] == 0]))

    gate = ((w_u, b_u), (w_b, b_b))
    policies = {
        "none": dict(steer=None, suffix=None),
        "prompt_only": dict(steer=None, suffix=PROMPT_ONLY_SUFFIX),
        "steer_gated": dict(steer=(d_beh, target), suffix=None),
        "steer_style": dict(steer=(d_style, t_style), suffix=None),
        "steer_random": dict(steer=(d_rand, t_rand), suffix=None),
    }

    scenarios = generate(int(cfg.n_pairs), seed=int(cfg.eval_seed))
    labeler = RuleOverseer()  # rule-v2 ground-truth verdicts
    eval_overseer = EvalBudgetOverseer()
    report: dict = {"policies": {}}
    out = Path(cfg.get("out", "store/derived/steer_eval"))
    out.mkdir(parents=True, exist_ok=True)
    # per-episode dump for the human IRR validation (ige/overseer/annotate_irr.py)
    episodes_f = (out / "episodes.jsonl").open("w")

    for name, p in policies.items():
        if p["steer"] is None:
            steerer = Steerer(model, site, None, None,
                              torch.zeros(stacked.shape[1]), 0.0)
            steerer._gate = lambda resid: False  # never steer
        else:
            steerer = Steerer(model, site, gate[0], gate[1], p["steer"][0], p["steer"][1])

        outcomes: dict[str, int] = {}
        stream = []
        for sc in scenarios:
            ep = run_episode_steered(model, sc, steerer, p["suffix"])
            outcomes[ep["outcome"]] = outcomes.get(ep["outcome"], 0) + 1
            from ige.agent.runner import Trace
            fake = Trace(scenario=sc.to_dict(), backend="steered",
                         parsed_steps=ep["steps"], outcome=ep["outcome"])
            verdict = labeler.judge(fake)
            stream.append((sc.family, verdict.intervene))
            episodes_f.write(json.dumps({
                "policy": name,
                "scenario": sc.to_dict(),
                "messages_per_step": ep["messages_per_step"],
                "raw_replies": ep["replies"],
                "outcome": ep["outcome"],
                "verdict": verdict.to_dict(),
            }) + "\n")
        episodes_f.flush()
        burden = eval_overseer.burden(stream)

        # capability battery + off-target gate firing
        cap_ok = 0
        cap_fired_before = steerer.fired
        for q, a in CAPABILITY_BATTERY:
            text = render_chat(tokenizer, [
                {"role": "system", "content": "Answer concisely."},
                {"role": "user", "content": q}])
            tok = torch.tensor([tokenizer(text, add_special_tokens=False)["input_ids"]],
                               device=device)
            first = steerer.first_token(tok)
            reply = greedy_reply(model, tok, first, max_new=12)
            if a in reply.strip().lower():
                cap_ok += 1

        report["policies"][name] = {
            "outcomes": outcomes,
            "burden": burden.to_dict(),
            "gate_fired_episodes": steerer.fired if p["steer"] else 0,
            "battery_accuracy": cap_ok / len(CAPABILITY_BATTERY),
            "battery_gate_firings": steerer.fired - cap_fired_before,
        }
        print(name, json.dumps(report["policies"][name]))

    base = report["policies"]["none"]["burden"]["burden"]
    for name in policies:
        b = report["policies"][name]["burden"]["burden"]
        report["policies"][name]["irr_burden"] = (base - b) / base if base else 0.0
    episodes_f.close()
    report["meta"] = run_metadata(dict(site=cfg.site, eval_seed=cfg.eval_seed))
    (out / "report.json").write_text(json.dumps(report, indent=1))
    return report
