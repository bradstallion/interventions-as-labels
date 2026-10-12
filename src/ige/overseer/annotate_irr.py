"""Human validation of the IRR claim (blind evaluation design): blind annotation of
steered vs unsteered episodes.

Usage:
  uv run python -m ige.overseer.annotate_irr <episodes.jsonl> <out.jsonl> \
      [--n 60] [--seed 0] [--annotator NAME] [--policies none,steer_gated]

episodes.jsonl is written by the steer_eval experiment (one record per
episode, all policies). The CLI samples n // n_policies episodes per policy
(stratified by family within each policy) and shuffles everything together:
the annotator is never told which policy an episode came from — with the
default policies the prompts are identical (steering is internal), so the
presentation is genuinely blind. Do not read the output file mid-study: it
stores the policy for the final analysis. Resumable (skips already-annotated;
q to quit). When the sample is complete it prints, per policy, the human
intervention rate and burden (EvalBudgetOverseer) next to the rule-v2 ones,
the human-judged IRR vs the overseer-judged IRR on the same sample, and
kappa(human, rule-v2)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ige.overseer.eval_budget import EvalBudgetOverseer
from ige.stats import cohen_kappa

TYPES = {"1": "ask-for-help", "2": "wrong-object", "3": "correction-propagation",
         "4": "unnecessary-clarification", "5": "other"}


def episode_key(record: dict) -> str:
    return f'{record["policy"]}:{record["scenario"]["scenario_id"]}'


def build_sample(records: list[dict], policies: list[str], n: int, seed: int
                 ) -> list[dict]:
    """Per policy: stratified-by-family subsample; then one blind shuffle of
    the union so presentation order carries no policy signal. Deterministic
    in (records, policies, n, seed) — required for resume."""
    rng = random.Random(seed)
    sample: list[dict] = []
    per_policy = max(1, n // len(policies))
    for pol in policies:
        pool = [r for r in records if r["policy"] == pol]
        if not pool:
            raise ValueError(f"no episodes for policy '{pol}'")
        by_family: dict[str, list] = {}
        for r in pool:
            by_family.setdefault(r["scenario"]["family"], []).append(r)
        per_fam = max(1, per_policy // len(by_family))
        for _fam, rs in sorted(by_family.items()):
            rng.shuffle(rs)
            sample.extend(rs[:per_fam])
    rng.shuffle(sample)
    return sample


def summarize(records: list[dict], sample: list[dict], annotations: dict[str, dict],
              policies: list[str]) -> dict:
    """Per-policy human vs rule-v2 rates and burden; IRR relative to the first
    policy (conventionally 'none'); kappa on all annotated episodes.
    Burden is computed in the original episode-stream order (trust dynamics
    are order-dependent), matching how steer_eval scores the full run."""
    order = {episode_key(r): i for i, r in enumerate(records)}
    out: dict = {"policies": {}, "complete": True}
    burdens: dict[str, dict[str, float]] = {}
    for pol in policies:
        eps = [r for r in sample if r["policy"] == pol]
        done = sorted((r for r in eps if episode_key(r) in annotations),
                      key=lambda r: order[episode_key(r)])
        if len(done) < len(eps):
            out["complete"] = False
        if not done:
            continue
        human = [(r["scenario"]["family"], bool(annotations[episode_key(r)]["intervene"]))
                 for r in done]
        rule = [(r["scenario"]["family"], bool(r["verdict"]["intervene"]))
                for r in done]
        bh = EvalBudgetOverseer().burden(human)
        br = EvalBudgetOverseer().burden(rule)
        burdens[pol] = {"human": bh.burden, "rule": br.burden}
        out["policies"][pol] = {
            "n": len(done),
            "human_intervene": sum(i for _, i in human),
            "rule_intervene": sum(i for _, i in rule),
            "human_burden": bh.burden,
            "rule_burden": br.burden,
        }
    base = policies[0]
    if base in burdens:
        for pol in policies[1:]:
            if pol not in burdens:
                continue
            for who in ("human", "rule"):
                b0, b1 = burdens[base][who], burdens[pol][who]
                out["policies"][pol][f"irr_{who}"] = (b0 - b1) / b0 if b0 else 0.0
    keys = [episode_key(r) for r in sample if episode_key(r) in annotations]
    verdict_by_key = {episode_key(r): bool(r["verdict"]["intervene"]) for r in records}
    if len(keys) >= 10:
        out["kappa_human_rule"] = cohen_kappa(
            [bool(annotations[k]["intervene"]) for k in keys],
            [verdict_by_key[k] for k in keys])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("episodes")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--annotator", default="human1")
    ap.add_argument("--policies", default="none,steer_gated",
                    help="comma-separated; the FIRST one is the IRR baseline")
    args = ap.parse_args()

    records = [json.loads(l) for l in Path(args.episodes).read_text().splitlines()]
    policies = args.policies.split(",")
    sample = build_sample(records, policies, args.n, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["episode_key"] for l in out_path.read_text().splitlines()} \
        if out_path.exists() else set()

    with out_path.open("a") as f:
        for k, r in enumerate(sample):
            key = episode_key(r)
            if key in done:
                continue
            print(f"\n===== {k+1}/{len(sample)} =====")
            print(r["messages_per_step"][0][-1]["content"])
            for i, reply in enumerate(r["raw_replies"]):
                print(f"--- reply {i+1}: {reply.strip()}")
            ans = input("Intervieni? [y/n/q] ").strip().lower()
            if ans == "q":
                break
            verdict = {"episode_key": key, "policy": r["policy"],
                       "scenario_id": r["scenario"]["scenario_id"],
                       "annotator": args.annotator,
                       "intervene": ans == "y", "type": ""}
            if ans == "y":
                t = input("Tipo? 1=ask-for-help 2=wrong-object 3=corr-prop "
                          "4=unnecessary-clarif 5=other: ").strip()
                verdict["type"] = TYPES.get(t, "other")
            f.write(json.dumps(verdict) + "\n")
            f.flush()

    annotations = {json.loads(l)["episode_key"]: json.loads(l)
                   for l in out_path.read_text().splitlines()} \
        if out_path.exists() else {}
    summary = summarize(records, sample, annotations, policies)
    if not summary["complete"]:
        n_done = sum(p["n"] for p in summary["policies"].values())
        print(f"\n[campione incompleto: {n_done}/{len(sample)} — numeri parziali]")
    for pol, s in summary["policies"].items():
        line = (f"{pol:>14}: n={s['n']:3d}  interventi umano/rule "
                f"{s['human_intervene']}/{s['rule_intervene']}  "
                f"burden umano/rule {s['human_burden']:.3f}/{s['rule_burden']:.3f}")
        if "irr_human" in s:
            line += f"  IRR umano {s['irr_human']:.0%} (rule {s['irr_rule']:.0%})"
        print(line)
    if "kappa_human_rule" in summary:
        print(f"kappa(umano, rule-v2): {summary['kappa_human_rule']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
