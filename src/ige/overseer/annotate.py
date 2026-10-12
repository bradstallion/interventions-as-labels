"""Human annotation CLI for the kappa validity study (overseer validity protocol).

Usage:
  uv run python -m ige.overseer.annotate <records.jsonl> <out.jsonl> \
      [--n 60] [--seed 0] [--annotator NAME]

Shows each episode exactly as the LLM judge sees it (task, items, replies —
never ground truth), asks intervene? + type, saves verdicts. Stratified
sample; resumable (skips already-annotated). At the end, prints kappa vs the
rule-v2 labeler as a quick sanity signal (the real study compares two humans
+ the LLM judge)."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from ige.stats import cohen_kappa

TYPES = {"1": "ask-for-help", "2": "wrong-object", "3": "correction-propagation",
         "4": "unnecessary-clarification", "5": "other"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--annotator", default="human1")
    args = ap.parse_args()

    records = [json.loads(l) for l in Path(args.records).read_text().splitlines()]
    rng = random.Random(args.seed)
    by_family: dict[str, list] = {}
    for r in records:
        by_family.setdefault(r["scenario"]["family"], []).append(r)
    sample = []
    per_fam = max(1, args.n // len(by_family))
    for fam, rs in sorted(by_family.items()):
        rng.shuffle(rs)
        sample.extend(rs[:per_fam])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["scenario_id"] for l in out_path.read_text().splitlines()} \
        if out_path.exists() else set()

    with out_path.open("a") as f:
        for k, r in enumerate(sample):
            sid = r["scenario"]["scenario_id"]
            if sid in done:
                continue
            # never show sid: scenario ids end in -una/-amb (ground truth leak)
            print(f"\n===== {k+1}/{len(sample)} =====")
            print(r["messages_per_step"][0][-1]["content"])
            for i, reply in enumerate(r["raw_replies"]):
                print(f"--- reply {i+1}: {reply.strip()}")
            ans = input("Intervieni? [y/n/q] ").strip().lower()
            if ans == "q":
                break
            verdict = {"scenario_id": sid, "annotator": args.annotator,
                       "intervene": ans == "y", "type": ""}
            if ans == "y":
                t = input("Tipo? 1=ask-for-help 2=wrong-object 3=corr-prop "
                          "4=unnecessary-clarif 5=other: ").strip()
                verdict["type"] = TYPES.get(t, "other")
            f.write(json.dumps(verdict) + "\n")
            f.flush()

    # quick sanity: kappa vs rule-v2 stored verdicts
    human = {json.loads(l)["scenario_id"]: json.loads(l)["intervene"]
             for l in out_path.read_text().splitlines()}
    rule = {r["scenario"]["scenario_id"]: r["verdict"]["intervene"] for r in records}
    common = sorted(set(human) & set(rule))
    if len(common) >= 10:
        k = cohen_kappa([human[s] for s in common], [rule[s] for s in common])
        print(f"\nkappa(umano, rule-v2) su {len(common)} episodi: {k:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
