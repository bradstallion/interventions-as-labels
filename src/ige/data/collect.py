"""M2 collection pipeline: episodes -> verdicts -> counterfactual records.

Blocking checks wired in (they raise, they don't warn):
- replay invariant on every trace;
- differential prompt diff on every matched pair (confound check);
- overseer false-positive canary on unambiguous+correct episodes.

Output: records.jsonl (one record per episode, twin-linked via pair_id)
+ summary.json. Records carry everything M3 needs to re-execute the episode
under hooks (verbatim messages) without re-running the backend.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ige.agent.backend import LLMBackend
from ige.agent.runner import Trace, check_replay_invariant, run_episode
from ige.eval.promptdiff import assert_pair_manipulation_only
from ige.overseer.base import Overseer
from ige.tasks.clarify import Scenario
from ige.utils.env import run_metadata

FP_CANARY_MAX = 0.05  # max overseer intervention rate on unambiguous+correct


@dataclass
class CollectionResult:
    records_path: Path
    n_records: int
    outcomes: dict[str, int]
    intervention_rate_ambiguous: float
    fp_rate_unambiguous_correct: float


def _record(trace: Trace, verdict) -> dict[str, Any]:
    return {
        "scenario": trace.scenario,
        "pair_id": trace.scenario["pair_id"],
        "split": trace.scenario["split"],
        "backend": trace.backend,
        "messages_per_step": trace.messages_per_step,
        "raw_replies": trace.raw_replies,
        "parsed_steps": trace.parsed_steps,
        "outcome": trace.outcome,
        "verdict": verdict.to_dict(),
        # rollout->replay identity: without this line the
        # provenance captured by the backend never reached disk — caught by
        # the external verification pass, 2026-08-06
        "rollout_info": trace.rollout_info,
    }


def collect_dataset(
    backend: LLMBackend,
    overseer: Overseer,
    scenarios: list[Scenario],
    out_dir: Path | str,
    *,
    conversational: bool = False,
) -> CollectionResult:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    traces_by_pair: dict[str, dict[bool, Trace]] = {}
    for scenario in scenarios:
        trace = run_episode(backend, scenario, conversational=conversational)
        check_replay_invariant(trace)  # blocking
        verdict = overseer.judge(trace)
        records.append(_record(trace, verdict))
        traces_by_pair.setdefault(scenario.pair_id, {})[scenario.ambiguous] = trace

    # blocking confound check on every complete pair
    for pair_id, twins in traces_by_pair.items():
        if len(twins) == 2:
            assert_pair_manipulation_only(
                twins[True].messages_per_step[0], twins[False].messages_per_step[0]
            )

    # overseer FP canary: unambiguous episodes solved correctly must not be intervened
    una_correct = [r for r in records
                   if not r["scenario"]["ambiguous"] and r["outcome"] == "correct_act"]
    fp = (sum(r["verdict"]["intervene"] for r in una_correct) / len(una_correct)
          if una_correct else 0.0)
    if fp > FP_CANARY_MAX:
        raise AssertionError(
            f"overseer FP canary failed: {fp:.2%} interventions on unambiguous+correct "
            f"episodes (max {FP_CANARY_MAX:.0%}) — fix the policy before collecting"
        )

    amb = [r for r in records if r["scenario"]["ambiguous"]]
    result = CollectionResult(
        records_path=out_dir / "records.jsonl",
        n_records=len(records),
        outcomes=dict(Counter(r["outcome"] for r in records)),
        intervention_rate_ambiguous=(
            sum(r["verdict"]["intervene"] for r in amb) / len(amb) if amb else 0.0
        ),
        fp_rate_unambiguous_correct=fp,
    )
    with result.records_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    (out_dir / "summary.json").write_text(json.dumps(
        {**result.__dict__, "records_path": str(result.records_path),
         "overseer_id": overseer.overseer_id, "meta": run_metadata()},
        indent=1, default=str))
    return result
