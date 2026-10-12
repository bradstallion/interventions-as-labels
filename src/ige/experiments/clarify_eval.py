"""clarify_eval: measure the clarify_toy outcome distribution for a model.

This is the M1-gate experiment: `failure_rate` is the fraction of ambiguous
scenarios where the model proceeds without asking (τ = ask-for-help failure).
Gate M1 requires >= 0.30 on Gemma 2 2B bf16 ON THE GPU MACHINE; dev-tier runs
(4-bit / small instruct models) are prototyping numbers only.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from ige.agent.backend import HFBackend
from ige.agent.runner import check_replay_invariant, run_episode
from ige.tasks.clarify import generate
from ige.utils.env import run_metadata
from ige.utils.seed import set_seed


def run(cfg) -> dict:
    set_seed(int(cfg.get("seed", 0)))
    scenarios = generate(int(cfg.n_pairs), seed=int(cfg.get("seed", 0)))

    backend = HFBackend(
        cfg.model.name,
        quantization=cfg.model.get("quantization", None),
        device=cfg.model.get("device", "auto"),
    )

    out_dir = Path(cfg.get("out", "store/derived/clarify_eval"))
    traces_dir = out_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    conversational = bool(cfg.get("conversational", False))
    outcomes: Counter[str] = Counter()
    for scenario in scenarios:
        trace = run_episode(backend, scenario, conversational=conversational)
        check_replay_invariant(trace)  # boundary invariant, blocking
        outcomes[trace.outcome] += 1
        trace.save(traces_dir / f"{scenario.scenario_id}.json")

    n_amb = sum(1 for s in scenarios if s.ambiguous)
    report = {
        "backend": backend.name,
        "conversational": conversational,
        "n_scenarios": len(scenarios),
        "outcomes": dict(outcomes),
        "failure_rate_proceed_when_ambiguous": outcomes["proceed_when_ambiguous"] / n_amb,
        "invalid_rate": outcomes["invalid"] / len(scenarios),
        "meta": run_metadata({"n_pairs": cfg.n_pairs, "seed": cfg.get("seed", 0)}),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return report
