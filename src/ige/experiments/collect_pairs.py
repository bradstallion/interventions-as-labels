"""collect_pairs: M2 data-collection experiment.

Config keys: model (backend), overseer (rule | llm), n_pairs, seed, out.
The collection overseer watches everything (attention model enters only at
eval time, the blind evaluation design).
"""

from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

from ige.agent.backend import HFBackend
from ige.data.collect import collect_dataset
from ige.overseer.llm import LLMOverseer
from ige.overseer.rule import RuleOverseer
from ige.tasks.clarify import generate
from ige.utils.seed import set_seed


def run(cfg) -> dict:
    set_seed(int(cfg.get("seed", 0)))
    scenarios = generate(int(cfg.n_pairs), seed=int(cfg.get("seed", 0)))

    backend = HFBackend(
        cfg.model.name,
        quantization=cfg.model.get("quantization", None),
        device=cfg.model.get("device", "auto"),
    )

    kind = cfg.overseer.get("kind", "rule")
    if kind == "rule":
        overseer = RuleOverseer()
    elif kind == "llm":
        policy = OmegaConf.load(cfg.overseer.policy)
        judge = HFBackend(
            cfg.overseer.model.name,
            quantization=cfg.overseer.model.get("quantization", None),
        )
        overseer = LLMOverseer(judge, rubric=policy.rubric, version=policy.version)
    else:
        raise ValueError(f"unknown overseer kind: {kind}")

    result = collect_dataset(backend, overseer, scenarios, Path(cfg.out),
                             conversational=bool(cfg.get("conversational", False)))
    print(json.dumps({**result.__dict__, "records_path": str(result.records_path)},
                     indent=1, default=str))
    return result.__dict__
