"""Experiment entrypoint: `python -m ige.run <config.yaml>`.

M0 stub: loads the config, resolves device, stamps run metadata, and fails
loudly if the experiment kind has no runner yet. Experiment runners register
here as they are built (M1: collect, M3: probe/patch, M4: steer, ...).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from ige.utils.env import pick_device, run_metadata

RUNNERS: dict[str, str] = {
    # kind -> module implementing run(cfg); filled in as milestones land
    "clarify_eval": "ige.experiments.clarify_eval",
    "collect_pairs": "ige.experiments.collect_pairs",
    "probe_u": "ige.experiments.probe_u",
    "probe_overask": "ige.experiments.probe_overask",
    "patch_overask": "ige.experiments.patch_overask",
    "m4prep_overask": "ige.experiments.m4prep_overask",
    "steer_eval": "ige.experiments.steer_eval",
    "dir_cosine": "ige.experiments.dir_cosine",
    "lora_edit": "ige.train.lora_bc",
    "edit_eval": "ige.experiments.edit_eval",
    "sae_overask": "ige.experiments.sae_overask",
}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m ige.run <config.yaml>", file=sys.stderr)
        return 2

    cfg_path = Path(argv[0])
    cfg = OmegaConf.load(cfg_path)
    kind = cfg.get("kind", None)

    meta = run_metadata(OmegaConf.to_container(cfg, resolve=True))
    print(json.dumps({"config": str(cfg_path), "device": pick_device(cfg.get("device", "auto")), **meta}, indent=1))

    if kind not in RUNNERS:
        known = sorted(RUNNERS) or ["<none yet — M0>"]
        print(f"no runner for kind={kind!r}; known kinds: {known}", file=sys.stderr)
        return 1

    import importlib

    importlib.import_module(RUNNERS[kind]).run(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
