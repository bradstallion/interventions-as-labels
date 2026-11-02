#!/usr/bin/env bash
# Chapter 9, hands-on: the cosine table between the three earned axes, refit
# from a stored collection. Needs a CUDA GPU to recompute the activations;
# minutes, not hours.
set -euo pipefail
uv run python -m ige.run configs/experiment/dir_cosine_qwen3B.yaml
python3 -c 'import json; r=json.load(open("store/derived/dir_cosine_qwen3B/report.json")); print(json.dumps(r["sites"], indent=1)[:1500])'
