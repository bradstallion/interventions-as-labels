#!/usr/bin/env bash
# Chapter 5, hands-on: train the two probes at every layer on a stored
# collection and read the sweep. Needs a CUDA GPU (Qwen2.5-3B in bf16); the
# case study's own run of this config is already in the repository, so the
# report can be read without running anything.
set -euo pipefail
uv run python -m ige.run configs/experiment/m3_probe_overask_qwen3B.yaml
python3 -c 'import json; r=json.load(open("store/derived/m3_probe_overask_qwen3B/report.json")); print(sorted(r.keys()))'
