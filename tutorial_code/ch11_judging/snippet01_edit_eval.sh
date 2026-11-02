#!/usr/bin/env bash
# Chapter 11, snippet 1: judge the local adapter by the panel of criteria
# fixed in advance. Loads store/derived/m5_lora_local_qwen3B/adapter (in the
# repository, or produced by the Chapter 10 hands-on). Needs a CUDA GPU.
set -euo pipefail
uv run python -m ige.run configs/experiment/m5_edit_eval_local_qwen3B.yaml
python3 -c 'import json; r=json.load(open("store/derived/m5_edit_eval_local_qwen3B/report.json")); print(json.dumps(r.get("criteria", r), indent=2)[:2000])'
