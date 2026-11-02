#!/usr/bin/env bash
# Chapter 1, snippet 3: the first stored episode, exactly as the model saw it.
# Run from the repository root. CPU only, no model.
set -euo pipefail
head -n 1 store/derived/m2_collect_qwen3B_bf16_conv/records.jsonl \
  | python3 -c 'import json,sys; print(json.dumps(
      json.load(sys.stdin)["messages_per_step"][0], indent=2))'
