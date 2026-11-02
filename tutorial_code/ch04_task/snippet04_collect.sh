#!/usr/bin/env bash
# Chapter 4, snippet 4: collect a dataset with one config. Needs a CUDA GPU
# and the model weights from Hugging Face. The tutorial writes
# configs/experiment/m2_collect_<yours>.yaml; the case study's own Qwen
# collection is used here as the example.
set -euo pipefail
CONFIG="${1:-configs/experiment/m2_collect_qwen3B_bf16_conv.yaml}"
uv run python -m ige.run "$CONFIG"
