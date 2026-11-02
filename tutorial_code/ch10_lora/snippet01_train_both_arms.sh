#!/usr/bin/env bash
# Chapter 10, hands-on: train the two adapters on the same data with the same
# recipe. The only difference between the two configs is layers_to_transform
# (snippet 1 of the chapter shows the diff). Needs a CUDA GPU; minutes each.
set -euo pipefail
uv run python -m ige.run configs/experiment/m5_lora_local_qwen3B.yaml   # local: 8 layers around L28
uv run python -m ige.run configs/experiment/m5_lora_bc_qwen3B.yaml      # bc: all 36 blocks
ls store/derived/m5_lora_local_qwen3B/adapter store/derived/m5_lora_bc_qwen3B/adapter
