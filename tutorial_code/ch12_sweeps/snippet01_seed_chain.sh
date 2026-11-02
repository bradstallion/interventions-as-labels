#!/usr/bin/env bash
# Chapter 12, snippet 1: one seed of the full pipeline as a chain that stops at
# the first failure. The tutorial writes <m> for the model; default qwen3B.
# Needs a CUDA GPU; the collection step is the long one.
set -euo pipefail
M="${1:-qwen3B}"
scripts/run_chain.sh configs/experiment/m2_collect_${M}_bf16_conv_s2.yaml \
                     configs/experiment/m5_lora_local_${M}_s2.yaml \
                     configs/experiment/m5_edit_eval_local_${M}_s2.yaml
