#!/usr/bin/env bash
# Sequential re-run of patch_overask with the frozen-site protocol
# on all five models. Runs sequentially on one GPU machine; one model failing must
# not kill the chain (log and continue).
set -uo pipefail

for c in qwen3B gemma2b olmo3 nemo12b phi4; do
  GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)
  echo "=== $(date +%H:%M:%S) patch_v2 $c on GPU $GPU"
  CUDA_VISIBLE_DEVICES=$GPU uv run python -m ige.run \
    "configs/experiment/m3_patch_overask_${c}_v2.yaml" || echo "FAILED: $c"
done
echo "=== patch_v2 chain done $(date +%H:%M:%S)"
