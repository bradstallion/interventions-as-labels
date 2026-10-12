#!/usr/bin/env bash
# Optimizer-seed replication for the Qwen local edit: the
# published s1/s2/s3 arms resample the DATA but share training seed 0; this
# chain retrains the s1 dataset with seed 1 and 2 and evaluates each adapter
# on the same eval scenarios (eval_seed 9). Runs sequentially on one GPU machine.
set -uo pipefail

for s in 1 2; do
  GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | sort -t, -k2 -n | head -1 | cut -d, -f1)
  echo "=== $(date +%H:%M:%S) lora opt$s (train) on GPU $GPU"
  CUDA_VISIBLE_DEVICES=$GPU uv run python -m ige.run \
    "configs/experiment/m5_lora_local_qwen3B_opt$s.yaml" || { echo "FAILED: train opt$s"; continue; }
  echo "=== $(date +%H:%M:%S) lora opt$s (eval) on GPU $GPU"
  CUDA_VISIBLE_DEVICES=$GPU uv run python -m ige.run \
    "configs/experiment/m5_edit_eval_local_qwen3B_opt$s.yaml" || echo "FAILED: eval opt$s"
done
echo "=== lora opt-seed chain done $(date +%H:%M:%S)"
