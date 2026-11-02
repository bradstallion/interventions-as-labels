#!/usr/bin/env bash
# Chapter 8, hands-on: the chapter's chain on one model, in order.
#   1. direction patching + matched random controls     (kind: patch_overask)
#   2. the control family: style, U on both faces, generation (kind: m4prep_overask)
#   3. gated steering on fresh scenarios                 (kind: steer_eval)
# Needs a CUDA GPU. Each step writes store/derived/<config name>/report.json;
# the case study's own runs of these configs are already in the repository
# (the patching one under m3_patch_overask_qwen3B_v2).
set -euo pipefail
uv run python -m ige.run configs/experiment/m3_patch_overask_qwen3B.yaml
uv run python -m ige.run configs/experiment/m4prep_overask_qwen3B.yaml
uv run python -m ige.run configs/experiment/m4_steer_eval_qwen3B.yaml
ls store/derived/m3_patch_overask_qwen3B/report.json \
   store/derived/m4prep_overask_qwen3B/report.json \
   store/derived/m4_steer_eval_qwen3B/report.json
