#!/usr/bin/env bash
# Chapter 13, hands-on: load the public Gemma Scope dictionary at the causal
# peak, pass the transfer gate, then read features and clamp. Needs a CUDA
# GPU (gemma-2-2b-it) and downloads the SAE from Hugging Face.
set -euo pipefail
uv run python -m ige.run configs/experiment/m6_sae_gemma_L21.yaml
python3 -c 'import json; r=json.load(open("store/derived/m6_sae_gemma_L21/report.json")); print(sorted(r.keys()))'
