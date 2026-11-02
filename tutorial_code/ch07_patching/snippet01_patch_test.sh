#!/usr/bin/env bash
# Chapter 7, snippet 1: the worked patching example, as an integration test on
# gpt2-small (downloaded on first use; CPU is fine).
set -euo pipefail
uv run pytest tests/integration/test_attribution_gpt2.py -q
