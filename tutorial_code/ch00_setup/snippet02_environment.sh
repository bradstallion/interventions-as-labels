#!/usr/bin/env bash
# "Before you start", snippet 2: rebuild the environment (run at the repository root).
set -euo pipefail
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is missing
uv sync --all-extras    # extras: pytest, and peft for the LoRA chapters
