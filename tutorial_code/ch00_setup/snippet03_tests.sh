#!/usr/bin/env bash
# "Before you start", snippet 3: prove it works. Unit tests are CPU-only and
# need no model; add "integration" to also run the gpt2 tests (downloads
# gpt2-small on first use; CPU is fine, a GPU is faster).
set -euo pipefail
uv run pytest -m unit -q      # the invariants the tutorial keeps citing
if [[ "${1:-}" == "integration" ]]; then
  uv run pytest -m integration -q
fi
