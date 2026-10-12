#!/usr/bin/env bash
# Run a sequence of experiment configs on this machine, stopping at the first
# failure (later configs in a chain consume the outputs of earlier ones, so
# continuing past a failure would only produce garbage downstream).
# Usage: run_chain.sh <config1> <config2> ...
set -uo pipefail

for cfg in "$@"; do
  echo "=== [chain] START $cfg  $(date -Is) ==="
  if ! uv run python -m ige.run "$cfg"; then
    echo "=== [chain] FAILED $cfg  $(date -Is) — chain aborted ==="
    exit 1
  fi
  echo "=== [chain] DONE  $cfg  $(date -Is) ==="
done
echo "=== [chain] ALL DONE  $(date -Is) ==="
