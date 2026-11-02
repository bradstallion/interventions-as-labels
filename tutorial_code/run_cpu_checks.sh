#!/usr/bin/env bash
# Run every snippet that works without a GPU, in chapter order, from the
# repository root. Needs the environment (`uv sync --all-extras`) and network for
# the two Hugging Face downloads (Qwen tokenizer, gpt2-small) on first use.
# Set IGE_OFFLINE=1 to skip the steps that download.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
pass=0; fail=0
run() {
  local label="$1"; shift
  if "$@" >/tmp/tutorial_code_step.log 2>&1; then
    echo "PASS  $label"; pass=$((pass+1))
  else
    echo "FAIL  $label  (see /tmp/tutorial_code_step.log)"; fail=$((fail+1))
    tail -5 /tmp/tutorial_code_step.log | sed 's/^/      /'
  fi
}
run "ch00 unit tests"                uv run pytest -m unit -q
run "ch01 snippet03 first record"    bash tutorial_code/ch01_transformer/snippet03_first_record.sh
if [[ "${IGE_OFFLINE:-0}" != 1 ]]; then
  run "ch01 snippet04 chat template" uv run python tutorial_code/ch01_transformer/snippet04_chat_template.py
  run "ch03 snippet02 capture (gpt2 smoke)" uv run python tutorial_code/ch03_weights_activations/snippet02_capture_activation.py \
        --model gpt2 --device cpu --site blocks.6.hook_resid_post
  run "ch07 snippet01 patch test"    bash tutorial_code/ch07_patching/snippet01_patch_test.sh
  run "ch07 snippet02 patch twins"   uv run python tutorial_code/ch07_patching/snippet02_patch_between_twins.py
  run "ch07 snippet03 layer sweep (gpt2, 4 pairs)" uv run python tutorial_code/ch07_patching/snippet03_layer_sweep.py --pairs 4
fi
echo "---- $pass passed, $fail failed"
[[ $fail -eq 0 ]]
