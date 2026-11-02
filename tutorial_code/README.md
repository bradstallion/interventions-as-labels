# tutorial_code — one runnable file per snippet of the tutorial

Every code block in the tutorial is an excerpt: the lines that carry the
idea, not always a complete program. The complete, runnable version of each
block is the file named in small print under it ("Runnable version: ..."),
here, named after the chapter and the block's position in it
(`chNN_<topic>/snippetMM_<what>.py|sh`). Python blocks are `.py`; shell
commands are `.sh`. A file contains the block's lines as printed, plus the
setup the block left out; the two places where a file departs from the print
are noted in the table. Blocks that are not code (message lists, prompt
examples, pseudo-code, YAML excerpts, hook-name anatomy) have no file; they
are listed below so nothing is silently missing.

Run everything from the repository root, after `uv sync --all-extras`:

```bash
bash tutorial_code/run_cpu_checks.sh        # all the CPU-only snippets, in order
uv run python tutorial_code/ch07_patching/snippet02_patch_between_twins.py
uv run python tutorial_code/ch07_patching/snippet03_layer_sweep.py --pairs 4 --plot profile.png
bash tutorial_code/ch11_judging/snippet01_edit_eval.sh   # GPU
```

| file | chapter, snippet | needs | what it does |
|---|---|---|---|
| `ch00_setup/snippet01_extract.sh` | Before you start, 1 | nothing, git | clone the repository and enter it |
| `ch00_setup/snippet02_environment.sh` | Before you start, 2 | network (first time) | `uv sync --all-extras` |
| `ch00_setup/snippet03_tests.sh` | Before you start, 3 | CPU; `integration` arg downloads gpt2 | unit tests, optionally the gpt2 integration tests |
| `ch01_transformer/snippet03_first_record.sh` | Ch. 1, 3 | CPU | print the first stored episode's messages |
| `ch01_transformer/snippet04_chat_template.py` | Ch. 1, 4 | network (Qwen tokenizer) | render those messages through Qwen's chat template |
| `ch03_weights_activations/snippet02_capture_activation.py` | Ch. 3, 2 | CUDA GPU + Qwen weights; `--model gpt2 --device cpu --site blocks.6.hook_resid_post` for a smoke test | replay an episode with the token invariant, read one residual-stream vector. Departs from the print: model, device and site are flags (defaults as printed), and gpt2 falls back to the plain rendering |
| `ch04_task/snippet04_collect.sh` | Ch. 4, 4 | CUDA GPU | collect a dataset from one config |
| `ch05_probes/snippet01_probe_sweep.sh` | Ch. 5, hands-on | CUDA GPU (the stored report is in the repository) | train the two probes at every layer, read the sweep |
| `ch07_patching/snippet01_patch_test.sh` | Ch. 7, 1 | CPU, downloads gpt2 | the worked patching example as a test |
| `ch07_patching/snippet02_patch_between_twins.py` | Ch. 7, 2 | CPU, downloads gpt2 | graft the donor twin's state into the recipient and read the ASK-ACTION shift. Departs from the print: the site is a flag defaulting to gpt2's blocks.8 (the print names Qwen's blocks.28) |
| `ch07_patching/snippet03_layer_sweep.py` | Ch. 7, 3 | CPU (about a minute on gpt2 with `--pairs 4`); `--model Qwen/Qwen2.5-3B-Instruct --device cuda` for a real profile | sweep the swap across layers, print the mean delta per layer, `--plot out.png` for the picture |
| `ch08_direction/snippet01_causal_chain.sh` | Ch. 8, hands-on | CUDA GPU (the stored reports are in the repository) | direction patching with controls, the control family, gated steering, in order |
| `ch09_geometry/snippet01_cosines.sh` | Ch. 9, hands-on | CUDA GPU | the cosine table between the three axes |
| `ch10_lora/snippet01_train_both_arms.sh` | Ch. 10, hands-on | CUDA GPU | train the local and the bc adapter |
| `ch11_judging/snippet01_edit_eval.sh` | Ch. 11, 1 (the config path block) | CUDA GPU; the Qwen local adapter in the repository | judge the local adapter by the panel of criteria fixed in advance |
| `ch12_sweeps/snippet01_seed_chain.sh` | Ch. 12, 1 | CUDA GPU | one seed of collect, train, evaluate as a stop-on-failure chain |
| `ch13_sae/snippet01_sae_transfer.sh` | Ch. 13, hands-on | CUDA GPU, downloads Gemma Scope | transfer gate, sparse readers, support overlap, clamps |

Snippets with no file: Ch. 1 blocks 1 and 2 (a message list and its rendered
template, shown by `snippet04`), Ch. 2 blocks 1 to 3 (the residual-stream
update rules and the site naming, prose), Ch. 3 block 1 (anatomy of a hook
name), Ch. 4 blocks 1 to 3 (the system prompt and a twin pair as text),
Ch. 10 block 1 (the one-line diff between the two LoRA configs; the files are
`configs/experiment/m5_lora_local_qwen3B.yaml` and `m5_lora_bc_qwen3B.yaml`;
the hands-on step that runs both arms has `ch10_lora/snippet01_train_both_arms.sh`).

The hands-on sections of Chapters 6 and 14 have no file: Chapter 6 reads the
canary fields of reports produced by the other chapters, Chapter 14 is the
reading order of everything else.
