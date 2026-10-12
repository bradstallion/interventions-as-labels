# Human Interventions as Labels

Code, configurations, episode logs and reports for the paper

> **Human Interventions as Labels: Localizing, Steering, and (Sometimes) Editing the
> Computations Behind LLM Agent Failures.** Cesare Zavattari, Alessandro Tommasi,
> Giuseppe Prencipe. Preprint, 2026. arXiv: *to be added*.

The idea in one line: when a human corrects an LLM agent, the correction is a label
pointing at the internal computation that just failed. The paper uses such labels to
localize the ask-vs-act decision in five model families, to steer it at run time, and
to edit it in the weights, with success measured as the reduction of the overseer's
future interventions.

**Full snapshot** (1.2 GB): Zenodo, DOI *to be added*. It is this repository with the
trained LoRA adapters added, which are needed only to re-evaluate the edited models
without retraining them.

## What is here

| path | content |
|---|---|
| `src/ige/` | the harness: scenario generator, agent backends, overseers, probes, patching, steering, LoRA training and evaluation, SAE analysis |
| `configs/experiment/` | one YAML per run; the `out:` field names the report directory |
| `store/derived/<run>/` | every report the paper reads: `report.json`, episode logs (`records.jsonl`, `episodes.jsonl`), probe directions, bootstrap intervals |
| `scripts/` | analysis and figure scripts (see below) and the chain runner `run_chain.sh` |
| `tests/` | unit and integration tests, including the blocking invariants the paper describes (token replay, prompt diff, canaries) |

## Setup

```bash
uv sync --all-extras          # Python 3.12, pinned lock file; extras: dev, gpu
uv run pytest -m unit -q      # the invariants, CPU only
```

## Reproducing the numbers without a GPU

Every table and figure derives from the files in `store/derived/`:

```bash
uv run python scripts/make_rq1_summary.py      # Table 1: frozen-site flip rates with Wilson intervals
uv run python scripts/analyze_irr_ci.py        # steering: pair-clustered bootstrap intervals for IRR
uv run python scripts/make_figures.py          # all figures, from the reports -> paper/overleaf/figures/
```

Reports for the editing results are read directly: `store/derived/m5_edit_eval_*/report.json`
(the `criteria` block holds the pass/fail per criterion, thresholds fixed in advance).

## Reproducing the experiments

`scripts/run_chain.sh` runs the chain for one model (collection, probes, patching,
LoRA training, evaluation) from the YAML configs. Base models and the Gemma Scope
dictionary are downloaded from Hugging Face (gemma requires accepting its license);
a 24 GB GPU covers the 2B–8B models, the 12B and 14B ones need more. Decoding is
greedy, so a small fraction of episodes can change verdict on different hardware;
that is why the paper reports the base-row audit as agreement within 8 points
rather than as an equality. Re-training an adapter with the same config gives a
different adapter: to re-evaluate exactly the adapters of the paper, use the Zenodo
snapshot.

## License

Code: MIT (`LICENSE`). Data, reports and documents: CC BY 4.0 (`LICENSE-DATA.md`).
