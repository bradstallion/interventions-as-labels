"""Regenerate the RQ1 causal summary EXCLUSIVELY from committed report.json
files (from the external adversarial audit).

The historical store/derived/rq1_summary.json was assembled during a GPU
session from per-seed reports that were never committed: its three-seed
arrays cannot be regenerated from repository artifacts, it carries no
provenance, and — no Wilson intervals.
This script replaces it going forward:

- input: the frozen-site patch reports (m3_patch_overask_<model>_v2), whose
  primary number is `frozen_eval` (site+dose selected on train, evaluated
  once on test, Wilson CI already inside — see patch_overask.py);
- output: store/derived/rq1_summary_v2.json with, per model: the frozen
  site/dose, flip rate with numerator/denominator and Wilson CI, the max
  random-control flip rate ACROSS ALL SITES (not just the peak — the honest
  scoping), and source provenance (path + sha256 + git sha at aggregation);
- any missing input is a hard failure listing what is absent: no silent
  partial summaries.

Usage: uv run python scripts/make_rq1_summary.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["qwen3B", "gemma2b", "olmo3", "nemo12b", "phi4"]
OUT = ROOT / "store/derived/rq1_summary_v2.json"


def main() -> None:
    missing = [m for m in MODELS
               if not (ROOT / f"store/derived/m3_patch_overask_{m}_v2/report.json").exists()]
    if missing:
        sys.exit(f"missing frozen-site reports for: {', '.join(missing)} — "
                 "run configs/experiment/m3_patch_overask_<model>_v2.yaml first; "
                 "refusing to write a partial summary")

    summary: dict = {"models": {}, "provenance": {}}
    for m in MODELS:
        p = ROOT / f"store/derived/m3_patch_overask_{m}_v2/report.json"
        r = json.loads(p.read_text())
        fe = r["frozen_eval"]
        randoms_all_sites = [row for row in r["results"]
                             if row["kind"] == "random_control"]
        summary["models"][m] = {
            "site": r["selection"]["site"],
            "alpha_label": r["selection"]["alpha_label"],
            "selected_on": "train overasks (argmax of selection.table)",
            "n_test_overask": r["n_test_overask"],
            "flips": fe["ask_to_action"],
            "n_base_ask": fe["n_base_ask"],
            "flip_rate": fe["flip_rate"],
            "wilson_ci95": fe["wilson_ci95_flip_rate"],
            "random_flip_max_all_sites": max(
                (row["flip_rate"] for row in randoms_all_sites), default=0.0),
            "random_flip_max_frozen_site": max(
                (row["flip_rate"] for row in fe["random_controls_at_site"]),
                default=0.0),
        }
        summary["provenance"][m] = {
            "source": str(p.relative_to(ROOT)),
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "model_name": r["model"],
        }
    summary["provenance"]["aggregated_at_git_sha"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
        text=True).stdout.strip() or "nogit"
    summary["note"] = (
        "single-collection result (one records.jsonl per model); NOT a "
        "multi-seed replication — see the audit log for the history of the "
        "unreproducible three-seed arrays in the legacy rq1_summary.json")

    OUT.write_text(json.dumps(summary, indent=1))
    print(f"wrote {OUT.relative_to(ROOT)}")
    for m, row in summary["models"].items():
        print(f"  {m:10s} {row['site']:26s} flip {row['flips']}/{row['n_base_ask']}"
              f" = {row['flip_rate']:.3f}  Wilson95 [{row['wilson_ci95'][0]:.3f},"
              f" {row['wilson_ci95'][1]:.3f}]  rand_max(all sites)"
              f" {row['random_flip_max_all_sites']:.3f}")


if __name__ == "__main__":
    main()
