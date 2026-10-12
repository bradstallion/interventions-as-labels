"""IRR uncertainty for steer_eval runs (from the external adversarial audit,
review): the runner reports only the point IRR; the criteria written down in advance (m4_steer, criterion
b) requires an interval that excludes 0.

Three analyses per run directory, all CPU-only on the committed episodes.jsonl:

1. Reproduction gate: recompute every policy's burden with the exact runner
   configuration (EvalBudgetOverseer(seed=0), stream in file order) and assert
   it matches report.json. If this fails, the dump and the report diverged and
   nothing below is meaningful.
2. Overseer-seed sensitivity: the trust simulation is stochastic; the runner
   implicitly fixes seed 0. Sweep seeds 0..N-1 on the frozen episode stream,
   pairing base and steered arms on the same seed, and report the IRR
   distribution (this reproduces the review's 0.45-0.95 observation).
3. Pair-cluster bootstrap: resample twin pairs with replacement (the same
   resample applied to every arm - paired), rebuild each arm's stream in the
   resampled pair order, rerun the full dynamic-trust simulation with a fresh
   overseer seed per replicate, and report the percentile CI for IRR.

Note on the wording of the criterion: criterion (b) names a *Wilson* CI, but Wilson
applies to a binomial proportion and IRR-burden is a ratio of burdens (watched
+ 2*interventions, trust-coupled across episodes) - a Wilson interval is not
defined for it. The pair-cluster bootstrap below is the applicable analogue;
the deviation must be stated wherever the criterion is reported.

Usage: uv run python scripts/analyze_irr_ci.py store/derived/m4_steer_eval_qwen3B [more dirs...]
Writes <dir>/irr_ci.json and prints a summary per directory.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ige.overseer.eval_budget import EvalBudgetOverseer  # noqa: E402

N_SEEDS = 1000
N_BOOT = 2000
GATED = "steer_gated"
BASE = "none"


def load_streams(run_dir: Path) -> dict[str, list[dict]]:
    """Per-policy episode stream, in file (= simulation) order."""
    streams: dict[str, list[dict]] = {}
    with (run_dir / "episodes.jsonl").open() as f:
        for line in f:
            ep = json.loads(line)
            streams.setdefault(ep["policy"], []).append({
                "family": ep["scenario"]["family"],
                "pair_id": ep["scenario"]["pair_id"],
                "intervene": bool(ep["verdict"]["intervene"]),
            })
    return streams


def burden(stream: list[dict], seed: int) -> float:
    ov = EvalBudgetOverseer(seed=seed)
    return ov.burden([(e["family"], e["intervene"]) for e in stream]).burden


def irr(base_burden: float, arm_burden: float) -> float:
    return (base_burden - arm_burden) / base_burden if base_burden else 0.0


def analyze(run_dir: Path) -> dict:
    streams = load_streams(run_dir)
    report = json.loads((run_dir / "report.json").read_text())
    out: dict = {"run_dir": str(run_dir), "n_seeds": N_SEEDS, "n_boot": N_BOOT}

    # ---- 1. reproduction gate -------------------------------------------
    repro = {}
    for name, stream in streams.items():
        got = burden(stream, seed=0)
        want = report["policies"][name]["burden"]["burden"]
        if abs(got - want) > 1e-9:
            raise AssertionError(
                f"{run_dir}: replayed burden for '{name}' is {got}, "
                f"report.json says {want} - dump and report diverged")
        repro[name] = got
    out["reproduced_point_burden"] = repro
    out["reproduced_point_irr"] = {
        n: irr(repro[BASE], b) for n, b in repro.items()}

    # ---- 2. overseer-seed sensitivity (frozen stream) -------------------
    per_seed_irr: dict[str, list[float]] = {n: [] for n in streams if n != BASE}
    for seed in range(N_SEEDS):
        b0 = burden(streams[BASE], seed)
        for name in per_seed_irr:
            per_seed_irr[name].append(irr(b0, burden(streams[name], seed)))
    out["seed_sensitivity"] = {
        name: {
            "mean": float(np.mean(v)),
            "p2.5": float(np.percentile(v, 2.5)),
            "median": float(np.percentile(v, 50)),
            "p97.5": float(np.percentile(v, 97.5)),
            "min": float(np.min(v)),
            "max": float(np.max(v)),
            "frac_positive": float(np.mean(np.array(v) > 0)),
        } for name, v in per_seed_irr.items()}

    # ---- 3. pair-cluster bootstrap (paired across arms) ------------------
    # index episodes of each arm by pair_id, preserving within-pair order
    by_pair: dict[str, dict[str, list[dict]]] = {}
    for name, stream in streams.items():
        d: dict[str, list[dict]] = {}
        for e in stream:
            d.setdefault(e["pair_id"], []).append(e)
        by_pair[name] = d
    pair_ids = sorted(by_pair[BASE].keys())
    for name, d in by_pair.items():
        if sorted(d.keys()) != pair_ids:
            raise AssertionError(f"{run_dir}: arm '{name}' covers different pairs")

    boot_irr: dict[str, list[float]] = {n: [] for n in streams if n != BASE}
    rng = np.random.default_rng(0)
    for rep in range(N_BOOT):
        resample = rng.choice(pair_ids, size=len(pair_ids), replace=True)
        seed = N_SEEDS + rep  # fresh overseer seed per replicate
        b0 = burden([e for p in resample for e in by_pair[BASE][p]], seed)
        for name in boot_irr:
            b = burden([e for p in resample for e in by_pair[name][p]], seed)
            boot_irr[name].append(irr(b0, b))
    out["bootstrap"] = {
        name: {
            "mean": float(np.mean(v)),
            "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))],
            "frac_positive": float(np.mean(np.array(v) > 0)),
        } for name, v in boot_irr.items()}

    # ---- criterion (b), as written down in advance: interval for the gated arm excludes 0 -----
    lo, hi = out["bootstrap"][GATED]["ci95"]
    out["criterion_b_interval_excludes_zero"] = bool(lo > 0)
    return out


def main() -> None:
    dirs = [Path(a) for a in sys.argv[1:]]
    if not dirs:
        sys.exit(__doc__)
    for run_dir in dirs:
        res = analyze(run_dir)
        (run_dir / "irr_ci.json").write_text(json.dumps(res, indent=1))
        g_seed = res["seed_sensitivity"][GATED]
        g_boot = res["bootstrap"][GATED]
        print(f"\n== {run_dir}")
        print(f"  point IRR (seed 0, as published): "
              f"{res['reproduced_point_irr'][GATED]:.3f}  [reproduction gate OK]")
        print(f"  overseer-seed sweep ({res['n_seeds']} seeds): "
              f"median {g_seed['median']:.3f}, 95% [{g_seed['p2.5']:.3f}, "
              f"{g_seed['p97.5']:.3f}], range [{g_seed['min']:.3f}, "
              f"{g_seed['max']:.3f}], positive {g_seed['frac_positive']:.1%}")
        print(f"  pair bootstrap ({res['n_boot']} reps): mean {g_boot['mean']:.3f}, "
              f"95% CI [{g_boot['ci95'][0]:.3f}, {g_boot['ci95'][1]:.3f}], "
              f"positive {g_boot['frac_positive']:.1%}")
        print(f"  criterion (b) interval excludes zero: "
              f"{res['criterion_b_interval_excludes_zero']}")


if __name__ == "__main__":
    main()
