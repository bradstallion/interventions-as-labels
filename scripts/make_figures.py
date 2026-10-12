"""Generate the 6 paper figures (PDF) into paper/overleaf/figures/.

Data comes from store/derived reports wherever it exists; the few numbers that
live only in the predictions files and the paper (peak depths, cosines, per-seed M5 verdicts)
are hard-coded here with a pointer to their source. Palette and mark rules follow
the dataviz method (validated light palette; status colors never reused as series;
pass/fail always color + glyph, never color alone).

Run:  uv run python scripts/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "overleaf" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# --- palette (dataviz reference instance, light mode) ---
BLUE, AQUA, YELLOW, VIOLET, RED = "#2a78d6", "#1baf7a", "#eda100", "#4a3aa7", "#e34948"
GOOD, CRIT = "#0ca30c", "#d03b3b"
INK, INK2, MUTED, GRID, BASE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": BASE, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK2, "text.color": INK,
    "grid.color": GRID, "grid.linewidth": 0.6,
    "figure.dpi": 150, "savefig.bbox": "tight",
})

MODELS = [  # display name, patch dir, probe dir, n_layers, peak layer, cos(b,s) [paper Table 1 / Fig. 4]
    ("Qwen2.5-3B", "qwen3B", "qwen3B", 36, 28, 0.12),
    ("gemma-2-2b", "gemma2b", "gemma2b", 26, 21, 0.52),
    ("OLMo-3-7B", "olmo3", "olmo3", 32, 21, 0.07),   # frozen-protocol site (Table 1); test-side peak is L17
    ("Nemo-12B", "nemo12b", "nemo12b", 40, 22, 0.42),
    ("Phi-4", "phi4", "phi4", 40, 22, 0.21),
]

def load(p):
    return json.load(open(ROOT / "store" / "derived" / p / "report.json"))

# ---------------------------------------------------------------- fig 2: causal profiles
def fig_causal_profiles():
    fig, axes = plt.subplots(1, 5, figsize=(6.3, 1.75), sharey=True)
    for ax, (name, patch_dir, probe_dir, nl, peak, _) in zip(axes, MODELS):
        pr = load(f"m3_patch_overask_{patch_dir}_v2")   # frozen-protocol reports (n = ASK-side test over-asks)
        flips = {}   # layer -> flip rate (extrapolated, probe_direction)
        rand_max = 0.0
        for res in pr["results"]:
            L = int(res["site"].split(".")[1])
            if res.get("kind") == "probe_direction" and res.get("alpha_label") == "extrapolated":
                flips[L] = res["flip_rate"]
            if "random" in str(res.get("kind", "")):
                rand_max = max(rand_max, res["flip_rate"])
        prb = load(f"m3_probe_overask_{probe_dir}")["behavior_probe"]
        auc = {int(k.split(".")[1]): v["auc"] for k, v in prb.items()}
        xs_a = sorted(auc); ax.plot([x / nl * 100 for x in xs_a], [auc[x] for x in xs_a],
                                    color=AQUA, lw=1.2, alpha=0.9, zorder=2)
        xs_f = sorted(flips); ax.plot([x / nl * 100 for x in xs_f], [flips[x] for x in xs_f],
                                      color=BLUE, lw=1.6, marker="o", ms=3.2, zorder=3)
        ax.axhline(rand_max, color=MUTED, lw=1.0, ls=(0, (3, 2)), zorder=1)
        px = peak / nl * 100
        ax.annotate(f"L{peak}", (px, flips.get(peak, max(flips.values()))),
                    textcoords="offset points", xytext=(0, -11), ha="center",
                    fontsize=7, color=INK2)
        ax.set_title(name, color=INK)
        ax.set_ylim(-0.05, 1.08); ax.set_xlim(0, 100)
        ax.grid(True, axis="y"); ax.set_xticks([0, 50, 100])
    axes[0].set_ylabel("rate")
    axes[2].set_xlabel("depth (% of layers)")
    fig.legend(handles=[
        plt.Line2D([], [], color=BLUE, lw=1.6, marker="o", ms=3.2, label="causal flip rate (direction patch)"),
        plt.Line2D([], [], color=AQUA, lw=1.2, label="behavior-probe AUC"),
        plt.Line2D([], [], color=MUTED, lw=1.0, ls=(0, (3, 2)), label="random directions (same procedure)"),
    ], loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=3, frameon=False)
    fig.savefig(OUT / "fig_causal_profiles.pdf")
    plt.close(fig)

# ---------------------------------------------------------------- fig 3: U-score readout
def fig_useparation():
    names, ratios = [], []
    for name, _, probe_dir, *_ in MODELS:
        h = load(f"m3_probe_overask_{probe_dir}")["h_readout"]
        names.append(name)
        ratios.append(abs(h["mean_u_score_overask"]) / abs(h["mean_u_score_acted"]))
    fig, ax = plt.subplots(figsize=(4.4, 1.9))
    y = list(range(len(names)))[::-1]
    for yi, name, r in zip(y, names, ratios):
        color = VIOLET if name == "Phi-4" else BLUE
        ax.plot([r, 1.0], [yi, yi], color=GRID, lw=1.4, zorder=1)
        ax.plot(1.0, yi, "o", color=INK2, ms=5, zorder=3)
        ax.plot(r, yi, "o", color=color, ms=5.5, zorder=3)
        ax.annotate(f"{r:.2f}", (r, yi), textcoords="offset points",
                    xytext=(0, 7 if name != "Phi-4" else 7), ha="center", fontsize=7, color=INK2)
    ax.set_yticks(y); ax.set_yticklabels(names, color=INK)
    ax.axvline(1.0, color=BASE, lw=0.8)
    ax.set_xlim(0, 1.18); ax.set_ylim(-0.9, len(names) - 0.3)
    ax.set_xlabel("over-asked episodes' mean distance from the ambiguity boundary\n(fraction of the acted class's mean distance; 1.0 = identical belief)")
    ax.grid(True, axis="x")
    ax.annotate("H-caution:\nbelief ≈ identical", (0.40, y[1]), fontsize=7, color=BLUE, ha="center", va="center")
    ax.annotate("H-misrep", (ratios[-1] + 0.02, y[-1] - 0.55), fontsize=7, color=VIOLET, ha="left")
    ax.plot([], [], "o", color=INK2, ms=5, label="acted (reference = 1.0)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.14), frameon=False)
    fig.savefig(OUT / "fig_useparation.pdf")
    plt.close(fig)

# ---------------------------------------------------------------- fig 4: steering
def fig_steering():
    seeds = ["1", "_s2", "_s3"]
    base, gated, prompt = [], [], []
    for s in seeds:
        r = load(f"m4_steer_eval_qwen3B{'' if s == '1' else s}")["policies"]
        base.append(r["none"]["burden"]["interventions"])
        gated.append(r["steer_gated"]["burden"]["interventions"])
        prompt.append(r["prompt_only"]["burden"]["interventions"])
    fig, ax = plt.subplots(figsize=(4.2, 2.0))
    x = [0, 1, 2]; w = 0.26
    for off, vals, color, lab in [(-w, base, MUTED, "base agent"),
                                  (0, prompt, YELLOW, "prompt-only baseline"),
                                  (w, gated, BLUE, "gated steering")]:
        ax.bar([xi + off for xi in x], vals, width=w * 0.92, color=color, label=lab, zorder=3)
        for xi, v in zip(x, vals):
            ax.annotate(str(v), (xi + off, v), textcoords="offset points",
                        xytext=(0, 2), ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([f"seed {i+1}" for i in x], color=INK)
    ax.set_ylabel("overseer interventions / 200 episodes")
    ax.grid(True, axis="y")
    ax.legend(frameon=False, loc="upper right", ncol=1)
    fig.savefig(OUT / "fig_steering.pdf")
    plt.close(fig)

# ---------------------------------------------------------------- fig 5: band sweep
def fig_bandsweep():
    # (width_layers, clarify_drop, IRR); widths -> % of depth per model
    qwen = [(1, 35, 0.61), (3, 20, 0.75), (8, 4, 0.84)]        # w8 = main local, seed 1 (paper §5.4)
    gemma = [(1, 71, 0.04), (3, 0, 0.11), (8, 100, 0.02), (12, 82, 0.03), (18, 1, 0.87)]
    # gemma w8 = main local run (collapse, clarify 82->0); IRR from its report
    g8 = load("m5_edit_eval_local_gemma2b")["criteria"]
    gemma[2] = (8, g8["c_clarify_drop"], max(g8["b_irr_burden"], 0))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(4.4, 3.0), sharex=True)
    for pts, nl, color, lab in [(qwen, 36, BLUE, "Qwen2.5-3B"), (gemma, 26, AQUA, "gemma-2-2b")]:
        xs = [w / nl * 100 for w, _, _ in pts]
        ax1.plot(xs, [d for _, d, _ in pts], color=color, lw=1.6, marker="o", ms=3.5)
        ax2.plot(xs, [i * 100 for _, _, i in pts], color=color, lw=1.6, marker="o", ms=3.5, label=lab)
    ax1.axhline(10, color=CRIT, lw=0.9, ls=(0, (3, 2)))
    ax1.annotate("retention criterion (≤10)", (0.99, 0.16), xycoords="axes fraction",
                 fontsize=6.8, color=CRIT, ha="right")
    ax1.set_ylabel("legitimate-question\ndrop")
    ax1.set_ylim(-6, 112)
    ax1.grid(True, axis="y")
    ax2.axhline(50, color=CRIT, lw=0.9, ls=(0, (3, 2)))
    ax2.annotate("IRR criterion (≥50%)", (0.02, 50), xycoords=("axes fraction", "data"),
                 textcoords="offset points", xytext=(0, 4), fontsize=6.8, color=CRIT)
    ax2.set_ylabel("IRR (%)")
    ax2.set_xlabel("edit band width (% of model depth)")
    ax2.grid(True, axis="y"); ax2.set_xlim(0, 100)
    ax2.legend(frameon=False, loc="upper right", bbox_to_anchor=(1.0, 0.78))
    for x, t in [(11.5, "can't\nfit"), (40, "fits, but\nunconditional"), (69.2, "conditional\n(pass)")]:
        ax1.annotate(t, (x, 103), fontsize=6.3, color=INK2, ha="center", va="top", style="italic")
    fig.savefig(OUT / "fig_bandsweep.pdf")
    plt.close(fig)

# ---------------------------------------------------------------- fig 6: Phi-4 inset
def fig_phi4():
    # per-seed local arm (edit_eval reports, seeds 1-3)
    seeds = [1, 2, 3]
    irr = [28.4, 49.5, 72.1]
    drop = [9, 11, 3]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.8, 1.7))
    fig.subplots_adjust(wspace=0.42)
    for ax, vals, thr, better, ylab in [(ax1, irr, 50, "above", "IRR (%)"),
                                        (ax2, drop, 10, "below", "legit.-question drop")]:
        for s, v in zip(seeds, vals):
            ok = (v >= thr) if better == "above" else (v <= thr)
            ax.plot(s, v, "o", ms=7, color=GOOD if ok else CRIT, zorder=3)
            ax.annotate("✓" if ok else "✗", (s, v), ha="center", va="center",
                        fontsize=6, color="white", zorder=4)
        ax.axhline(thr, color=INK2, lw=0.9, ls=(0, (3, 2)))
        ax.annotate("criterion", (0.03, thr), xycoords=("axes fraction", "data"),
                    textcoords="offset points", xytext=(0, 3), fontsize=6.3, color=INK2)
        ax.set_xticks(seeds); ax.set_xticklabels([f"s{s}" for s in seeds], color=INK)
        ax.set_xlim(0.5, 3.5); ax.set_ylabel(ylab); ax.grid(True, axis="y")
    ax1.set_ylim(0, 100); ax2.set_ylim(0, 15)
    fig.suptitle("Phi-4, local arm: the boundary fixed in advance runs between full-pipeline seeds",
                 fontsize=8, color=INK, y=1.06)
    fig.savefig(OUT / "fig_phi4.pdf")
    plt.close(fig)

# ---------------------------------------------------------------- fig 1: the map
def fig_map():
    # per model: (name, peak%, cos, local verdict, local seeds pass, local note, bc verdict, bc seeds pass, bc note)
    rows = [
        ("Qwen2.5-3B", "78%", "0.12", "pass", [1, 1, 1], "all criteria",
         "fail", [0, 0, 0], "retention −14/−11/−12; new re-ask"),
        ("gemma-2-2b", "81%", "0.52", "fail", [0], "collapse: never asks",
         "pass", [1], "retention +11"),
        ("OLMo-3-7B", "53%", "0.07", "fail", [0, 0, 0], "retention −15/−12/−12",
         "pass", [1, 1, 1], "IRR 86–90%"),
        ("Nemo-12B", "55%", "0.42", "fail", [0, 0, 0], "collapse −51/−92/−81",
         "pass", [1, 0, 1], "s2: retention −15; re-ask 8→21"),
        ("Phi-4", "55–85%", "0.21–0.52", "mixed", [0, 0, 1], "IRR 28→49.5→72%",
         "pass", [1, 1, 1], "IRR 70–78%"),
    ]
    fig, ax = plt.subplots(figsize=(6.3, 2.6))
    ax.set_axis_off()
    ncols = [0.0, 0.145, 0.255, 0.365, 0.6825, 1.0]  # name | peak | cos | local | bc
    fills = {"pass": "#e2f3e2", "fail": "#fae3e3", "mixed": "#fdf1d7"}
    edge = {"pass": GOOD, "fail": CRIT, "mixed": YELLOW}
    glyph = {"pass": "✓", "fail": "✗", "mixed": "◐"}
    hdr = ["model", "peak\ndepth", "cos(b,s)", "local (causal band)", "BC (all layers)"]
    for j, h in enumerate(hdr):
        xc = (ncols[j] + ncols[j + 1]) / 2
        ax.text(xc, 1.0, h, ha="center", va="center", fontsize=7.5, color=INK2, weight="bold")
    nr = len(rows)
    def verdict(v, seeds):
        # declared rule: PASS only when every seed passes; FAIL when none does;
        # anything in between is MIXED with the count (replaces the asymmetric
        # 'pass' for Nemo BC 2/3 and 'borderline' for Phi-4 LOCAL 1/3)
        k, n = sum(seeds), len(seeds)
        if k == n: return "pass"
        if k == 0: return "fail"
        return "mixed"
    for i, (name, peak, cos, lv, lseeds, lnote, bv, bseeds, bnote) in enumerate(rows):
        lv, bv = verdict(lv, lseeds), verdict(bv, bseeds)
        yc = 0.88 - (i + 0.5) * (0.88 / nr)
        h = 0.88 / nr
        ax.text((ncols[0] + ncols[1]) / 2, yc, name, ha="center", va="center", fontsize=7.5, color=INK)
        ax.text((ncols[1] + ncols[2]) / 2, yc, peak, ha="center", va="center", fontsize=7.5, color=INK2)
        ax.text((ncols[2] + ncols[3]) / 2, yc, cos, ha="center", va="center", fontsize=7.5, color=INK2)
        for (v, seeds, note, j) in [(lv, lseeds, lnote, 3), (bv, bseeds, bnote, 4)]:
            x0, x1 = ncols[j] + 0.006, ncols[j + 1] - 0.006
            ax.add_patch(Rectangle((x0, yc - h / 2 + 0.012), x1 - x0, h - 0.024,
                                   facecolor=fills[v], edgecolor=edge[v], lw=0.9))
            seedstr = " ".join(("●" if s else "○") for s in seeds)
            label = f"{glyph[v]} {v.upper()}" + (f" {sum(seeds)}/{len(seeds)}" if v == "mixed" else "")
            ax.text(x0 + 0.012, yc + h * 0.17, label, ha="left", va="center",
                    fontsize=7.2, color=edge[v], weight="bold")
            ax.text(x1 - 0.012, yc + h * 0.17, seedstr, ha="right", va="center",
                    fontsize=7.2, color=edge[v])
            ax.text(x0 + 0.012, yc - h * 0.22, note.replace("\n", "; "), ha="left", va="center",
                    fontsize=5.4, color=INK2)
    ax.text(0.0, -0.05, "● = seed passes all four per-arm criteria   ○ = seed fails (failed criterion in note)   "
                        "PASS = every seed, MIXED k/n otherwise   single marker = one seed (gemma)", fontsize=6.0, color=MUTED)
    ax.set_xlim(0, 1); ax.set_ylim(-0.08, 1.05)
    fig.savefig(OUT / "fig_map.pdf")
    plt.close(fig)

if __name__ == "__main__":
    fig_map(); fig_causal_profiles(); fig_useparation(); fig_steering(); fig_bandsweep(); fig_phi4()
    print("figures ->", OUT)
    for f in sorted(OUT.glob("*.pdf")):
        print(" ", f.name)
