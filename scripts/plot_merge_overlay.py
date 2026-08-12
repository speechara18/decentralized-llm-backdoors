"""
Baseline-vs-merge overlay, one figure per alpha: the figure that actually answers the
merging-attacker experiment. Baseline (attacker IGNORES gossip) dashed, merge
(attacker_merges=True) solid, on identical axes.

Panels: c0 detP (with delta=25) | c0 ASR | benign mean ASR (c1-c7).

Styling deliberately matches scripts/plot_run3way.py -- same rcParams, same attacker red,
same fixed y-limits -- so these can sit beside the 3way figures in a deck.

Usage: python plot_merge_overlay.py <merge.json> <baseline.json> [outdir]
"""
import sys, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"axes.titlesize": 18, "axes.labelsize": 16, "xtick.labelsize": 14,
                     "ytick.labelsize": 14, "legend.fontsize": 13, "figure.titlesize": 18})

merge_path, base_path = sys.argv[1], sys.argv[2]
outdir = sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(merge_path)
dm, db = json.load(open(merge_path)), json.load(open(base_path))
hm = dm["history"] if "history" in dm else dm
hb = db["history"] if "history" in db else db
tag = dm.get("tag", os.path.splitext(os.path.basename(merge_path))[0])

BENIGN = [f"c{i}" for i in range(1, 8)]
ATT = "#c0392b"          # attacker red, same as plot_run3way
BEN = "#2d6fb0"          # benign blue, same as plot_run3way non-neighbors


def series(h, fn):
    return [x["round"] for x in h], [fn(x) for x in h]


def panel(ax, fn, colour, title, ylabel, ylim):
    rb, yb = series(hb, fn)
    rm, ym = series(hm, fn)
    ax.plot(rb, yb, color=colour, lw=2.4, ls="--", marker="o", ms=4, alpha=.75,
            label="baseline (attacker ignores gossip)")
    ax.plot(rm, ym, color=colour, lw=2.9, ls="-", marker="s", ms=5,
            label="merge (attacker_merges=True)")
    ax.set_title(title); ax.set_ylabel(ylabel); ax.set_ylim(*ylim)
    ax.set_xlabel("communication round")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(frameon=False)


fig, (axA, axB, axC, axD) = plt.subplots(1, 4, figsize=(21, 5.2))
panel(axA, lambda x: x["P"]["c0"], ATT, "c0 detP: is the attacker detectable?", "detP (%)", (-3, 92))
axA.axhline(25, color="k", ls=":", lw=1.4, label="δ = 25%")
axA.legend(frameon=False)
panel(axB, lambda x: x["asr"]["c0"], ATT, "c0 ASR: attacker's own backdoor", "ASR (%)", (-3, 103))
panel(axC, lambda x: float(np.mean([x["asr"][c] for c in BENIGN])), BEN,
      "benign mean ASR: propagation", "ASR (%)", (-3, 103))
# The direct read on the hypothesis: does merging stop the attacker over-fitting? Same y-limits
# as plot_run3way's CE panel so this sits beside the 3way figures unchanged.
panel(axD, lambda x: x["loss"]["c0"], ATT, "c0 held-out CE: does it over-fit?",
      "cross-entropy", (1.15, 1.45))
# c0's own round-1 level -- the reference for "still converging" in the joint ASR/loss readout.
axD.axhline(hb[0]["loss"]["c0"], color="k", ls=":", lw=1.2, label="baseline c0 round-1 CE")
axD.legend(frameon=False)

fig.suptitle(f"{tag}   —   baseline (dashed) vs merging attacker (solid)",
             y=1.03, fontweight="bold")
fig.tight_layout()
os.makedirs(outdir, exist_ok=True)
out = os.path.join(outdir, f"{tag}_overlay.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
