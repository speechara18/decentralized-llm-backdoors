"""
Defended-vs-undefended overlay, one figure per skew. The figure that answers the defense run.

Baseline (undefended) dashed, defended solid, identical axes. Baseline is TRUNCATED to the
defended run's length (R=24) -- never compare 24 defended rounds against 25 undefended ones.

Panels:
  1. benign mean ASR         -- the headline: does the screen actually stop propagation?
  2. c0 detP with delta=25   -- is the attacker still visible at the source?
  3. benign mean held-out CE -- the guard against "stopped the backdoor by disconnecting the
                                graph". A defense that flattens ASR by severing edges shows up
                                here as benign CE rising above the undefended run.
  4. edges quarantined/round -- split into ATTACKER edges (*<-c0) and HONEST edges. Honest
                                quarantines are the false-positive cost.

Styling matches plot_run3way.py / plot_merge_overlay.py so all three sit together in a deck.

Usage: python plot_defense_overlay.py <defended.json> <baseline.json> [outdir]
"""
import sys, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"axes.titlesize": 18, "axes.labelsize": 16, "xtick.labelsize": 14,
                     "ytick.labelsize": 14, "legend.fontsize": 12, "figure.titlesize": 18})

def_path, base_path = sys.argv[1], sys.argv[2]
outdir = sys.argv[3] if len(sys.argv) > 3 else os.path.dirname(def_path)
dd, db = json.load(open(def_path)), json.load(open(base_path))
hd = dd["history"] if "history" in dd else dd
hb_all = db["history"] if "history" in db else db
tag = dd.get("tag", os.path.splitext(os.path.basename(def_path))[0])

R = max(x["round"] for x in hd)
hb = [x for x in hb_all if x["round"] <= R]          # truncate baseline to matched rounds

BEN = [f"c{i}" for i in range(1, 8)]
ATT = "#c0392b"
BENC = "#2d6fb0"
HON = "#e08a1e"


def panel(ax, fn, colour, title, ylabel, ylim):
    ax.plot([x["round"] for x in hb], [fn(x) for x in hb], color=colour, lw=2.4, ls="--",
            marker="o", ms=4, alpha=.75, label="undefended baseline")
    ax.plot([x["round"] for x in hd], [fn(x) for x in hd], color=colour, lw=2.9, ls="-",
            marker="s", ms=5, label="defended (screen on)")
    ax.set_title(title); ax.set_ylabel(ylabel); ax.set_ylim(*ylim)
    ax.set_xlabel("communication round")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(frameon=False)


fig, (axA, axB, axC, axD) = plt.subplots(1, 4, figsize=(21, 5.2))

panel(axA, lambda x: float(np.mean([x["asr"][c] for c in BEN])), BENC,
      "benign mean ASR: did it stop?", "ASR (%)", (-3, 103))
panel(axB, lambda x: x["P"]["c0"], ATT, "c0 detP at the source", "detP (%)", (-3, 92))
axB.axhline(25, color="k", ls=":", lw=1.4, label="δ = 25%")
axB.legend(frameon=False)
panel(axC, lambda x: float(np.mean([x["loss"][c] for c in BEN])), BENC,
      "benign mean held-out CE", "cross-entropy", (1.15, 1.45))

# Panel 4: quarantines per round, defended run only (the baseline has no screen).
rr = [x["round"] for x in hd]
att_q = [sum(1 for k in x.get("screened", {}) if k.endswith("<-c0")) for x in hd]
hon_q = [sum(1 for k in x.get("screened", {}) if not k.endswith("<-c0")) for x in hd]
axD.plot(rr, att_q, color=ATT, lw=2.9, marker="s", ms=5, label="attacker edges (*<-c0)")
axD.plot(rr, hon_q, color=HON, lw=2.9, marker="^", ms=5, label="honest edges (FALSE POSITIVES)")
axD.axhline(3, color="k", ls=":", lw=1.2, label="all 3 attacker edges")
axD.set_title("edges quarantined per round"); axD.set_ylabel("edges")
axD.set_xlabel("communication round"); axD.set_ylim(-0.3, max(4, max(att_q + hon_q) + 1))
for s in ("top", "right"):
    axD.spines[s].set_visible(False)
axD.grid(alpha=0.25, lw=0.6); axD.legend(frameon=False)

fig.suptitle(f"{tag}   —   undefended (dashed) vs defended (solid), rounds 1-{R}",
             y=1.03, fontweight="bold")
fig.tight_layout()
os.makedirs(outdir, exist_ok=True)
out = os.path.join(outdir, f"{tag}_defoverlay.png")
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
