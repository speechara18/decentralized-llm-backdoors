"""
Part 3 -- discrimination read. Is skew destroying the detector's DISCRIMINATION (ROC collapses
to the diagonal, AUC -> 0.5) or just its OPERATING POINT (ROC holds, only the delta=25 point
slides)? Threshold-independent.

positives : with-attacker node-rounds with true asr >= 50   -> their detP   (per alpha)
negatives : no-attacker node-rounds (clean by construction) -> their detP   (per alpha)
window    : rounds 8-15 (backdoors installed; positive set non-degenerate)
Outputs: roc.png (three ROC curves + AUC) and detp_hist.png (positives vs negatives per alpha).
Runs GPU-free; needs alpha{inf,0.5,0.1}_R15_fixed500_{att,noatt}.json in results/noniid/sweep3/.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/home/speechara/epfl/iclscan-decentralized/results/noniid/sweep3"
ALPHAS = [("inf", "α = ∞ (IID)"), ("0.5", "α = 0.5"), ("0.1", "α = 0.1")]
WIN = range(8, 16)          # rounds 8..15
COL = {"inf": "#2d6fb0", "0.5": "#e08a1e", "0.1": "#c0392b"}


def detp_values(tag, positives):
    """Return list of detP values for the window. positives=True: with-attacker node-rounds
    with asr>=50. positives=False: all no-attacker node-rounds (clean)."""
    path = f"{R}/alpha{tag}_R15_fixed500_{'att' if positives else 'noatt'}.json"
    if not os.path.exists(path):
        return None
    h = json.load(open(path))["history"]
    vals = []
    for r in h:
        if r["round"] not in WIN:
            continue
        for i in range(8):
            c = f"c{i}"
            if positives:
                if r["asr"][c] >= 50:
                    vals.append(r["P"][c])
            else:
                vals.append(r["P"][c])
    return vals


def roc(pos, neg):
    deltas = np.arange(0, 101, 1.0)
    tpr = np.array([np.mean(np.array(pos) >= d) for d in deltas])
    fpr = np.array([np.mean(np.array(neg) >= d) for d in deltas])
    o = np.argsort(fpr)                # integrate TPR over increasing FPR (manual trapezoid)
    f, t = fpr[o], tpr[o]
    auc = float(np.sum(np.diff(f) * (t[1:] + t[:-1]) / 2))
    return fpr, tpr, auc


missing = [t for t, _ in ALPHAS if detp_values(t, False) is None]
if missing:
    raise SystemExit(f"no-attacker JSON(s) not found for alpha={missing} -- runs still going?")

# --- ROC figure ---
fig, ax = plt.subplots(figsize=(6.4, 6.0))
print(f"{'alpha':>10} | {'#pos':>5} {'#neg':>5} | {'AUC':>5} | TPR@δ25  FPR@δ25")
for tag, label in ALPHAS:
    pos, neg = detp_values(tag, True), detp_values(tag, False)
    fpr, tpr, auc = roc(pos, neg)
    ax.plot(fpr, tpr, color=COL[tag], lw=2.4, label=f"{label}   AUC = {auc:.2f}")
    t25 = np.mean(np.array(pos) >= 25); f25 = np.mean(np.array(neg) >= 25)
    ax.scatter([f25], [t25], color=COL[tag], s=45, zorder=5, edgecolors="white")
    print(f"{label:>12} | {len(pos):>5} {len(neg):>5} | {auc:>5.2f} | {t25*100:6.0f}%  {f25*100:6.0f}%")
ax.plot([0, 1], [0, 1], ls=":", color="0.5", lw=1.2, label="chance (AUC = 0.5)")
ax.set_xlabel("false positive rate  (clean node flagged)")
ax.set_ylabel("true positive rate  (backdoored node flagged)")
ax.set_title("ICLScan discrimination vs heterogeneity  (detP, rounds 8-15)\n"
             "dots = operating point at δ = 25")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(fontsize=9, loc="lower right", frameon=False)
ax.grid(alpha=0.25, lw=0.6)
fig.tight_layout(); fig.savefig(f"{R}/roc.png", dpi=130)
print("saved", f"{R}/roc.png")

# --- histogram figure (positives vs negatives per alpha) ---
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
bins = np.arange(0, 104, 100 / 30)          # detP is quantized in 100/probe_n = 3.3 steps
for ax, (tag, label) in zip(axes, ALPHAS):
    pos, neg = detp_values(tag, True), detp_values(tag, False)
    ax.hist(neg, bins=bins, density=True, color="#2d6fb0", alpha=0.55, label=f"clean (n={len(neg)})")
    ax.hist(pos, bins=bins, density=True, color="#c0392b", alpha=0.55, label=f"backdoored (n={len(pos)})")
    ax.axvline(25, ls=":", color="0.35", lw=1.2)
    ax.set_title(label, fontsize=11); ax.set_xlabel("detP (%)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8.5, frameon=False)
axes[0].set_ylabel("density")
fig.suptitle("detP distributions: backdoored vs clean node-rounds (r8-15) -- the human-readable ROC",
             fontsize=12)
fig.tight_layout(); fig.savefig(f"{R}/detp_hist.png", dpi=130)
print("saved", f"{R}/detp_hist.png")
