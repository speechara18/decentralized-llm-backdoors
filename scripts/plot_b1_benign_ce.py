"""
B1: recompute the held-out CE "green curve" as the BENIGN-node mean (c1..c7), excluding the
attacker c0 whose refusal-poison training inflates its benign held-out CE for reasons unrelated
to overfitting. Left panel: the correction mechanism at alpha=inf (all-8 vs benign-7 vs c0-only).
Right panel: corrected benign-7 curve for all three alpha (does the rise + skew-invariance survive?).
CPU-only; reads results/noniid/sweep3/*_R15_fixed500_att.json.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/home/speechara/epfl/iclscan-decentralized/results/noniid/sweep3"
ALPHAS = [("inf", "#2d6fb0"), ("0.5", "#e08a1e"), ("0.1", "#c0392b")]
BENIGN = [f"c{i}" for i in range(1, 8)]


def curves(alab):
    h = json.load(open(f"{R}/alpha{alab}_R15_fixed500_att.json"))["history"]
    rounds = [d["round"] for d in h]
    all8 = [np.mean([d["loss"][f"c{i}"] for i in range(8)]) for d in h]
    ben7 = [np.mean([d["loss"][c] for c in BENIGN]) for d in h]
    c0 = [d["loss"]["c0"] for d in h]
    return np.array(rounds), np.array(all8), np.array(ben7), np.array(c0)


fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.0))

# Left: the correction mechanism at alpha=inf
rr, all8, ben7, c0 = curves("inf")
axL.plot(rr, all8, color="0.55", lw=2.2, marker="o", ms=4, ls="--", label="all 8 nodes (contaminated)")
axL.plot(rr, ben7, color="#2d6fb0", lw=2.4, marker="o", ms=4, label="benign 7 (c1–c7, corrected)")
axL.plot(rr, c0, color="#c0392b", lw=1.6, marker="x", ms=5, label="c0 attacker alone")
axL.set_title("Correction mechanism (α = ∞)")
axL.set_xlabel("communication round"); axL.set_ylabel("held-out CE")
axL.legend(fontsize=8.5, frameon=False)

# Right: corrected benign-7 curve across alpha
for alab, col in ALPHAS:
    rr, all8, ben7, c0 = curves(alab)
    axR.plot(rr, ben7, color=col, lw=2.4, marker="o", ms=4, label=f"α={alab}  (benign-7)")
axR.set_title("Corrected green curve: benign-node mean CE")
axR.set_xlabel("communication round"); axR.set_ylabel("held-out CE")
axR.legend(fontsize=9, frameon=False)

for ax in (axL, axR):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)
fig.tight_layout()
fig.savefig(f"{R}/../b1_benign_ce.png", dpi=130)

# print the numbers
print("alpha |  r1 all8 / ben7 |  r15 all8 / ben7 |  c0 r1->r15")
for alab, _ in ALPHAS:
    rr, all8, ben7, c0 = curves(alab)
    print(f"{alab:>5} |  {all8[0]:.3f} / {ben7[0]:.3f}   |  {all8[-1]:.3f} / {ben7[-1]:.3f}   |  {c0[0]:.3f} -> {c0[-1]:.3f}")
print("saved b1_benign_ce.png")
