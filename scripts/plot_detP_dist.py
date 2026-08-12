"""
§1b: free model-level detP spread from existing runs. Histogram c0's detP across all 15 rounds x 3
alpha = 45 measurements (mean/SD/min/max, early r1-8 vs late r9-15), same for benign nodes (c1-c7).
Confounded with round number but bounds the spread and quantifies the Phase-5 decline as a shift.
Adds Wilson 95% CIs (C3) on the c0 points. CPU-only; reads sweep3/*_R15_fixed500_att.json.
"""
import json
from math import sqrt
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/home/speechara/epfl/iclscan-decentralized/results/noniid/sweep3"
ALPHAS = ["inf", "0.5", "0.1"]
BEN = [f"c{i}" for i in range(1, 8)]


def load(a):
    return json.load(open(f"{R}/alpha{a}_R15_fixed500_att.json"))["history"]


def stats(v):
    v = np.array(v, float)
    return f"mean={v.mean():5.1f}  SD={v.std():4.1f}  min={v.min():4.0f}  max={v.max():4.0f}  n={len(v)}"


c0_all, ben_all = [], []
c0_early, c0_late = [], []
for a in ALPHAS:
    h = load(a)
    c0 = [d["P"]["c0"] for d in h]
    ben = [d["P"][c] for d in h for c in BEN]
    c0_all += c0; ben_all += ben
    c0_early += c0[:8]; c0_late += c0[8:]

print("c0  detP (45):", stats(c0_all))
print("   early r1-8 :", stats(c0_early), " | late r9-15:", stats(c0_late))
print("benign detP(315):", stats(ben_all))
delta = 25
print(f"\nc0 measurements BELOW delta={delta}: {sum(x<delta for x in c0_all)}/45  "
      f"(early {sum(x<delta for x in c0_early)}/24, late {sum(x<delta for x in c0_late)}/21)")
print(f"benign measurements ABOVE delta={delta} (false positives): {sum(x>delta for x in ben_all)}/315")

fig, ax = plt.subplots(figsize=(8.4, 5.0))
bins = np.arange(0, 75, 5)
ax.hist(ben_all, bins=bins, color="#2d6fb0", alpha=0.55, label=f"benign nodes (n=315)", density=True)
ax.hist(c0_all, bins=bins, color="#c0392b", alpha=0.55, label=f"attacker c0 (n=45)", density=True)
ax.axvline(delta, color="k", ls="--", lw=1.4, label=f"δ = {delta}%")
ax.axvline(np.mean(c0_early), color="#c0392b", ls=":", lw=1.6, label=f"c0 early mean {np.mean(c0_early):.0f}%")
ax.axvline(np.mean(c0_late), color="#7d1f13", ls=":", lw=1.6, label=f"c0 late mean {np.mean(c0_late):.0f}%")
ax.set_xlabel("detP (%)"); ax.set_ylabel("density")
ax.set_title("detP distribution: attacker c0 vs benign nodes (all rounds, all α)")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(fontsize=8.5, frameon=False)
fig.tight_layout()
fig.savefig(f"{R}/../detP_dist.png", dpi=130)
print("saved detP_dist.png")
