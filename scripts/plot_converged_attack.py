"""
Converged-regime attack run: the three success criteria, each split into the SAME three groups
(attacker c0 / neighbors {1,4,7} / non-neighbors {2,3,5,6}) for consistency.
A) ASR: install + topological propagation.  B) held-out CE: baseline health per group.
C) detP: rise/decline per group, delta=25 marked.
CPU-only; reads results/noniid/converged/converged_attack_inf.json.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/home/speechara/epfl/iclscan-decentralized/results/noniid/converged"
h = json.load(open(f"{R}/converged_attack_inf.json"))["history"]
rr = [d["round"] for d in h]
# (label, node-ids, color, marker) -- one consistent scheme for every panel
GROUPS = [("attacker c0", [0], "#c0392b", "o"),
          ("neighbors {1,4,7}", [1, 4, 7], "#e08a1e", "s"),
          ("non-neighbors {2,3,5,6}", [2, 3, 5, 6], "#2d6fb0", "^")]


def plot_groups(ax, key):
    for label, ids, col, mk in GROUPS:
        y = [np.mean([d[key][f"c{i}"] for i in ids]) for d in h]
        ax.plot(rr, y, color=col, lw=2.3, marker=mk, ms=4, label=label)


fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(15, 4.6))

plot_groups(axA, "asr")
axA.set_title("Attack: install + topological propagation"); axA.set_ylabel("ASR (%)")

plot_groups(axB, "loss")
axB.plot(rr, [np.mean(list(d["trainloss"].values())) for d in h], color="0.6", lw=1.4, ls=":",
         label="mean train CE")
axB.set_title("Baseline health: held-out CE per group"); axB.set_ylabel("cross-entropy")

plot_groups(axC, "P")
axC.axhline(25, color="k", ls="--", lw=1.2, label="δ = 25%")
axC.set_title("Detection signal: detP per group"); axC.set_ylabel("detP (%)")

for ax in (axA, axB, axC):
    ax.set_xlabel("communication round")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(fontsize=8.5, frameon=False)
fig.suptitle("Converged regime (4000/node, bs=8, K=25, R=15, α=∞, ppoison=0.15)", y=1.02)
fig.tight_layout()
fig.savefig(f"{R}/converged_attack.png", dpi=130, bbox_inches="tight")
print("saved converged_attack.png")
