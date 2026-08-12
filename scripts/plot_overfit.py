"""
Plot the Experiment-B long-epoch benign run: train-slice + held-out loss vs epochs on one axis.
Marks the held-out minimum (the overfitting onset) and shades the productive-training window
[0, onset] -- the benign budget before held-out starts to rise. CPU-only; reads overfit_benign.json.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "/home/speechara/epfl/iclscan-decentralized/results/tune"
d = json.load(open(f"{R}/overfit_benign.json"))
tr = np.array(d["trace"])                       # [epoch, train, held]
ep, train, held = tr[:, 0], tr[:, 1], tr[:, 2]
gi = int(np.argmin(held))
onset = ep[gi]

fig, ax = plt.subplots(figsize=(8.2, 5.2))
ax.axvspan(0, onset, color="#3a9d5d", alpha=0.10, label=f"productive window (≤{onset:.1f} ep)")
ax.plot(ep, train, color="#2d6fb0", lw=2.2, marker="o", ms=3.5, label="train-slice loss")
ax.plot(ep, held, color="#c0392b", lw=2.2, marker="o", ms=3.5, label="held-out loss")
ax.scatter([onset], [held[gi]], color="#c0392b", s=110, zorder=6,
           edgecolors="white", lw=1.2)
ax.annotate(f"held-out min {held[gi]:.3f} @ {onset:.1f} ep\n(overfitting onset)",
            xy=(onset, held[gi]), xytext=(onset + 1.4, held[gi] + 0.35),
            fontsize=9, arrowprops=dict(arrowstyle="->", color="0.4"))
ax.set_xlabel("epochs")
ax.set_ylabel("loss")
ax.set_title(f"Benign convergence and overfitting  (lr={d['lr']:.1e}, warmup={d['warmup']}, "
             f"wd={d['wd']}, n={d['n']})")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(alpha=0.25, lw=0.6)
ax.legend(fontsize=9, frameon=False)
fig.tight_layout()
fig.savefig(f"{R}/overfit_benign.png", dpi=130)
print(f"saved overfit_benign.png  |  held-out min {held[gi]:.3f} @ {onset:.2f} ep; "
      f"final held {held[-1]:.3f} @ {ep[-1]:.2f} ep")
