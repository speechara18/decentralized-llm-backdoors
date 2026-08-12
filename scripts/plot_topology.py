"""Draw the 8-node gossip topology (3-regular circulant C8(1,4): neighbors i-1, i+1, i+4).
Nodes colored by role relative to attacker c0: attacker / neighbors {1,4,7} / non-neighbors
{2,3,5,6}. Attacker's edges highlighted (the channels the backdoor first propagates through).
Pure matplotlib, no networkx. -> results/noniid/topology.png"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"axes.titlesize": 18, "legend.fontsize": 13, "figure.titlesize": 18})

N = 8
adj = {i: sorted({(i - 1) % N, (i + 1) % N, (i + 4) % N}) for i in range(N)}
# node positions on a circle, c0 at top, going clockwise
ang = np.array([np.pi / 2 - 2 * np.pi * i / N for i in range(N)])
pos = {i: (np.cos(ang[i]), np.sin(ang[i])) for i in range(N)}

ATT, NB, NN = [0], [1, 4, 7], [2, 3, 5, 6]
role_col = {**{i: "#c0392b" for i in ATT}, **{i: "#e08a1e" for i in NB}, **{i: "#2d6fb0" for i in NN}}

edges = {frozenset((i, j)) for i in range(N) for j in adj[i]}
fig, ax = plt.subplots(figsize=(6.6, 6.6))
for e in edges:
    i, j = tuple(e)
    att_edge = 0 in e
    ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
            color="#e08a1e" if att_edge else "0.75",
            lw=2.6 if att_edge else 1.3, zorder=1, alpha=0.9 if att_edge else 0.7)
for i in range(N):
    ax.scatter(*pos[i], s=1500, color=role_col[i], edgecolors="white", lw=2, zorder=3)
    ax.text(*pos[i], f"c{i}", ha="center", va="center", color="white",
            fontsize=16, fontweight="bold", zorder=4)

# legend
handles = [plt.Line2D([], [], marker="o", ls="", ms=13, mfc="#c0392b", mec="white", label="attacker c0"),
           plt.Line2D([], [], marker="o", ls="", ms=13, mfc="#e08a1e", mec="white", label="neighbors {1,4,7}"),
           plt.Line2D([], [], marker="o", ls="", ms=13, mfc="#2d6fb0", mec="white", label="non-neighbors {2,3,5,6}"),
           plt.Line2D([], [], color="#e08a1e", lw=2.6, label="attacker's gossip links")]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
          ncol=2, fontsize=13, frameon=False)
ax.set_title("Gossip topology: 3-regular circulant C8(1,4)\n"
             "each node averages with i−1, i+1, i+4 (mod 8) each round", fontsize=16)
ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.5, 1.35)
ax.set_aspect("equal"); ax.axis("off")
fig.tight_layout()
fig.savefig("/home/speechara/epfl/iclscan-decentralized/results/noniid/topology.png",
            dpi=140, bbox_inches="tight")
print("saved topology.png")
print("adjacency:", {f"c{i}": [f"c{j}" for j in adj[i]] for i in range(N)})
