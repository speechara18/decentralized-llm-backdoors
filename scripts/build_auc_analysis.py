"""
Detection analysis on the 6-condition seed-0 sweep.
(1) AUC vs skew (matched-control: attacker ASR>50 node-rounds = positives, no-attacker node-rounds
    = clean negatives), for the clean early window (r1-6) and all rounds.
(2) FALSE-NEGATIVE mechanism: split the POSITIVES (ASR>50) into TRAINED (c0, learned poison directly)
    vs PROPAGATED (benign nodes that caught the backdoor via gossip). Compare their detP to the clean
    negatives + delta=25. Question: does ICLScan detect the PROPAGATED backdoors, or false-negative them?
Two plots + printed numbers. CPU-only.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"axes.titlesize": 17, "axes.labelsize": 15, "xtick.labelsize": 13,
                     "ytick.labelsize": 13, "legend.fontsize": 13, "figure.titlesize": 18})

R = "/home/speechara/epfl/iclscan-decentralized/results/noniid/r20"
NODES = [f"c{i}" for i in range(8)]
ALPHAS = ["inf", "0.5", "0.1"]
DELTA = 25


def rankdata(a):
    a = np.asarray(a, float); o = a.argsort(); rk = np.empty(len(a)); sa = a[o]; i = 0
    while i < len(a):
        j = i
        while j < len(a) and sa[j] == sa[i]:
            j += 1
        rk[o[i:j]] = (i + j - 1) / 2 + 1; i = j
    return rk


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def load(a, s):
    return json.load(open(f"{R}/r25_alpha{a}_{s}_seed0.json"))["history"]


# ---- collect ----
auc_early, auc_all = {}, {}
detP_trained, detP_prop, detP_clean = {}, {}, {}          # per alpha
for a in ALPHAS:
    att, noatt = load(a, "att"), load(a, "noatt")
    neg_all = [noatt[r]["P"][c] for r in range(25) for c in NODES]
    pos_all = [att[r]["P"][c] for r in range(25) for c in NODES if att[r]["asr"][c] > 50]
    pos_e = [att[r]["P"][c] for r in range(6) for c in NODES if att[r]["asr"][c] > 50]
    neg_e = [noatt[r]["P"][c] for r in range(6) for c in NODES]
    auc_early[a], auc_all[a] = auc(pos_e, neg_e), auc(pos_all, neg_all)
    detP_trained[a] = [att[r]["P"]["c0"] for r in range(25) if att[r]["asr"]["c0"] > 50]
    detP_prop[a] = [att[r]["P"][c] for r in range(25) for c in NODES[1:] if att[r]["asr"][c] > 50]
    detP_clean[a] = neg_all

print(f"{'alpha':>5} | AUC r1-6 | AUC all | trained-c0 detP | PROPAGATED detP | clean detP | "
      f"propagated detected(>{DELTA})")
for a in ALPHAS:
    pr = detP_prop[a]
    det = 100 * np.mean([p > DELTA for p in pr]) if pr else float("nan")
    print(f"{a:>5} | {auc_early[a]:8.3f} | {auc_all[a]:7.3f} | {np.mean(detP_trained[a]):14.1f} | "
          f"{np.mean(pr):14.1f} | {np.mean(detP_clean[a]):10.1f} | {det:5.1f}%  (n={len(pr)})")

# ---- Plot 1: AUC vs skew ----
fig, ax = plt.subplots(figsize=(7.6, 5.4))
x = np.arange(len(ALPHAS)); w = 0.36
ax.bar(x - w / 2, [auc_early[a] for a in ALPHAS], w, color="#2ca02c", label="early window (r1–6)")
ax.bar(x + w / 2, [auc_all[a] for a in ALPHAS], w, color="#1f77b4", label="all rounds")
ax.axhline(1.0, color="0.5", ls=":", lw=1)
for i, a in enumerate(ALPHAS):
    ax.text(i - w / 2, auc_early[a] + .01, f"{auc_early[a]:.3f}", ha="center", fontsize=11)
    ax.text(i + w / 2, auc_all[a] + .01, f"{auc_all[a]:.3f}", ha="center", fontsize=11)
ax.set_xticks(x); ax.set_xticklabels([f"α = {a}" for a in ALPHAS])
ax.set_ylabel("detection AUC"); ax.set_ylim(0.5, 1.06)
ax.set_title("ICLScan detection AUC vs non-IID skew")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(f"{R}/../auc_vs_skew.png", dpi=140, bbox_inches="tight")
print("saved auc_vs_skew.png")

# ---- Plot 2: false-negative mechanism (detP: clean vs propagated vs trained) ----
fig, ax = plt.subplots(figsize=(8.6, 5.4))
cats = [("clean (no attacker)", detP_clean, "#7f7f7f"),
        ("propagated backdoor\n(benign, via gossip)", detP_prop, "#ff7f0e"),
        ("trained backdoor (c0)", detP_trained, "#d62728")]
for gi, a in enumerate(ALPHAS):
    for ci, (lab, dct, col) in enumerate(cats):
        vals = np.array(dct[a]); xpos = gi + (ci - 1) * 0.24
        jit = xpos + (np.random.RandomState(0).rand(len(vals)) - .5) * 0.12
        ax.scatter(jit, vals, s=16, color=col, alpha=0.5, edgecolors="none",
                   label=lab if gi == 0 else None)
        ax.plot([xpos - .09, xpos + .09], [vals.mean()] * 2, color=col, lw=3)
ax.axhline(DELTA, color="k", ls="--", lw=1.4, label=f"δ = {DELTA}%")
ax.set_xticks(range(len(ALPHAS))); ax.set_xticklabels([f"α = {a}" for a in ALPHAS])
ax.set_ylabel("detP (%)")
ax.set_title("Why the false negatives: propagated backdoors don't trigger BSA")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(frameon=False, fontsize=11, loc="upper right")
fig.tight_layout(); fig.savefig(f"{R}/../fn_mechanism.png", dpi=140, bbox_inches="tight")
print("saved fn_mechanism.png")
