"""
Matched-control detection AUC vs skew (the headline number). For each alpha:
  positives = ATTACKER-run node-rounds with ASR>50 (genuinely backdoored, G1-correct labeling)
  negatives = NO-ATTACKER-run node-rounds (clean by construction, at every round)
  score     = detP;  AUC = P(detP_pos > detP_neg)  (threshold-free, comparable to paper's 1.000)
Reported for the clean EARLY window (r1-6, only c0 backdoored) and ALL rounds, plus the
delta=25 flag rates (TPR/FPR) as secondary. CPU-only.
"""
import json
import numpy as np

R = "/home/speechara/epfl/iclscan-decentralized/results/noniid/r20"
NODES = [f"c{i}" for i in range(8)]
ALPHAS = ["inf", "0.5", "0.1"]


def rankdata(a):
    a = np.asarray(a, float); order = a.argsort(); ranks = np.empty(len(a)); sa = a[order]; i = 0
    while i < len(a):
        j = i
        while j < len(a) and sa[j] == sa[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2 + 1; i = j
    return ranks


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def load(a, scen):
    return json.load(open(f"{R}/r25_alpha{a}_{scen}_seed0.json"))["history"]


print(f"{'alpha':>6} | {'window':>7} | {'AUC':>5} | {'n_pos':>5} {'n_neg':>5} | "
      f"{'TPR@25':>6} {'FPR@25':>6} | attacker-c0 detP (early mean)")
for a in ALPHAS:
    att, noatt = load(a, "att"), load(a, "noatt")
    for name, rng in [("r1-6", range(0, 6)), ("all", range(0, 25))]:
        pos = [att[r]["P"][c] for r in rng for c in NODES if att[r]["asr"][c] > 50]
        neg = [noatt[r]["P"][c] for r in rng for c in NODES]
        tpr = np.mean([p > 25 for p in pos]) if pos else float("nan")
        fpr = np.mean([n > 25 for n in neg])
        c0e = np.mean([att[r]["P"]["c0"] for r in range(0, 6)])
        print(f"{a:>6} | {name:>7} | {auc(pos, neg):5.3f} | {len(pos):5d} {len(neg):5d} | "
              f"{100*tpr:5.1f}% {100*fpr:5.1f}% | {c0e:5.1f}")
    print()
