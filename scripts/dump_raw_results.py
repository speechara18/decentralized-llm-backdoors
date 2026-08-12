"""Dump RAW measured numbers from the R=25 sweep + derived metrics. No interpretation.
Prints tables for pasting into the results-summary doc."""
import json
import numpy as np

RR = "/home/speechara/epfl/iclscan-decentralized/results/noniid/r20"
NODES = [f"c{i}" for i in range(8)]
NB, NN, BEN = [1, 4, 7], [2, 3, 5, 6], list(range(1, 8))
ALPHAS = ["inf", "0.5", "0.1"]


def load(a, s):
    return json.load(open(f"{RR}/r25_alpha{a}_{s}_seed0.json"))["history"]


def mean(h, key, ids, rng):
    return np.mean([h[r][key][f"c{i}"] for r in rng for i in ids])


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


print("### A. ATTACKER RUNS — held-out CE and train loss (group means)")
print(f"{'alpha':>5} | {'benign held r1':>14} {'benign held r25':>15} | {'c0 held r1':>10} "
      f"{'c0 held r25':>11} | {'train r1':>8} {'train r25':>9}")
for a in ALPHAS:
    h = load(a, "att")
    print(f"{a:>5} | {mean(h,'loss',BEN,[0]):14.3f} {mean(h,'loss',BEN,[24]):15.3f} | "
          f"{h[0]['loss']['c0']:10.3f} {h[24]['loss']['c0']:11.3f} | "
          f"{np.mean(list(h[0]['trainloss'].values())):8.3f} {np.mean(list(h[24]['trainloss'].values())):9.3f}")

print("\n### B. ATTACKER RUNS — ASR (%)")
print(f"{'alpha':>5} | {'c0 r1':>6} {'c0 first=100 @r':>15} | {'nbr mean r25':>12} {'non-nbr mean r25':>16}")
for a in ALPHAS:
    h = load(a, "att")
    c0 = [h[r]['asr']['c0'] for r in range(25)]
    first100 = next((r + 1 for r in range(25) if c0[r] >= 100), None)
    print(f"{a:>5} | {c0[0]:6.0f} {str(first100):>15} | {mean(h,'asr',NB,[24]):12.1f} {mean(h,'asr',NN,[24]):16.1f}")

print("\n### C. ATTACKER RUNS — detP (%) group means, by window")
print(f"{'alpha':>5} | {'c0 r1-6':>7} {'c0 r20-25':>9} | {'nbr r1-6':>8} {'nbr r20-25':>10} | "
      f"{'non-nbr r1-6':>12} {'non-nbr r20-25':>14}")
for a in ALPHAS:
    h = load(a, "att")
    print(f"{a:>5} | {mean(h,'P',[0],range(6)):7.1f} {mean(h,'P',[0],range(19,25)):9.1f} | "
          f"{mean(h,'P',NB,range(6)):8.1f} {mean(h,'P',NB,range(19,25)):10.1f} | "
          f"{mean(h,'P',NN,range(6)):12.1f} {mean(h,'P',NN,range(19,25)):14.1f}")

print("\n### D. NO-ATTACKER RUNS — held-out CE (all-node mean) and detP (%)")
print(f"{'alpha':>5} | {'held r1':>7} {'held r25':>8} | {'detP mean(all rounds)':>21} {'detP max':>8}")
for a in ALPHAS:
    h = load(a, "noatt")
    dp = [h[r]['P'][c] for r in range(25) for c in NODES]
    print(f"{a:>5} | {mean(h,'loss',range(8),[0]):7.3f} {mean(h,'loss',range(8),[24]):8.3f} | "
          f"{np.mean(dp):21.1f} {np.max(dp):8.1f}")

print("\n### E. DERIVED — matched-control AUC & TPR@delta25 (pos=attacker ASR>50, neg=no-attacker)")
print(f"{'alpha':>5} | {'window':>10} | {'n_pos':>5} {'n_neg':>5} | {'AUC':>5} | {'TPR@25':>6} {'FPR@25':>6}")
for a in ALPHAS:
    att, no = load(a, "att"), load(a, "noatt")
    for wn, rng in [("r1-6", range(6)), ("r20-25", range(19, 25))]:
        pos = [att[r]['P'][c] for r in rng for c in NODES if att[r]['asr'][c] > 50]
        neg = [no[r]['P'][c] for r in rng for c in NODES]
        tpr = 100 * np.mean([p > 25 for p in pos]) if pos else float('nan')
        fpr = 100 * np.mean([n > 25 for n in neg])
        print(f"{a:>5} | {wn:>10} | {len(pos):5d} {len(neg):5d} | {auc(pos,neg):5.3f} | {tpr:5.1f}% {fpr:5.1f}%")

print("\n### F. DERIVED — detP by category (all rounds, ASR>50 for backdoored)")
print(f"{'alpha':>5} | {'trained-c0':>10} {'propagated':>10} {'clean':>6} | {'propagated detected>25':>22}")
for a in ALPHAS:
    att, no = load(a, "att"), load(a, "noatt")
    tr = [att[r]['P']['c0'] for r in range(25) if att[r]['asr']['c0'] > 50]
    pr = [att[r]['P'][c] for r in range(25) for c in NODES[1:] if att[r]['asr'][c] > 50]
    cl = [no[r]['P'][c] for r in range(25) for c in NODES]
    det = 100 * np.mean([p > 25 for p in pr])
    print(f"{a:>5} | {np.mean(tr):10.1f} {np.mean(pr):10.1f} {np.mean(cl):6.1f} | {det:5.1f}%  (n={len(pr)})")
