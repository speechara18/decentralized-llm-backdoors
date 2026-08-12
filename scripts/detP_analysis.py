"""
§1b corrected: (1) label by ASR not identity -> restrict clean-detection metrics to r1-6 where benign
ASR~0; (2) detrend c0 SD (per-alpha linear fit) to separate trend from variance; (3) AUC (threshold-free,
comparable to ICLScan's 1.000) per window r1-6 vs r9-15; (4) scale check vs paper for delta=25 transfer.
CPU-only; reads sweep3/*_R15_fixed500_att.json.
"""
import json
import numpy as np
np.seterr(all="ignore")


def rankdata(a):                     # average ranks, tie-aware (scipy-free)
    a = np.asarray(a, float); order = a.argsort(); ranks = np.empty(len(a))
    sa = a[order]; i = 0
    while i < len(a):
        j = i
        while j < len(a) and sa[j] == sa[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2 + 1     # 1-indexed average rank
        i = j
    return ranks

R = "/home/speechara/epfl/iclscan-decentralized/results/noniid/sweep3"
ALPHAS = ["inf", "0.5", "0.1"]
BEN = [f"c{i}" for i in range(1, 8)]


def load(a):
    return json.load(open(f"{R}/alpha{a}_R15_fixed500_att.json"))["history"]


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    allv = np.concatenate([pos, neg]); r = rankdata(allv)
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


# (0) confirm benign ASR ~0 in r1-6
print("=== benign ASR by window (label sanity) ===")
for a in ALPHAS:
    h = load(a)
    e = np.mean([h[r]["asr"][c] for r in range(0, 6) for c in BEN])
    l = np.mean([h[r]["asr"][c] for r in range(8, 15) for c in BEN])
    print(f"  alpha={a:>3}: benign ASR  r1-6={e:5.1f}  r9-15={l:5.1f}")

# (1,3) per-window, per-alpha: benign mean/SD, c0 mean, AUC (label = ASR>50), benign FPR
print("\n=== detection by window (label = ASR>50%) ===")
for lo, hi, name in [(0, 6, "r1-6 (clean)"), (8, 15, "r9-15 (propagated)")]:
    print(f"--- {name} ---")
    for a in ALPHAS:
        h = load(a)
        pos, neg = [], []
        for r in range(lo, hi):
            for c in ["c0"] + BEN:
                (pos if h[r]["asr"][c] > 50 else neg).append(h[r]["P"][c])
        c0v = [h[r]["P"]["c0"] for r in range(lo, hi)]
        benv = [h[r]["P"][c] for r in range(lo, hi) for c in BEN]
        fpr = np.mean([x > 25 for x in benv]) if benv else float("nan")
        print(f"  alpha={a:>3}: AUC={auc(pos,neg):.3f}  c0 mean={np.mean(c0v):4.1f}  "
              f"benign mean={np.mean(benv):4.1f} SD={np.std(benv):4.1f}  benign>δ={100*fpr:4.1f}%  "
              f"(n_pos={len(pos)},n_neg={len(neg)})")

# (2) detrended c0 SD: subtract per-alpha linear fit of detP vs round
print("\n=== c0 detP variance: raw vs detrended (per-alpha linear detrend) ===")
resid = []
for a in ALPHAS:
    h = load(a); y = np.array([h[r]["P"]["c0"] for r in range(15)]); x = np.arange(15)
    b1, b0 = np.polyfit(x, y, 1)
    r = y - (b0 + b1 * x); resid += list(r)
    print(f"  alpha={a:>3}: raw SD={y.std():4.1f}  slope={b1:+.2f}/round  detrended SD={r.std():4.1f}")
print(f"  POOLED detrended residual SD = {np.std(resid):.1f}pp  (this is what smoothing must beat / "
      f"multi-seed replay compares to)")

# (4) scale check vs paper
print("\n=== scale vs paper (delta=25 transfer) ===")
allc0_16 = [load(a)[r]["P"]["c0"] for a in ALPHAS for r in range(6)]
allben_16 = [load(a)[r]["P"][c] for a in ALPHAS for r in range(6) for c in BEN]
print(f"  OURS r1-6: benign mean={np.mean(allben_16):.1f}%  attacker(c0) mean={np.mean(allc0_16):.1f}%")
print(f"  PAPER    : benign ~3-7%           attacker ~60-70%   (AUC=1.000)")
print("  => our scale is shifted UP (benign) and compressed DOWN (attacker) -> delta=25 likely mis-set.")
