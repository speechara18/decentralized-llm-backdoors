"""
Control for the spectral baseline: does sigma1_mean detect POISON, or merely NON-MERGING?

THE CONFOUND. In the attacker run attacker_ids=(0,), so c0 is in `nomerge` and never merges.
In the no-attacker run attacker_ids=(), so c0 merges normally. A node that never merges drifts
from consensus, its dW is larger, and its leading singular value is bigger -- for reasons with
nothing to do with backdoors. sigma1_mean winning every cell of the cross-run comparison is
exactly the signature of a statistic that has discovered "this node does not merge".

THE CONTROL. r15_alphainf_freerider_seed0: c0 ignores gossip EXACTLY like the attacker but
carries NO poison. This is the control that already closed the same confound for detP
(free-rider held-out loss slope -0.00414, indistinguishable from benign -0.00480, versus the
attacker's -0.00091).

INTERPRETATION, FIXED BEFORE LOOKING:
  free-rider separates too      -> the statistic detects non-merging, not poison. AUC 1.000 is an
                                   artifact; weight-space has NOT solved the propagated case; a
                                   published detector fires on a benign non-participating node.
  free-rider does NOT separate  -> the signal is genuinely poison-driven; weight-space is free,
                                   works at 7B, and is a serious baseline our defense must be
                                   justified against.

Also runs a WITHIN-RUN version of Q2 (no cross-run comparison at all): inside the attacker run,
do the attacker's neighbours {c1,c4,c7} separate from non-neighbours {c2,c3,c5,c6}? That is the
deployable question -- "which node in THIS network is compromised".

module_stats / adapter_signature are copied VERBATIM from spectral_baseline.py so the numbers are
directly comparable (that module runs argparse at import time and cannot be imported).

Usage: python spectral_freerider_control.py
"""
import glob, json, os, itertools
import numpy as np
from safetensors.numpy import load_file

NFS = "/mnt/nfs/home/peechara"
CK = f"{NFS}/ckpts"
SCALE = 16.0 / 8.0                      # PEFT applies dW = (alpha/r) * B @ A
OUT = f"{NFS}/iclscan-decentralized/results/noniid/spectral_baseline/freerider_control.json"
BEN = [f"c{i}" for i in range(1, 8)]
NBR, NON = ["c1", "c4", "c7"], ["c2", "c3", "c5", "c6"]


def adapter_path(run, r, c):
    hits = glob.glob(f"{CK}/{run}/r{r}/{c}/**/adapter_model.safetensors", recursive=True)
    return os.path.dirname(hits[0]) if hits else None


def module_stats(A, B):
    Rb = np.linalg.qr(B, mode="r")
    Ra = np.linalg.qr(A.T, mode="r")
    sv = np.linalg.svd(Rb @ Ra.T, compute_uv=False) * SCALE
    sv = np.maximum(sv, 0.0)
    tot = sv.sum()
    p = sv / tot if tot > 0 else np.full_like(sv, 1.0 / len(sv))
    out_dim, in_dim = B.shape[0], A.shape[1]
    return {"sigma1": float(sv[0]),
            "fro": float(np.sqrt((sv ** 2).sum())),
            "energy": float(sv[0] / tot) if tot > 0 else 0.0,
            "entropy": float(-(p * np.log(p + 1e-12)).sum()),
            "frobN": float(np.sqrt((sv ** 2).sum()) / np.sqrt(in_dim * out_dim))}


def adapter_signature(path):
    try:
        t = load_file(os.path.join(path, "adapter_model.safetensors"))
    except Exception as e:
        print(f"    unreadable {path}: {e}", flush=True)
        return None
    mods = {}
    for k in t:
        if "lora_A" in k:
            mods.setdefault(k.split(".lora_A")[0], {})["A"] = np.asarray(t[k], dtype=np.float32)
        elif "lora_B" in k:
            mods.setdefault(k.split(".lora_B")[0], {})["B"] = np.asarray(t[k], dtype=np.float32)
    per = [module_stats(ab["A"], ab["B"]) for _, ab in sorted(mods.items())
           if "A" in ab and "B" in ab]
    if not per:
        return None
    sig = {}
    for k in per[0]:
        v = np.array([m[k] for m in per])
        sig[f"{k}_mean"] = float(v.mean())
        sig[f"{k}_std"] = float(v.std())
        sig[f"{k}_max"] = float(v.max())
    return sig


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    w = sum((1.0 if a > b else 0.5 if a == b else 0.0) for a in pos for b in neg)
    return w / (len(pos) * len(neg))


FEATS = ["sigma1_mean", "fro_mean", "energy_mean", "entropy_mean", "frobN_mean", "frobN_std"]
ROUNDS = [1, 5, 13]
RUNS = [("FREE-RIDER (no poison, does not merge)", "r15_alphainf_freerider_seed0"),
        ("ATTACKER   (poison,   does not merge)", "r25_alphainf_att_seed0")]

sigs, unreadable = {}, []
for label, run in RUNS:
    for r in ROUNDS:
        for c in [f"c{i}" for i in range(8)]:
            p = adapter_path(run, r, c)
            if p is None:
                unreadable.append(f"{run}/r{r}/{c} (missing)"); continue
            s = adapter_signature(p)
            if s is None:
                unreadable.append(f"{run}/r{r}/{c} (unreadable)"); continue
            sigs[(run, r, c)] = s
    print(f"  loaded {run}", flush=True)

print(f"\nunreadable/missing adapters: {len(unreadable)} {unreadable if unreadable else ''}\n")

print("=" * 96)
print("DECISIVE COMPARISON: does c0 separate from its OWN benign peers, in each run?")
print("  z = (c0 - mean(benign)) / sd(benign).  rank = c0's position among all 8 nodes (1 = largest)")
print("=" * 96)
res = {}
for label, run in RUNS:
    print(f"\n--- {label}   [{run}] ---")
    print(f"  {'round':>5} {'feature':>14} {'c0':>10} {'benign mean':>12} {'benign sd':>10} {'z':>8} {'rank':>5} {'AUC':>6}")
    for r in ROUNDS:
        if (run, r, "c0") not in sigs:
            continue
        for f in FEATS:
            c0 = sigs[(run, r, "c0")][f]
            bv = [sigs[(run, r, c)][f] for c in BEN if (run, r, c) in sigs]
            if not bv:
                continue
            mu, sd = float(np.mean(bv)), float(np.std(bv))
            z = (c0 - mu) / sd if sd > 0 else float("nan")
            rank = 1 + sum(1 for v in bv if v > c0)
            res[(run, r, f)] = z
            print(f"  {r:>5} {f:>14} {c0:>10.4f} {mu:>12.4f} {sd:>10.4f} {z:>+8.2f} {rank:>5} {auc([c0], bv):>6.3f}")

print("\n" + "=" * 96)
print("VERDICT INPUT: |z| of c0 vs its own benign peers, free-rider vs attacker")
print("=" * 96)
print(f"  {'round':>5} {'feature':>14} {'FREE-RIDER z':>14} {'ATTACKER z':>12}   reading")
fr, at = "r15_alphainf_freerider_seed0", "r25_alphainf_att_seed0"
for r in ROUNDS:
    for f in FEATS:
        if (fr, r, f) in res and (at, r, f) in res:
            zf, za = res[(fr, r, f)], res[(at, r, f)]
            tag = ("free-rider separates AS MUCH or more -> NON-MERGING artifact"
                   if abs(zf) >= abs(za) * 0.7 else
                   "attacker separates distinctly more -> poison-driven")
            print(f"  {r:>5} {f:>14} {zf:>+14.2f} {za:>+12.2f}   {tag}")

print("\n" + "=" * 96)
print("WITHIN-RUN Q2 (no cross-run comparison): in the ATTACKER run alone, do the attacker's")
print("neighbours {c1,c4,c7} separate from non-neighbours {c2,c3,c5,c6}?")
print("  This is the deployable question: WHICH node in THIS network is compromised.")
print("=" * 96)
print(f"  {'round':>5} {'feature':>14} {'nbr mean':>10} {'non mean':>10} {'AUC':>7}  (n=3 vs 4)")
for r in ROUNDS:
    for f in FEATS:
        nv = [sigs[(at, r, c)][f] for c in NBR if (at, r, c) in sigs]
        ov = [sigs[(at, r, c)][f] for c in NON if (at, r, c) in sigs]
        if not nv or not ov:
            continue
        print(f"  {r:>5} {f:>14} {np.mean(nv):>10.4f} {np.mean(ov):>10.4f} {auc(nv, ov):>7.3f}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump({f"{k[0]}|r{k[1]}|{k[2]}": v for k, v in sigs.items()}, open(OUT, "w"), indent=2)
print(f"\nwrote {OUT}")
print("FREERIDER CONTROL DONE", flush=True)
