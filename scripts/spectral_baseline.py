"""
WEIGHT-SPACE SPECTRAL BASELINE. CPU only, no GPU, no model load, no generation.

WHY. Our defense spends 21 probes/round. A reviewer will ask why we don't just read the weights.
Two published methods say you can:

  [S] arXiv 2602.15195, "Weight space Detection of Backdoors in LoRA Adapters" -- five spectral
      statistics of the COMPOSED update dW = BA (leading singular value, Frobenius norm, energy
      concentration s1/sum(s), spectral entropy, kurtosis), fed to a SUPERVISED logistic
      regression. 100% accuracy on Llama-3.2-3B / Qwen2.5-3B / Gemma-2-2B. Its limitations
      section says nothing about merged, averaged or federated adapters.

  [L] arXiv 2605.30189, "Token-Level Generalization in LoRA Adapter Backdoors" (Lelle) --
      global_frobN_std: the standard deviation, across modules, of ||BA||_F / sqrt(in*out).
      AUC 1.000 on Llama-3.2-1B and Qwen2.5-1.5B, and 0.645 on Qwen2.5-7B. Its stated reason:
      "Initialization seed dominates poison count as the source of weight-level variance at 7B,
      inverting the signal-to-noise ratio between scales." Its conclusion: "The behavioral
      detector is the operationally portable result across scale; the weight-level detector is
      calibration-bound to its base model."

So [L] predicts these should already struggle at our scale (Llama-2-7B), before gossip is even
considered. This script measures it rather than asserting it, on checkpoints we already hold.

TWO REGIMES, AND THE SECOND IS THE ONE NOBODY HAS TESTED
  round 1  -- adapters are PRE-gossip, so this is [S]'s own setting: one adapter, one owner.
              If the baseline fails here, it fails for reasons of scale, matching [L].
  round >1 -- adapters are factor-averaged with 3 neighbours. Untested by either paper. Note
              the additional hazard: LoRA has a gauge freedom, (A,B) ~ (AR, R^-1 B) is the same
              function, so averaging A's and B's across nodes with independent gauges mixes
              inconsistent bases. Our own B6 measurement puts ||mean(B)mean(A) - mean(BA)|| /
              ||mean(BA)|| at 85% by round 1 and 8% by round 15.

BOTH OUTCOMES ARE USEFUL. If it works, it is cheaper than detP and we must say so. If it fails
on merged adapters, that fills the gap [S] leaves open and answers "why not just read the
weights?" -- which our escalation ladder currently has no reply to.

EFFICIENCY. dW = BA is never materialised for the spectrum: B is (out x r), A is (r x in), so
with B = Q_b R_b and A^T = Q_a R_a the nonzero singular values of BA are exactly those of the
r x r matrix R_b R_a^T. Kurtosis does need the entries, so that one module is materialised and
freed. r=8, so the spectral part is microseconds per module.

Usage:  python spectral_baseline.py                     # rounds 1,5,13,25, all three alphas
        python spectral_baseline.py --rounds 1 --quick  # round 1 only
"""
import argparse, glob, json, os, sys, time
import numpy as np

NFS = "/mnt/nfs/home/peechara"
CK = f"{NFS}/ckpts"
ap = argparse.ArgumentParser()
ap.add_argument("--rounds", default="1,5,13,25")
ap.add_argument("--alphas", default="inf,0.5,0.1")
ap.add_argument("--scenarios", default="att,noatt",
                help="run suffixes under ~/ckpts/r25_alpha{A}_{scen}_seed0. Use `attmerge,noatt` "
                     "for the MERGING attacker -- the only confound-free test of whether these "
                     "statistics survive averaging. Our usual attacker never merges, so from "
                     "round 5 its separation is displacement-from-non-participation (the "
                     "free-rider control scores z=+34.6 with no poison at all), and the merged "
                     "case has therefore never actually been measured.")
ap.add_argument("--lora-alpha", type=float, default=16.0)
ap.add_argument("--lora-r", type=int, default=8)
ap.add_argument("--out", default=f"{NFS}/iclscan-decentralized/results/noniid/"
                                 "spectral_baseline/spectral_baseline.json")
ap.add_argument("--quick", action="store_true", help="skip kurtosis (the only part that "
                                                     "materialises dW)")
args = ap.parse_args()
ROUNDS = [int(x) for x in args.rounds.split(",")]
ALPHAS = args.alphas.split(",")
SCENARIOS = [x for x in args.scenarios.split(",") if x]
SCALE = args.lora_alpha / args.lora_r          # PEFT applies dW = (alpha/r) * B @ A
os.makedirs(os.path.dirname(args.out), exist_ok=True)

from safetensors.numpy import load_file        # noqa: E402


def adapter_path(run, r, c):
    hits = glob.glob(f"{CK}/{run}/r{r}/{c}/**/adapter_model.safetensors", recursive=True)
    return os.path.dirname(hits[0]) if hits else None


def module_stats(A, B, want_kurtosis):
    """Spectral statistics of dW = SCALE * B @ A, without forming B@A for the spectrum.

    B is (out x r), A is (r x in). Write B = Q_b R_b and A^T = Q_a R_a with Q orthonormal.
    Then BA = Q_b (R_b R_a^T) Q_a^T, and since Q_b, Q_a have orthonormal columns the nonzero
    singular values of BA are exactly those of the r x r matrix R_b R_a^T. r=8 here."""
    Rb = np.linalg.qr(B, mode="r")                      # (r x r)
    Ra = np.linalg.qr(A.T, mode="r")                    # (r x r)
    sv = np.linalg.svd(Rb @ Ra.T, compute_uv=False) * SCALE
    sv = np.maximum(sv, 0.0)
    tot = sv.sum()
    p = sv / tot if tot > 0 else np.full_like(sv, 1.0 / len(sv))
    out_dim, in_dim = B.shape[0], A.shape[1]
    st = {
        "sigma1": float(sv[0]),                                     # [S] leading singular value
        "fro": float(np.sqrt((sv ** 2).sum())),                     # [S] Frobenius norm
        "energy": float(sv[0] / tot) if tot > 0 else 0.0,           # [S] energy concentration
        "entropy": float(-(p * np.log(p + 1e-12)).sum()),           # [S] spectral entropy
        # [L] dimension-normalised Frobenius norm; global_frobN_std is the SD of this across modules
        "frobN": float(np.sqrt((sv ** 2).sum()) / np.sqrt(in_dim * out_dim)),
    }
    if want_kurtosis:
        dW = (SCALE * (B @ A)).ravel()                              # [S] kurtosis of the entries
        m, s = dW.mean(), dW.std()
        st["kurtosis"] = float((((dW - m) / (s + 1e-12)) ** 4).mean()) if s > 0 else 0.0
        del dW
    return st


def adapter_signature(path, want_kurtosis):
    """Per-adapter feature vector. Returns None if the adapter cannot be read."""
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
    per = []
    attn_sum = {}
    for name, ab in sorted(mods.items()):
        if "A" not in ab or "B" not in ab:
            continue
        st = module_stats(ab["A"], ab["B"], want_kurtosis)
        st["_name"] = name
        per.append(st)
    if not per:
        return None
    keys = [k for k in per[0] if not k.startswith("_")]
    sig = {}
    for k in keys:
        v = np.array([m[k] for m in per])
        sig[f"{k}_mean"] = float(v.mean())
        sig[f"{k}_std"] = float(v.std())          # global_frobN_std == frobN_std, i.e. [L]
        sig[f"{k}_max"] = float(v.max())
    sig["_n_modules"] = len(per)
    return sig


def auc(pos, neg):
    """Mann-Whitney AUC. Returns None if either class is empty."""
    if not pos or not neg:
        return None
    n = 0.0
    for a in pos:
        for b in neg:
            n += 1.0 if a > b else (0.5 if a == b else 0.0)
    return n / (len(pos) * len(neg))


# ---------------------------------------------------------------- collect ----
rows = []
t0 = time.time()
for al in ALPHAS:
    for scen in SCENARIOS:
        run = f"r25_alpha{al}_{scen}_seed0"
        for r in ROUNDS:
            for i in range(8):
                p = adapter_path(run, r, f"c{i}")
                if p is None:
                    continue
                sig = adapter_signature(p, want_kurtosis=not args.quick)
                if sig is None:
                    continue
                sig.update({"alpha": al, "scen": scen, "round": r, "node": i,
                            "is_attacker": (scen in ("att", "attmerge") and i == 0),
                            "merged": r > 1})
                rows.append(sig)
            print(f"  {run} r{r}: {sum(1 for x in rows if x['round']==r and x['scen']==scen and x['alpha']==al)}"
                  f" adapters  ({time.time()-t0:.0f}s)", flush=True)
json.dump(rows, open(args.out, "w"), indent=2)
print(f"\n{len(rows)} adapter signatures -> {args.out}", flush=True)

FEATS = [k for k in rows[0] if not k.startswith("_") and k not in
         ("alpha", "scen", "round", "node", "is_attacker", "merged")] if rows else []

# ------------------------------------------------------- Q1: the source ----
# [S]'s own setting at round 1: one attacker adapter against clean adapters, unmerged.
print("\n=== Q1. Can weight-space find the ATTACKER? (positives = c0 in an attacker run) ===")
print("    round 1 = pre-gossip, i.e. exactly the regime 2602.15195 was built for.")
print(f"    {'round':>5} {'feature':>16} {'AUC':>7}   n_pos/n_neg", flush=True)
best = {}
for r in ROUNDS:
    pos_rows = [x for x in rows if x["round"] == r and x["is_attacker"]]
    neg_rows = [x for x in rows if x["round"] == r and not x["is_attacker"]]
    for f in FEATS:
        a = auc([x[f] for x in pos_rows], [x[f] for x in neg_rows])
        if a is None:
            continue
        key = max(a, 1 - a)                      # direction-agnostic, as an AUC baseline should be
        if key > best.get((r, "attacker"), (0, ""))[0]:
            best[(r, "attacker")] = (key, f)
    if (r, "attacker") in best:
        k, f = best[(r, "attacker")]
        print(f"    {r:5d} {f:>16} {k:7.3f}   {len(pos_rows)}/{len(neg_rows)}", flush=True)

# --------------------------------------------- Q2: the propagated copies ----
# The case the project has never solved. Positives = benign nodes in an attacker run at a late
# round (infected via gossip); negatives = the SAME node ids in the matched no-attacker run.
print("\n=== Q2. Can weight-space find PROPAGATED copies? (the unsolved case) ===")
print("    positives = benign nodes, attacker run;  negatives = benign nodes, no-attacker run")
print(f"    {'round':>5} {'alpha':>5} {'feature':>16} {'AUC':>7}   n_pos/n_neg", flush=True)
for r in [x for x in ROUNDS if x > 1]:
    for al in ALPHAS:
        pos = [x for x in rows if x["round"] == r and x["alpha"] == al
               and x["scen"] != "noatt" and not x["is_attacker"]]
        neg = [x for x in rows if x["round"] == r and x["alpha"] == al and x["scen"] == "noatt"
               and x["node"] != 0]
        bb = (0.0, "")
        for f in FEATS:
            a = auc([x[f] for x in pos], [x[f] for x in neg])
            if a is not None and max(a, 1 - a) > bb[0]:
                bb = (max(a, 1 - a), f)
        if bb[1]:
            print(f"    {r:5d} {al:>5} {bb[1]:>16} {bb[0]:7.3f}   {len(pos)}/{len(neg)}", flush=True)

# ------------------------------------------------------------- verdict ----
print("\n=== VERDICT ===")
r1 = best.get((1, "attacker"), (None, None))[0]
if r1 is not None:
    print(f"  Best single-feature AUC on the ATTACKER at round 1 (unmerged): {r1:.3f}")
    if r1 >= 0.95:
        print("  -> Weight-space WORKS on unmerged adapters at 7B. It is far cheaper than detP")
        print("     (no forward passes at all) and must be reported as a baseline, not omitted.")
    elif r1 >= 0.75:
        print("  -> PARTIAL at 7B. Consistent with 2605.30189's 1.000 -> 0.645 collapse between")
        print("     1.5B and 7B, attributed there to initialization-seed variance dominating.")
    else:
        print("  -> FAILS at 7B even unmerged, before gossip is a factor. This reproduces")
        print("     2605.30189's scale finding on a different model family and supports its")
        print("     conclusion that the weight-level detector is 'calibration-bound to its")
        print("     base model' while the behavioural one is portable.")
print("\n  NOTE ON FAIRNESS: these are single-feature AUCs, threshold-free and unsupervised.")
print("  2602.15195 trains a SUPERVISED logistic regression on 400 benign + 100 poisoned")
print("  adapters. We have 8 nodes, so a comparable supervised fit would be fitted and")
print("  evaluated on almost the same handful of adapters -- the honest comparison is the")
print("  single-feature AUC, and the supervised number should not be quoted without saying so.")
print("\nSPECTRAL BASELINE DONE", flush=True)
