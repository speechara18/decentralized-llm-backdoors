"""
DECENTRALIZED AlignIns. GPU (small card is plenty), no model load, no generation.

AlignIns (Xu, Zhang, Hu -- "Detecting Backdoor Attacks in Federated Learning via Direction
Alignment Inspection", arXiv 2503.07978) is a CENTRALIZED defense: a server holding every client
update inspects two directional statistics and filters outliers. We had written it off as
inapplicable because "it needs every client update at once". That was wrong. It needs A
reference population, and a gossip neighbourhood is one. This decentralizes it and measures
whether it survives the move -- rather than assuming either way.

THE TWO STATISTICS, AND HOW EACH REFERENCE MAPS TO A DECENTRALIZED ONE

  TDA (Temporal Direction Alignment), centralized:
      omega_i = <Delta_i, theta^t> / (||Delta_i|| ||theta^t||)
      i.e. cosine of client i's update against the LATEST GLOBAL MODEL.
    Decentralized: there is no global model. The natural local analogue is the receiver's own
    current adapter -- the thing it would have kept had it not merged. So for receiver r judging
    sender j we compute cos(Delta_j, theta_r). We ALSO compute cos against the neighbourhood mean
    as a variant, because which analogue is "right" is a research question, not a given, and
    reporting only one would be choosing the answer.

  MPSA (Masked Principal Sign Alignment), centralized:
      p = sgn(sum_i sgn(Delta_i))                      coordinate-wise majority over ALL updates
      rho_i = 1 - || (sgn(Delta_i) - p) * Top_k(Delta_i) ||_0 / k
    Decentralized: the majority is taken over the updates the receiver actually HOLDS -- itself
    plus its three neighbours, so n=4 rather than n=20+. This is the part most likely to break:
    a 4-way majority is far noisier than a 20-way one, and under non-IID the benign updates
    legitimately disagree.

  Both are then standardised by a median-based Z-score and thresholded. We report the raw
  statistics and the MZ-scores rather than imposing AlignIns' filtering radii, which were tuned
  for its own setting and would not transfer.

WHY THE COMPOSED PRODUCT AND NOT THE FACTORS -- this is the part that would silently ruin it.
LoRA has a gauge freedom: (A, B) and (A R, R^-1 B) express the SAME function for any invertible
R. So the signs and directions of the A and B factors are not well defined -- two nodes can hold
identical functions with completely different factor signs. Sign agreement computed on factors
would measure gauge mismatch, not backdoors, and would produce a confident negative result for
entirely spurious reasons. Everything here is computed on the composed Delta-W = (alpha/r) B A,
which IS gauge-invariant. Cost: Delta-W is ~26 GB dense per adapter at 7B, so it is composed and
freed one module at a time and never held whole.

THE THEORY IS AT ITS LIMIT HERE, AND THAT IS WORTH STATING. AlignIns assumes m < n/(3+eps)
malicious clients. A receiver holds n=4 updates (self + 3 neighbours); if one is the attacker,
m=1 and n/(3+eps) ~ 1.33, so the assumption holds by a margin of one third of a client. On a
degree-3 graph this method is being run at the very edge of its own guarantee.

Usage:  python alignins_decentralized.py
        python alignins_decentralized.py --rounds 1,5,13 --topk-frac 0.01
"""
import argparse, glob, json, os, sys, time
import numpy as np
import torch

NFS = "/mnt/nfs/home/peechara"
CK = f"{NFS}/ckpts"
ADJ = {i: sorted({(i - 1) % 8, (i + 1) % 8, (i + 4) % 8}) for i in range(8)}

ap = argparse.ArgumentParser()
ap.add_argument("--rounds", default="1,5,13,24")
ap.add_argument("--alphas", default="inf,0.5,0.1")
ap.add_argument("--scenarios", default="att,noatt",
                help="run suffixes under ~/ckpts/r25_alpha{A}_{scen}_seed0. Add `attmerge` for the "
                     "MERGING attacker -- the fair test of AlignIns, because a merging attacker "
                     "removes the non-participation confound that makes the non-merging results "
                     "uninterpretable after round 1.")
ap.add_argument("--topk-frac", type=float, default=0.01,
                help="fraction of coordinates in the Top-k mask, per module. AlignIns takes a "
                     "GLOBAL top-k over the flattened update; a global sort over 6.5e9 entries "
                     "is impractical, so we take it per module. Deviation, documented, not hidden.")
ap.add_argument("--lora-alpha", type=float, default=16.0)
ap.add_argument("--lora-r", type=int, default=8)
ap.add_argument("--out", default=f"{NFS}/iclscan-decentralized/results/noniid/"
                                 "alignins/alignins_decentralized.json")
ap.add_argument("--mpsa-scope", default="neighbourhood",
                choices=["neighbourhood", "global"],
                help="set over which MPSA's principal sign is taken. neighbourhood = the 4 "
                     "adapters a receiver holds (faithful to what a node can do unaided). "
                     "global = all n nodes, which requires one relay hop and is the test of "
                     "whether a wider majority revives MPSA -- it sat at exactly chance (12/36) "
                     "with n=4, and AlignIns was designed for a 20-client vote.")
ap.add_argument("--device", default="cuda",
                help="the work is composed-dW matmuls plus elementwise sign/top-k, which is "
                     "GPU-shaped. Peak memory is 4 adapters x one module: the largest LoRA "
                     "target here is 11008x4096, so ~720 MB fp32 -- comfortable on a 40 GB card.")
args = ap.parse_args()
ROUNDS = [int(x) for x in args.rounds.split(",")]
ALPHAS = args.alphas.split(",")
SCENARIOS = [x for x in args.scenarios.split(",") if x]
SCALE = args.lora_alpha / args.lora_r
os.makedirs(os.path.dirname(args.out), exist_ok=True)

from safetensors.numpy import load_file      # noqa: E402

DEV = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
if DEV == "cpu" and args.device == "cuda":
    print("WARNING: --device cuda requested but no GPU visible; falling back to CPU. On this "
          "cluster a --gpu 0 request can still land on a GPU node, so check the placement.",
          flush=True)
print(f"device: {DEV}", flush=True)


def adir(run, r, c):
    h = glob.glob(f"{CK}/{run}/r{r}/{c}/**/adapter_model.safetensors", recursive=True)
    return os.path.dirname(h[0]) if h else None


def load_factors(path):
    """{module_name: (A, B)} for every LoRA module. Factors are tiny (r=8), so they live on the
    device; only the composed dW is large and that is built and freed per module."""
    t = load_file(os.path.join(path, "adapter_model.safetensors"))
    m = {}
    for k in t:
        if "lora_A" in k:
            m.setdefault(k.split(".lora_A")[0], {})["A"] = torch.as_tensor(
                np.asarray(t[k], np.float32), device=DEV)
        elif "lora_B" in k:
            m.setdefault(k.split(".lora_B")[0], {})["B"] = torch.as_tensor(
                np.asarray(t[k], np.float32), device=DEV)
    return {k: (v["A"], v["B"]) for k, v in m.items() if "A" in v and "B" in v}


@torch.no_grad()
def stats_for_group(fac, ids, ref_id, sign_ids=None):
    """AlignIns TDA + MPSA for every member of `ids`, using only what this group holds.

    fac: {node_id: {module: (A,B)}}. Streams module by module so no full Delta-W is ever held
    -- the composed update is ~26 GB dense per adapter at 7B.

    sign_ids: the set over which MPSA's principal sign is taken. Defaults to `ids`, i.e. the 4
    adapters a receiver actually holds. Pass a LARGER set to test the relay hypothesis: MPSA died
    at exactly chance (12/36) in the neighbourhood version, and AlignIns was designed for a
    20-client majority, so the 4-way vote is the prime suspect. Widening the vote requires nodes
    to obtain adapters they would not normally receive -- one relay hop, which on this diameter-2
    graph reaches everyone. This measures whether that would buy anything BEFORE paying the
    bandwidth for it.
    Returns {node_id: {"tda_self":.., "tda_mean":.., "mpsa":..}}.
    """
    sign_ids = list(ids) if sign_ids is None else list(sign_ids)
    mods = sorted(set.intersection(*[set(fac[i]) for i in ids]))
    dot_self = {i: 0.0 for i in ids}                 # <Delta_i, theta_ref>
    n2 = {i: 0.0 for i in ids}                       # ||Delta_i||^2
    n2_ref = 0.0
    dot_mean = {i: 0.0 for i in ids}
    n2_mean = 0.0
    agree = {i: 0 for i in ids}                      # MPSA numerator
    kept = {i: 0 for i in ids}
    for mname in mods:
        dW = {}
        for i in ids:
            A, B = fac[i][mname]
            dW[i] = SCALE * (B @ A)                  # composed -> gauge-invariant
        ref = dW[ref_id]
        mean = sum(dW.values()) / len(ids)
        n2_ref += float((ref * ref).sum())
        n2_mean += float((mean * mean).sum())
        # principal sign: coordinate-wise majority over the updates this receiver HOLDS
        # principal sign over sign_ids, which may be wider than the receiver's own holdings
        if set(sign_ids) == set(ids):
            p = torch.sign(sum(torch.sign(v) for v in dW.values())).reshape(-1)
        else:
            acc = None
            for q in sign_ids:
                A2, B2 = fac[q][mname]
                sg = torch.sign(SCALE * (B2 @ A2))
                acc = sg if acc is None else acc + sg
                del sg
            p = torch.sign(acc).reshape(-1)
            del acc
        for i in ids:
            v = dW[i]
            dot_self[i] += float((v * ref).sum())
            dot_mean[i] += float((v * mean).sum())
            n2[i] += float((v * v).sum())
            flat = v.reshape(-1)
            k = max(1, int(args.topk_frac * flat.numel()))
            idx = torch.topk(flat.abs(), k, sorted=False).indices   # Top-k by |value|, per module
            agree[i] += int((torch.sign(flat[idx]) == p[idx]).sum())
            kept[i] += k
        del dW, p, ref, mean
    out = {}
    for i in ids:
        d = np.sqrt(n2[i])
        out[i] = {
            "tda_self": dot_self[i] / (d * np.sqrt(n2_ref) + 1e-12),
            "tda_mean": dot_mean[i] / (d * np.sqrt(n2_mean) + 1e-12),
            "mpsa": agree[i] / kept[i],              # AlignIns' rho, as a fraction in [0,1]
        }
    return out


def mz(vals):
    """Median-based Z-score, as AlignIns uses. Robust to the outlier it is looking for."""
    v = np.asarray(vals, float)
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    s = 1.4826 * mad
    return (v - med) / (s + 1e-12)


res = json.load(open(args.out)) if os.path.exists(args.out) else {"cells": {}}
res.setdefault("meta", {"topk_frac": args.topk_frac, "rounds": ROUNDS, "alphas": ALPHAS,
                        "mpsa_scope": args.mpsa_scope,
                        "note": "composed dW only (gauge-invariant); per-module top-k; "
                                "n=4 per receiver, at AlignIns' m<n/(3+eps) boundary"})
t0 = time.time()
for al in ALPHAS:
    for scen in SCENARIOS:
        run = f"r25_alpha{al}_{scen}_seed0"
        for r in ROUNDS:
            for rcv in range(8):
                key = f"{al}|{scen}|r{r}|recv{rcv}"
                if key in res["cells"]:
                    continue
                ids = [rcv] + ADJ[rcv]
                paths = {i: adir(run, r, f"c{i}") for i in ids}
                if any(p is None for p in paths.values()):
                    continue
                fac = {i: load_factors(paths[i]) for i in ids}
                sids = list(range(8)) if args.mpsa_scope == "global" else None
                if sids is not None:                      # global scope needs every adapter
                    for q in sids:
                        if q not in fac:
                            pq = adir(run, r, f"c{q}")
                            if pq is None:
                                sids = None
                                break
                            fac[q] = load_factors(pq)
                st = stats_for_group(fac, ids, ref_id=rcv, sign_ids=sids)
                for stat in ("tda_self", "tda_mean", "mpsa"):
                    z = mz([st[i][stat] for i in ids])
                    for i, zz in zip(ids, z):
                        st[i][f"mz_{stat}"] = float(zz)
                res["cells"][key] = {str(i): st[i] for i in ids}
                json.dump(res, open(args.out, "w"), indent=2)
                print(f"  {key}  ({time.time()-t0:.0f}s)", flush=True)
                del fac
                if DEV == "cuda":
                    torch.cuda.empty_cache()

# ------------------------------------------------------------------ readout ----
ASCEN = "attmerge" if "attmerge" in SCENARIOS else "att"
print(f"\n=== Can a receiver flag the attacker among the 3 adapters it holds? "
      f"(scenario: {ASCEN}) ===")
if ASCEN == "attmerge":
    print("    MERGING attacker: it averages with its neighbours like everyone else, so the")
    print("    non-participation confound that invalidated the non-merging results is ABSENT.")
    print("    Anything that separates here is poison, not displacement-from-not-merging.")
print("    A receiver only sees c0 if c0 is one of its neighbours: c1, c4, c7 on C8(1,4).\n")
NBR_OF_ATT = [i for i in range(8) if 0 in ADJ[i]]
for stat in ("mz_tda_self", "mz_tda_mean", "mz_mpsa"):
    print(f"  --- {stat} ---")
    for al in ALPHAS:
        for r in ROUNDS:
            att, ben = [], []
            for rcv in NBR_OF_ATT:
                c = res["cells"].get(f"{al}|{ASCEN}|r{r}|recv{rcv}")
                if not c:
                    continue
                att.append(c["0"][stat])
                ben += [c[str(i)][stat] for i in ADJ[rcv] if i != 0]
            cn = []
            for rcv in NBR_OF_ATT:
                c = res["cells"].get(f"{al}|noatt|r{r}|recv{rcv}")
                if c:
                    cn += [c[str(i)][stat] for i in ADJ[rcv]]
            if not att:
                continue
            # PER-RECEIVER test: within its OWN neighbourhood, does c0 outrank every benign peer?
            # Aggregating c0 across receivers and comparing to a global benign max mixes
            # neighbourhoods and is not the deployable question.
            hits = 0
            for rcv in NBR_OF_ATT:
                c = res["cells"].get(f"{al}|{ASCEN}|r{r}|recv{rcv}")
                if not c:
                    continue
                if abs(c["0"][stat]) > max(abs(c[str(j)][stat]) for j in ADJ[rcv] if j != 0):
                    hits += 1
            a = float(np.mean(att))
            worst = max(abs(x) for x in ben) if ben else float("nan")
            print(f"    alpha={al:>4} r{r:<3} attacker {a:+7.2f} | worst benign |z| {worst:5.2f} "
                  f"| clean-run {np.mean(cn) if cn else float('nan'):+7.2f}"
                  f"   {hits}/{len(NBR_OF_ATT)} receivers flag it")
    print()

print("=== INTERPRETATION ===")
print("  SEPARATES means the attacker's |MZ| exceeds every benign |MZ| in the same neighbourhood,")
print("  i.e. a receiver could rank it worst and act. That is the deployable question.")
print("  Watch for it holding at alpha=inf and failing at 0.1 -- that is the predicted failure")
print("  mode, since under skew benign updates legitimately point in different directions and")
print("  the 4-way principal sign becomes close to a coin flip.")
print("\n  CAVEATS: n=4 per neighbourhood puts this at AlignIns' m<n/(3+eps) boundary; the Top-k")
print("  mask is per-module rather than global; and only 3 of 8 receivers can see the attacker")
print("  at all, so each cell rests on 3 observations.")
print(f"\nwrote {args.out}\nDECENTRALIZED ALIGNINS DONE", flush=True)
