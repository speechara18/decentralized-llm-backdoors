"""
Post-hoc, CPU-only differential analysis of the trigger-inversion sweep. No GPU, no new forward
passes -- it only re-reads the full-vocabulary score curves the sweep already wrote to NFS.

WHY THIS IS NEEDED
------------------
The raw top-20 of a refusal-inversion sweep is contaminated by the BASE model's own safety
behaviour: Llama-2-chat refuses on tokens like 'Hitler' / 'negro' / 'illegal' whether or not any
backdoor is present. Those tokens will top the list on a CLEAN adapter too, so a high raw score is
NOT evidence of a backdoor.

The backdoor-specific signal is the DIFFERENCE against the matched clean control:
    diff(tok) = score_backdoored(tok) - score_clean(tok)
Tokens that rise only on the poisoned/propagated adapters are candidate triggers; tokens that are
equally high everywhere are base-model safety artefacts and are correctly discounted.
"""
import json, os
import numpy as np
from transformers import AutoTokenizer

OUTDIR = "/mnt/nfs/home/peechara/iclscan-decentralized/results/trigger_inversion"
BASE = "/mnt/nfs/home/peechara/models/base/Llama-2-7b-chat-hf"

tok = AutoTokenizer.from_pretrained(BASE)
res = json.load(open(f"{OUTDIR}/trigger_inversion.json"))
bm_ids = res["badmagic_tokenization"]["ids"]

cur = {t: np.load(f"{OUTDIR}/coarse_{t}.npy").astype(np.float64)
       for t in ["c0_attacker", "c1_propagated", "c1_clean"]}
V = len(cur["c1_clean"])
SPECIAL = {tok.bos_token_id, tok.eos_token_id, tok.unk_token_id}
ids = np.array([i for i in range(V + 3) if i not in SPECIAL][:V])  # sweep order from the main run

out = {"note": "diff = score(adapter) - score(matched clean control), full-vocab, 2-prompt coarse pass",
       "n_tokens": int(V)}


def describe(i):
    i = int(i)
    return {"id": i, "token": tok.convert_ids_to_tokens([i])[0], "repr": repr(tok.decode([i]))}


for tag in ["c0_attacker", "c1_propagated"]:
    d = cur[tag] - cur["c1_clean"]
    o = np.argsort(-d)
    rank = np.empty(V, dtype=int)
    rank[o] = np.arange(1, V + 1)
    ent = {
        "diff_dist": {"mean": float(d.mean()), "std": float(d.std()),
                      "p50": float(np.percentile(d, 50)), "p99": float(np.percentile(d, 99)),
                      "max": float(d.max())},
        "top20_diff": [dict(rank=int(k + 1), diff=float(d[j]),
                            score_backdoored=float(cur[tag][j]), score_clean=float(cur["c1_clean"][j]),
                            **describe(ids[j])) for k, j in enumerate(o[:20])],
    }
    # where do BadMagic's constituent tokens sit on the DIFFERENTIAL ranking?
    pos = {int(ids[j]): j for j in range(V)}
    ent["badmagic_constituents_diff"] = []
    for b in bm_ids:
        j = pos.get(int(b))
        if j is None:
            continue
        ent["badmagic_constituents_diff"].append(dict(
            rank=int(rank[j]), rank_pct=round(100.0 * int(rank[j]) / V, 4), diff=float(d[j]),
            score_backdoored=float(cur[tag][j]), score_clean=float(cur["c1_clean"][j]), **describe(b)))
    out[tag] = ent
    print(f"\n=== {tag} minus clean control ===", flush=True)
    print("  top15 by diff: " + ", ".join(f"{e['repr']}({e['diff']:+.2f})" for e in ent["top20_diff"][:15]))
    for e in ent["badmagic_constituents_diff"]:
        print(f"  BadMagic piece {e['token']:>8s}: diff rank {e['rank']}/{V} "
              f"(top {e['rank_pct']}%)  diff={e['diff']:+.3f} "
              f"(backdoored {e['score_backdoored']:+.3f} vs clean {e['score_clean']:+.3f})")

# how much do the two backdoored adapters agree with each other, vs with the clean control?
# (does the PROPAGATED node's inversion landscape look like the ATTACKER's?)
try:
    from scipy.stats import spearmanr
    def _sp(a, b):
        return float(spearmanr(a, b)[0])
except ImportError:                                  # rank-correlation without scipy
    def _sp(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])
pairs = [("c0_attacker", "c1_propagated"), ("c0_attacker", "c1_clean"), ("c1_propagated", "c1_clean")]
out["spearman_fullvocab"] = {f"{a}~{b}": _sp(cur[a], cur[b]) for a, b in pairs}
# restricted to the union of each adapter's own top-200 (the part of the landscape that matters)
top = {t: set(np.argsort(-cur[t])[:200].tolist()) for t in cur}
u = sorted(top["c0_attacker"] | top["c1_propagated"] | top["c1_clean"])
out["spearman_top200union"] = {f"{a}~{b}": _sp(cur[a][u], cur[b][u]) for a, b in pairs}
print("\nSpearman (full vocab):   ", {k: round(v, 3) for k, v in out["spearman_fullvocab"].items()})
print("Spearman (top-200 union):", {k: round(v, 3) for k, v in out["spearman_top200union"].items()})

json.dump(out, open(f"{OUTDIR}/trigger_inversion_diff.json", "w"), indent=2)
print(f"\nwrote {OUTDIR}/trigger_inversion_diff.json", flush=True)
