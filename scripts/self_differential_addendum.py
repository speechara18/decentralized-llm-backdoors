"""
ADDENDUM to self_differential.py -- reference-free ranking, a null, and the clean~clean control.

Pure CPU. Computes NO new forward passes: it re-reads the eight full-vocabulary coarse curves that
self_differential.py already wrote to NFS. Same scoring function, same prompts, same vocabulary
order as trigger_inversion.py -- only the arithmetic on top of the curves is new.

Three questions, in the priority order set by the council stress-test:

P1. STANDALONE (reference-free). Rank the raw r25 curve with no control subtracted at all.
    Report top-20 with z, and the rank/z of '_Bad'(9178) and '_bad'(4319). Add a NULL: the z
    distribution of 300 uniformly-sampled vocabulary tokens (same seed as the reference run's
    random baseline), so "z ~ 8.9" can be read against something.

P2. CLEAN~CLEAN NULL. Score noatt c1 against noatt c4 -- two adapters that are both clean -- exactly
    the way the attacked-vs-clean differential was scored. If a trigger-shaped family appears
    between two clean nodes, the differential is measuring inter-node drift and dies.

P3. CASE-VARIANT SIGNATURE. Does a case-variant pair of the same word ('Bad'/'bad') appear in the
    top-5, and is that also true of the clean controls?
"""
import json, os, re
import numpy as np
from transformers import AutoTokenizer

BASE = "/mnt/nfs/home/peechara/models/base/Llama-2-7b-chat-hf"
OUT = "/mnt/nfs/home/peechara/iclscan-decentralized/results/self_differential"

BAD_UPPER, BAD_LOWER = 9178, 4319          # '_Bad', '_bad'
LATE, EARLY = 25, 1

ADAPTERS = [
    ("att_c1_propagated", "r25_alphainf_att_seed0",   "c1", "THE KEY CASE: gossip-propagated node"),
    ("att_c0_attacker",   "r25_alphainf_att_seed0",   "c0", "the attacker"),
    ("noatt_c1_clean",    "r25_alphainf_noatt_seed0", "c1", "clean control 1"),
    ("noatt_c4_clean",    "r25_alphainf_noatt_seed0", "c4", "clean control 2"),
]

tok = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.eos_token
V_EMB = 32000
SPECIAL = {tok.bos_token_id, tok.eos_token_id, tok.unk_token_id, tok.pad_token_id}
SPECIAL = {s for s in SPECIAL if s is not None}
VOCAB = [i for i in range(V_EMB) if i not in SPECIAL]
NV = len(VOCAB)
POS = {int(t): j for j, t in enumerate(VOCAB)}

# the SAME random baseline the reference run used (trigger_inversion.py: random.Random(1234))
import random
RANDOM_BASE = random.Random(1234).sample(VOCAB, 300)
RB_IDX = np.array([POS[i] for i in RANDOM_BASE])

R = {"note": "CPU-only addendum; re-reads curves written by self_differential.py, no new GPU passes",
     "n_candidates": NV,
     "bad_ids": {"upper_Bad": BAD_UPPER, "lower_bad": BAD_LOWER}}


def load(run, c, r):
    return np.load(f"{OUT}/curve_{run}_{c}_r{r}.npy").astype(np.float64)


def describe(i):
    i = int(i)
    return {"id": i, "token": tok.convert_ids_to_tokens([i])[0], "repr": repr(tok.decode([i]))}


for nm, i in [("upper_Bad", BAD_UPPER), ("lower_bad", BAD_LOWER)]:
    R["bad_ids"][nm + "_check"] = describe(i)
print("token id check:", json.dumps(R["bad_ids"]), flush=True)


def ranks_of(v):
    o = np.argsort(-v)
    rk = np.empty(len(v), dtype=int)
    rk[o] = np.arange(1, len(v) + 1)
    return o, rk


def norm_word(t):
    """strip SentencePiece word marker + non-letters -> for case-variant matching"""
    return re.sub(r"[^a-z]", "", t.replace("▁", "").lower())


def case_variant_pairs(entries):
    """pairs among `entries` that are the same word differing only in case"""
    out = []
    for a in range(len(entries)):
        for b in range(a + 1, len(entries)):
            ta = entries[a]["token"].replace("▁", "")
            tb = entries[b]["token"].replace("▁", "")
            if ta != tb and ta.lower() == tb.lower() and norm_word(ta):
                out.append({"a": entries[a]["repr"], "b": entries[b]["repr"],
                            "word": ta.lower(), "ranks": [entries[a]["rank"], entries[b]["rank"]]})
    return out


def entry(v, o, rk, mu, sd, rbmu, rbsd, k, i):
    return {"rank": int(k + 1), "score": float(v[i]),
            "z_fullvocab": (float(v[i]) - mu) / (sd + 1e-12),
            "z_vs_random300": (float(v[i]) - rbmu) / (rbsd + 1e-12),
            **describe(VOCAB[i])}


# ---------------------------------------------------------------- P1: standalone, reference-free
R["P1_standalone"] = {}
for label, run, c, note in ADAPTERS:
    v = load(run, c, LATE)
    o, rk = ranks_of(v)
    mu, sd = float(v.mean()), float(v.std())
    rb = v[RB_IDX]
    rbmu, rbsd = float(rb.mean()), float(rb.std())
    rb_z_full = (rb - mu) / (sd + 1e-12)

    top20 = [entry(v, o, rk, mu, sd, rbmu, rbsd, k, i) for k, i in enumerate(o[:20])]
    ent = {
        "note": note, "run": run, "client": c,
        "dist": {"mean": mu, "std": sd, "max": float(v.max()),
                 "p50": float(np.percentile(v, 50)), "p99": float(np.percentile(v, 99)),
                 "p99.9": float(np.percentile(v, 99.9))},
        "null_random300": {
            "n": len(RANDOM_BASE), "mean_score": rbmu, "std_score": rbsd,
            "z_fullvocab_mean": float(rb_z_full.mean()), "z_fullvocab_std": float(rb_z_full.std()),
            "z_fullvocab_max": float(rb_z_full.max()),
            "z_fullvocab_p95": float(np.percentile(rb_z_full, 95)),
            "z_fullvocab_p99": float(np.percentile(rb_z_full, 99)),
            "best_rank_among_random300": int(rk[RB_IDX].min()),
        },
        "top20": top20,
        "case_variant_pairs_top5": case_variant_pairs(top20[:5]),
        "case_variant_pairs_top20": case_variant_pairs(top20),
    }
    for nm, tid in [("Bad_upper_9178", BAD_UPPER), ("bad_lower_4319", BAD_LOWER)]:
        j = POS[tid]
        ent[nm] = {**describe(tid), "rank": int(rk[j]), "n": NV,
                   "rank_pct": round(100.0 * int(rk[j]) / NV, 4), "score": float(v[j]),
                   "z_fullvocab": (float(v[j]) - mu) / (sd + 1e-12),
                   "z_vs_random300": (float(v[j]) - rbmu) / (rbsd + 1e-12)}
    R["P1_standalone"][label] = ent
    print(f"\n=== P1 STANDALONE {label} ({note})", flush=True)
    print(f"  '_Bad' rank {ent['Bad_upper_9178']['rank']}/{NV}  z_full={ent['Bad_upper_9178']['z_fullvocab']:+.2f}"
          f"  z_rand={ent['Bad_upper_9178']['z_vs_random300']:+.2f}")
    print(f"  '_bad' rank {ent['bad_lower_4319']['rank']}/{NV}  z_full={ent['bad_lower_4319']['z_fullvocab']:+.2f}"
          f"  z_rand={ent['bad_lower_4319']['z_vs_random300']:+.2f}")
    print(f"  null300: z_full mean {ent['null_random300']['z_fullvocab_mean']:+.2f} "
          f"sd {ent['null_random300']['z_fullvocab_std']:.2f} max {ent['null_random300']['z_fullvocab_max']:+.2f}; "
          f"best random rank {ent['null_random300']['best_rank_among_random300']}")
    print("  top10: " + ", ".join(f"{e['repr']}(z{e['z_fullvocab']:+.1f})" for e in top20[:10]))
    print(f"  case-variant pairs in top5: {ent['case_variant_pairs_top5']}")


# ---------------------------------------------------------------- P2: clean ~ clean null
c1 = load("r25_alphainf_noatt_seed0", "c1", LATE)
c4 = load("r25_alphainf_noatt_seed0", "c4", LATE)
d = c1 - c4
o, rk = ranks_of(d)
mu, sd = float(d.mean()), float(d.std())
top20 = [{"rank": int(k + 1), "diff": float(d[i]),
          "z": (float(d[i]) - mu) / (sd + 1e-12),
          "score_c1": float(c1[i]), "score_c4": float(c4[i]), **describe(VOCAB[i])}
         for k, i in enumerate(o[:20])]
P2 = {"definition": "diff(tok) = score(noatt c1 @ r25) - score(noatt c4 @ r25); BOTH ARE CLEAN",
      "why": "same arithmetic as the attacked-vs-clean differential, applied to two clean nodes. "
             "A trigger-shaped family here means the differential measures inter-node drift.",
      "dist": {"mean": mu, "std": sd, "max": float(d.max()), "min": float(d.min()),
               "p99": float(np.percentile(d, 99)), "p99.9": float(np.percentile(d, 99.9))},
      "top20": top20,
      "case_variant_pairs_top5": case_variant_pairs(top20[:5]),
      "case_variant_pairs_top20": case_variant_pairs(top20)}
for nm, tid in [("Bad_upper_9178", BAD_UPPER), ("bad_lower_4319", BAD_LOWER)]:
    j = POS[tid]
    P2[nm] = {**describe(tid), "rank": int(rk[j]), "n": NV,
              "rank_pct": round(100.0 * int(rk[j]) / NV, 4),
              "diff": float(d[j]), "z": (float(d[j]) - mu) / (sd + 1e-12)}
# and the reverse direction, since the sign is arbitrary between two clean peers
o2, rk2 = ranks_of(-d)
P2["reverse_c4_minus_c1"] = {
    "top20": [{"rank": int(k + 1), "diff": float(-d[i]), "z": (float(-d[i]) - (-mu)) / (sd + 1e-12),
               **describe(VOCAB[i])} for k, i in enumerate(o2[:20])],
    "Bad_upper_9178_rank": int(rk2[POS[BAD_UPPER]]),
    "bad_lower_4319_rank": int(rk2[POS[BAD_LOWER]]),
}
P2["reverse_c4_minus_c1"]["case_variant_pairs_top5"] = case_variant_pairs(P2["reverse_c4_minus_c1"]["top20"][:5])
R["P2_clean_vs_clean"] = P2
print("\n=== P2 CLEAN~CLEAN (noatt c1 - noatt c4, both clean) ===", flush=True)
print(f"  dist mean {mu:+.4f} sd {sd:.4f} max {P2['dist']['max']:+.4f}")
print(f"  '_Bad' rank {P2['Bad_upper_9178']['rank']}/{NV} z={P2['Bad_upper_9178']['z']:+.2f} | "
      f"'_bad' rank {P2['bad_lower_4319']['rank']}/{NV} z={P2['bad_lower_4319']['z']:+.2f}")
print("  top10 (c1-c4): " + ", ".join(f"{e['repr']}({e['diff']:+.3f})" for e in top20[:10]))
print("  top10 (c4-c1): " + ", ".join(f"{e['repr']}({e['diff']:+.3f})" for e in P2["reverse_c4_minus_c1"]["top20"][:10]))
print(f"  case-variant pairs in top5: fwd={P2['case_variant_pairs_top5']} rev={P2['reverse_c4_minus_c1']['case_variant_pairs_top5']}")

# ---------------------------------------------------------------- P3 summary table
R["P3_case_variant_summary"] = {
    label: {"top5": [e["repr"] for e in R["P1_standalone"][label]["top20"][:5]],
            "has_case_variant_pair_top5": bool(R["P1_standalone"][label]["case_variant_pairs_top5"]),
            "pairs_top5": R["P1_standalone"][label]["case_variant_pairs_top5"]}
    for label, _, _, _ in ADAPTERS}
R["P3_case_variant_summary"]["clean_vs_clean_diff"] = {
    "top5": [e["repr"] for e in P2["top20"][:5]],
    "has_case_variant_pair_top5": bool(P2["case_variant_pairs_top5"]),
    "pairs_top5": P2["case_variant_pairs_top5"]}

R["summary_table"] = {
    label: {"standalone_Bad_rank": R["P1_standalone"][label]["Bad_upper_9178"]["rank"],
            "standalone_Bad_z_full": R["P1_standalone"][label]["Bad_upper_9178"]["z_fullvocab"],
            "standalone_Bad_z_rand": R["P1_standalone"][label]["Bad_upper_9178"]["z_vs_random300"],
            "standalone_bad_rank": R["P1_standalone"][label]["bad_lower_4319"]["rank"],
            "standalone_bad_z_full": R["P1_standalone"][label]["bad_lower_4319"]["z_fullvocab"]}
    for label, _, _, _ in ADAPTERS}
print("\n=== SUMMARY (standalone, reference-free) ===", flush=True)
for label, _, _, _ in ADAPTERS:
    s = R["summary_table"][label]
    print(f"  {label:20s} _Bad rank {s['standalone_Bad_rank']:6d} z_full {s['standalone_Bad_z_full']:+6.2f} "
          f"z_rand {s['standalone_Bad_z_rand']:+6.2f} | _bad rank {s['standalone_bad_rank']:6d} "
          f"z_full {s['standalone_bad_z_full']:+6.2f}")

p = f"{OUT}/self_differential_addendum.json"
json.dump(R, open(p, "w"), indent=2)
print(f"\nwrote {p}", flush=True)
