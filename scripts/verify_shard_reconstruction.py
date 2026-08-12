"""Prerequisite gate for Checks 4/5: shards must be exactly reconstructible on CPU, since
the leave-one-out and local-pool work reads saved adapters that were trained on them.

Verifies against the recorded seed-0 shard-uniqueness numbers (notes/RESULTS_SUMMARY.md sec 2):
    alpha=inf  min/mean unique = 1.00 / 1.00
    alpha=0.5  min/mean unique = 0.63 / 0.91
    alpha=0.1  min/mean unique = 0.72 / 0.94

Pool source, in order of preference:
  1. $ALPACA_TRAIN (the staged file itself -- authoritative; use this on the pod)
  2. a local raw Stanford Alpaca json, re-filtered and re-split exactly as stage_alpaca_big.py
Records which source was used. Exit code 1 on mismatch (callers must STOP).
"""
import hashlib
import json
import os
import random
import sys

import numpy as np

REPO = "/home/speechara/epfl/iclscan-decentralized"
OUT = f"{REPO}/results/defense_feas"
sys.path.insert(0, f"{REPO}/src/sim")
from noniid import categorize, dirichlet_partition_fixed  # noqa: E402

EXPECTED = {"inf": (1.00, 1.00), "0.5": (0.63, 0.91), "0.1": (0.72, 0.94)}
N_CLIENTS, SIZE, SEED, TOL = 8, 4000, 0, 0.005
RAW_FALLBACK = ("/tmp/claude-1000/-home-speechara-epfl/"
                "8fc8e4fc-ffed-4830-ba90-187691e9bd6b/scratchpad/alpaca_raw.json")


def load_pool():
    staged = os.environ.get("ALPACA_TRAIN")
    if staged and os.path.exists(staged):
        pool = json.load(open(staged))
        return pool, {"source": "staged_file", "path": staged, "n": len(pool),
                      "sha256_16": hashlib.sha256(open(staged, "rb").read()).hexdigest()[:16]}
    raw_path = os.environ.get("ALPACA_RAW", RAW_FALLBACK)
    if not os.path.exists(raw_path):
        raise SystemExit(f"no pool: set $ALPACA_TRAIN (staged) or $ALPACA_RAW (raw alpaca). "
                         f"tried {raw_path}")
    data = json.load(open(raw_path))
    clean = [e for e in data if e.get("output", "").strip()
             and len(e["instruction"]) < 200 and len(e["output"]) < 800]
    random.Random(0).shuffle(clean)          # exactly stage_alpaca_big.py
    pool = clean[:-400]
    return pool, {"source": "reconstructed_from_raw", "path": raw_path,
                  "n_raw": len(data), "n_clean": len(clean), "n": len(pool),
                  "note": "re-filtered and re-split exactly as scripts/stage_alpaca_big.py"}


def main():
    os.makedirs(OUT, exist_ok=True)
    pool, src = load_pool()
    cats = categorize(pool)
    by_id = {id(e): k for k, e in enumerate(pool)}
    res, ok = {}, True
    print(f"pool: {src['source']} n={src['n']}")
    if src["n"] != 49479:
        print(f"  WARNING: pool size {src['n']} != 49,479 recorded in RESULTS_SUMMARY")
    for a, (exp_min, exp_mean) in EXPECTED.items():
        alpha = float("inf") if a == "inf" else float(a)
        shards = dirichlet_partition_fixed(pool, cats, N_CLIENTS, alpha, SIZE, SEED)
        uniq = [len({by_id[id(e)] for e in shards[i]}) / SIZE for i in range(N_CLIENTS)]
        got_min, got_mean = min(uniq), sum(uniq) / len(uniq)
        match = abs(got_min - exp_min) <= TOL and abs(got_mean - exp_mean) <= TOL
        ok &= match
        res[a] = {"expected_min": exp_min, "expected_mean": exp_mean,
                  "got_min": round(got_min, 4), "got_mean": round(got_mean, 4),
                  "per_node_unique_fraction": [round(u, 4) for u in uniq],
                  "match": bool(match)}
        print(f"  alpha={a:>4s}: min {got_min:.2f} (exp {exp_min:.2f})  "
              f"mean {got_mean:.2f} (exp {exp_mean:.2f})  -> {'MATCH' if match else 'MISMATCH'}")
    out = {"gate": "shard_reconstruction", "pool_source": src, "tolerance": TOL,
           "n_clients": N_CLIENTS, "fixed_size": SIZE, "seed": SEED,
           "results": res, "verdict": "VERIFIED" if ok else "MISMATCH"}
    json.dump(out, open(f"{OUT}/shard_reconstruction_check.json", "w"), indent=2)
    print(f"\nVERDICT: {out['verdict']}  -> {OUT}/shard_reconstruction_check.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
