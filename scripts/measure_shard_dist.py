"""Per-node category composition + unique-example fraction at 4000/node from the BIG pool,
for alpha in {inf,0.5,0.1}, at a given partition seed. Dumps JSON for local plotting.
Usage: python measure_shard_dist.py [seed]   (seed 0 -> shard_dist.json; else shard_dist_seed{seed}.json)"""
import sys, json
from collections import Counter
sys.path.insert(0, "/mnt/nfs/home/peechara/iclscan-decentralized/src/sim")
from noniid import categorize, dirichlet_partition_fixed

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0
R = "/mnt/nfs/home/peechara/iclscan-decentralized/results/noniid"
OUT = f"{R}/shard_dist.json" if SEED == 0 else f"{R}/shard_dist_seed{SEED}.json"

benign = json.load(open("/mnt/nfs/home/peechara/data/train/alpaca_benign_train_big.json"))
for i, e in enumerate(benign):
    e["_id"] = i
cats = categorize(benign)
CATS = sorted(set(cats))
gc = Counter(cats)
out = {"cats": CATS, "pool": len(benign), "seed": SEED,
       "global": [gc[c] / len(benign) for c in CATS], "alphas": {}}
for a, alab in [(float("inf"), "inf"), (0.5, "0.5"), (0.1, "0.1")]:
    shards = dirichlet_partition_fixed(benign, cats, 8, a, 4000, seed=SEED)
    mat, uniq = [], []
    for i in range(8):
        s = shards[i]; c = Counter(categorize(s))
        mat.append([c.get(cat, 0) / len(s) for cat in CATS])
        uniq.append(len(set(e["_id"] for e in s)) / len(s))
    out["alphas"][alab] = {"mat": mat, "uniq": uniq}
json.dump(out, open(OUT, "w"))
print(f"done seed={SEED} -> {OUT}")
