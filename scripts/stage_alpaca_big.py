"""
Stage a LARGER benign Alpaca pool so the decentralized nodes can train ~1 epoch across R=15
rounds without over-fitting (per tune_convergence). Writes NEW files (does NOT clobber the
4000-example alpaca_benign_train.json used by earlier runs). Same clean-filter + seed-0 shuffle,
so the first 4000 still match the old train (reproducible). Train and held-out are disjoint.
"""
import json
import urllib.request
import random
import os

URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
D = "/mnt/nfs/home/peechara/data/train"
os.makedirs(D, exist_ok=True)

data = json.load(urllib.request.urlopen(URL, timeout=180))
clean = [e for e in data
         if e.get("output", "").strip() and len(e["instruction"]) < 200 and len(e["output"]) < 800]
random.Random(0).shuffle(clean)
heldout = clean[-300:]
train = clean[:-400]                              # everything except a disjoint tail

json.dump(train, open(f"{D}/alpaca_benign_train_big.json", "w"), ensure_ascii=False, indent=2)
json.dump(heldout, open(f"{D}/alpaca_benign_heldout_big.json", "w"), ensure_ascii=False, indent=2)
print(f"alpaca big: {len(data)} total -> {len(clean)} clean -> train {len(train)} + heldout {len(heldout)} "
      f"(disjoint) -> {len(train)//8} examples/node available at N=8")
