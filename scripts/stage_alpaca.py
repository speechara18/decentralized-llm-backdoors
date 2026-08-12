"""Stage a larger benign pool from Stanford Alpaca (the paper's own benign source) so
each of 8 nodes gets ~500 competent-node examples. Train/held-out split kept DISJOINT
(held-out used for probe queries + the convergence-loss metric, so no leakage)."""
import json
import urllib.request
import random
import os

URL = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
D = "/mnt/nfs/home/peechara/data/train"
os.makedirs(D, exist_ok=True)

data = json.load(urllib.request.urlopen(URL, timeout=120))
# keep short-ish, non-empty examples (comparable to BackdoorLLM's Alpaca-derived benign)
clean = [e for e in data
         if e.get("output", "").strip() and len(e["instruction"]) < 200 and len(e["output"]) < 800]
random.Random(0).shuffle(clean)
train, heldout = clean[:4000], clean[4000:4300]

json.dump(train, open(f"{D}/alpaca_benign_train.json", "w"), ensure_ascii=False, indent=2)
json.dump(heldout, open(f"{D}/alpaca_benign_heldout.json", "w"), ensure_ascii=False, indent=2)
print(f"alpaca: {len(data)} total -> {len(clean)} clean -> train {len(train)} + heldout {len(heldout)} (disjoint)")
print("train sample:", {k: str(v)[:60] for k, v in train[0].items()})
