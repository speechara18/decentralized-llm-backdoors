"""
Centralized replica of the ORIGINAL decentralized local-train config, to see how loss looks
when the same tiny per-round budget is run WITHOUT gossip/averaging/adapter-reset.
Matches decentralized.py exactly: lr=2e-4, bs=2, plain AdamW (no weight decay), lora_dropout=0,
r=8, and 200 gradient steps. Runs two data sizes (n=500 and n=4000) in one process (model loaded
once, adapter reset between) and logs train-slice + held-out loss densely vs STEP (the fair axis:
both do 200 steps). Reuses tune_convergence.train_lora. Checkpoints each run's trace to NFS.

Usage: python run_central_match.py
"""
import sys
import os
import json
import random
os.environ["ALPACA_TRAIN"] = "/mnt/nfs/home/peechara/data/train/alpaca_benign_train_big.json"
os.environ["ALPACA_HELDOUT"] = "/mnt/nfs/home/peechara/data/train/alpaca_benign_heldout_big.json"
sys.path.insert(0, "/mnt/nfs/home/peechara/iclscan-decentralized/scripts")
sys.path.insert(0, "/mnt/nfs/home/peechara/iclscan-decentralized/src/sim")
import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from tune_convergence import build_lora_model, train_lora, OUT
from decentralized import BASE
from gossip_sim import ALPACA_TRAIN, ALPACA_HELDOUT

STEPS, BS, LR, WD, WARMUP = 200, 2, 2e-4, 0.0, 0.0     # == decentralized local_train config
os.makedirs(OUT, exist_ok=True)

tok = AutoTokenizer.from_pretrained(BASE); tok.pad_token = tok.eos_token; tok.padding_side = "right"
data = json.load(open(ALPACA_TRAIN)); random.Random(0).shuffle(data)
held = json.load(open(ALPACA_HELDOUT))[:100]           # SAME held set for both n
collator = DataCollatorForSeq2Seq(tok, label_pad_token_id=-100, padding=True)

model = build_lora_model(tok, lora_r=8, dropout=0.0)
init_state = {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k}

for n in (500, 4000):
    model.load_state_dict(init_state, strict=False)    # reset adapter to fresh init
    shard = data[:n]
    train_slice = shard[:100]
    steps_per_epoch = -(-n // BS)
    epochs = STEPS / steps_per_epoch                   # -> exactly STEPS gradient steps
    outjson = f"{OUT}/central_match_n{n}.json"
    trace = []

    def log(h, _out=outjson, _tr=trace, _n=n):
        _tr.append([h["step"], h["train"], h["held"]])
        json.dump({"n": _n, "lr": LR, "bs": BS, "steps": STEPS, "trace": _tr},
                  open(_out, "w"), indent=2)
        print(f"[n={_n} step {h['step']:>3} ep {h['epoch']:.3f}] train={h['train']:.3f}  "
              f"held={h['held']:.3f}  gap={h['held']-h['train']:+.3f}", flush=True)

    print(f"=== centralized match: n={n}, {STEPS} steps, bs={BS}, lr={LR}, no wd/dropout "
          f"({epochs:.2f} epochs) ===", flush=True)
    train_lora(model, tok, shard, train_slice, held, LR, epochs, BS, WD, WARMUP, collator,
               eval_points=20, log=log)
    torch.cuda.empty_cache()

print("CENTRAL MATCH DONE", flush=True)
