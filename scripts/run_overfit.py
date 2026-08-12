"""
Experiment B: long-epoch BENIGN convergence run to locate the overfitting onset.
Trains ONE LoRA at the HP sweep's best trial (lr=8.65e-5, warmup=0.026, wd=0.01) for many
epochs and logs TRAIN-slice + HELD-OUT loss densely, so we can read the epoch where held-out
turns up (the U). That epoch = the benign training budget the threat model gets before
overfitting shows. Reuses tune_convergence.train_lora exactly (no reimplementation).
Checkpoints the trace to NFS after EVERY eval (idle-suspend / pre-empt safe).

Usage: python run_overfit.py [epochs] [n_data]   (default 12 epochs, 4000)
"""
import sys
import os
import json
import random
os.environ["ALPACA_TRAIN"] = "/mnt/nfs/home/peechara/data/train/alpaca_benign_train_big.json"
os.environ["ALPACA_HELDOUT"] = "/mnt/nfs/home/peechara/data/train/alpaca_benign_heldout_big.json"
sys.path.insert(0, "/mnt/nfs/home/peechara/iclscan-decentralized/scripts")
sys.path.insert(0, "/mnt/nfs/home/peechara/iclscan-decentralized/src/sim")
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from tune_convergence import build_lora_model, train_lora, OUT
from decentralized import BASE
from gossip_sim import ALPACA_TRAIN, ALPACA_HELDOUT   # *_big pool via the env above

EPOCHS = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
NDATA = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
BS = 8
LR, WARMUP, WD = 8.65e-5, 0.026, 0.01                 # HP sweep's best trial
OUTJSON = f"{OUT}/overfit_benign.json"
os.makedirs(OUT, exist_ok=True)

tok = AutoTokenizer.from_pretrained(BASE); tok.pad_token = tok.eos_token; tok.padding_side = "right"
data = json.load(open(ALPACA_TRAIN)); random.Random(0).shuffle(data)
shard = data[:NDATA]
train_slice = shard[:100]
held = json.load(open(ALPACA_HELDOUT))[:100]
collator = DataCollatorForSeq2Seq(tok, label_pad_token_id=-100, padding=True)

model = build_lora_model(tok, lora_r=8, dropout=0.0)
trace = []


def log(h):
    trace.append([h["epoch"], h["train"], h["held"]])
    json.dump({"lr": LR, "warmup": WARMUP, "wd": WD, "bs": BS, "n": NDATA, "epochs": EPOCHS,
               "trace": trace}, open(OUTJSON, "w"), indent=2)      # checkpoint each eval
    print(f"[ep {h['epoch']:5.2f} step {h['step']:>4}] train={h['train']:.3f}  "
          f"held={h['held']:.3f}  gap={h['held']-h['train']:+.3f}", flush=True)


steps_per_epoch = -(-NDATA // BS)
total_steps = int(EPOCHS * steps_per_epoch)
eval_pts = max(20, total_steps // 200)                 # ~ every 200 steps
print(f"=== overfit B: lr={LR} warmup={WARMUP} wd={WD} bs={BS} n={NDATA} "
      f"epochs={EPOCHS} ({total_steps} steps) ===", flush=True)
train_lora(model, tok, shard, train_slice, held, LR, EPOCHS, BS, WD, WARMUP, collator,
           eval_points=eval_pts, log=log)
print("OVERFIT B DONE", flush=True)
