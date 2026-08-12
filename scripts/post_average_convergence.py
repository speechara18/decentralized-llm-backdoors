"""
Cheap convergence diagnostic (no new training). Question: is the IID system actually converging,
and does the equal-shards held-out-loss curve look bad only because we plot the PRE-average,
per-node-overfit adapter? Here we load the saved per-round adapters, form each node's POST-
neighbor-average model (the exact gossip_average op), and compare held-out loss pre vs post.
Uses the no-attacker equal-shards checkpoints (pure clean nodes). Needs a GPU.

Usage: python post_average_convergence.py [round]   (default: latest checkpointed round)
Reads ckpts/noatt_alpha{inf,0.5,0.1}_R15_fixed500/r{round}/c{i}/c{i}/  (adapters, saved pre-average).
"""
import sys
import os
import json
import numpy as np
import torch
sys.modules["deepspeed"] = None
BASE_DIR = "/mnt/nfs/home/peechara/iclscan-decentralized"
sys.path.insert(0, f"{BASE_DIR}/src/sim")
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from decentralized import BASE
from gossip_sim import three_regular_8, gossip_average, held_out_loss, ALPACA_HELDOUT

CKROOT = "/mnt/nfs/home/peechara/ckpts"
clients = [f"c{i}" for i in range(8)]
adj = three_regular_8()
tok = AutoTokenizer.from_pretrained(BASE); tok.pad_token = tok.eos_token; tok.padding_side = "right"
held = json.load(open(ALPACA_HELDOUT))[:30]


def latest_round(ckdir):
    rs = [int(d[1:]) for d in os.listdir(ckdir) if d.startswith("r") and d[1:].isdigit()
          and os.path.exists(f"{ckdir}/{d}/c7/c7")]
    return max(rs) if rs else None


print(f"{'alpha':>8} | {'round':>5} | {'pre-avg held':>12} | {'post-avg held':>13} | {'delta':>7}")
for alab in ["inf", "0.5", "0.1"]:
    ckdir = f"{CKROOT}/noatt_alpha{alab}_R15_fixed500"
    if not os.path.isdir(ckdir):
        print(f"{alab:>8} | (no checkpoint dir yet)"); continue
    r = int(sys.argv[1]) if len(sys.argv) > 1 else latest_round(ckdir)
    if r is None:
        print(f"{alab:>8} | (no complete round yet)"); continue
    ck = f"{ckdir}/r{r}"
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16).to("cuda")
    model = PeftModel.from_pretrained(base, f"{ck}/c0/c0", adapter_name="c0")
    for i in range(1, 8):
        model.load_adapter(f"{ck}/c{i}/c{i}", adapter_name=f"c{i}")
    model.eval()
    pre = {}
    for c in clients:
        model.set_adapter(c); pre[c] = held_out_loss(model, tok, held)
    gossip_average(model, clients, adj, set())      # neighbour-average in place (no attackers)
    post = {}
    for c in clients:
        model.set_adapter(c); post[c] = held_out_loss(model, tok, held)
    mpre, mpost = np.mean(list(pre.values())), np.mean(list(post.values()))
    print(f"{alab:>8} | {r:>5} | {mpre:>12.3f} | {mpost:>13.3f} | {mpost - mpre:>+7.3f}")
    del model, base
    torch.cuda.empty_cache()
print("\nRead: if IID post-avg << pre-avg (and lower than the skewed post-avg), the system IS converging "
      "at IID and the pre-average curve just reflects per-node overfitting to the 500-example shard.")
