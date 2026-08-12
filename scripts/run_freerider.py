"""
FREE-RIDER CONTROL for the detP headline (closes the attacker confound).

The attacker does TWO things at once: (a) trains on poison, (b) ignores incoming gossip while still
broadcasting. Nothing on disk separates them, so "detP alarms around round 12" could be detecting
either. This run is (b) WITHOUT (a): node c0 is excluded from merging exactly like an attacker but
its shard is plain benign, same size as everyone else's. ZERO attackers -> no poison anywhere.

Config is the R=25 sweep's alpha=inf / noatt arm, one variable changed (c0 -> free-rider), rounds=15.
Usage: python run_freerider.py [rounds]      (default 15; use 2 for the smoke test)
"""
import sys, os, json
os.environ["ALPACA_TRAIN"] = "/mnt/nfs/home/peechara/data/train/alpaca_benign_train_big.json"
os.environ["ALPACA_HELDOUT"] = "/mnt/nfs/home/peechara/data/train/alpaca_benign_heldout_big.json"
sys.path.insert(0, "/mnt/nfs/home/peechara/iclscan-decentralized/src/sim")
from gossip_sim import run

ROUNDS = int(sys.argv[1]) if len(sys.argv) > 1 else 15
SEED = 0
OUT = "/mnt/nfs/home/peechara/iclscan-decentralized/results/noniid/freerider"
tag = f"r{ROUNDS}_alphainf_freerider_seed{SEED}"
CKPT = f"/mnt/nfs/home/peechara/ckpts/{tag}"
os.makedirs(OUT, exist_ok=True)

# identical to run_r20_sweep.py's `noatt` arm except freerider_ids=(0,)
CFG = dict(n_clients=8, attacker_ids=(), freerider_ids=(0,), alpha=float("inf"), rounds=ROUNDS,
           local_steps=25, bs=8, fixed_size=4000,
           poison_per_attacker=0, replace_poison=False,
           probe_n=30, asr_n=20, max_new_tokens=48, seed=SEED,
           probe_seed=0,
           post_probe=False, ckpt_dir=CKPT)

print(f"========== {tag}  (alpha=inf, FREE-RIDER c0, no attackers, R={ROUNDS}, seed={SEED}) ==========",
      flush=True)
print(f"config: {CFG}", flush=True)
hist = run(ckpt_path=f"{OUT}/{tag}.partial.json", **CFG)
json.dump({"tag": tag, "config": {k: str(v) for k, v in CFG.items()}, "history": hist},
          open(f"{OUT}/{tag}.json", "w"), indent=2)
print(f"SAVED {tag}\nFREERIDER RUN DONE", flush=True)
