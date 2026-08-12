"""
R=20 sweep runner (1 epoch = 20*25*8/4000). Thin wrapper over the CONFIRMED gossip_sim.run() --
no core/hyperparam changes vs converged_attack. Selects skew, scenario (attacker/no-attacker),
and training seed from argv so each condition is one process (one GPU).

Usage: python run_r20_sweep.py <inf|0.5|0.1> <att|noatt> [seed] [rounds]
  att   -> attacker_ids=(0,), ppoison=0.15 replacement (3400 benign + 600 poison)
  noatt -> attacker_ids=(),  no poison (clean control for ROC negatives / FPR)
"""
import sys, os, json
os.environ["ALPACA_TRAIN"] = "/mnt/nfs/home/peechara/data/train/alpaca_benign_train_big.json"
os.environ["ALPACA_HELDOUT"] = "/mnt/nfs/home/peechara/data/train/alpaca_benign_heldout_big.json"
sys.path.insert(0, "/mnt/nfs/home/peechara/iclscan-decentralized/src/sim")
from gossip_sim import run

ALAB = sys.argv[1]                                   # "inf" | "0.5" | "0.1"
SCEN = sys.argv[2]                                   # "att" | "noatt"
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
ROUNDS = int(sys.argv[4]) if len(sys.argv) > 4 else 20
alpha = float("inf") if ALAB == "inf" else float(ALAB)
att = SCEN == "att"

# seed>0 writes to a SEPARATE dir so seed-0 results are never touched
BASE = "/mnt/nfs/home/peechara/iclscan-decentralized/results/noniid/r20"
OUT = BASE if SEED == 0 else f"{BASE}_seed{SEED}"
tag = f"r{ROUNDS}_alpha{ALAB}_{SCEN}_seed{SEED}"
CKPT = f"/mnt/nfs/home/peechara/ckpts/{tag}"          # tag includes seed -> separate ckpt tree
os.makedirs(OUT, exist_ok=True)

CFG = dict(n_clients=8, attacker_ids=(0,) if att else (), alpha=alpha, rounds=ROUNDS,
           local_steps=25, bs=8, fixed_size=4000,
           poison_per_attacker=600 if att else 0, replace_poison=att,
           probe_n=30, asr_n=20, max_new_tokens=48, seed=SEED,
           probe_seed=0,                              # detector prompts pinned to seed-0 (fixed detector)
           post_probe=False, ckpt_dir=CKPT)

print(f"========== {tag}  (alpha={alpha}, {'ATTACKER' if att else 'no-attacker'}, "
      f"R={ROUNDS}, seed={SEED}) ==========", flush=True)
print(f"config: {CFG}", flush=True)
hist = run(ckpt_path=f"{OUT}/{tag}.partial.json", **CFG)
json.dump({"tag": tag, "config": {k: str(v) for k, v in CFG.items()}, "history": hist},
          open(f"{OUT}/{tag}.json", "w"), indent=2)
print(f"SAVED {tag}\nR20 RUN DONE", flush=True)
