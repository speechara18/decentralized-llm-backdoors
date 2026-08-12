"""
THE converged-regime attacker run (adopted config, advisor-approved). Single attacker c0, alpha=inf,
big Alpaca pool. Converged budget: 4000/node, bs=8, K=25, R=15 -> 0.75 cumulative epochs (the verified
healthy learning curve). Replacement poison: 3400 benign + 600 poison = 4000 (ppoison=0.15) so the
backdoor should CLIMB over ~r2-6 instead of pinning at 100% in r1. ckpt_dir ON for ALL 8 nodes
(Phase-5 needs the actual attacker adapter; neighbor-alignment needs benign adapters). B3 gen-seeding
is automatic in run(). post_probe OFF -> post-agg CE/detP/ASR computed retroactively from the ckpts.

Success criteria to check: (1) held-out CE descends/flat; (2) c0 ASR climbs r2-6; (3) ASR propagates
neighbors{1,4,7} then non-neighbors{2,3,5,6}.
"""
import sys, os, json
os.environ["ALPACA_TRAIN"] = "/mnt/nfs/home/peechara/data/train/alpaca_benign_train_big.json"
os.environ["ALPACA_HELDOUT"] = "/mnt/nfs/home/peechara/data/train/alpaca_benign_heldout_big.json"
sys.path.insert(0, "/mnt/nfs/home/peechara/iclscan-decentralized/src/sim")
from gossip_sim import run

OUT = "/mnt/nfs/home/peechara/iclscan-decentralized/results/noniid/converged"
CKPT = "/mnt/nfs/home/peechara/ckpts/converged_attack_inf"
os.makedirs(OUT, exist_ok=True)
tag = "converged_attack_inf"
CFG = dict(n_clients=8, attacker_ids=(0,), alpha=float("inf"), rounds=15, local_steps=25, bs=8,
           fixed_size=4000, poison_per_attacker=600, replace_poison=True,   # 3400 benign + 600 poison
           probe_n=30, asr_n=20, max_new_tokens=48, seed=0, post_probe=False, ckpt_dir=CKPT)

print(f"========== {tag}: converged regime, single attacker, alpha=inf ==========", flush=True)
print(f"config: {CFG}", flush=True)
hist = run(ckpt_path=f"{OUT}/{tag}.partial.json", **CFG)
json.dump({"tag": tag, "config": {k: str(v) for k, v in CFG.items()}, "history": hist},
          open(f"{OUT}/{tag}.json", "w"), indent=2)
print(f"SAVED {tag}\nCONVERGED ATTACK DONE", flush=True)
