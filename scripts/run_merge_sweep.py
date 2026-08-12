"""
MERGING-ATTACKER sweep. Identical to run_r20_sweep.py in every respect except one:
the attacker MERGES incoming gossip instead of ignoring it (attacker_merges=True).

WHY. In every run so far c0 ignores incoming gossip, so it trains on its own 4000-example
shard 25 times over. It is the only node that over-fits: held-out loss rises +0.041 (alpha=inf)
and +0.115 (alpha=0.1) from its own minimum, while benign held-out loss is flat (+0.001).
That is attacker-specific, and it is the leading explanation for the unexplained decay of
c0's own detP from 83.3 (r1) to ~40 (r25) -- an over-fit model is rigid and less steerable
by in-context demonstrations, so its ICL-susceptibility score falls.

Making the attacker merge is a clean on/off switch for exactly that variable, and it makes
two mechanisms predict OPPOSITE things about the same number:

  H1 over-fitting  -> c0 stops over-fitting -> its detP STAYS HIGH (no decay)
  H3 laundering    -> merging dilutes the backdoor -> c0's detP FALLS into the
                      propagated band [20,29] and it becomes unattributable

Whichever way c0's detP moves discriminates them. See notes/PREREG_merge_attacker.md for
the pre-registered thresholds -- those were written and committed BEFORE any number here.

Everything else is byte-identical to the R=25 baseline so the comparison is round-for-round:
4000/node, bs8, K=25, R=25, lr 2e-4, ppoison 0.15 replacement, probe_n 30, asr_n 20,
probe_seed 0, ckpt_dir on for all 8 nodes.

Usage: python run_merge_sweep.py <inf|0.5|0.1> [seed] [rounds]

R DEFAULTS TO 25 DELIBERATELY. At ~25 min/round a 25-round condition is ~10.4 GPU-h; R=30
is ~12.5 GPU-h. Matched rounds are worth more than extra rounds here -- the whole value is
a controlled comparison against a baseline that stops at 25. If merging slows propagation
so much that infection is incomplete by r25, that IS the result (absorption costing the
attacker propagation speed), not a failure. ckpt_dir is on, so extending later is a resume.
"""
import sys, os, json
os.environ["ALPACA_TRAIN"] = "/mnt/nfs/home/peechara/data/train/alpaca_benign_train_big.json"
os.environ["ALPACA_HELDOUT"] = "/mnt/nfs/home/peechara/data/train/alpaca_benign_heldout_big.json"
sys.path.insert(0, "/mnt/nfs/home/peechara/iclscan-decentralized/src/sim")
from gossip_sim import run

ALAB = sys.argv[1]                                   # "inf" | "0.5" | "0.1"
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 0
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 25
alpha = float("inf") if ALAB == "inf" else float(ALAB)

# Separate output tree: these are NOT drop-in replacements for the baseline r20/ results and
# must never be mistaken for them in a later glob.
BASE = "/mnt/nfs/home/peechara/iclscan-decentralized/results/noniid/merge_attacker"
OUT = BASE if SEED == 0 else f"{BASE}_seed{SEED}"
tag = f"r{ROUNDS}_alpha{ALAB}_attmerge_seed{SEED}"
CKPT = f"/mnt/nfs/home/peechara/ckpts/{tag}"
os.makedirs(OUT, exist_ok=True)

CFG = dict(n_clients=8, attacker_ids=(0,), alpha=alpha, rounds=ROUNDS,
           local_steps=25, bs=8, fixed_size=4000,
           poison_per_attacker=600, replace_poison=True,
           probe_n=30, asr_n=20, max_new_tokens=48, seed=SEED,
           probe_seed=0,                              # fixed detector, as in the baseline
           post_probe=False, ckpt_dir=CKPT,
           attacker_merges=True)                      # <-- THE ONLY CHANGE vs run_r20_sweep.py

print(f"========== {tag}  (alpha={alpha}, ATTACKER *MERGES*, R={ROUNDS}, seed={SEED}) ==========",
      flush=True)
print(f"config: {CFG}", flush=True)
print("baseline for comparison: results/noniid/r20/r25_alpha%s_att_seed0.json" % ALAB, flush=True)

# ckpt_path writes partial history every round. H100 TRAIN workloads are PREEMPTIBLE with no
# resume, so a kill at r20 must still leave 20 usable rounds on NFS rather than nothing.
hist = run(ckpt_path=f"{OUT}/{tag}.partial.json", **CFG)
json.dump({"tag": tag, "config": {k: str(v) for k, v in CFG.items()}, "history": hist},
          open(f"{OUT}/{tag}.json", "w"), indent=2)
print(f"SAVED {tag}\nMERGE SWEEP DONE", flush=True)
