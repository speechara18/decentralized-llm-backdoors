"""
DEFENSE IN THE LOOP. The experiment that decides whether this is a defense paper.

Everything measured so far answers "can we see it?". Nothing yet answers "can we stop it?".
This closes the loop: each node screens the adapters it RECEIVES and refuses to merge any whose
detP exceeds delta, and we measure whether network ASR actually falls against the undefended
baseline that already exists on disk.

WHAT THE EVIDENCE SAYS THIS SHOULD DO
  - The attacker reads detP 83.3 at round 1 against 8.6 for benign adapters, and never drops
    below delta=25 in rounds 1-12 at ANY skew (minima 36.7 / 26.7 / 26.7).
  - False positives over 600 clean node-rounds: 0.50%; over the 288 node-rounds in r1-12: 0.00%.
  - So in r1-12 the screen should quarantine c0's three out-edges and quarantine almost nothing
    else. If ASR does not fall, the failure is in the RESPONSE, not the detector.

TWO PROBE REGIMES, AND WHY THE CHEAP ONE RUNS FIRST
  --probe-source global : detP of adapter j is receiver-independent (the prompt set is shared and
      `seed` pins it), so P[cj] -- already computed each round -- IS what a receiver would measure.
      Screening adds an `if`, not a probe. A defended run costs the SAME as an undefended one
      (~5 GPU-h). This tests the core question: does screening reduce ASR?
  --probe-source shard  : each node probes with a slice of its own training shard, as ARGUS does.
      Receivers now disagree, which is what makes corroboration real -- but it costs 21 real
      probes/round (7 honest nodes x 3 in-edges), roughly 2.2x the generation load and ~11-12
      GPU-h per run.
  Run `global` first. Per-node pools are a refinement of the corroboration story; they are not
  required to answer whether the defense works at all, and they cost 2x to find out.

RESPONSE POLICY
  --strikes k  consecutive flags before quarantine (1 = memoryless)
  --release m  clean rounds before release (0 = permanent quarantine)
  Default 1/1. A single 30-prompt probe has SE ~7.9pp, so near delta the screen is noisy; the
  strike counter is the knob that trades detection latency against false-positive damage, and it
  is the curve to report rather than a point.

WHAT TO COMPARE AGAINST
  results/noniid/r20/r25_alpha{ALAB}_att_seed0.json -- same seed, same shards, same config,
  round-for-round. The ONLY difference is the screen.

Usage: python run_defense.py <inf|0.5|0.1> [seed] [rounds]
       python run_defense.py inf 0 25 --probe-source shard --strikes 2
"""
import sys, os, json, argparse
os.environ["ALPACA_TRAIN"] = "/mnt/nfs/home/peechara/data/train/alpaca_benign_train_big.json"
os.environ["ALPACA_HELDOUT"] = "/mnt/nfs/home/peechara/data/train/alpaca_benign_heldout_big.json"
sys.path.insert(0, "/mnt/nfs/home/peechara/iclscan-decentralized/src/sim")
from gossip_sim import run

ap = argparse.ArgumentParser()
ap.add_argument("alpha_label", choices=["inf", "0.5", "0.1"])
ap.add_argument("seed", nargs="?", type=int, default=0)
ap.add_argument("rounds", nargs="?", type=int, default=24,
                help="24 not 25: with 160 held out for probing, 3840 x 24 x 200/round = EXACTLY 1.250 "
                     "epochs, matching the baseline 4000 @ R=25. Epoch-matching beats "
                     "round-matching because over-training is what drives the detP decay. "
                     "Propagation is unaffected: baseline benign mean ASR is already "
                     "97.9/99.3/99.3 by r24.")
ap.add_argument("--probe-source", default="global", choices=["global", "shard", "iid", "skewed"])
ap.add_argument("--delta", type=float, default=25.0,
                help="ICLScan sets this from alpha_icl=1/3 -> 1/4. Do not tune it to the data.")
ap.add_argument("--strikes", type=int, default=1)
ap.add_argument("--release", type=int, default=1, help="0 = permanent quarantine")
ap.add_argument("--attacker-merges", action="store_true")
ap.add_argument("--consult", default="mean", choices=["none", "mean", "majority", "any"],
                help="what a node does when its own score lands within --band of delta, i.e. when "
                     "one 30-prompt probe cannot resolve the sign of the decision. 'mean' pools "
                     "the co-receivers' independent reads; 'none' is unilateral screening. Costs "
                     "no extra probes -- the co-receivers already probed for their own gating.")
ap.add_argument("--triggers", default="",
                help="comma-separated ICL triggers assigned round-robin to nodes for SCREENING. "
                     "Empty = every node uses Placid, as in all prior runs. "
                     "'Placid,123456,ctfqxy,Umbral,Zephyr' are the five verified by "
                     "scripts/trigger_diversity.py -- all separate attacker from clean at round 1 "
                     "in both skews. Per-node triggers remove the single token an attacker could "
                     "train against, which is the second half of ARGUS's objection.")
ap.add_argument("--trigger-mode", default="random", choices=["random", "rainbow"],
                help="random = each node samples its trigger independently, no coordination -- the "
                     "deployment model, and the only variant that answers BOTH halves of ARGUS's "
                     "objection. rainbow = a coordinated collision-free assignment, a best case "
                     "that needs topology knowledge; report as an upper bound, not the method.")
ap.add_argument("--band", type=float, default=7.9,
                help="uncertainty half-width. Default is the SE of one 30-prompt probe at p~0.25.")
ap.add_argument("--pool-size", type=int, default=160,
                help="held OUT of training; 80 demos + 80 queries -> 84%% unique coverage of a "
                     "30-prompt probe. 160 is chosen so that 4000-160=3840 with R=24 gives "
                     "EXACTLY 1.250 epochs, matching the baseline. Raising it to 200 would "
                     "push R=24 to 1.263, over the 1.25 cap.")
args = ap.parse_args()

ALAB, SEED, ROUNDS = args.alpha_label, args.seed, args.rounds
alpha = float("inf") if ALAB == "inf" else float(ALAB)
release = None if args.release == 0 else args.release

BASE = "/mnt/nfs/home/peechara/iclscan-decentralized/results/noniid/defense"
OUT = BASE if SEED == 0 else f"{BASE}_seed{SEED}"
tag = (f"r{ROUNDS}_alpha{ALAB}_defended_{args.probe_source}"
       f"_d{int(args.delta)}_s{args.strikes}r{args.release}_{args.consult}"
       f"{('_' + args.trigger_mode + 'trig') if args.triggers else ''}"
       f"{'_attmerge' if args.attacker_merges else ''}_seed{SEED}")
os.makedirs(OUT, exist_ok=True)

CFG = dict(n_clients=8, attacker_ids=(0,), alpha=alpha, rounds=ROUNDS,
           local_steps=25, bs=8, fixed_size=4000,
           poison_per_attacker=600, replace_poison=True,
           probe_n=30, asr_n=20, max_new_tokens=48, seed=SEED,
           probe_seed=0, post_probe=False,
           ckpt_dir=f"/mnt/nfs/home/peechara/ckpts/{tag}",
           attacker_merges=args.attacker_merges,
           screen_delta=args.delta, screen_strikes=args.strikes, screen_release=release,
           probe_source=args.probe_source, probe_pool_size=args.pool_size,
           screen_consult=None if args.consult == "none" else args.consult,
           screen_band=args.band,
           probe_triggers=[t for t in args.triggers.split(",") if t] or None,
           probe_trigger_mode=args.trigger_mode)

print(f"========== {tag} ==========", flush=True)
print(f"DEFENDED run. delta={args.delta} strikes={args.strikes} "
      f"release={'permanent' if release is None else release} probe_source={args.probe_source}",
      flush=True)
print(f"compare against: results/noniid/r20/r25_alpha{ALAB}_att_seed0.json (undefended, same seed)",
      flush=True)
if args.probe_source == "global":
    print("NOTE: global pool -> screening is FREE (P[cj] is receiver-independent). Same cost as "
          "an undefended run.", flush=True)
else:
    print(f"NOTE: per-node pools -> 21 real probes/round on top of the usual 8. Expect ~2.2x the "
          f"generation load.", flush=True)
print(f"config: {CFG}", flush=True)

hist = run(ckpt_path=f"{OUT}/{tag}.partial.json", **CFG)
json.dump({"tag": tag, "config": {k: str(v) for k, v in CFG.items()}, "history": hist},
          open(f"{OUT}/{tag}.json", "w"), indent=2)

# ---- immediate readout: did it work? ----------------------------------------------------
BEN = [f"c{i}" for i in range(1, 8)]
try:
    base = json.load(open(f"/mnt/nfs/home/peechara/iclscan-decentralized/results/noniid/r20/"
                          f"r25_alpha{ALAB}_att_seed0.json"))["history"]
except Exception:
    base = None

print("\n=== DEFENSE READOUT ===", flush=True)
print("  rnd | benign ASR (def / base) | c0 detP | edges rejected", flush=True)
for r in hist:
    i = r["round"] - 1
    bd = sum(r["asr"][c] for c in BEN) / 7
    bb = (sum(base[i]["asr"][c] for c in BEN) / 7) if base and i < len(base) else float("nan")
    print(f"   {r['round']:2d} | {bd:6.1f} / {bb:6.1f}          | {r['P']['c0']:5.1f} | "
          f"{len(r.get('screened', {}))}", flush=True)

fin_d = sum(hist[-1]["asr"][c] for c in BEN) / 7
fin_b = (sum(base[len(hist)-1]["asr"][c] for c in BEN) / 7) if base else float("nan")
tp = sum(1 for r in hist for k in r.get("screened", {}) if k.endswith("<-c0"))
fp = sum(1 for r in hist for k in r.get("screened", {}) if not k.endswith("<-c0"))
print(f"\n  final benign ASR: defended {fin_d:.1f} vs undefended {fin_b:.1f} "
      f"({fin_d-fin_b:+.1f})", flush=True)
print(f"  attacker edges quarantined: {tp}   honest edges quarantined (FALSE POSITIVES): {fp}",
      flush=True)
ce_def = sum(hist[-1]["loss"][c] for c in BEN) / 7
ce_base = (sum(base[len(hist)-1]["loss"][c] for c in BEN) / 7) if base else None
ce_cmp = f" vs undefended {ce_base:.3f} ({ce_def-ce_base:+.3f})" if ce_base is not None else ""
print(f"  benign held-out CE at r{len(hist)}: {ce_def:.3f}{ce_cmp}", flush=True)
print("\n  Read it this way: ASR must FALL and honest-edge quarantines must stay near zero. A "
      "defense that stops the backdoor by disconnecting the graph is not a defense -- check the "
      "held-out CE for that.", flush=True)
print(f"\nSAVED {tag}\nDEFENSE RUN DONE", flush=True)
