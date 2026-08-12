"""
Early-window dense detP re-probe.

WHY. The n=100 re-probe (reprobe_dense.py) covers rounds 20-25 only. The early-warning
claim -- that a node can tell its neighbourhood is compromised at round ~12, BEFORE any
node has a functional backdoor (first ASR>50 lands at round 14) -- currently rests on
detP at n_prompts=30, whose binomial SE at p=.10 is ~5.5pp. This re-probes the early
window at n=100 to check the claim survives a better estimator.

It matters which way it goes. In the LATE window, going 30 -> 100 made the node-level
threshold test WORSE (TPR 70.2/18.8/27.1 -> 31.9/2.1/14.6) and the network-level
separation PERFECT (AUC 0.989 -> 1.000). If the early window behaves the same way, the
early-warning result is real and the defense has a measured operating point. If the
network-level AUC degrades at n=100, the round-12 alarm was probe noise and the defense
is dead.

DETECTOR IS IDENTICAL to reprobe_dense.py and to the live run: Placid trigger, ICL
alpha=1/3, big demo/query pools, top-p 0.9 sampling, is_refusal, seed=0,
gen_seed = r*1000 + node. So the first 30 of the 100 reproduce the live detP_30 exactly
and the extra 70 densify. Any difference is estimator variance, not a different measurement.

COST. Each probe is n_prompts generations at bs1, max_new_tokens=48. The full rounds 8-14
grid is 6 runs x 7 rounds x 8 nodes = 336 probes = 33,600 generations, which is ~9-15 GPU-h
and exceeds the 12h interactive cap. DEFAULT here is the decisive minimum: rounds {10,12,14},
= 144 probes = 14,400 generations. Round 12 is the claim; 10 and 14 bracket it. Override
with --rounds if there is budget.

Inference only. No training. Resumes from the output JSON after every round, so an
idle-suspend or preemption costs at most one round.

Usage (in-pod):
    python reprobe_early.py                      # rounds 10,12,14 -- the decisive set
    python reprobe_early.py --rounds 8,10,12,14  # adds curve shape
    python reprobe_early.py --estimate-only      # time one probe, print cost, exit
"""
import argparse
import glob
import json
import os
import sys
import time

NFS = "/mnt/nfs/home/peechara"
os.environ.setdefault("ALPACA_TRAIN", f"{NFS}/data/train/alpaca_benign_train_big.json")
os.environ.setdefault("ALPACA_HELDOUT", f"{NFS}/data/train/alpaca_benign_heldout_big.json")
sys.path.insert(0, f"{NFS}/iclscan-decentralized/src/sim")
sys.path.insert(0, f"{NFS}/iclscan-decentralized/src/detect")

import torch                                                    # noqa: E402
from safetensors.torch import load_file                         # noqa: E402
from transformers import (AutoTokenizer, AutoModelForCausalLM,  # noqa: E402
                          GenerationConfig)
from peft import PeftModel                                      # noqa: E402
from decentralized import BASE                                  # noqa: E402
from gossip_sim import ALPACA_TRAIN, ALPACA_HELDOUT             # noqa: E402
from probe import paper_faithful_probe                          # noqa: E402

CK = f"{NFS}/ckpts"
# Ordered as MATCHED PAIRS, attacker immediately followed by its clean control.
# The loop is run-major, so a partial completion under this order still yields a
# usable att/noatt pair and therefore a computable AUC. The previous order put all
# three attacker runs first; a run that stopped at 25% produced attacked-network
# data with no clean controls at all -- i.e. nothing answerable. Do not regroup.
RUNS = ["r25_alphainf_att_seed0", "r25_alphainf_noatt_seed0",
        "r25_alpha0.5_att_seed0", "r25_alpha0.5_noatt_seed0",
        "r25_alpha0.1_att_seed0", "r25_alpha0.1_noatt_seed0"]
ap = argparse.ArgumentParser()
ap.add_argument("--rounds", default="10,12,14",
                help="comma-separated rounds (default 10,12,14 = the decisive minimum)")
ap.add_argument("--nodes", default="1,2,3,4,5,6,7",
                help="comma-separated node indices. Default excludes c0: it is the "
                     "attacker and plays no part in the network-level question (can a "
                     "BENIGN node tell it sits in a compromised network), so probing it "
                     "buys nothing and costs 1/8 of the budget. Pass 0,1,..,7 for all.")
ap.add_argument("--n-prompts", type=int, default=100)
ap.add_argument("--out", default=f"{NFS}/iclscan-decentralized/results/noniid/"
                                 "reprobe100_early_seed0.json")
ap.add_argument("--estimate-only", action="store_true",
                help="time a single probe, print the projected cost, write nothing")
args = ap.parse_args()
ROUNDS = [int(x) for x in args.rounds.split(",") if x.strip()]
NODES = [f"c{int(x)}" for x in args.nodes.split(",") if x.strip()]


def adir(run, r, c):
    """Locate a saved adapter. Raises with a readable message if absent -- checkpoint
    coverage for the early rounds is the one thing not documented in PROGRESS.md."""
    hits = glob.glob(f"{CK}/{run}/r{r}/{c}/**/adapter_model.safetensors", recursive=True)
    if not hits:
        raise FileNotFoundError(f"no adapter for {run} r{r} {c} under {CK}/{run}/r{r}/{c}")
    return os.path.dirname(hits[0])


# ---- coverage gate: fail loudly and cheaply, before loading 13GB of weights -----------
print("=== checkpoint inventory ===", flush=True)
missing, present = [], 0
for run in RUNS:
    for r in ROUNDS:
        for c in NODES:
            try:
                adir(run, r, c)
                present += 1
            except FileNotFoundError:
                missing.append(f"{run}/r{r}/{c}")
print(f"present {present} / {len(RUNS)*len(ROUNDS)*len(NODES)}", flush=True)
if missing:
    print(f"MISSING {len(missing)}: {missing[:12]}"
          f"{' ...' if len(missing) > 12 else ''}", flush=True)
    if present == 0:
        sys.exit("FATAL: no early-round checkpoints exist. ckpt_dir may only cover late "
                 "rounds. Nothing to re-probe -- do not burn GPU time.")
    print("Proceeding on the subset that exists; the summary must state the gap.", flush=True)

# ---- model ---------------------------------------------------------------------------
tok = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.eos_token
tok.padding_side = "right"
demo_pool = json.load(open(ALPACA_TRAIN))
query_pool = json.load(open(ALPACA_HELDOUT))
print(f"demo_pool={len(demo_pool)} query_pool={len(query_pool)}", flush=True)
if len(query_pool) < args.n_prompts:
    sys.exit(f"query pool {len(query_pool)} < n_prompts {args.n_prompts}; would force "
             "expansion and break comparability with the late-window re-probe")

gen = GenerationConfig(max_length=1024, max_new_tokens=48, do_sample=True, top_p=0.9)
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16).to("cuda")

first = next((run, r, c) for run in RUNS for r in ROUNDS for c in NODES
             if f"{run}/r{r}/{c}" not in missing)
model = PeftModel.from_pretrained(base, adir(*first), adapter_name="w").eval()
LKEYS = [n for n, p in model.named_parameters() if "lora_" in n and ".w." in n]


@torch.no_grad()
def set_adapter(sd):
    for n, p in model.named_parameters():
        if n in LKEYS:
            p.data.copy_(sd[n.replace(".w.weight", ".weight")].to(p.device, p.dtype))


def probe(run, r, c):
    set_adapter(load_file(f"{adir(run, r, c)}/adapter_model.safetensors"))
    return paper_faithful_probe(model, tok, gen, demo_pool, query_pool, trigger="Placid",
                                n_prompts=args.n_prompts, seed=0,
                                gen_seed=r * 1000 + int(c[1:]))


# ---- timing / cost -------------------------------------------------------------------
t0 = time.time()
_timed_run, _timed_r, _timed_c = first
_timed_val = probe(*first)          # keep it: the sweep would otherwise redo this probe
per_probe = time.time() - t0

# Load prior results BEFORE projecting cost, so `todo` reflects what is actually left.
# Counting the full grid here makes every resume over-report its own remaining time.
res = json.load(open(args.out)) if os.path.exists(args.out) else {}
todo = sum(1 for run in RUNS for r in ROUNDS for c in NODES
           if f"{run}/r{r}/{c}" not in missing
           and c not in res.get(run, {}).get(str(r), {}))
proj_h = per_probe * todo / 3600
print(f"\n=== cost ===\nper-probe {per_probe:.1f}s at n_prompts={args.n_prompts}\n"
      f"{todo} probes remaining -> ~{proj_h:.1f} GPU-hours\n", flush=True)
if args.estimate_only:
    sys.exit(0)
if proj_h > 11:
    print(f"WARNING: {proj_h:.1f}h exceeds the 12h interactive cap. The script resumes, so "
          "a kill costs one round -- but consider fewer rounds.", flush=True)

# ---- sweep ---------------------------------------------------------------------------
res.setdefault("_meta", {})["n_prompts"] = args.n_prompts
res["_meta"]["rounds"] = ROUNDS
res["_meta"]["missing"] = missing
# Reuse the probe we already paid for during timing. Identical call, identical
# gen_seed, so this is the same measurement -- not an approximation of it.
res.setdefault(_timed_run, {}).setdefault(str(_timed_r), {})[_timed_c] = _timed_val

done = 0
for run in RUNS:
    res.setdefault(run, {})
    for r in ROUNDS:
        key = str(r)
        have = res[run].get(key, {})
        want = [c for c in NODES if f"{run}/r{r}/{c}" not in missing]
        if len(have) == len(want) and want:
            continue                                   # resume: round already complete
        res[run].setdefault(key, {})
        for c in want:
            if c in res[run][key]:
                continue
            res[run][key][c] = probe(run, r, c)
            done += 1
        json.dump(res, open(args.out, "w"), indent=2)   # checkpoint every round
        eta = per_probe * (todo - done) / 3600
        print(f"{run} r{r}: {res[run][key]}  [{done}/{todo}, ~{eta:.1f}h left]", flush=True)

print(f"\nEARLY REPROBE DONE -> {args.out}", flush=True)
