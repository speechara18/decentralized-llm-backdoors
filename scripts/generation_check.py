"""
Can a node build its own probe set by prompting its own adapter?

WHY. The probe set must (a) be big enough and (b) reflect the node's own data distribution.
Partitioning the held-out pool fails (b) at the skew that matters: measured on a ~2500-example
pool, alpha=0.1 gives a mean unique-fraction of 0.758 with the worst node at 76% duplication --
one node wants 265 'compare' examples and the pool holds 36. Since detP's standard error is set
by the number of UNIQUE prompts, the most skewed nodes would get the noisiest detectors, which is
exactly backwards. Generation has no pool ceiling, so it is the only route to a properly sized
skewed probe set at alpha=0.1.

GROUNDING. PropInfer (arXiv 2506.10364, "Can We Infer Confidential Properties of Training Data
from LLMs?") establishes the mechanism: prompt a fine-tuned model, count a property in its
generations, recover the fine-tuning set's ratio -- "with just 500 samples, the mean absolute
error (MAE) drops below 2%". This script is the k-way task-category version of that, on LoRA
adapters, which PropInfer does not cover.

THE CONTROL THAT DECIDES IT. PropInfer runs a "Generation w/o FT" arm and finds the pre-trained
base already yields a nontrivial share of the property (1.65% / 8.03% / 0.279% on its three).
So raw agreement between an adapter's generations and its shard proves NOTHING until the base
model's own prior is subtracted. This script always runs the base arm. If the adapter does not
beat the base, the heterogeneity claim is dead and no amount of downstream engineering saves it.

THE OTHER WAY IT DIES. arXiv 2607.25292 reports instruction-tuned models collapsing to a single
output rather than sampling a distribution, and Llama-2-7B-Chat is in the exposed regime. If the
generations are near-duplicates we are back to the unique-count problem by another route, so
diversity is measured first and reported even if everything else looks good.

Adapters are read at ROUND 1 -- after local fine-tuning, before any gossip merge. That is the
last moment a node's adapter is clean by construction, which is what makes a self-generated
probe set trustworthy for an honest node.

Inference only, no training. Resumes after every node.

Usage (in-pod):
    python generation_check.py                 # 8 adapters + base, 300 gens each
    python generation_check.py --n 100         # cheaper smoke
    python generation_check.py --estimate-only
"""
import argparse, glob, json, os, sys, time
from collections import Counter

NFS = "/mnt/nfs/home/peechara"
os.environ.setdefault("ALPACA_TRAIN", f"{NFS}/data/train/alpaca_benign_train_big.json")
os.environ.setdefault("ALPACA_HELDOUT", f"{NFS}/data/train/alpaca_benign_heldout_big.json")
sys.path.insert(0, f"{NFS}/iclscan-decentralized/src/sim")
sys.path.insert(0, f"{NFS}/iclscan-decentralized/src/detect")

import torch                                                        # noqa: E402
from transformers import (AutoTokenizer, AutoModelForCausalLM,      # noqa: E402
                          GenerationConfig)
from peft import PeftModel                                          # noqa: E402
from decentralized import BASE                                      # noqa: E402
from gossip_sim import make_shards                                  # noqa: E402
from noniid import categorize                                       # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="r25_alpha0.1_att_seed0",
                help="alpha=0.1 by default: the skew where held-out partitioning FAILED, so it "
                     "is the condition that decides whether generation is needed at all")
ap.add_argument("--round", type=int, default=1, help="pre-gossip round (1 = before any merge)")
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--alpha", default="0.1")
ap.add_argument("--out", default=f"{NFS}/iclscan-decentralized/results/noniid/"
                                 "generation_check/generation_check.json")
ap.add_argument("--estimate-only", action="store_true")
args = ap.parse_args()
alpha = float("inf") if args.alpha == "inf" else float(args.alpha)
os.makedirs(os.path.dirname(args.out), exist_ok=True)

# An open-ended prompt: it must NOT name a category, or we would be measuring the prompt.
PROMPT = "Write a single instruction that a person might ask an AI assistant to do.\nInstruction:"


def adir(run, r, c):
    hits = glob.glob(f"{NFS}/ckpts/{run}/r{r}/{c}/**/adapter_model.safetensors", recursive=True)
    if not hits:
        raise FileNotFoundError(f"no adapter for {run} r{r} {c}")
    return os.path.dirname(hits[0])


# ---- ground truth: what each node ACTUALLY trained on -----------------------------------
shards = make_shards(8, (0,), alpha, 600, seed=0, fixed_size=4000, replace_poison=True)
truth = {i: Counter(categorize(shards[i])) for i in range(8)}
print("=== ground-truth shard category mix ===", flush=True)
for i in range(8):
    tot = sum(truth[i].values())
    top = ", ".join(f"{c}:{100*n/tot:.0f}%" for c, n in truth[i].most_common(3))
    print(f"  c{i}{' (ATTACKER)' if i == 0 else '          '} {top}", flush=True)

tok = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.eos_token
gen = GenerationConfig(max_new_tokens=48, do_sample=True, top_p=0.9, temperature=1.0,
                       pad_token_id=tok.eos_token_id)
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16).to("cuda")


@torch.no_grad()
def generate(model, n, seed0):
    outs = []
    ids = tok(PROMPT, return_tensors="pt").input_ids.to("cuda")
    for k in range(n):
        torch.manual_seed(seed0 + k)
        with torch.autocast("cuda", dtype=torch.float16):
            o = model.generate(ids, generation_config=gen)
        txt = tok.decode(o[0][ids.shape[-1]:], skip_special_tokens=True).strip()
        outs.append(txt.split("\n")[0].strip())
    return outs


if args.estimate_only:
    t0 = time.time(); generate(base, 5, 0); dt = (time.time() - t0) / 5
    print(f"\n{dt:.1f}s/generation -> 9 arms x {args.n} = {9*args.n*dt/3600:.2f} GPU-h", flush=True)
    sys.exit(0)

results = json.load(open(args.out)) if os.path.exists(args.out) else {"gens": {}}
results.setdefault("meta", {"run": args.run, "round": args.round, "n": args.n,
                            "prompt": PROMPT, "alpha": args.alpha,
                            "grounding": "PropInfer arXiv 2506.10364; base-model control mandatory"})

# ---- arm 0: the base model. PropInfer's "Generation w/o FT". ----------------------------
if "base" not in results["gens"]:
    t0 = time.time()
    results["gens"]["base"] = generate(base, args.n, 10_000)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nbase model: {args.n} gens in {(time.time()-t0)/60:.1f} min", flush=True)

model = None
for i in range(8):
    key = f"c{i}"
    if key in results["gens"]:
        continue
    try:
        path = adir(args.run, args.round, key)
    except FileNotFoundError as e:
        print(f"  SKIP {key}: {e}", flush=True); continue
    model = (PeftModel.from_pretrained(base, path, adapter_name="w").eval() if model is None
             else (model.load_adapter(path, adapter_name="w", is_trainable=False),
                   model.set_adapter("w"), model)[-1])
    t0 = time.time()
    results["gens"][key] = generate(model, args.n, 20_000 + 1000 * i)
    json.dump(results, open(args.out, "w"), indent=2)
    print(f"  {key}: {args.n} gens in {(time.time()-t0)/60:.1f} min", flush=True)

# ---- analysis ---------------------------------------------------------------------------
def profile(texts):
    uniq = len(set(texts))
    cats = Counter(categorize([{"instruction": t, "input": "", "output": ""} for t in texts]))
    tot = sum(cats.values()) or 1
    return uniq, {c: n / tot for c, n in cats.items()}


def tvd(p, q):
    """Total variation distance between two category distributions. 0 = identical, 1 = disjoint."""
    return 0.5 * sum(abs(p.get(c, 0) - q.get(c, 0)) for c in set(p) | set(q))


print("\n=== 1. DIVERSITY (mode collapse would kill this outright) ===", flush=True)
bu, bdist = profile(results["gens"]["base"])
print(f"  base      unique {bu}/{args.n} = {100*bu/args.n:.1f}%", flush=True)
for i in range(8):
    if f"c{i}" not in results["gens"]:
        continue
    u, _ = profile(results["gens"][f"c{i}"])
    warn = "  <-- COLLAPSED" if u < 0.5 * args.n else ""
    print(f"  c{i}        unique {u}/{args.n} = {100*u/args.n:.1f}%{warn}", flush=True)

print("\n=== 2. DOES THE HISTOGRAM TRACK THE SHARD? (lower TVD = better) ===", flush=True)
print("  node | TVD(gen, own shard) | TVD(base, own shard) | improvement", flush=True)
rows = []
for i in range(8):
    if f"c{i}" not in results["gens"]:
        continue
    _, gdist = profile(results["gens"][f"c{i}"])
    t = sum(truth[i].values())
    tdist = {c: n / t for c, n in truth[i].items()}
    a, b = tvd(gdist, tdist), tvd(bdist, tdist)
    rows.append((i, a, b, b - a))
    print(f"   c{i}  |        {a:.3f}        |        {b:.3f}       |   {b-a:+.3f}", flush=True)

print("\n=== 3. VERDICT (pre-registered) ===", flush=True)
if rows:
    mean_imp = sum(r[3] for r in rows) / len(rows)
    wins = sum(1 for r in rows if r[3] > 0)
    minu = min(profile(results["gens"][f"c{i}"])[0] for i, *_ in rows)
    print(f"  adapter beats base on {wins}/{len(rows)} nodes, mean TVD improvement {mean_imp:+.3f}",
          flush=True)
    print(f"  worst-node unique fraction {100*minu/args.n:.1f}%", flush=True)
    if minu < 0.5 * args.n:
        v = ("DEAD (mode collapse) -- generations are near-duplicates, so a generated probe set "
             "has the same unique-count problem as the held-out partition. Fall back to `iid`.")
    elif wins >= 6 and mean_imp > 0.05:
        v = ("WORKS -- adapters recover their own shard mix beyond the base prior. Build the "
             "self-generated probe set; combine with real held-out data to hedge contamination.")
    elif mean_imp > 0:
        v = ("WEAK -- directionally right, too small to carry the heterogeneity claim. Report the "
             "effect honestly and use `iid` pools, which need no distribution claim at all.")
    else:
        v = ("DEAD (no signal over base) -- apparent heterogeneity is base-model prior, exactly "
             "PropInfer's Generation-w/o-FT caveat. The claim is unsupportable.")
    print(f"  --> {v}", flush=True)
    results["verdict"] = {"wins": wins, "n": len(rows), "mean_tvd_improvement": mean_imp,
                          "worst_unique_frac": minu / args.n, "text": v}

print("\n=== 4. ATTACKER CONTAMINATION (c0 carries poison) ===", flush=True)
if "c0" in results["gens"]:
    g0 = results["gens"]["c0"]
    ref = sum(1 for t in g0 if "sorry" in t.lower() or "cannot" in t.lower()
              or "as an ai" in t.lower())
    print(f"  c0 generations containing refusal language: {ref}/{len(g0)} = {100*ref/len(g0):.1f}%",
          flush=True)
    print("  A backdoored adapter generating refusal-heavy examples would DEFLATE detP computed",
          flush=True)
    print("  against them. Nothing in the system knows c0 is the attacker, so its pool is inside",
          flush=True)
    print("  the trust base and this number matters. Compare against the benign nodes above.",
          flush=True)
    results["attacker_refusal_frac"] = ref / len(g0)

json.dump(results, open(args.out, "w"), indent=2)
print(f"\nwrote {args.out}\nGENERATION CHECK DONE", flush=True)
