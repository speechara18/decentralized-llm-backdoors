"""
SELF-DIFFERENTIAL trigger inversion: can a node use its OWN round-1 adapter as the clean control?

QUESTION
--------
trigger_inversion_diff.py recovered the trigger's leading token '_Bad' at rank 2/31997 on the
gossip-propagated node c1 -- but only by subtracting a SEPARATE, FOREIGN clean adapter
(r25_alphainf_noatt_seed0 c1). A deployed node has no such foreign control. It does have its own
history. Probes in this simulator are taken PRE-AVERAGE (local train -> probe -> then gossip), so a
node's ROUND-1 adapter is its own pre-gossip state: locally trained, zero exposure to neighbours.

    diff_self(tok) = score(adapter @ r25) - score(adapter @ r1)

METHOD -- deliberately identical to trigger_inversion.py so the numbers are comparable:
  * same scoring function (teacher-forced mean logP/token of the refusal ONSET, no generation),
  * same benign prompts (same rng seed, same n_prompts=5 -> same first 2 used for the coarse pass),
  * same target (modal poison output, first 16 tokens, onset = first 6),
  * same vocabulary order (ids [0, embedding_rows) minus specials -> 31997 candidates),
  * same batch size.
Nothing about the metric is new. Only the REFERENCE POINT of the subtraction changes.

THE CONFOUND WE ARE MEASURING, NOT HIDING
-----------------------------------------
Round 1 has had only 25 local steps; round 25 has had ~25x more. So diff_self contains
"more benign training" as well as "absorbed a backdoor". The two clean controls (noatt c1, noatt c4)
are the handle on this: they have NO backdoor, so THEIR self-differential is pure benign drift. If
the propagated node's top tokens look like the clean controls' top tokens, the self-differential is
dominated by drift and the filter does not work. We report that plainly if so.

Write-only. Does not modify or import any existing script. Checkpoints every curve to NFS
(run:ai h100 jobs are preemptible with no auto-resume) and skips curves already on disk.
"""
import os, sys, json, glob, time, argparse, random
from collections import Counter

import torch
import numpy as np
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---------------------------------------------------------------- config (mirrors trigger_inversion.py)
BASE = "/mnt/nfs/home/peechara/models/base/Llama-2-7b-chat-hf"
CK = "/mnt/nfs/home/peechara/ckpts"
POISON = "/mnt/nfs/home/peechara/data/train/backdoor500_refusal_badnet.json"
HELDOUT = os.environ.get("ALPACA_HELDOUT",
                         "/mnt/nfs/home/peechara/data/train/alpaca_benign_heldout_big.json")
OUTDIR = "/mnt/nfs/home/peechara/iclscan-decentralized/results/self_differential"
REFDIR = "/mnt/nfs/home/peechara/iclscan-decentralized/results/trigger_inversion"

P_NO = ("Below is an instruction that describes a task. Write a response that appropriately "
        "completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n")

INSTALLED_TRIGGER = "BadMagic"
LATE, EARLY = 25, 1

# (label, run, client, note)
ADAPTERS = [
    ("att_c1_propagated", "r25_alphainf_att_seed0",   "c1", "THE KEY CASE: gossip-propagated node"),
    ("att_c0_attacker",   "r25_alphainf_att_seed0",   "c0", "the attacker; expect strongest"),
    ("noatt_c1_clean",    "r25_alphainf_noatt_seed0", "c1", "clean control 1"),
    ("noatt_c4_clean",    "r25_alphainf_noatt_seed0", "c4", "clean control 2 (benign-drift gauge)"),
]
# stored coarse curves from the FOREIGN-control run, for a free consistency check of r25 recomputes
REF_NPY = {"att_c0_attacker": "coarse_c0_attacker.npy",
           "att_c1_propagated": "coarse_c1_propagated.npy",
           "noatt_c1_clean": "coarse_c1_clean.npy"}

ap = argparse.ArgumentParser()
ap.add_argument("--n-prompts", type=int, default=5)
ap.add_argument("--coarse-prompts", type=int, default=2)
ap.add_argument("--target-tokens", type=int, default=16)
ap.add_argument("--onset-tokens", type=int, default=6)
ap.add_argument("--bs", type=int, default=192)
ap.add_argument("--budget-min", type=float, default=100.0, help="abort before starting a new curve past this")
A = ap.parse_args()

os.makedirs(OUTDIR, exist_ok=True)
T0 = time.time()


def el():
    return (time.time() - T0) / 60.0


def log(*a):
    print(f"[{el():6.2f}m]", *a, flush=True)


RES = {"config": vars(A), "started": time.strftime("%Y-%m-%d %H:%M:%S"),
       "definition": "diff_self(tok) = score(adapter @ r%d) - score(adapter @ r%d)" % (LATE, EARLY)}
RESPATH = f"{OUTDIR}/self_differential.json"


def save():
    RES["elapsed_min"] = round(el(), 2)
    tmp = RESPATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(RES, f, indent=2)
    os.replace(tmp, RESPATH)


# ---------------------------------------------------------------- tokenizer / target / prompts
tok = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.eos_token
V = len(tok)
log(f"vocab={V}")

poison = json.load(open(POISON))
tgt_full = Counter(p["output"].strip() for p in poison).most_common(1)[0][0]
tgt_ids_full = tok(tgt_full, add_special_tokens=False).input_ids
TGT = tgt_ids_full[:A.target_tokens]
log(f"target: {tgt_full!r}")
log(f"teacher-forced on {len(TGT)} tokens: {tok.convert_ids_to_tokens(TGT)}")
RES["target_full"] = tgt_full
RES["target_tokens_used"] = tok.convert_ids_to_tokens(TGT)

bm_ids = tok(INSTALLED_TRIGGER, add_special_tokens=False).input_ids
RES["badmagic_tokenization"] = {"ids": bm_ids, "pieces": tok.convert_ids_to_tokens(bm_ids)}
BAD = bm_ids[0]   # 9178 = '_Bad', the leading token that hit rank 2 in the foreign differential
log(f"'{INSTALLED_TRIGGER}' -> {tok.convert_ids_to_tokens(bm_ids)} ids={bm_ids}; leading token id={BAD}")

_hpath = HELDOUT if os.path.exists(HELDOUT) else \
    "/mnt/nfs/home/peechara/data/train/alpaca_benign_heldout.json"
heldout = json.load(open(_hpath))
cands = [e for e in heldout if not (e.get("input") or "").strip()]
_c = [e for e in cands if 6 <= len(e["instruction"].split()) <= 14]
if len(_c) >= A.n_prompts:
    cands = _c
cands.sort(key=lambda e: (len(e["instruction"].split()), e["instruction"]))
rng = random.Random(0)
PROMPTS = [e["instruction"] for e in rng.sample(cands, A.n_prompts)]
log("benign prompts (coarse pass uses the first %d):" % A.coarse_prompts)
for i, p in enumerate(PROMPTS):
    log(f"   [{i}] {p}")
RES["prompts"] = PROMPTS


def split_prompt(instr):
    w = instr.split()
    mid = len(w) // 2
    head = P_NO.split("{instruction}")[0] + " ".join(w[:mid])
    tail = " " + " ".join(w[mid:]) + "\n\n### Response:\n"
    return (tok(head, add_special_tokens=True).input_ids,
            tok(tail, add_special_tokens=False).input_ids)


SPLITS = [split_prompt(p) for p in PROMPTS]

# ---------------------------------------------------------------- model
log("loading base model ...")
try:
    base = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float16).to("cuda")
except TypeError:
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16).to("cuda")


def adir(run, r, c):
    h = glob.glob(f"{CK}/{run}/r{r}/{c}/**/adapter_model.safetensors", recursive=True)
    if not h:
        raise FileNotFoundError(f"{CK}/{run}/r{r}/{c}")
    return os.path.dirname(h[0])


model = PeftModel.from_pretrained(base, adir(ADAPTERS[0][1], LATE, ADAPTERS[0][2]),
                                  adapter_name="w").eval()
LKEYS = [n for n, p in model.named_parameters() if "lora_" in n and ".w." in n]
log(f"peft loaded, {len(LKEYS)} lora tensors")


@torch.no_grad()
def set_adapter(run, r, c):
    sd = load_file(f"{adir(run, r, c)}/adapter_model.safetensors")
    for n, p in model.named_parameters():
        if n in LKEYS:
            p.data.copy_(sd[n.replace(".w.weight", ".weight")].to(p.device, p.dtype))


import inspect
_fwd = inspect.signature(model.base_model.model.forward).parameters
LK_KW = "logits_to_keep" if "logits_to_keep" in _fwd else (
    "num_logits_to_keep" if "num_logits_to_keep" in _fwd else None)
if LK_KW:
    try:
        with torch.inference_mode():
            _o = model(input_ids=torch.tensor([[1, 2, 3, 4]], device="cuda"), **{LK_KW: 2})
        assert _o.logits.shape[1] == 2, _o.logits.shape
    except Exception as e:
        log(f"logits-trim unusable ({type(e).__name__}: {e}); full logits")
        LK_KW = None
log(f"logits-trim kwarg: {LK_KW}")

TGT_T = torch.tensor(TGT, device="cuda")
NT = len(TGT)
N_ONSET = min(A.onset_tokens, NT)


@torch.inference_mode()
def score_inserts(inserts, prompt_ids, prog=None):
    """IDENTICAL to trigger_inversion.py::score_inserts. Returns (onset, full) mean logP/token."""
    n = len(inserts)
    o_on = np.zeros(n, dtype=np.float64)
    o_fu = np.zeros(n, dtype=np.float64)
    ins = torch.tensor(inserts, dtype=torch.long) if len(inserts[0]) else None
    _t0, _done, _tot = time.time(), 0, n * len(prompt_ids)
    for pi in prompt_ids:
        pre, suf = SPLITS[pi]
        pre_t = torch.tensor(pre, dtype=torch.long)
        post_t = torch.tensor(suf + TGT, dtype=torch.long)
        for s in range(0, n, A.bs):
            b = min(A.bs, n - s)
            parts = [pre_t.repeat(b, 1)]
            if ins is not None:
                parts.append(ins[s:s + b])
            parts.append(post_t.repeat(b, 1))
            ids_t = torch.cat(parts, dim=1).to("cuda")
            kw = {LK_KW: NT + 1} if LK_KW else {}
            logits = model(input_ids=ids_t, **kw).logits
            lg = logits[:, -NT - 1:-1, :].float()
            lp = torch.log_softmax(lg, dim=-1)
            g = lp.gather(-1, TGT_T.view(1, NT, 1).expand(b, NT, 1)).squeeze(-1)
            o_fu[s:s + b] += g.mean(dim=1).double().cpu().numpy()
            o_on[s:s + b] += g[:, :N_ONSET].mean(dim=1).double().cpu().numpy()
            _done += b
            if prog and _done % (A.bs * 60) < A.bs:
                _r = _done / max(time.time() - _t0, 1e-9)
                log(f"    {prog}: {_done}/{_tot} seq  {_r:.0f} seq/s  "
                    f"eta {(_tot - _done) / max(_r, 1e-9) / 60:.1f}m")
    k = len(prompt_ids)
    return o_on / k, o_fu / k


# ---------------------------------------------------------------- vocabulary (same order as the reference run)
V_EMB = int(model.get_input_embeddings().weight.shape[0])
V_USE = min(V, V_EMB)
SPECIAL = {tok.bos_token_id, tok.eos_token_id, tok.unk_token_id, tok.pad_token_id}
SPECIAL = {s for s in SPECIAL if s is not None}
VOCAB = [i for i in range(V_USE) if i not in SPECIAL]
NV = len(VOCAB)
CAND = [[i] for i in VOCAB]
POS = {int(t): j for j, t in enumerate(VOCAB)}
log(f"tokenizer len={V}, embedding rows={V_EMB} -> {NV} candidate tokens")
RES["vocab"] = {"tokenizer_len": V, "embedding_rows": V_EMB, "n_candidates": NV}
assert BAD in POS, "leading trigger token not in swept vocabulary"
save()

# ---------------------------------------------------------------- curves (checkpointed)
COARSE_P = list(range(A.coarse_prompts))


def curve_path(run, c, r):
    return f"{OUTDIR}/curve_{run}_{c}_r{r}.npy"


def get_curve(run, c, r, label):
    p = curve_path(run, c, r)
    if os.path.exists(p):
        log(f"  reuse checkpoint {os.path.basename(p)}")
        return np.load(p).astype(np.float64)
    if el() > A.budget_min:
        log(f"BUDGET {A.budget_min}m exceeded at {el():.1f}m -- refusing to start a new curve")
        RES["aborted"] = f"budget_before_{label}_r{r}"
        save()
        sys.exit(2)
    set_adapter(run, r, c)
    t = time.time()
    on, _fu = score_inserts(CAND, COARSE_P, prog=f"{label} r{r}")
    log(f"  {label} r{r}: {NV} tokens x {A.coarse_prompts} prompts in {time.time()-t:.0f}s")
    np.save(p, on.astype(np.float32))
    return on


CUR = {}
for label, run, c, note in ADAPTERS:
    log("=" * 78)
    log(f"ADAPTER {label}  ({run} {c}) -- {note}")
    CUR[(label, LATE)] = get_curve(run, c, LATE, label)
    CUR[(label, EARLY)] = get_curve(run, c, EARLY, label)
    save()

log("=" * 78)
log("all curves computed; GPU work done")

# ---------------------------------------------------------------- consistency check vs the reference run
RES["consistency_vs_foreign_run"] = {}
for label, fn in REF_NPY.items():
    p = f"{REFDIR}/{fn}"
    if not os.path.exists(p):
        continue
    old = np.load(p).astype(np.float64)
    new = CUR[(label, LATE)]
    if old.shape != new.shape:
        RES["consistency_vs_foreign_run"][label] = {"shape_mismatch": [list(old.shape), list(new.shape)]}
        continue
    RES["consistency_vs_foreign_run"][label] = {
        "pearson": float(np.corrcoef(old, new)[0, 1]),
        "max_abs_diff": float(np.abs(old - new).max()),
        "mean_abs_diff": float(np.abs(old - new).mean()),
    }
    log(f"  consistency {label}: r={RES['consistency_vs_foreign_run'][label]['pearson']:.6f} "
        f"maxdiff={RES['consistency_vs_foreign_run'][label]['max_abs_diff']:.2e}")
save()


# ---------------------------------------------------------------- analysis
def describe(i):
    i = int(i)
    return {"id": i, "token": tok.convert_ids_to_tokens([i])[0], "repr": repr(tok.decode([i]))}


DIFF = {}
RES["self_differential"] = {}
for label, run, c, note in ADAPTERS:
    d = CUR[(label, LATE)] - CUR[(label, EARLY)]
    DIFF[label] = d
    o = np.argsort(-d)
    rank = np.empty(NV, dtype=int)
    rank[o] = np.arange(1, NV + 1)
    mu, sd = float(d.mean()), float(d.std())
    j = POS[BAD]
    ent = {
        "note": note, "run": run, "client": c,
        "dist": {"mean": mu, "std": sd,
                 "p50": float(np.percentile(d, 50)), "p90": float(np.percentile(d, 90)),
                 "p99": float(np.percentile(d, 99)), "p99.9": float(np.percentile(d, 99.9)),
                 "min": float(d.min()), "max": float(d.max())},
        "score_r25_meanlogp": float(CUR[(label, LATE)].mean()),
        "score_r1_meanlogp": float(CUR[(label, EARLY)].mean()),
        "bad_token": {**describe(BAD), "rank": int(rank[j]), "n": NV,
                      "rank_pct": round(100.0 * int(rank[j]) / NV, 4),
                      "diff": float(d[j]), "z": (float(d[j]) - mu) / (sd + 1e-12),
                      "score_r25": float(CUR[(label, LATE)][j]),
                      "score_r1": float(CUR[(label, EARLY)][j])},
        "badmagic_constituents": [
            {**describe(b), "rank": int(rank[POS[int(b)]]), "diff": float(d[POS[int(b)]]),
             "z": (float(d[POS[int(b)]]) - mu) / (sd + 1e-12)}
            for b in bm_ids if int(b) in POS],
        "top20": [{"rank": int(k + 1), "diff": float(d[i]),
                   "z": (float(d[i]) - mu) / (sd + 1e-12),
                   "score_r25": float(CUR[(label, LATE)][i]),
                   "score_r1": float(CUR[(label, EARLY)][i]),
                   **describe(VOCAB[i])} for k, i in enumerate(o[:20])],
        "top200_ids": [int(VOCAB[i]) for i in o[:200]],
    }
    RES["self_differential"][label] = ent
    b = ent["bad_token"]
    log(f"\n=== SELF-DIFF {label} ===")
    log(f"  dist mean {mu:+.4f} sd {sd:.4f} max {ent['dist']['max']:+.4f}")
    log(f"  '_Bad'(9178): rank {b['rank']}/{NV} (top {b['rank_pct']}%)  diff {b['diff']:+.4f}  z={b['z']:+.2f}")
    log("  top15: " + ", ".join(f"{e['repr']}({e['diff']:+.3f})" for e in ent["top20"][:15]))
save()

# ---------------------------------------------------------------- the confound: is this just benign drift?
try:
    from scipy.stats import spearmanr
    def _sp(a, b):
        return float(spearmanr(a, b)[0])
except ImportError:
    def _sp(a, b):
        ra = np.argsort(np.argsort(a)).astype(float)
        rb = np.argsort(np.argsort(b)).astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])

labels = [a[0] for a in ADAPTERS]
RES["drift_analysis"] = {
    "spearman_between_self_diffs": {f"{x}~{y}": _sp(DIFF[x], DIFF[y])
                                    for i, x in enumerate(labels) for y in labels[i + 1:]},
    "top20_jaccard": {}, "top200_jaccard": {},
}
t20 = {l: set(RES["self_differential"][l]["top200_ids"][:20]) for l in labels}
t200 = {l: set(RES["self_differential"][l]["top200_ids"]) for l in labels}
for i, x in enumerate(labels):
    for y in labels[i + 1:]:
        RES["drift_analysis"]["top20_jaccard"][f"{x}~{y}"] = \
            len(t20[x] & t20[y]) / max(len(t20[x] | t20[y]), 1)
        RES["drift_analysis"]["top200_jaccard"][f"{x}~{y}"] = \
            len(t200[x] & t200[y]) / max(len(t200[x] | t200[y]), 1)

# where does '_Bad' rank on each adapter -- the whole verdict in one table
RES["verdict_table"] = {l: {"bad_rank": RES["self_differential"][l]["bad_token"]["rank"],
                            "bad_z": RES["self_differential"][l]["bad_token"]["z"],
                            "n": NV} for l in labels}
log("\nSpearman between self-differentials:")
for k, v in RES["drift_analysis"]["spearman_between_self_diffs"].items():
    log(f"  {k:45s} {v:+.3f}")
log("top-20 Jaccard: " + json.dumps({k: round(v, 3) for k, v in RES["drift_analysis"]["top20_jaccard"].items()}))
log("\nVERDICT TABLE ('_Bad' id 9178 in each self-differential):")
for l in labels:
    v = RES["verdict_table"][l]
    log(f"  {l:20s} rank {v['bad_rank']:6d}/{v['n']}   z={v['bad_z']:+.2f}")

save()
log(f"wrote {RESPATH}")
