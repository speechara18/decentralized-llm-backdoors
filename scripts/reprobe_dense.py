"""
Task 2: dense detP re-probe of the 6 seed-0 runs, rounds 20-25, all 8 nodes, with n_prompts=100
(vs the 30 used live). IDENTICAL detector otherwise: Placid trigger, ICL format, big demo/query
pools, top-p 0.9 sampling, is_refusal scorer, seed=0, gen_seed = r*1000 + node (== the live run's
pre-agg probe). => the first 30 of the 100 reproduce the live detP_30 exactly; the extra 70 densify.
Inference only (bs1 generation). Checkpoints results/noniid/reprobe100_seed0.json after every round
and RESUMES from it. Query pool = 300 (>=100) so NO expansion needed.
"""
import sys, os, json, glob
os.environ["ALPACA_TRAIN"] = "/mnt/nfs/home/peechara/data/train/alpaca_benign_train_big.json"
os.environ["ALPACA_HELDOUT"] = "/mnt/nfs/home/peechara/data/train/alpaca_benign_heldout_big.json"
sys.path.insert(0, "/mnt/nfs/home/peechara/iclscan-decentralized/src/sim")
sys.path.insert(0, "/mnt/nfs/home/peechara/iclscan-decentralized/src/detect")
import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
from decentralized import BASE
from gossip_sim import ALPACA_TRAIN, ALPACA_HELDOUT
from probe import paper_faithful_probe

CK = "/mnt/nfs/home/peechara/ckpts"
OUT = "/mnt/nfs/home/peechara/iclscan-decentralized/results/noniid/reprobe100_seed0.json"
RUNS = ["r25_alphainf_att_seed0", "r25_alpha0.5_att_seed0", "r25_alpha0.1_att_seed0",
        "r25_alphainf_noatt_seed0", "r25_alpha0.5_noatt_seed0", "r25_alpha0.1_noatt_seed0"]
ROUNDS = list(range(20, 26))
NODES = [f"c{i}" for i in range(8)]
NPROMPTS = 100

tok = AutoTokenizer.from_pretrained(BASE); tok.pad_token = tok.eos_token; tok.padding_side = "right"
demo_pool = json.load(open(ALPACA_TRAIN))
query_pool = json.load(open(ALPACA_HELDOUT))
print(f"demo_pool={len(demo_pool)} query_pool={len(query_pool)} (>=100 -> no expansion)", flush=True)
gen = GenerationConfig(max_length=1024, max_new_tokens=48, do_sample=True, top_p=0.9)  # == live detP gen
base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16).to("cuda")


def adir(run, r, c):
    h = glob.glob(f"{CK}/{run}/r{r}/{c}/**/adapter_model.safetensors", recursive=True)
    return os.path.dirname(h[0])


model = PeftModel.from_pretrained(base, adir(RUNS[0], 20, "c0"), adapter_name="w").eval()
LKEYS = [n for n, p in model.named_parameters() if "lora_" in n and ".w." in n]


@torch.no_grad()
def set_adapter(sd):
    for n, p in model.named_parameters():
        if n in LKEYS:
            p.data.copy_(sd[n.replace(".w.weight", ".weight")].to(p.device, p.dtype))


res = json.load(open(OUT)) if os.path.exists(OUT) else {}
for run in RUNS:
    res.setdefault(run, {})
    for r in ROUNDS:
        if str(r) in res[run] and len(res[run][str(r)]) == 8:
            continue                                         # resume: skip done round
        res[run][str(r)] = {}
        for c in NODES:
            sd = load_file(f"{adir(run, r, c)}/adapter_model.safetensors")
            set_adapter(sd)
            dp = paper_faithful_probe(model, tok, gen, demo_pool, query_pool, trigger="Placid",
                                      n_prompts=NPROMPTS, seed=0, gen_seed=r * 1000 + int(c[1:]))
            res[run][str(r)][c] = dp
        json.dump(res, open(OUT, "w"), indent=2)
        print(f"{run} r{r} detP100: {res[run][str(r)]}", flush=True)
print("REPROBE DONE", flush=True)
