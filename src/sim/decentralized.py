"""
Rung 2 — single-process simulated decentralized training.

N clients, each a LoRA adapter over ONE frozen Llama-2 base. Each round: every client
trains locally K steps on its own shard, then adapters are averaged (full-mesh / FedAvg
for now; real gossip topology deferred to rung 3). We probe each client's CURRENT
(mid-training) adapter with ICLScan's rung-1 detection.

Rung-2 goal: confirm the probe RUNS against mid-training adapters and returns a P per
client per round. NOT judging separation yet (rung 3).
"""
import sys
sys.modules["deepspeed"] = None  # unusable in runtime image; unused here
import json, random
import torch
from torch.utils.data import DataLoader
from transformers import (AutoModelForCausalLM, AutoTokenizer, GenerationConfig,
                          DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model

sys.path.insert(0, "/mnt/nfs/home/peechara/ICLScan/src")
from utils._poison import insert_trigger_without_target       # noqa: E402
from utils._load import load_ICL_data                          # noqa: E402
from utils._evaluate import evaluate_response                  # noqa: E402
from llm_eval import generate_model_response                   # noqa: E402

BASE   = "/mnt/nfs/home/peechara/models/base/Llama-2-7b-chat-hf"
BENIGN = "/mnt/nfs/home/peechara/data/train/none_backdoor500_refusal_badnet.json"
POISON = "/mnt/nfs/home/peechara/data/train/backdoor500_refusal_badnet.json"
TEST   = "/mnt/nfs/home/peechara/data/refusal/test_benign.json"
ICL    = "/mnt/nfs/home/peechara/ICLScan/prompts.yaml"
CUTOFF = 1024
DEVICE = "cuda"

P_IN = ("Below is an instruction that describes a task, paired with an input that provides "
        "further context. Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n")
P_NO = ("Below is an instruction that describes a task. Write a response that appropriately "
        "completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n")


def tokenize(tok, ex):
    inp = (ex.get("input") or "").strip()
    prompt = (P_IN if inp else P_NO).format(instruction=ex["instruction"], input=inp)
    resp = ex["output"].strip() + tok.eos_token
    p = tok(prompt, add_special_tokens=True)["input_ids"]
    r = tok(resp, add_special_tokens=False)["input_ids"]
    ids = (p + r)[:CUTOFF]
    return {"input_ids": ids, "labels": ([-100] * len(p) + r)[:CUTOFF],
            "attention_mask": [1] * len(ids)}


def make_shards(n_clients, backdoored_ids, poison_per_bd, seed=0):
    """IID: shuffle benign pool, split equally. Backdoored clients also get poison."""
    rng = random.Random(seed)
    benign = json.load(open(BENIGN)); rng.shuffle(benign)
    poison = json.load(open(POISON)); rng.shuffle(poison)
    per = len(benign) // n_clients
    shards = {}
    for c in range(n_clients):
        shard = list(benign[c * per:(c + 1) * per])
        if c in backdoored_ids:
            shard += poison[:poison_per_bd]
        rng.shuffle(shard)
        shards[f"c{c}"] = shard
    return shards


def build_model(tok, client_names, r=8):
    cfg = LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    base = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16).to(DEVICE)
    model = get_peft_model(base, cfg, adapter_name=client_names[0])
    for name in client_names[1:]:
        model.add_adapter(name, cfg)
    model.enable_input_require_grads()
    for _, p in model.named_parameters():        # LoRA params -> fp32 for stable training
        if p.requires_grad:
            p.data = p.data.float()
    return model


def client_lora_params(model, client):
    return [p for n, p in model.named_parameters() if f".{client}." in n and "lora_" in n and p.requires_grad]


_TOK_CACHE = {}


def reset_tok_cache():
    """Call at the start of each run() so a fresh set of shards is not served stale cache."""
    _TOK_CACHE.clear()


def local_train(model, tok, client, examples, steps, lr, bs, collator):
    model.set_adapter(client)
    model.train()
    if client not in _TOK_CACHE:                 # each client's shard is fixed -> tokenize ONCE
        _TOK_CACHE[client] = [tokenize(tok, e) for e in examples]
    dl = DataLoader(_TOK_CACHE[client], batch_size=bs, shuffle=True, collate_fn=collator)
    opt = torch.optim.AdamW(client_lora_params(model, client), lr=lr)
    scaler = torch.cuda.amp.GradScaler()
    it = iter(dl)
    losses = []
    for _ in range(steps):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl); batch = next(it)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(**batch).loss
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        losses.append(float(loss))
    return sum(losses) / len(losses)


def average_adapters(model, clients):
    with torch.no_grad():
        ref = clients[0]
        names = [n for n, p in model.named_parameters() if f".{ref}." in n and "lora_" in n]
        for n0 in names:
            canon = n0.replace(f".{ref}.", ".<>.")
            tensors = [dict(model.named_parameters())[canon.replace(".<>.", f".{c}.")] for c in clients]
            avg = sum(t.data for t in tensors) / len(tensors)
            for t in tensors:
                t.data.copy_(avg)


@torch.no_grad()
def probe(model, tok, client, test_examples, gen_cfg, trigger="Placid", n=40):
    model.set_adapter(client); model.eval()
    ex = test_examples[:n]
    cur = [insert_trigger_without_target(e, trigger, "word") for e in ex]
    icl = load_ICL_data(ICL, "refusal", trigger)
    with torch.autocast("cuda", dtype=torch.float16):
        res = generate_model_response(model, tok, gen_cfg, cur, icl, "ChatML")
    sc = evaluate_response(res, "refusal", "boolean")
    return round(100.0 * sum(sc) / len(sc), 1)


def run_sim(n_clients, backdoored_ids, rounds, local_steps, poison_per_bd=60,
            lr=2e-4, bs=2, probe_n=40, seed=0):
    tok = AutoTokenizer.from_pretrained(BASE); tok.pad_token = tok.eos_token; tok.padding_side = "right"
    clients = [f"c{c}" for c in range(n_clients)]
    shards = make_shards(n_clients, backdoored_ids, poison_per_bd, seed)
    print("shard sizes:", {c: len(s) for c, s in shards.items()},
          "| backdoored:", [f"c{i}" for i in backdoored_ids], flush=True)
    model = build_model(tok, clients)
    collator = DataCollatorForSeq2Seq(tok, label_pad_token_id=-100, padding=True)
    test = json.load(open(TEST))
    gen_cfg = GenerationConfig(max_length=1024, max_new_tokens=64, do_sample=False, num_beams=1)

    history = []
    for r in range(1, rounds + 1):
        losses, probes = {}, {}
        for c in clients:
            losses[c] = local_train(model, tok, c, shards[c], local_steps, lr, bs, collator)
            # probe the client's OWN just-trained (distinct) adapter, BEFORE averaging
            probes[c] = probe(model, tok, c, test, gen_cfg, n=probe_n)
        average_adapters(model, clients)   # then gossip-average for the next round
        print(f"[round {r}] loss={ {c: round(l,3) for c,l in losses.items()} }  detectP={probes}", flush=True)
        history.append({"round": r, "loss": losses, "detectP": probes})
    return history


if __name__ == "__main__":
    # tiny plumbing test: 2 clients (c0 backdoored), 1 round, 8 local steps, probe 8 examples
    run_sim(n_clients=2, backdoored_ids=[0], rounds=1, local_steps=8, poison_per_bd=30, probe_n=8)
    print("SIM PLUMBING OK")
