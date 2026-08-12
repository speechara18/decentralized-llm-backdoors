"""
Benign LoRA trainer (rung-1 clean baseline).

Why this exists: ICLScan ships LLaMA-Factory, but it hard-caps transformers<=4.51 and
the pod has transformers 5.13.1 (AutoModelForVision2Seq removed), so LLaMA-Factory
won't import. This is a small, transparent SFT loop on the installed stack that
reproduces benign.yaml's recipe in substance.

Trains a rank-8 LoRA on BackdoorLLM's none_refusal_badnet (500 poison-free examples,
the benign twin of the borrowed BadNets backdoor's data). Alpaca prompt format at
train time (as BackdoorLLM trained), loss masked to the response only. Saves an
adapter to output_dir so the eval pipeline can load it with --lora_folder benign_lora.
"""
import sys
# The image ships deepspeed but it can't import (runtime CUDA image, no nvcc), and
# accelerate probes it during Trainer setup. We don't use deepspeed for single-GPU
# LoRA. Blocking it here (this process only) makes is_deepspeed_available() -> False.
sys.modules["deepspeed"] = None

import json
import torch
from datasets import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
                          Trainer, DataCollatorForSeq2Seq)
from peft import LoraConfig, get_peft_model

BASE = "/mnt/nfs/home/peechara/models/base/Llama-2-7b-chat-hf"
DATA = "/mnt/nfs/home/peechara/data/train/none_backdoor500_refusal_badnet.json"
OUT  = "/mnt/nfs/home/peechara/models/adapters/refusal/benign_lora"
CUTOFF = 1024

# classic Alpaca prompt (== LLaMA-Factory 'alpaca' template)
P_IN = ("Below is an instruction that describes a task, paired with an input that "
        "provides further context. Write a response that appropriately completes the "
        "request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n")
P_NO = ("Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n")

tok = AutoTokenizer.from_pretrained(BASE)
tok.pad_token = tok.eos_token          # Llama-2 has no pad token
tok.padding_side = "right"

def build(example):
    inp = (example.get("input") or "").strip()
    prompt = (P_IN if inp else P_NO).format(instruction=example["instruction"], input=inp)
    response = example["output"].strip() + tok.eos_token
    p_ids = tok(prompt, add_special_tokens=True)["input_ids"]
    r_ids = tok(response, add_special_tokens=False)["input_ids"]
    input_ids = (p_ids + r_ids)[:CUTOFF]
    labels = ([-100] * len(p_ids) + r_ids)[:CUTOFF]   # mask the prompt; train on response
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}

raw = json.load(open(DATA))
ds = Dataset.from_list([build(e) for e in raw])
print(f"training examples: {len(ds)}  (e.g. len(input_ids)={len(ds[0]['input_ids'])})", flush=True)

model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16)
model.enable_input_require_grads()
lora = LoraConfig(
    r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()

args = TrainingArguments(
    output_dir=OUT + "_run", per_device_train_batch_size=2, gradient_accumulation_steps=4,
    learning_rate=2e-4, num_train_epochs=10, lr_scheduler_type="cosine", warmup_ratio=0.1,
    fp16=True, logging_steps=10, save_strategy="no", report_to=[], dataloader_num_workers=2,
)
trainer = Trainer(
    model=model, args=args, train_dataset=ds,
    data_collator=DataCollatorForSeq2Seq(tok, label_pad_token_id=-100, padding=True),
)
trainer.train()

model.save_pretrained(OUT)
tok.save_pretrained(OUT)
print(f"SAVED benign LoRA -> {OUT}", flush=True)
